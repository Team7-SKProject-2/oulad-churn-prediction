"""최근 4주 학습행동으로 다음 주 중도이탈을 예측하는 GRU 모델.

CatBoost와 동일하게 학생 단위 3-Fold OOF 평가를 수행한다. 초기 1~3주는
제거하지 않고 Sequence 뒤쪽을 0으로 Padding한 뒤 실제 마지막 관찰 시점의
GRU 출력만 사용한다.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


DL_DIR = Path(__file__).resolve().parent
MODELS_DIR = DL_DIR.parent
PROJECT_ROOT = MODELS_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

try:
    from common_weekly_metrics import (
        calculate_metrics,
        fold_metadata,
        make_group_folds,
        validate_oof,
    )
except ModuleNotFoundError:
    from models.common_weekly_metrics import (
        calculate_metrics,
        fold_metadata,
        make_group_folds,
        validate_oof,
    )


DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "ML"
    / "used_data"
    / "weekly_next_week_with_vle_enhanced.csv"
)
OUTPUT_DIR = DL_DIR

TARGET_COL = "target_next_week_withdrawn"
ID_COL = "id_student"
SEQUENCE_KEYS = ["code_module", "code_presentation", ID_COL]
SORT_COLS = [*SEQUENCE_KEYS, "prediction_week"]

# GRU는 주간 흐름에 필요한 핵심 행동 Feature만 사용한다.
# vle_cum_unique_sites는 대규모 비정상 결측 때문에 제외한다.
DYNAMIC_FEATURES = [
    "current_total_clicks",
    "current_active_days",
    "current_unique_sites",
    "current_activity_type_count",
    "current_forumng_clicks",
    "current_oucontent_clicks",
    "current_quiz_clicks",
    "current_resource_clicks",
    "current_other_clicks",
    "current_no_activity",
    "weeks_since_last_activity",
]

# 0/1 지표를 제외한 비음수 카운트 Feature는 긴 꼬리를 완화한다.
LOG1P_FEATURES = [
    feature for feature in DYNAMIC_FEATURES if feature != "current_no_activity"
]

SEQUENCE_LENGTH = 4
N_SPLITS = 3
RANDOM_STATE = 42
TOP_FRACTION = 0.20

BATCH_SIZE = 4096
MAX_EPOCHS = 12
PATIENCE = 3
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 32
DROPOUT_RATE = 0.20
GRADIENT_CLIP_NORM = 1.0

EXPECTED_ROWS = 895_005
EXPECTED_POSITIVES = 6_672


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="주간 Demo1 학습 CSV 경로",
    )
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="앞부분 50,000행으로 Sequence와 1개 Batch 학습만 검증",
    )
    return parser.parse_args()


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_weekly_data(
    data_path: Path,
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """필요한 컬럼만 읽고 복합키·Target·주차 연속성을 검증한다."""

    if not data_path.is_file():
        raise FileNotFoundError(f"학습 데이터가 없습니다: {data_path}")

    required_cols = list(dict.fromkeys([*SORT_COLS, TARGET_COL, *DYNAMIC_FEATURES]))
    dtype_map: dict[str, str] = {
        ID_COL: "int64",
        "prediction_week": "int16",
        TARGET_COL: "int8",
    }
    dtype_map.update({feature: "float32" for feature in DYNAMIC_FEATURES})

    print("\n1. GRU 입력 데이터 불러오기")
    data = pd.read_csv(
        data_path,
        usecols=required_cols,
        dtype=dtype_map,
        nrows=max_rows,
        low_memory=False,
    )
    data = data.sort_values(SORT_COLS, kind="mergesort", ignore_index=True)

    if max_rows is None and len(data) != EXPECTED_ROWS:
        raise ValueError(f"행 수가 다릅니다: {len(data):,}")

    duplicate_count = int(data.duplicated(SORT_COLS).sum())
    if duplicate_count:
        raise ValueError(f"복합키 중복이 있습니다: {duplicate_count:,}건")

    target_count = int(data[TARGET_COL].sum())
    if max_rows is None and target_count != EXPECTED_POSITIVES:
        raise ValueError(f"Target 양성 수가 다릅니다: {target_count:,}")

    numeric_values = data[DYNAMIC_FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(numeric_values).all():
        raise ValueError("GRU 입력 Feature에 NaN 또는 무한값이 있습니다.")
    if (numeric_values < 0).any():
        raise ValueError("log1p 적용 대상에 음수 Feature가 있습니다.")

    grouped = data.groupby(SEQUENCE_KEYS, observed=True, sort=False)
    week_gap = grouped["prediction_week"].diff()
    invalid_week_gap = int((week_gap.notna() & week_gap.ne(1)).sum())
    if invalid_week_gap:
        raise ValueError(
            "학생-강좌의 주차가 연속적이지 않은 행이 있습니다: "
            f"{invalid_week_gap:,}건"
        )

    # 원본은 유지하되 GRU용 값만 log1p 변환한다.
    data[LOG1P_FEATURES] = np.log1p(data[LOG1P_FEATURES]).astype("float32")

    print("행 수:", f"{len(data):,}")
    print("학생 수:", f"{data[ID_COL].nunique():,}")
    print("Target 양성:", f"{target_count:,}")
    print("복합키 중복:", duplicate_count)
    print("비연속 주차:", invalid_week_gap)
    return data


def build_sequences(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """각 행을 최근 최대 4주의 왼쪽 정렬 Sequence로 변환한다."""

    print("\n2. 최근 4주 Sequence 생성")
    grouped = data.groupby(SEQUENCE_KEYS, observed=True, sort=False)
    sequence_lengths = (
        grouped.cumcount()
        .add(1)
        .clip(upper=SEQUENCE_LENGTH)
        .to_numpy(dtype=np.int64)
    )

    row_count = len(data)
    feature_count = len(DYNAMIC_FEATURES)
    sequences = np.zeros(
        (row_count, SEQUENCE_LENGTH, feature_count),
        dtype=np.float32,
    )

    # early week는 실제 값을 앞쪽에 두고 뒤쪽을 0으로 Padding한다.
    for lag in range(SEQUENCE_LENGTH):
        shifted = grouped[DYNAMIC_FEATURES].shift(lag).to_numpy(dtype=np.float32)
        valid_mask = sequence_lengths > lag
        valid_rows = np.flatnonzero(valid_mask)
        positions = sequence_lengths[valid_mask] - 1 - lag
        sequences[valid_rows, positions, :] = shifted[valid_mask]
        del shifted
        gc.collect()
        print(f"lag {lag} 생성 완료: {valid_mask.sum():,}행")

    if not np.isfinite(sequences).all():
        raise ValueError("생성된 Sequence에 NaN 또는 무한값이 있습니다.")

    target = data[TARGET_COL].to_numpy(dtype=np.float32)
    groups = data[ID_COL].to_numpy(dtype=np.int64)
    print("Sequence 크기:", sequences.shape)
    print("Sequence 메모리:", f"{sequences.nbytes / 1024**2:,.1f} MB")
    print("Sequence 길이 분포:")
    print(pd.Series(sequence_lengths).value_counts().sort_index())
    return sequences, sequence_lengths, target, groups


def fit_sequence_scaler(
    sequences: np.ndarray,
    lengths: np.ndarray,
    train_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """학습 Fold의 실제 관찰 시점만 이용해 평균과 표준편차를 계산한다."""

    feature_count = sequences.shape[-1]
    total = np.zeros(feature_count, dtype=np.float64)
    total_square = np.zeros(feature_count, dtype=np.float64)
    count = 0
    train_lengths = lengths[train_index]

    for step in range(SEQUENCE_LENGTH):
        valid_index = train_index[train_lengths > step]
        values = sequences[valid_index, step, :].astype(np.float64, copy=False)
        total += values.sum(axis=0)
        total_square += np.square(values).sum(axis=0)
        count += len(values)

    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def scale_sequences(
    sequences: np.ndarray,
    lengths: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Fold 학습 통계로 표준화하고 Padding 위치를 다시 0으로 만든다."""

    scaled = ((sequences - mean) / std).astype(np.float32)
    valid_mask = np.arange(SEQUENCE_LENGTH)[None, :] < lengths[:, None]
    scaled[~valid_mask] = 0.0
    if not np.isfinite(scaled).all():
        raise ValueError("표준화된 Sequence에 NaN 또는 무한값이 있습니다.")
    return scaled


class IndexedSequenceDataset(Dataset):
    """전체 Tensor를 공유하고 Fold에 해당하는 행만 반환한다."""

    def __init__(
        self,
        sequences: np.ndarray,
        lengths: np.ndarray,
        target: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.sequences = torch.from_numpy(sequences)
        self.lengths = torch.from_numpy(lengths.astype(np.int64, copy=True))
        self.target = torch.from_numpy(target.astype(np.float32, copy=True))
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = int(self.indices[item])
        return self.sequences[row], self.lengths[row], self.target[row]


class GRUNextWeekModel(nn.Module):
    """최근 주간 행동을 GRU로 요약해 다음 주 이탈 Logit을 반환한다."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_SIZE, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(16, 1),
        )

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(sequences)
        row_index = torch.arange(outputs.size(0), device=outputs.device)
        last_index = lengths.to(outputs.device) - 1
        last_hidden = outputs[row_index, last_index]
        return self.head(last_hidden).squeeze(1)


def make_loader(
    sequences: np.ndarray,
    lengths: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = IndexedSequenceDataset(sequences, lengths, target, indices)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=generator if shuffle else None,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0

    for batch_sequences, batch_lengths, batch_target in loader:
        batch_sequences = batch_sequences.to(device)
        batch_lengths = batch_lengths.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_sequences, batch_lengths)
        loss = criterion(logits, batch_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

        batch_rows = len(batch_target)
        total_loss += float(loss.detach().cpu()) * batch_rows
        total_rows += batch_rows

    return total_loss / total_rows


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []

    for batch_sequences, batch_lengths, _ in loader:
        batch_sequences = batch_sequences.to(device)
        batch_lengths = batch_lengths.to(device)
        logits = model(batch_sequences, batch_lengths)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probabilities).astype(np.float64, copy=False)


def precision_at_top_fraction(
    y_true: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    fraction: float = TOP_FRACTION,
) -> float:
    target = np.asarray(y_true, dtype=np.int8)
    values = np.asarray(probability, dtype=float)
    top_k = max(1, int(np.ceil(len(target) * fraction)))
    ranked_index = np.argsort(-values, kind="stable")[:top_k]
    return float(target[ranked_index].mean())


def train_fold(
    fold: int,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    sequences: np.ndarray,
    lengths: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, float | int], dict[str, object]]:
    """한 학생 Fold를 학습하고 최적 PR-AUC OOF 확률을 반환한다."""

    mean, std = fit_sequence_scaler(sequences, lengths, train_index)
    scaled_sequences = scale_sequences(sequences, lengths, mean, std)

    train_loader = make_loader(
        scaled_sequences,
        lengths,
        target,
        train_index,
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_STATE + fold,
    )
    validation_loader = make_loader(
        scaled_sequences,
        lengths,
        target,
        validation_index,
        batch_size=batch_size,
        shuffle=False,
        seed=RANDOM_STATE + fold,
    )

    model = GRUNextWeekModel(len(DYNAMIC_FEATURES)).to(device)
    positive_count = float(target[train_index].sum())
    negative_count = float(len(train_index) - positive_count)
    positive_weight = negative_count / max(positive_count, 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_pr_auc = -np.inf
    best_epoch = 0
    best_probability: np.ndarray | None = None
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation_probability = predict_probabilities(model, validation_loader, device)
        validation_pr_auc = float(
            average_precision_score(target[validation_index], validation_probability)
        )
        print(
            f"Fold {fold} | Epoch {epoch:02d} | "
            f"Loss={train_loss:.5f} | PR-AUC={validation_pr_auc:.6f}"
        )

        if validation_pr_auc > best_pr_auc + 1e-6:
            best_pr_auc = validation_pr_auc
            best_epoch = epoch
            best_probability = validation_probability.copy()
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= PATIENCE:
                print(f"Fold {fold} 조기 종료: best epoch={best_epoch}")
                break

    if best_probability is None or best_state is None:
        raise RuntimeError(f"Fold {fold}의 최적 모델이 저장되지 않았습니다.")

    row = fold_metadata(
        "GRU",
        fold,
        train_index,
        validation_index,
        target.astype(np.int8),
        groups,
    )
    row.update(calculate_metrics(target[validation_index], best_probability))
    row["precision_at_top_20pct"] = precision_at_top_fraction(
        target[validation_index], best_probability
    )
    row["best_epoch"] = best_epoch
    row["positive_weight"] = positive_weight

    checkpoint = {
        "model_state_dict": best_state,
        "input_features": DYNAMIC_FEATURES,
        "log1p_features": LOG1P_FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "fold": fold,
        "best_epoch": best_epoch,
    }

    del model, train_loader, validation_loader, scaled_sequences
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return best_probability, row, checkpoint


def run_smoke_test(
    sequences: np.ndarray,
    lengths: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> None:
    """Sequence·표준화·MPS forward/backward를 한 Batch로 검증한다."""

    print("\n3. GRU 1개 Batch Smoke Test")
    indices = np.arange(len(target), dtype=np.int64)
    mean, std = fit_sequence_scaler(sequences, lengths, indices)
    scaled = scale_sequences(sequences, lengths, mean, std)
    loader = make_loader(
        scaled,
        lengths,
        target,
        indices,
        batch_size=min(batch_size, 2048),
        shuffle=True,
        seed=RANDOM_STATE,
    )
    model = GRUNextWeekModel(len(DYNAMIC_FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    batch_sequences, batch_lengths, batch_target = next(iter(loader))
    batch_sequences = batch_sequences.to(device)
    batch_lengths = batch_lengths.to(device)
    batch_target = batch_target.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch_sequences, batch_lengths)
    loss = criterion(logits, batch_target)
    loss.backward()
    optimizer.step()

    print("Batch 크기:", tuple(batch_sequences.shape))
    print("Logit 크기:", tuple(logits.shape))
    print("Loss:", float(loss.detach().cpu()))
    print("Smoke Test 통과")


def save_results(
    data: pd.DataFrame,
    target: np.ndarray,
    probabilities: np.ndarray,
    fold_assignment: np.ndarray,
    fold_rows: list[dict[str, object]],
    checkpoints: list[dict[str, object]],
    fold_hash: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    oof = data[[*SORT_COLS, TARGET_COL]].copy()
    oof["gru_oof_probability"] = probabilities
    oof["fold"] = fold_assignment

    overall: dict[str, object] = {
        "model": "GRU recent 4-week behavior",
        "rows": len(data),
        "target_count": int(target.sum()),
        "target_rate": float(target.mean()),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": len(DYNAMIC_FEATURES),
    }
    overall.update(calculate_metrics(target, probabilities))
    overall["precision_at_top_20pct"] = precision_at_top_fraction(
        target, probabilities
    )

    pd.DataFrame([overall]).to_csv(
        OUTPUT_DIR / "gru_weekly_next_week_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fold_rows).to_csv(
        OUTPUT_DIR / "gru_weekly_next_week_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    oof.to_csv(
        OUTPUT_DIR / "gru_weekly_next_week_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    checkpoint_names: list[str] = []
    for checkpoint in checkpoints:
        fold = int(checkpoint["fold"])
        checkpoint_path = OUTPUT_DIR / f"gru_weekly_next_week_fold_{fold}.pt"
        torch.save(checkpoint, checkpoint_path)
        checkpoint_names.append(checkpoint_path.name)

    metadata = {
        "input_features": DYNAMIC_FEATURES,
        "log1p_features": LOG1P_FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "n_splits": N_SPLITS,
        "fold_assignment_sha256": fold_hash,
        "checkpoints": checkpoint_names,
        "data_rows": len(data),
        "target_count": int(target.sum()),
    }
    with (OUTPUT_DIR / "gru_weekly_next_week_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\n===== GRU OOF 평가 완료 =====")
    print(pd.DataFrame([overall]).to_string(index=False))
    print("결과 저장 위치:", OUTPUT_DIR)


def main() -> None:
    args = parse_args()
    seed_everything()
    device = choose_device()

    print("===== GRU 실행 환경 =====")
    print("PyTorch:", torch.__version__)
    print("장치:", device)
    print("데이터:", args.data_path)
    print("Sequence 길이:", SEQUENCE_LENGTH)
    print("입력 Feature 수:", len(DYNAMIC_FEATURES))

    max_rows = 50_000 if args.smoke_test else None
    data = load_weekly_data(args.data_path.resolve(), max_rows=max_rows)
    sequences, lengths, target, groups = build_sequences(data)

    if args.smoke_test:
        run_smoke_test(sequences, lengths, target, device, args.batch_size)
        return

    folds, fold_assignment, fold_hash = make_group_folds(
        groups, n_splits=N_SPLITS
    )
    probabilities = np.zeros(len(data), dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []

    for fold, train_index, validation_index in folds:
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")
        fold_probability, fold_row, checkpoint = train_fold(
            fold,
            train_index,
            validation_index,
            sequences,
            lengths,
            target,
            groups,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        probabilities[validation_index] = fold_probability
        fold_rows.append(fold_row)
        checkpoints.append(checkpoint)
        print(
            f"Fold {fold} 완료 | PR-AUC={fold_row['pr_auc']:.6f} | "
            f"Recall@20={fold_row['recall_at_top_20pct']:.4f}"
        )

    probabilities = validate_oof(probabilities, fold_assignment, len(data))
    save_results(
        data,
        target,
        probabilities,
        fold_assignment,
        fold_rows,
        checkpoints,
        fold_hash,
    )

if __name__ == "__main__":
    main()
    
'''
최근 4주 학습행동의 시계열 흐름을 반영하기 위해 GRU를 추가 실험했다. 
GRU도 무작위 기준보다 높은 예측력을 보였지만,
희소하고 불균형한 정형 데이터 특성상 CatBoost의 성능이 더 우수했다.
따라서 최종 서비스는 고정 임계값 0.065를 적용한 CatBoost를 사용하고,
GRU는 딥러닝 비교 실험으로 제시했다.
'''

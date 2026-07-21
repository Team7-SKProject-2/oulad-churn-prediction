"""최근 4주 행동 패턴으로 다음 주 이탈을 예측하는 TCN형 1D-CNN.

GRU와 동일한 데이터, 11개 Feature, 학생 단위 3-Fold 및 평가지표를 사용한다.
인과적 1D 합성곱은 클릭 급감이나 연속 무활동 같은 짧은 시간 패턴을 탐지한다.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = MODELS_DIR / "demo_1"
GRU_SCRIPT = MODELS_DIR / "06_gru_weekly_next_week.py"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from common_weekly_metrics import (  # noqa: E402
    calculate_metrics,
    fold_metadata,
    make_group_folds,
    validate_oof,
)


def load_sequence_common():
    """검증된 GRU 데이터·Sequence·DataLoader 구현을 그대로 공유한다."""

    spec = importlib.util.spec_from_file_location("gru_sequence_common", GRU_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"GRU 공통 코드를 불러올 수 없습니다: {GRU_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load_sequence_common()
DEFAULT_DATA_PATH = COMMON.DEFAULT_DATA_PATH
TARGET_COL = COMMON.TARGET_COL
SORT_COLS = COMMON.SORT_COLS
DYNAMIC_FEATURES = COMMON.DYNAMIC_FEATURES
LOG1P_FEATURES = COMMON.LOG1P_FEATURES
SEQUENCE_LENGTH = COMMON.SEQUENCE_LENGTH
N_SPLITS = COMMON.N_SPLITS
RANDOM_STATE = COMMON.RANDOM_STATE
TOP_FRACTION = COMMON.TOP_FRACTION

BATCH_SIZE = 4096
MAX_EPOCHS = 12
PATIENCE = 3
LEARNING_RATE = 1e-3
CHANNELS = 32
DROPOUT_RATE = 0.20
GRADIENT_CLIP_NORM = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CausalConv1d(nn.Module):
    """오른쪽 미래를 보지 않는 1D 합성곱."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.trim = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.trim,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        if self.trim:
            outputs = outputs[..., :-self.trim]
        return outputs


class TemporalResidualBlock(nn.Module):
    """두 개의 인과 합성곱과 잔차 연결을 갖는 TCN 블록."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=2,
            dilation=dilation,
        )
        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size=2,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.residual(inputs)
        outputs = self.dropout(self.activation(self.norm1(self.conv1(inputs))))
        outputs = self.dropout(self.activation(self.norm2(self.conv2(outputs))))
        return self.activation(outputs + residual)


class TCNNextWeekModel(nn.Module):
    """TCN의 마지막 관찰 상태와 기간 내 최대 반응을 함께 사용한다."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            TemporalResidualBlock(input_size, CHANNELS, dilation=1),
            TemporalResidualBlock(CHANNELS, CHANNELS, dilation=2),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(CHANNELS * 2),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(CHANNELS * 2, 32),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(32, 1),
        )

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(sequences.transpose(1, 2)).transpose(1, 2)
        device_lengths = lengths.to(encoded.device)
        row_index = torch.arange(encoded.size(0), device=encoded.device)
        last_hidden = encoded[row_index, device_lengths - 1]

        valid_mask = (
            torch.arange(encoded.size(1), device=encoded.device)[None, :]
            < device_lengths[:, None]
        )
        masked = encoded.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
        max_hidden = masked.max(dim=1).values
        representation = torch.cat([last_hidden, max_hidden], dim=1)
        return self.head(representation).squeeze(1)


def precision_at_top_fraction(
    target: np.ndarray,
    probability: np.ndarray,
    fraction: float = TOP_FRACTION,
) -> float:
    top_k = max(1, int(np.ceil(len(target) * fraction)))
    selected = np.argsort(-probability, kind="stable")[:top_k]
    return float(target[selected].mean())


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
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    mean, std = COMMON.fit_sequence_scaler(sequences, lengths, train_index)
    scaled = COMMON.scale_sequences(sequences, lengths, mean, std)
    train_loader = COMMON.make_loader(
        scaled,
        lengths,
        target,
        train_index,
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_STATE + fold,
    )
    validation_loader = COMMON.make_loader(
        scaled,
        lengths,
        target,
        validation_index,
        batch_size=batch_size,
        shuffle=False,
        seed=RANDOM_STATE + fold,
    )

    model = TCNNextWeekModel(len(DYNAMIC_FEATURES)).to(device)
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
        train_loss = COMMON.train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        validation_probability = COMMON.predict_probabilities(
            model, validation_loader, device
        )
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
        "TCN",
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

    checkpoint: dict[str, object] = {
        "model_state_dict": best_state,
        "input_features": DYNAMIC_FEATURES,
        "log1p_features": LOG1P_FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "channels": CHANNELS,
        "dropout_rate": DROPOUT_RATE,
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "fold": fold,
        "best_epoch": best_epoch,
    }

    del model, train_loader, validation_loader, scaled
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
    print("\n3. TCN 1개 Batch Smoke Test")
    indices = np.arange(len(target), dtype=np.int64)
    mean, std = COMMON.fit_sequence_scaler(sequences, lengths, indices)
    scaled = COMMON.scale_sequences(sequences, lengths, mean, std)
    loader = COMMON.make_loader(
        scaled,
        lengths,
        target,
        indices,
        batch_size=min(batch_size, 2048),
        shuffle=True,
        seed=RANDOM_STATE,
    )
    model = TCNNextWeekModel(len(DYNAMIC_FEATURES)).to(device)
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
    print("TCN Smoke Test 통과")


def save_results(
    data: pd.DataFrame,
    target: np.ndarray,
    probability: np.ndarray,
    fold_assignment: np.ndarray,
    fold_rows: list[dict[str, object]],
    checkpoints: list[dict[str, object]],
    fold_hash: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall: dict[str, object] = {
        "model": "TCN causal 1D-CNN recent 4-week behavior",
        "rows": len(data),
        "target_count": int(target.sum()),
        "target_rate": float(target.mean()),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": len(DYNAMIC_FEATURES),
    }
    overall.update(calculate_metrics(target, probability))
    overall["precision_at_top_20pct"] = precision_at_top_fraction(
        target, probability
    )
    pd.DataFrame([overall]).to_csv(
        OUTPUT_DIR / "tcn_weekly_next_week_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fold_rows).to_csv(
        OUTPUT_DIR / "tcn_weekly_next_week_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof = data[[*SORT_COLS, TARGET_COL]].copy()
    oof["tcn_oof_probability"] = probability
    oof["fold"] = fold_assignment
    oof.to_csv(
        OUTPUT_DIR / "tcn_weekly_next_week_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    checkpoint_names: list[str] = []
    for checkpoint in checkpoints:
        fold = int(checkpoint["fold"])
        checkpoint_path = OUTPUT_DIR / f"tcn_weekly_next_week_fold_{fold}.pt"
        torch.save(checkpoint, checkpoint_path)
        checkpoint_names.append(checkpoint_path.name)
    metadata = {
        "input_features": DYNAMIC_FEATURES,
        "log1p_features": LOG1P_FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "channels": CHANNELS,
        "n_splits": N_SPLITS,
        "fold_assignment_sha256": fold_hash,
        "checkpoints": checkpoint_names,
        "data_rows": len(data),
        "target_count": int(target.sum()),
    }
    with (OUTPUT_DIR / "tcn_weekly_next_week_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print("\n===== TCN OOF 평가 완료 =====")
    print(pd.DataFrame([overall]).to_string(index=False))
    print("결과 저장 위치:", OUTPUT_DIR)


def main() -> None:
    args = parse_args()
    seed_everything()
    device = choose_device()
    print("===== TCN 실행 환경 =====")
    print("PyTorch:", torch.__version__)
    print("장치:", device)
    print("데이터:", args.data_path)
    print("Sequence 길이:", SEQUENCE_LENGTH)
    print("입력 Feature 수:", len(DYNAMIC_FEATURES))

    max_rows = 50_000 if args.smoke_test else None
    data = COMMON.load_weekly_data(args.data_path.resolve(), max_rows=max_rows)
    sequences, lengths, target, groups = COMMON.build_sequences(data)
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

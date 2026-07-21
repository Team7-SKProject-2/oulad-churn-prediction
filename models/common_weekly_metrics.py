"""Demo1 주간 이탈 모델이 공유하는 데이터 검증, Fold, 평가지표 도구.

Dummy와 ElasticNet이 같은 안정 정렬, 같은 학생 GroupKFold, 같은 지표 정의를
사용하도록 한곳에서 관리한다. 기존 01·02·05 모델은 변경하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold


TARGET_COL = "target_next_week_withdrawn"
ID_COL = "id_student"
SORT_COLUMNS = ["code_module", "code_presentation", ID_COL, "prediction_week"]
EXPECTED_ROWS = 895_005
EXPECTED_COLUMNS = 126
EXPECTED_FEATURES = 124
N_SPLITS = 3
RANDOM_STATE = 42
TOP_FRACTION = 0.20


@dataclass
class PreparedWeeklyData:
    """검증과 안정 정렬이 끝난 공통 모델 입력 묶음."""

    data: pd.DataFrame
    features: pd.DataFrame
    target: np.ndarray
    groups: np.ndarray
    categorical: list[str]
    numeric: list[str]
    profile: dict[str, Any]


def resolve_data_path(cli_path: str | Path | None) -> Path:
    """CLI, 환경변수, 저장소 내 ignored 데이터 경로 순으로 CSV를 찾는다."""
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser())
    env_path = os.environ.get("OULAD_DATA_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(
        Path(__file__).resolve().parent
        / "demo_1"
        / "used_data"
        / "weekly_next_week_with_vle_enhanced.csv"
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "주간 학습 CSV를 찾지 못했습니다. --data-path 또는 OULAD_DATA_PATH를 "
        f"지정하세요.\n검색 경로:\n - {searched}"
    )


def _as_1d_binary_target(y_true: Sequence[int] | np.ndarray) -> np.ndarray:
    target = np.asarray(y_true)
    if target.ndim != 1:
        raise ValueError("Target은 1차원 배열이어야 합니다.")
    if target.size == 0:
        raise ValueError("Target 배열이 비어 있습니다.")
    try:
        numeric_target = target.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Target은 0과 1로만 구성되어야 합니다.") from exc
    if not np.isfinite(numeric_target).all():
        raise ValueError("Target에 NaN 또는 무한값이 있습니다.")
    if not np.isin(numeric_target, [0.0, 1.0]).all():
        raise ValueError("Target은 0과 1로만 구성되어야 합니다.")
    return numeric_target.astype(np.int8, copy=False)


def _as_1d_probability(
    probability: Sequence[float] | np.ndarray,
    expected_length: int,
) -> np.ndarray:
    try:
        values = np.asarray(probability, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("예측확률은 숫자형 1차원 배열이어야 합니다.") from exc
    if values.ndim != 1:
        raise ValueError("예측확률은 1차원 배열이어야 합니다.")
    if len(values) != expected_length:
        raise ValueError(
            f"Target과 예측확률 길이가 다릅니다: {expected_length} != {len(values)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("예측확률에 NaN 또는 무한값이 있습니다.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("예측확률은 0 이상 1 이하이어야 합니다.")
    return values


def validate_metric_inputs(
    y_true: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """공통 지표 입력을 검증하고 정규화한다."""
    target = _as_1d_binary_target(y_true)
    values = _as_1d_probability(probability, len(target))
    return target, values


def recall_at_top_fraction(
    y_true: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    fraction: float = TOP_FRACTION,
) -> float:
    """예측확률 상위 ``fraction``에 포함된 실제 양성의 비율을 계산한다.

    내림차순 stable sort를 사용하므로 확률 동률은 입력 행 순서를 유지한다.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("상위 위험군 비율은 0보다 크고 1 이하여야 합니다.")
    target, values = validate_metric_inputs(y_true, probability)
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    top_k = max(1, int(np.ceil(len(target) * fraction)))
    ranked_index = np.argsort(-values, kind="stable")
    return float(target[ranked_index[:top_k]].sum() / positives)


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    bins: int = 10,
) -> float:
    """[0, 1]을 동일 너비로 나눈 표본 가중 ECE를 계산한다."""
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("ECE bins는 양의 정수여야 합니다.")
    target, values = validate_metric_inputs(y_true, probability)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == bins - 1:
            mask = (values >= lower) & (values <= upper)
        else:
            mask = (values >= lower) & (values < upper)
        if mask.any():
            observed_rate = float(target[mask].mean())
            mean_probability = float(values[mask].mean())
            ece += float(mask.mean()) * abs(observed_rate - mean_probability)
    return float(ece)


def calculate_metrics(
    y_true: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """불균형 순위 성능과 확률 품질 지표 네 개를 같은 정의로 반환한다."""
    target, values = validate_metric_inputs(y_true, probability)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        pr_auc = float(average_precision_score(target, values))
    return {
        "recall_at_top_20pct": recall_at_top_fraction(target, values),
        "pr_auc": pr_auc,
        "brier_score": float(brier_score_loss(target, values)),
        "ece_10bin": expected_calibration_error(target, values, bins=10),
    }


def split_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """CSV dtype를 기준으로 범주형과 수치형 Feature를 구분한다."""
    categorical = [
        column
        for column in features.columns
        if pd.api.types.is_object_dtype(features[column].dtype)
        or pd.api.types.is_string_dtype(features[column].dtype)
        or isinstance(features[column].dtype, pd.CategoricalDtype)
    ]
    numeric = [column for column in features.columns if column not in categorical]
    return categorical, numeric


def _leakage_risks(columns: Sequence[str]) -> dict[str, str]:
    risks: dict[str, str] = {}
    for column in columns:
        if column in {TARGET_COL, ID_COL}:
            continue
        lowered = column.lower()
        if lowered == "final_result":
            risks[column] = "과정 종료 후 확정되는 최종 결과"
        elif lowered == "date_unregistration" or "unregistration" in lowered:
            risks[column] = "이탈 시점 또는 이탈 이후 정보"
        elif "withdraw_week" in lowered:
            risks[column] = "정답 이탈 주차를 직접 또는 간접적으로 노출"
        elif "future" in lowered or "next_week" in lowered:
            risks[column] = "예측 기준일 이후 미래 정보 가능성"
    return risks


def _schema_hash(data: pd.DataFrame) -> str:
    schema = [(column, str(dtype)) for column, dtype in data.dtypes.items()]
    payload = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_and_prepare_weekly_data(
    data_path: str | Path,
    *,
    max_rows: int | None = None,
) -> PreparedWeeklyData:
    """CSV를 읽고 스키마·누수·키를 검증한 뒤 안정 정렬한다.

    ``max_rows``는 출력 파일을 만들지 않는 smoke test 전용이다. 전체 실행에서는
    지정하지 않아 895,005행 × 126열 계약을 엄격하게 확인한다.
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows는 1 이상의 정수여야 합니다.")

    data = pd.read_csv(path, nrows=max_rows)
    strict_shape = max_rows is None
    if strict_shape and data.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(
            "전체 데이터 크기가 계약과 다릅니다: "
            f"실제={data.shape}, 기대=({EXPECTED_ROWS}, {EXPECTED_COLUMNS})"
        )
    if data.shape[1] != EXPECTED_COLUMNS:
        raise ValueError(
            f"전체 열 수가 {EXPECTED_COLUMNS}개가 아닙니다: {data.shape[1]}"
        )

    required = [*SORT_COLUMNS, TARGET_COL]
    missing_required = [column for column in required if column not in data.columns]
    if missing_required:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_required}")
    if data[TARGET_COL].isna().any():
        raise ValueError(f"{TARGET_COL}에 결측값이 있습니다.")
    if data[ID_COL].isna().any():
        raise ValueError(f"{ID_COL}에 결측값이 있습니다.")

    target_values = _as_1d_binary_target(data[TARGET_COL].to_numpy())
    if strict_shape and set(np.unique(target_values)) != {0, 1}:
        raise ValueError("전체 Target에는 0과 1이 모두 존재해야 합니다.")

    duplicate_count = int(data.duplicated(SORT_COLUMNS).sum())
    if duplicate_count:
        raise ValueError(
            "학생·과목·운영회차·예측주차 복합키 중복이 있습니다: "
            f"{duplicate_count}건"
        )

    leakage_risks = _leakage_risks(data.columns)
    excluded_columns = list(leakage_risks)
    features = data.drop(columns=[ID_COL, TARGET_COL, *excluded_columns]).copy()
    categorical, numeric = split_feature_types(features)
    all_missing = [column for column in features if features[column].isna().all()]
    if all_missing:
        raise ValueError(f"값이 전부 결측인 Feature가 있습니다: {all_missing}")

    original_feature_count = data.shape[1] - 2
    if original_feature_count != EXPECTED_FEATURES:
        raise ValueError(
            f"id_student와 Target 제외 전 Feature 수가 {EXPECTED_FEATURES}개가 아닙니다: "
            f"{original_feature_count}"
        )
    if not excluded_columns and features.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"모델 입력 Feature 수가 {EXPECTED_FEATURES}개가 아닙니다: {features.shape[1]}"
        )

    infinity_count = int(
        sum(np.isinf(features[column].to_numpy(dtype=float, copy=False)).sum() for column in numeric)
    )
    missing_value_count = int(features.isna().sum().sum())
    exclusion_reason = " | ".join(
        f"{column}: {reason}" for column, reason in leakage_risks.items()
    )

    data = data.sort_values(SORT_COLUMNS, kind="mergesort", ignore_index=True)
    features = data.drop(columns=[ID_COL, TARGET_COL, *excluded_columns]).copy()
    target = data[TARGET_COL].astype(np.int8).to_numpy()
    groups = data[ID_COL].to_numpy()
    profile: dict[str, Any] = {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "target_count": int(target.sum()),
        "target_rate": float(target.mean()),
        "feature_count": int(features.shape[1]),
        "categorical_feature_count": int(len(categorical)),
        "numeric_feature_count": int(len(numeric)),
        "missing_value_count": missing_value_count,
        "infinity_count_before_replacement": infinity_count,
        "duplicate_key_count": duplicate_count,
        "excluded_leakage_columns": "|".join(excluded_columns),
        "feature_exclusion_reason": exclusion_reason,
        "data_schema_hash": _schema_hash(data),
    }
    return PreparedWeeklyData(
        data=data,
        features=features,
        target=target,
        groups=groups,
        categorical=categorical,
        numeric=numeric,
        profile=profile,
    )


def make_group_folds(
    groups: Sequence[Any] | np.ndarray,
    n_splits: int = N_SPLITS,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], np.ndarray, str]:
    """학생별 GroupKFold와 OOF Fold 배정, 배정 해시를 생성·검증한다."""
    group_values = np.asarray(groups)
    if group_values.ndim != 1 or len(group_values) == 0:
        raise ValueError("그룹은 비어 있지 않은 1차원 배열이어야 합니다.")
    splitter = GroupKFold(n_splits=n_splits)
    folds: list[tuple[int, np.ndarray, np.ndarray]] = []
    assignment = np.zeros(len(group_values), dtype=np.int8)
    validation_count = np.zeros(len(group_values), dtype=np.int8)
    placeholder = np.zeros((len(group_values), 1), dtype=np.uint8)

    for fold, (train_index, validation_index) in enumerate(
        splitter.split(placeholder, groups=group_values), start=1
    ):
        overlap = np.intersect1d(
            np.unique(group_values[train_index]),
            np.unique(group_values[validation_index]),
            assume_unique=True,
        )
        if len(overlap):
            raise ValueError(f"Fold {fold}에 학습·검증 학생이 {len(overlap)}명 겹칩니다.")
        assignment[validation_index] = fold
        validation_count[validation_index] += 1
        folds.append((fold, train_index, validation_index))

    if not np.all(validation_count == 1):
        raise ValueError("모든 행이 OOF 검증에 정확히 한 번 포함되지 않았습니다.")
    fold_hash = hashlib.sha256(assignment.tobytes()).hexdigest()
    return folds, assignment, fold_hash


def fold_metadata(
    model_name: str,
    fold: int,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """모델 공통 Fold 규모·양성률·학생 중복 정보를 반환한다."""
    train_students = np.unique(groups[train_index])
    validation_students = np.unique(groups[validation_index])
    overlap_count = int(
        len(np.intersect1d(train_students, validation_students, assume_unique=True))
    )
    return {
        "model": model_name,
        "fold": fold,
        "train_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "train_students": int(len(train_students)),
        "validation_students": int(len(validation_students)),
        "student_overlap_count": overlap_count,
        "train_target_count": int(target[train_index].sum()),
        "validation_target_count": int(target[validation_index].sum()),
        "train_target_rate": float(target[train_index].mean()),
        "validation_target_rate": float(target[validation_index].mean()),
    }


def validate_oof(
    probability: Sequence[float] | np.ndarray,
    fold_assignment: Sequence[int] | np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    """OOF 누락·중복 배정과 확률 범위를 최종 확인한다."""
    placeholder_target = np.zeros(expected_rows, dtype=np.int8)
    values = _as_1d_probability(probability, expected_rows)
    assignments = np.asarray(fold_assignment)
    if assignments.ndim != 1 or len(assignments) != expected_rows:
        raise ValueError("OOF Fold 배정 길이가 전체 행 수와 다릅니다.")
    if not np.isin(assignments, np.arange(1, N_SPLITS + 1)).all():
        raise ValueError("OOF Fold 배정에 누락 또는 잘못된 값이 있습니다.")
    # 확률 검증이 Target 값과 독립적임을 분명히 하면서 공통 검증 경로를 재사용한다.
    validate_metric_inputs(placeholder_target, values)
    return values

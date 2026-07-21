"""분류 모델의 공통 평가와 threshold 탐색 함수."""

from __future__ import annotations

from decimal import Decimal
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_THRESHOLD = 0.5


def _validated_binary_arrays(y_true, probabilities) -> tuple[np.ndarray, np.ndarray]:
    y_array = np.asarray(y_true, dtype=int).reshape(-1)
    probability_array = np.asarray(probabilities, dtype=float).reshape(-1)
    if y_array.size == 0:
        raise ValueError("평가할 검증 데이터가 비어 있습니다.")
    if y_array.size != probability_array.size:
        raise ValueError("정답 라벨과 양성 확률의 개수가 다릅니다.")
    if not np.isin(y_array, [0, 1]).all():
        raise ValueError("정답 라벨은 양성=1, 음성=0인 이진 값이어야 합니다.")
    if not np.isfinite(probability_array).all():
        raise ValueError("양성 확률에 NaN 또는 무한대가 포함되어 있습니다.")
    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("양성 확률은 0과 1 사이여야 합니다.")
    return y_array, probability_array


def generate_thresholds(
    minimum: float = 0.05,
    maximum: float = 0.95,
    step: float = 0.05,
    default: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """양 끝을 포함하는 후보를 만들고 기본 threshold 0.5를 항상 추가한다."""
    minimum_decimal = Decimal(str(minimum))
    maximum_decimal = Decimal(str(maximum))
    step_decimal = Decimal(str(step))
    default_decimal = Decimal(str(default))
    if not (Decimal("0") <= minimum_decimal <= maximum_decimal <= Decimal("1")):
        raise ValueError("threshold 범위는 0 <= min <= max <= 1이어야 합니다.")
    if step_decimal <= 0:
        raise ValueError("threshold 간격은 0보다 커야 합니다.")

    values: list[Decimal] = []
    current = minimum_decimal
    while current <= maximum_decimal:
        values.append(current)
        current += step_decimal
    values.append(default_decimal)
    return np.asarray(sorted({float(value) for value in values}), dtype=float)


def threshold_metrics(y_true, probabilities, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """하나의 threshold에 대한 혼동행렬과 이진 분류 지표를 계산한다."""
    y_array, probability_array = _validated_binary_arrays(y_true, probabilities)
    roc_auc, pr_auc = _ranking_metrics(y_array, probability_array)
    return _threshold_metrics_from_arrays(
        y_array, probability_array, threshold, roc_auc, pr_auc
    )


def _ranking_metrics(
    y_array: np.ndarray, probability_array: np.ndarray
) -> tuple[float, float]:
    """threshold와 무관한 확률 순위 지표를 한 번 계산한다."""
    has_both_classes = np.unique(y_array).size == 2
    if not has_both_classes:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(y_array, probability_array)),
        float(average_precision_score(y_array, probability_array)),
    )


def _threshold_metrics_from_arrays(
    y_array: np.ndarray,
    probability_array: np.ndarray,
    threshold: float,
    roc_auc: float,
    pr_auc: float,
) -> dict:
    # 모든 threshold에서 동일한 규칙을 사용한다.
    predictions = (probability_array >= threshold).astype(int)

    tp = int(((y_array == 1) & (predictions == 1)).sum())
    fp = int(((y_array == 0) & (predictions == 1)).sum())
    tn = int(((y_array == 0) & (predictions == 0)).sum())
    fn = int(((y_array == 1) & (predictions == 0)).sum())
    total = y_array.size

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1_value = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    # ROC-AUC와 PR-AUC는 threshold로 만든 prediction이 아니라 원래 확률로
    # 계산하므로 threshold별 행에서 같은 값이 나오는 것이 정상이다.
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1_value),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "predicted_positive_count": int(predictions.sum()),
        "predicted_positive_ratio": float(predictions.mean()),
        "youden_j": float(recall + specificity - 1),
    }


def compare_thresholds(y_true, probabilities, thresholds) -> pd.DataFrame:
    """후보 threshold 전체의 지표를 표로 반환한다."""
    y_array, probability_array = _validated_binary_arrays(y_true, probabilities)
    roc_auc, pr_auc = _ranking_metrics(y_array, probability_array)
    rows = [
        _threshold_metrics_from_arrays(
            y_array,
            probability_array,
            float(threshold),
            roc_auc,
            pr_auc,
        )
        for threshold in thresholds
    ]
    result = pd.DataFrame(rows).sort_values("threshold", ignore_index=True)

    # threshold가 커질 때 양성 예측 수가 증가했다면 구현 오류다.
    counts = result["predicted_positive_count"].to_numpy()
    if (np.diff(counts) > 0).any():
        raise AssertionError("threshold 증가에 따라 양성 예측 수가 증가했습니다.")
    return result


def select_best_threshold(
    table: pd.DataFrame,
    metric: str,
    condition: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> dict | None:
    """지표 최대값을 고르고 동점이면 0.5 근접, 이후 높은 threshold를 택한다."""
    candidates = table if condition is None else table.loc[condition(table)]
    candidates = candidates.loc[candidates[metric].notna()]
    if candidates.empty:
        return None

    # 명시된 동점 규칙: (1) metric 최대, (2) 0.5에 가까움, (3) 높은 threshold.
    ordered = candidates.assign(
        _distance=(candidates["threshold"] - DEFAULT_THRESHOLD).abs()
    ).sort_values(
        [metric, "_distance", "threshold"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    return _metric_row_to_dict(ordered.drop(columns="_distance").iloc[0])


def _metric_row_to_dict(row: pd.Series) -> dict:
    """DataFrame 행의 혼동행렬 개수는 JSON에서도 정수로 유지한다."""
    result = row.to_dict()
    for column in ("TP", "FP", "TN", "FN", "predicted_positive_count"):
        if column in result:
            result[column] = int(result[column])
    return result


def select_thresholds(
    table: pd.DataFrame,
    min_recall: float | None = None,
    min_precision: float | None = None,
) -> dict[str, dict | None]:
    """요구된 각 기준으로 최적 threshold와 기본 0.5 행을 반환한다."""
    default_rows = table.loc[np.isclose(table["threshold"], DEFAULT_THRESHOLD)]
    if default_rows.empty:
        raise ValueError("후보 threshold에 기본값 0.5가 없습니다.")

    selected: dict[str, dict | None] = {
        "best_f1": select_best_threshold(table, "f1_score"),
        "best_accuracy": select_best_threshold(table, "accuracy"),
        "best_recall": select_best_threshold(table, "recall"),
        "best_youden_j": select_best_threshold(table, "youden_j"),
        "default_0_5": _metric_row_to_dict(default_rows.iloc[0]),
    }
    if min_recall is not None:
        selected["best_precision_at_min_recall"] = select_best_threshold(
            table, "precision", lambda frame: frame["recall"] >= min_recall
        )
    if min_precision is not None:
        selected["best_recall_at_min_precision"] = select_best_threshold(
            table, "recall", lambda frame: frame["precision"] >= min_precision
        )
    return selected


def binary_metrics(y_true, probabilities, threshold: float = DEFAULT_THRESHOLD) -> dict[str, float]:
    """기존 학습 코드가 사용하는 평가 결과 형식을 유지한다."""
    metrics = threshold_metrics(y_true, probabilities, threshold)
    return {
        "threshold": metrics["threshold"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1_score"],
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
    }


def best_f1_threshold(y_true, probabilities) -> tuple[float, dict[str, float]]:
    """Validation에서 F1이 가장 높은 threshold를 찾는 기존 공개 함수."""
    candidates = np.linspace(0.05, 0.95, 91)
    scored = [(threshold, binary_metrics(y_true, probabilities, threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[1]["f1"], item[1]["recall"]))

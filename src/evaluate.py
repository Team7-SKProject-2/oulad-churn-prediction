"""분류 모델의 공통 평가와 threshold 탐색 함수."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    y_array = np.asarray(y_true)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_array, predictions)),
        "precision": float(precision_score(y_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_array, predictions, zero_division=0)),
        "f1": float(f1_score(y_array, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_array, probabilities)),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_array, probabilities))
        if len(np.unique(y_array)) == 2
        else float("nan")
    )
    return metrics


def best_f1_threshold(y_true, probabilities) -> tuple[float, dict[str, float]]:
    """Validation에서 F1이 가장 높은 threshold를 찾는 기본 함수."""
    candidates = np.linspace(0.05, 0.95, 91)
    scored = [(threshold, binary_metrics(y_true, probabilities, threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[1]["f1"], item[1]["recall"]))


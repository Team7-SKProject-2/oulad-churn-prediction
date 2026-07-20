"""Demo1 공통 주간 평가지표와 Fold 유틸리티 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from models.common_weekly_metrics import (
    calculate_metrics,
    expected_calibration_error,
    make_group_folds,
    recall_at_top_fraction,
)


def test_perfect_ranking_recall_at_top_20_percent() -> None:
    target = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    probability = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    assert recall_at_top_fraction(target, probability) == pytest.approx(1.0)


def test_reverse_ranking_recall_at_top_20_percent() -> None:
    target = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    probability = np.array([0.0, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    assert recall_at_top_fraction(target, probability) == pytest.approx(0.0)


def test_equal_probabilities_preserve_input_order() -> None:
    target = np.array([1, 0, 1, 0, 0])
    probability = np.full(5, 0.4)
    # ceil(5 * 0.2) == 1이며 stable sort로 첫 행을 고른다.
    assert recall_at_top_fraction(target, probability) == pytest.approx(0.5)


def test_no_positive_target_returns_nan_recall() -> None:
    metrics = calculate_metrics(np.zeros(5, dtype=int), np.full(5, 0.1))
    assert np.isnan(metrics["recall_at_top_20pct"])
    assert metrics["pr_auc"] == pytest.approx(0.0)


def test_probabilities_at_zero_and_one_are_valid() -> None:
    metrics = calculate_metrics(np.array([0, 1]), np.array([0.0, 1.0]))
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(0.0)
    assert metrics["ece_10bin"] == pytest.approx(0.0)


def test_ece_bin_boundaries_follow_left_closed_contract() -> None:
    target = np.array([0, 1, 0, 1])
    probability = np.array([0.0, 0.1, 0.9, 1.0])
    # 각 값은 [0,.1), [.1,.2), [.9,1]에 각각 한 번만 포함된다.
    expected = (0.0 + 0.9 + 0.9 + 0.0) / 4
    assert expected_calibration_error(target, probability, bins=10) == pytest.approx(expected)


def test_different_target_and_probability_lengths_raise() -> None:
    with pytest.raises(ValueError, match="길이가 다릅니다"):
        calculate_metrics(np.array([0, 1]), np.array([0.2]))


def test_nan_probability_raises() -> None:
    with pytest.raises(ValueError, match="NaN 또는 무한값"):
        calculate_metrics(np.array([0, 1]), np.array([0.2, np.nan]))


@pytest.mark.parametrize("invalid", [np.array([-0.1, 0.5]), np.array([0.5, 1.1])])
def test_out_of_range_probability_raises(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="0 이상 1 이하"):
        calculate_metrics(np.array([0, 1]), invalid)


def test_group_folds_have_no_student_overlap_and_cover_every_row_once() -> None:
    groups = np.repeat(np.arange(12), 3)
    folds, assignment, fold_hash = make_group_folds(groups)
    assert len(folds) == 3
    assert set(assignment) == {1, 2, 3}
    assert len(fold_hash) == 64
    for _, train_index, validation_index in folds:
        assert set(groups[train_index]).isdisjoint(set(groups[validation_index]))

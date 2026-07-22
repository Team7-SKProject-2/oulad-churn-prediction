"""Dummy prior 기준선으로 다음 주 이탈 확률의 학생 단위 OOF 성능을 측정한다."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

MODELS_DIR = Path(__file__).resolve().parents[1]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from common_weekly_metrics import (  # noqa: E402
    ID_COL,
    N_SPLITS,
    RANDOM_STATE,
    SORT_COLUMNS,
    TARGET_COL,
    calculate_metrics,
    fold_metadata,
    load_and_prepare_weekly_data,
    make_group_folds,
    resolve_data_path,
    validate_oof,
)


MODEL_NAME = "Dummy Classifier (strategy=prior)"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def run_dummy(
    data_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_rows: int | None = None,
    write_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """검증된 데이터에서 3-Fold Dummy OOF 예측과 CSV 결과를 만든다."""
    prepared = load_and_prepare_weekly_data(data_path, max_rows=max_rows)
    folds, fold_assignment, fold_hash = make_group_folds(prepared.groups)
    probabilities = np.full(len(prepared.data), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    print("=== 데이터 검증 ===")
    print(pd.Series(prepared.profile).to_string())
    for fold, train_index, validation_index in folds:
        y_train = prepared.target[train_index]
        y_validation = prepared.target[validation_index]
        model = DummyClassifier(strategy="prior")
        fold_started = time.perf_counter()
        model.fit(prepared.features.iloc[train_index], y_train)
        raw_probability = model.predict_proba(prepared.features.iloc[validation_index])
        if 1 in model.classes_:
            positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
            fold_probability = raw_probability[:, positive_column]
        else:
            fold_probability = np.zeros(len(validation_index), dtype=float)
        fold_seconds = time.perf_counter() - fold_started
        probabilities[validation_index] = fold_probability

        if np.unique(fold_probability).size != 1:
            raise RuntimeError(f"Dummy Fold {fold}의 예측확률이 상수가 아닙니다.")
        row = fold_metadata(
            MODEL_NAME,
            fold,
            train_index,
            validation_index,
            prepared.target,
            prepared.groups,
        )
        row.update(calculate_metrics(y_validation, fold_probability))
        row.update(
            {
                "training_seconds": fold_seconds,
                "train_prior_probability": float(fold_probability[0]),
            }
        )
        fold_rows.append(row)
        print(
            f"Fold {fold} 완료: prior={fold_probability[0]:.8f}, "
            f"PR-AUC={row['pr_auc']:.6f}, "
            f"Recall@Top-20%={row['recall_at_top_20pct']:.6f}"
        )

    training_seconds = time.perf_counter() - started
    probabilities = validate_oof(probabilities, fold_assignment, len(prepared.data))
    overall_metrics = calculate_metrics(prepared.target, probabilities)
    overall: dict[str, Any] = {
        "model": MODEL_NAME,
        "rows": prepared.profile["rows"],
        "target_count": prepared.profile["target_count"],
        "target_rate": prepared.profile["target_rate"],
        "feature_count": prepared.profile["feature_count"],
        "categorical_feature_count": prepared.profile["categorical_feature_count"],
        **overall_metrics,
        "n_splits": N_SPLITS,
        "group_column": ID_COL,
        "random_state": RANDOM_STATE,
        "data_schema_hash": prepared.profile["data_schema_hash"],
        "fold_assignment_hash": fold_hash,
        "training_seconds": training_seconds,
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "mean_oof_probability": float(probabilities.mean()),
        "probability_overestimates_target_rate": bool(
            probabilities.mean() > prepared.target.mean() + np.finfo(float).eps
        ),
        "missing_value_count": prepared.profile["missing_value_count"],
        "infinity_count_before_replacement": prepared.profile[
            "infinity_count_before_replacement"
        ],
        "excluded_leakage_columns": prepared.profile["excluded_leakage_columns"],
        "feature_exclusion_reason": prepared.profile["feature_exclusion_reason"],
        "recall_at_top_20pct_note": (
            "동률 처리 및 안정 정렬된 행 순서의 영향을 받는 참고값"
        ),
    }
    metrics_frame = pd.DataFrame([overall])
    fold_frame = pd.DataFrame(fold_rows)
    oof_frame = prepared.data[[*SORT_COLUMNS, TARGET_COL]].copy()
    oof_frame["fold"] = fold_assignment
    oof_frame["dummy_oof_probability"] = probabilities

    if not np.array_equal(oof_frame[TARGET_COL].to_numpy(), prepared.target):
        raise RuntimeError("OOF Target 순서가 안정 정렬된 원본 Target과 다릅니다.")
    if oof_frame.duplicated(SORT_COLUMNS).any():
        raise RuntimeError("OOF 결과에 복합키 중복이 있습니다.")

    if write_outputs:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        metrics_frame.to_csv(
            destination / "dummy_weekly_next_week_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        fold_frame.to_csv(
            destination / "dummy_weekly_next_week_fold_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        oof_frame.to_csv(
            destination / "dummy_weekly_next_week_oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n=== Dummy 3-Fold OOF 완료 ===")
    print(metrics_frame.to_string(index=False))
    return {"metrics": metrics_frame, "fold_metrics": fold_frame, "oof": oof_frame}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default=None, help="원본 주간 CSV 경로")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="metrics와 ignored OOF CSV 저장 폴더",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=None,
        help="앞 N행으로만 실행하며 결과 CSV를 쓰지 않는 smoke test 옵션",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    run_dummy(
        data_path,
        output_dir=args.output_dir,
        max_rows=args.smoke_rows,
        write_outputs=args.smoke_rows is None,
    )


if __name__ == "__main__":
    main()

"""ElasticNet Logistic Regression의 학생 단위 3-Fold OOF 성능을 검증한다."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

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


MODEL_NAME = "ElasticNet Logistic Regression"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Candidate:
    name: str
    class_weight: str | None
    C: float
    l1_ratio: float
    max_iter: int = 500
    tol: float = 1e-2


# 895,005행에서 무분별한 Grid Search를 피하면서 class_weight와 규제 조합을 비교한다.
CANDIDATES = (
    Candidate("unweighted_mild_l1", None, C=0.10, l1_ratio=0.10),
    Candidate(
        "balanced_mixed_l1_l2",
        "balanced",
        C=0.05,
        l1_ratio=0.50,
        max_iter=200,
        tol=5e-2,
    ),
)


def _replace_infinity(values: Any) -> Any:
    """수치형 Pipeline 내부에서 양·음의 무한값을 결측값으로 바꾼다."""
    if hasattr(values, "replace"):
        return values.replace([np.inf, -np.inf], np.nan)
    array = np.asarray(values, dtype=float).copy()
    array[~np.isfinite(array)] = np.nan
    return array


def build_pipeline(
    categorical: list[str],
    numeric: list[str],
    candidate: Candidate,
) -> Pipeline:
    """Fold 학습 데이터에만 적합되는 희소 전처리와 ElasticNet 모델을 만든다."""
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="미상")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64),
            ),
        ]
    )
    numeric_pipeline = Pipeline(
        [
            (
                "replace_infinity",
                FunctionTransformer(_replace_infinity, feature_names_out="one-to-one"),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical_pipeline, categorical),
            ("numeric", numeric_pipeline, numeric),
        ],
        sparse_threshold=1.0,
    )
    classifier = LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        class_weight=candidate.class_weight,
        C=candidate.C,
        l1_ratio=candidate.l1_ratio,
        max_iter=candidate.max_iter,
        tol=candidate.tol,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _candidate_overall_row(
    candidate: Candidate,
    target: np.ndarray,
    probability: np.ndarray,
    fold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = calculate_metrics(target, probability)
    total_seconds = float(sum(row["training_seconds"] for row in fold_rows))
    warning_folds = int(sum(bool(row["convergence_warning"]) for row in fold_rows))
    n_iter_max = int(max(row["n_iter"] for row in fold_rows))
    return {
        "candidate": candidate.name,
        "class_weight": candidate.class_weight,
        "C": candidate.C,
        "l1_ratio": candidate.l1_ratio,
        "max_iter": candidate.max_iter,
        "tol": candidate.tol,
        **metrics,
        "mean_oof_probability": float(probability.mean()),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "convergence_warning_folds": warning_folds,
        "max_observed_n_iter": n_iter_max,
        "training_seconds": total_seconds,
    }


def run_elasticnet(
    data_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_rows: int | None = None,
    write_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """제한된 두 후보를 전체 3-Fold OOF로 비교하고 최종 CSV를 만든다."""
    prepared = load_and_prepare_weekly_data(data_path, max_rows=max_rows)
    folds, fold_assignment, fold_hash = make_group_folds(prepared.groups)
    candidate_probabilities = {
        candidate.name: np.full(len(prepared.data), np.nan, dtype=float)
        for candidate in CANDIDATES
    }
    candidate_fold_rows: dict[str, list[dict[str, Any]]] = {
        candidate.name: [] for candidate in CANDIDATES
    }
    search_started = time.perf_counter()

    print("=== 데이터 검증 ===")
    print(pd.Series(prepared.profile).to_string())
    for fold, train_index, validation_index in folds:
        for candidate in CANDIDATES:
            pipeline = build_pipeline(prepared.categorical, prepared.numeric, candidate)
            fold_started = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", category=ConvergenceWarning)
                pipeline.fit(
                    prepared.features.iloc[train_index],
                    prepared.target[train_index],
                )
            fold_seconds = time.perf_counter() - fold_started
            convergence_warning = any(
                issubclass(warning.category, ConvergenceWarning) for warning in captured
            )
            fold_probability = pipeline.predict_proba(
                prepared.features.iloc[validation_index]
            )[:, 1]
            candidate_probabilities[candidate.name][validation_index] = fold_probability

            classifier = pipeline.named_steps["classifier"]
            row = fold_metadata(
                MODEL_NAME,
                fold,
                train_index,
                validation_index,
                prepared.target,
                prepared.groups,
            )
            row.update(calculate_metrics(prepared.target[validation_index], fold_probability))
            row.update(
                {
                    "candidate": candidate.name,
                    "class_weight": candidate.class_weight,
                    "C": candidate.C,
                    "l1_ratio": candidate.l1_ratio,
                    "max_iter": candidate.max_iter,
                    "tol": candidate.tol,
                    "convergence_warning": convergence_warning,
                    "n_iter": int(classifier.n_iter_[0]),
                    "training_seconds": fold_seconds,
                    "mean_probability": float(fold_probability.mean()),
                    "probability_min": float(fold_probability.min()),
                    "probability_max": float(fold_probability.max()),
                }
            )
            candidate_fold_rows[candidate.name].append(row)
            print(
                f"Fold {fold} / {candidate.name} 완료: "
                f"PR-AUC={row['pr_auc']:.6f}, n_iter={row['n_iter']}, "
                f"warning={convergence_warning}, seconds={fold_seconds:.1f}"
            )

    candidate_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        probability = validate_oof(
            candidate_probabilities[candidate.name],
            fold_assignment,
            len(prepared.data),
        )
        candidate_rows.append(
            _candidate_overall_row(
                candidate,
                prepared.target,
                probability,
                candidate_fold_rows[candidate.name],
            )
        )
    candidate_frame = pd.DataFrame(candidate_rows)
    # 순위 성능(PR-AUC)을 우선하고 동률이면 Brier와 ECE가 낮은 후보를 선택한다.
    selected_row = candidate_frame.sort_values(
        ["pr_auc", "brier_score", "ece_10bin"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    selected_name = str(selected_row["candidate"])
    selected_candidate = next(c for c in CANDIDATES if c.name == selected_name)
    probabilities = candidate_probabilities[selected_name]
    selected_fold_frame = pd.DataFrame(candidate_fold_rows[selected_name])
    search_seconds = time.perf_counter() - search_started

    overall: dict[str, Any] = {
        "model": MODEL_NAME,
        "rows": prepared.profile["rows"],
        "target_count": prepared.profile["target_count"],
        "target_rate": prepared.profile["target_rate"],
        "feature_count": prepared.profile["feature_count"],
        "categorical_feature_count": prepared.profile["categorical_feature_count"],
        **calculate_metrics(prepared.target, probabilities),
        "n_splits": N_SPLITS,
        "group_column": ID_COL,
        "random_state": RANDOM_STATE,
        "data_schema_hash": prepared.profile["data_schema_hash"],
        "fold_assignment_hash": fold_hash,
        "training_seconds": float(selected_row["training_seconds"]),
        "candidate_search_seconds": search_seconds,
        "candidate": selected_candidate.name,
        **{key: value for key, value in asdict(selected_candidate).items() if key != "name"},
        "selection_basis": "OOF PR-AUC 최대; 동률 시 Brier Score, ECE 최소",
        "convergence_warning_folds": int(selected_row["convergence_warning_folds"]),
        "max_observed_n_iter": int(selected_row["max_observed_n_iter"]),
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
    }
    metrics_frame = pd.DataFrame([overall])
    oof_frame = prepared.data[[*SORT_COLUMNS, TARGET_COL]].copy()
    oof_frame["fold"] = fold_assignment
    oof_frame["elasticnet_logistic_oof_probability"] = probabilities

    if not np.array_equal(oof_frame[TARGET_COL].to_numpy(), prepared.target):
        raise RuntimeError("OOF Target 순서가 안정 정렬된 원본 Target과 다릅니다.")
    if oof_frame.duplicated(SORT_COLUMNS).any():
        raise RuntimeError("OOF 결과에 복합키 중복이 있습니다.")

    if write_outputs:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        metrics_frame.to_csv(
            destination / "elasticnet_logistic_weekly_next_week_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        selected_fold_frame.to_csv(
            destination / "elasticnet_logistic_weekly_next_week_fold_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        oof_frame.to_csv(
            destination / "elasticnet_logistic_weekly_next_week_oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        candidate_frame.to_csv(
            destination / "elasticnet_logistic_candidate_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n=== ElasticNet 후보 비교 ===")
    print(candidate_frame.to_string(index=False))
    print("\n=== 선택 모델 ===")
    print(metrics_frame.to_string(index=False))
    return {
        "metrics": metrics_frame,
        "fold_metrics": selected_fold_frame,
        "candidate_metrics": candidate_frame,
        "oof": oof_frame,
    }


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
    run_elasticnet(
        data_path,
        output_dir=args.output_dir,
        max_rows=args.smoke_rows,
        write_outputs=args.smoke_rows is None,
    )


if __name__ == "__main__":
    main()

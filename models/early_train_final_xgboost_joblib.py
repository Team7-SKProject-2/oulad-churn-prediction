"""기존 전체 주차 데이터로 XGBoost를 학습해 1~10주차 운영 artifact를 저장한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

try:
    from .early_final_artifact_common import (
        FinalArtifactConfig,
        add_final_artifact_cli_arguments,
        load_final_training_data,
        print_artifact_result,
        resolve_decision_threshold,
        save_final_artifact,
    )
except ImportError:  # 직접 실행 지원
    from early_final_artifact_common import (
        FinalArtifactConfig,
        add_final_artifact_cli_arguments,
        load_final_training_data,
        print_artifact_result,
        resolve_decision_threshold,
        save_final_artifact,
    )


CONFIG = FinalArtifactConfig(
    model_name="Early weighted XGBoost",
    artifact_filename="early_xgboost.joblib",
    profiles_filename="early_xgboost_cohort_profiles.csv",
    threshold_results_path=(
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "threshold_analysis"
        / "early_xgboost"
        / "early_xgboost_optimal_f1_summary.json"
    ),
    probability_column="early_xgboost_scaled_oof_probability",
)


def build_final_pipeline(
    categorical: list[str], numeric: list[str], scale_pos_weight: float
) -> Pipeline:
    """One-Hot 전처리와 전체 Early 양성 가중치를 포함한 XGBoost Pipeline을 만든다."""
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ],
        sparse_threshold=0.3,
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        scale_pos_weight=float(scale_pos_weight),
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])


def run(args: argparse.Namespace) -> dict[str, str]:
    data_path, prepared = load_final_training_data(args)
    threshold, threshold_source = resolve_decision_threshold(
        args.threshold, args.eval_results_json
    )
    positive_count = int(prepared.target.sum())
    if positive_count == 0:
        raise ValueError("Early 전체 학습 데이터에 양성 클래스가 없습니다.")
    scale_pos_weight = float((len(prepared.target) - positive_count) / positive_count)
    features = prepared.features.copy()
    for column in prepared.categorical:
        features[column] = features[column].fillna("미상").astype(str)
    features[prepared.numeric] = features[prepared.numeric].replace(
        [np.inf, -np.inf], np.nan
    )
    pipeline = build_final_pipeline(
        prepared.categorical, prepared.numeric, scale_pos_weight
    )
    pipeline.fit(features, prepared.target)
    classifier = pipeline.named_steps["model"]
    training_parameters = classifier.get_params()
    training_parameters["scale_pos_weight_source"] = (
        "early_all_negative_count / early_all_positive_count"
    )
    paths = save_final_artifact(
        config=CONFIG,
        model=pipeline,
        prepared=prepared,
        data_path=data_path,
        artifact_dir=args.artifact_dir,
        threshold=threshold,
        threshold_source=threshold_source,
        training_parameters=training_parameters,
        categorical_features=prepared.categorical,
        preprocessing={
            "type": "sklearn Pipeline",
            "categorical": "OneHotEncoder(handle_unknown='ignore')",
            "numeric": "passthrough",
            "categorical_missing_value": "미상",
            "numeric_infinity_to_nan": True,
        },
    )
    print_artifact_result(paths, prepared, threshold)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_final_artifact_cli_arguments(parser, CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

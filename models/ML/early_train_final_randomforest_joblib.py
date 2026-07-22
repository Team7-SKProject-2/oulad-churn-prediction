"""기존 전체 주차 데이터로 RandomForest를 학습해 1~10주차 운영 artifact를 저장한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ML_DIR = Path(__file__).resolve().parent
MODELS_DIR = ML_DIR.parent
PROJECT_ROOT = MODELS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.early_final_artifact_common import (  # noqa: E402
    FinalArtifactConfig,
    add_final_artifact_cli_arguments,
    load_final_training_data,
    print_artifact_result,
    resolve_decision_threshold,
    save_final_artifact,
)


CONFIG = FinalArtifactConfig(
    model_name="Early RandomForest",
    artifact_filename="early_randomforest.joblib",
    profiles_filename="early_randomforest_cohort_profiles.csv",
    threshold_results_path=(
        PROJECT_ROOT
        / "outputs"
        / "threshold_analysis"
        / "early_randomforest"
        / "early_randomforest_optimal_f1_summary.json"
    ),
    probability_column="early_randomforest_oof_probability",
)


def build_final_pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
    """One-Hot 전처리와 최종 RandomForest를 함께 저장할 Pipeline을 만든다."""
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ],
        sparse_threshold=0.3,
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        criterion="entropy",
        max_depth=18,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=0.7,
        bootstrap=True,
        max_samples=0.8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])


def run(args: argparse.Namespace) -> dict[str, str]:
    data_path, prepared = load_final_training_data(args)
    threshold, threshold_source = resolve_decision_threshold(
        args.threshold, args.eval_results_json
    )
    features = prepared.features.copy()
    for column in prepared.categorical:
        features[column] = features[column].fillna("미상").astype(str)
    features[prepared.numeric] = features[prepared.numeric].replace(
        [np.inf, -np.inf], np.nan
    )
    pipeline = build_final_pipeline(prepared.categorical, prepared.numeric)
    pipeline.fit(features, prepared.target)
    classifier = pipeline.named_steps["model"]
    paths = save_final_artifact(
        config=CONFIG,
        model=pipeline,
        prepared=prepared,
        data_path=data_path,
        artifact_dir=args.artifact_dir,
        threshold=threshold,
        threshold_source=threshold_source,
        training_parameters=classifier.get_params(),
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

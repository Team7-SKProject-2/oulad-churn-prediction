"""기존 전체 주차 데이터로 CatBoost를 학습해 1~10주차 운영 artifact를 저장한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

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
    model_name="Early CatBoost",
    artifact_filename="early_catboost.joblib",
    profiles_filename="early_catboost_cohort_profiles.csv",
    threshold_results_path=(
        PROJECT_ROOT
        / "outputs"
        / "threshold_analysis"
        / "early_catboot"
        / "early_catboost_optimal_f1_summary.json"
    ),
    probability_column="early_catboost_oof_probability",
)
DEFAULT_BASE_ARTIFACT = PROJECT_ROOT / "artifacts" / "catboost.joblib"
DEFAULT_BASE_PROFILES = (
    PROJECT_ROOT / "artifacts" / "catboost_cohort_profiles.csv"
)


def build_final_model() -> CatBoostClassifier:
    """OOF 후보와 동일한 설정으로 Early 전체 데이터를 학습할 최종 모델을 만든다."""
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=500,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def run(args: argparse.Namespace) -> dict[str, str]:
    threshold, threshold_source = resolve_decision_threshold(
        args.threshold, args.eval_results_json
    )
    if not args.retrain:
        if not args.base_artifact.is_file() or not args.base_profiles_csv.is_file():
            raise FileNotFoundError(
                "재사용할 기존 CatBoost artifact 또는 프로필 CSV가 없습니다. "
                "--retrain과 --data-path를 사용해 다시 학습하세요."
            )
        artifact = joblib.load(args.base_artifact)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError(f"기존 CatBoost joblib 형식이 올바르지 않습니다: {args.base_artifact}")
        if not hasattr(artifact["model"], "predict_proba"):
            raise ValueError("기존 CatBoost artifact의 model이 predict_proba를 지원하지 않습니다.")
        profiles = pd.read_csv(args.base_profiles_csv)
        if "prediction_week" not in profiles.columns:
            raise ValueError("기존 코호트 CSV에 prediction_week가 없습니다.")
        profiles = profiles.loc[
            pd.to_numeric(profiles["prediction_week"], errors="coerce").between(
                args.operating_start_week, args.operating_end_week
            )
        ].copy()
        if profiles.empty:
            raise ValueError("기존 코호트 CSV에 1~10주차 프로필이 없습니다.")

        destination = args.artifact_dir
        destination.mkdir(parents=True, exist_ok=True)
        model_path = destination / CONFIG.artifact_filename
        profiles_path = destination / CONFIG.profiles_filename
        early_artifact = dict(artifact)
        early_artifact.update(
            {
                "model_name": CONFIG.model_name,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "training_scope": "full_existing_weekly_data",
                "start_week": args.operating_start_week,
                "end_week": args.operating_end_week,
                "probability_column": CONFIG.probability_column,
                "cohort_profiles_file": profiles_path.name,
                "source_artifact": str(args.base_artifact.resolve()),
                "smoke_test": False,
            }
        )
        joblib.dump(early_artifact, model_path, compress=3)
        profiles.to_csv(profiles_path, index=False, encoding="utf-8-sig")
        loaded = joblib.load(model_path)
        if loaded["threshold"] != threshold or not hasattr(
            loaded["model"], "predict_proba"
        ):
            raise RuntimeError("Early CatBoost artifact 재로딩 검증에 실패했습니다.")
        print(f"기존 CatBoost 모델 재사용: {args.base_artifact}")
        print(f"저장 모델: {model_path}")
        print(f"코호트 프로필: {profiles_path}")
        print(f"1~10주차 운영 임계값: {threshold:.9f}")
        return {
            "joblib": str(model_path.resolve()),
            "cohort_profiles_csv": str(profiles_path.resolve()),
        }

    data_path, prepared = load_final_training_data(args)
    features = prepared.features.copy()
    for column in prepared.categorical:
        features[column] = features[column].fillna("미상").astype(str)
    features[prepared.numeric] = features[prepared.numeric].replace(
        [np.inf, -np.inf], np.nan
    )
    model = build_final_model()
    model.fit(
        features,
        prepared.target,
        cat_features=prepared.categorical,
        verbose=False,
    )
    paths = save_final_artifact(
        config=CONFIG,
        model=model,
        prepared=prepared,
        data_path=data_path,
        artifact_dir=args.artifact_dir,
        threshold=threshold,
        threshold_source=threshold_source,
        training_parameters=model.get_params(),
        categorical_features=prepared.categorical,
        preprocessing={
            "type": "CatBoost native categorical",
            "categorical_missing_value": "미상",
            "numeric_infinity_to_nan": True,
        },
    )
    print_artifact_result(paths, prepared, threshold)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_final_artifact_cli_arguments(parser, CONFIG)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="기존 CatBoost artifact를 재사용하지 않고 전체 CSV로 다시 학습합니다.",
    )
    parser.add_argument(
        "--base-artifact", type=Path, default=DEFAULT_BASE_ARTIFACT
    )
    parser.add_argument(
        "--base-profiles-csv", type=Path, default=DEFAULT_BASE_PROFILES
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

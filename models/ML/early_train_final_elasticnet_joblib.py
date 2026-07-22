"""기존 전체 주차 데이터로 ElasticNet을 학습해 1~10주차 운영 artifact를 저장한다."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

ML_DIR = Path(__file__).resolve().parent
MODELS_DIR = ML_DIR.parent
PROJECT_ROOT = MODELS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_elasticnet_module = importlib.import_module(
    "models.ML.04_elasticnet_logistic_weekly_next_week"
)
CANDIDATES = _elasticnet_module.CANDIDATES
Candidate = _elasticnet_module.Candidate
build_pipeline = _elasticnet_module.build_pipeline

from models.early_final_artifact_common import (  # noqa: E402
    FinalArtifactConfig,
    add_final_artifact_cli_arguments,
    load_final_training_data,
    print_artifact_result,
    resolve_decision_threshold,
    save_final_artifact,
)
DEFAULT_MODEL_METRICS_CSV = (
    PROJECT_ROOT
    / "models"
    / "ML"
    / "elasticnet_logistic_weekly_next_week_metrics.csv"
)
CONFIG = FinalArtifactConfig(
    model_name="Early ElasticNet Logistic Regression",
    artifact_filename="early_elasticnet.joblib",
    profiles_filename="early_elasticnet_cohort_profiles.csv",
    threshold_results_path=(
        PROJECT_ROOT
        / "outputs"
        / "threshold_analysis"
        / "early_elasticNet"
        / "early_elasticnet_optimal_f1_summary.json"
    ),
    probability_column="early_elasticnet_logistic_oof_probability",
)


def resolve_candidate(
    candidate_name: str | None, model_metrics_csv: Path
) -> tuple[Candidate, str]:
    """명시값 또는 기존 전체 주차 모델 metrics CSV에서 최종 후보를 결정한다."""
    if candidate_name:
        selected = next(
            (candidate for candidate in CANDIDATES if candidate.name == candidate_name),
            None,
        )
        if selected is None:
            raise ValueError(f"지원하지 않는 ElasticNet 후보입니다: {candidate_name}")
        return selected, "user_supplied_candidate"

    path = Path(model_metrics_csv)
    if not path.is_file():
        raise FileNotFoundError(
            "기존 ElasticNet 모델 metrics CSV가 없습니다. "
            f"실행하거나 --candidate를 지정하세요: {path}"
        )
    metrics = pd.read_csv(path)
    if metrics.empty or "candidate" not in metrics.columns:
        raise ValueError(f"metrics CSV에 candidate 값이 없습니다: {path}")
    selected_name = str(metrics.iloc[0]["candidate"])
    selected = next(
        (candidate for candidate in CANDIDATES if candidate.name == selected_name),
        None,
    )
    if selected is None:
        raise ValueError(f"metrics CSV의 ElasticNet 후보를 확인할 수 없습니다: {selected_name}")
    return selected, str(path.resolve())


def run(args: argparse.Namespace) -> dict[str, str]:
    data_path, prepared = load_final_training_data(args)
    threshold, threshold_source = resolve_decision_threshold(
        args.threshold, args.eval_results_json
    )
    candidate, candidate_source = resolve_candidate(
        args.candidate, args.model_metrics_csv
    )
    pipeline = build_pipeline(prepared.categorical, prepared.numeric, candidate)
    pipeline.fit(prepared.features, prepared.target)
    classifier = pipeline.named_steps["classifier"]
    training_parameters = classifier.get_params()
    training_parameters.update(
        {"candidate": candidate.name, "candidate_source": candidate_source}
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
            "categorical": "constant imputer('미상') + OneHotEncoder(handle_unknown='ignore')",
            "numeric": "infinity-to-NaN + median imputer + StandardScaler(with_mean=False)",
        },
    )
    print_artifact_result(paths, prepared, threshold)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_final_artifact_cli_arguments(parser, CONFIG)
    parser.add_argument(
        "--candidate",
        choices=[candidate.name for candidate in CANDIDATES],
        help="최종 후보. 생략하면 기존 전체 주차 모델 metrics CSV에서 읽습니다.",
    )
    parser.add_argument(
        "--model-metrics-csv", type=Path, default=DEFAULT_MODEL_METRICS_CSV
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

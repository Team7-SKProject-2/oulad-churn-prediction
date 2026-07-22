"""Early 최종 모델의 joblib 딕셔너리와 코호트 프로필 CSV 저장 계약."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from .common_weekly_metrics import (
        ID_COL,
        TARGET_COL,
        PreparedWeeklyData,
        load_and_prepare_weekly_data,
    )
    from .early_weekly_common import (
        DEFAULT_END_WEEK,
        DEFAULT_START_WEEK,
        resolved_data_path,
    )
except ImportError:  # ``python models/ML/early_train_final_*.py`` 직접 실행 지원
    from common_weekly_metrics import (
        ID_COL,
        TARGET_COL,
        PreparedWeeklyData,
        load_and_prepare_weekly_data,
    )
    from early_weekly_common import (
        DEFAULT_END_WEEK,
        DEFAULT_START_WEEK,
        resolved_data_path,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
GROUP_COLUMNS = ["code_module", "code_presentation", "prediction_week"]


@dataclass(frozen=True)
class FinalArtifactConfig:
    """모델별 최종 artifact 파일명과 확률 의미를 정의한다."""

    model_name: str
    artifact_filename: str
    profiles_filename: str
    threshold_results_path: Path
    probability_column: str


def add_final_artifact_cli_arguments(
    parser: argparse.ArgumentParser,
    config: FinalArtifactConfig,
) -> None:
    """네 최종 학습 스크립트가 공유하는 실행 옵션을 추가한다."""
    parser.add_argument(
        "--data-path",
        type=Path,
        help="학습 CSV. 생략하면 OULAD_DATA_PATH 또는 저장소 기본 경로를 확인합니다.",
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--operating-start-week", type=int, default=DEFAULT_START_WEEK
    )
    parser.add_argument("--operating-end-week", type=int, default=DEFAULT_END_WEEK)
    parser.add_argument(
        "--threshold",
        type=float,
        help="운영 임계값. 생략하면 해당 Early OOF eval_results JSON에서 읽습니다.",
    )
    parser.add_argument(
        "--eval-results-json",
        type=Path,
        default=config.threshold_results_path,
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        help="개발 확인용 CSV 선두 행 수. 이 artifact는 운영에 사용하면 안 됩니다.",
    )


def load_final_training_data(args: argparse.Namespace) -> tuple[Path, PreparedWeeklyData]:
    """기존 OOF와 같은 전체 주차 CSV를 그대로 불러오고 운영 주차만 기록한다."""
    data_path = resolved_data_path(args.data_path)
    if args.operating_start_week < 1 or args.operating_end_week < args.operating_start_week:
        raise ValueError("운영 주차는 1 <= start <= end여야 합니다.")
    prepared = load_and_prepare_weekly_data(data_path, max_rows=args.smoke_rows)
    prediction_week = pd.to_numeric(
        prepared.data["prediction_week"], errors="raise"
    )
    prepared.profile.update(
        {
            "training_scope": "full_existing_weekly_data",
            "training_start_week": int(prediction_week.min()),
            "training_end_week": int(prediction_week.max()),
            "operating_start_week": int(args.operating_start_week),
            "operating_end_week": int(args.operating_end_week),
            "smoke_test": args.smoke_rows is not None,
            "smoke_rows_requested": args.smoke_rows,
        }
    )
    return data_path, prepared


def resolve_decision_threshold(
    explicit_threshold: float | None,
    eval_results_json: Path,
) -> tuple[float, str]:
    """명시값 또는 Early OOF 결과에서 운영 임계값을 안전하게 결정한다."""
    if explicit_threshold is not None:
        if not 0 <= explicit_threshold <= 1:
            raise ValueError("--threshold는 0과 1 사이여야 합니다.")
        return float(explicit_threshold), "user_supplied_threshold"

    path = Path(eval_results_json)
    if not path.is_file():
        raise FileNotFoundError(
            "Early OOF 임계값 결과 JSON이 없습니다. 먼저 해당 early_*_weekly_next_week.py를 "
            f"실행하거나 --threshold를 지정하세요: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation_data", {}).get("smoke_test"):
        raise ValueError(
            f"스모크 테스트 결과의 임계값은 최종 모델에 사용할 수 없습니다: {path}"
        )
    if "selected_threshold" in payload:
        threshold_value = payload["selected_threshold"]
    elif "selected" in payload and "threshold" in payload["selected"]:
        threshold_value = payload["selected"]["threshold"]
    else:
        raise ValueError(
            f"selected_threshold 또는 selected.threshold가 결과 JSON에 없습니다: {path}"
        )
    threshold = float(threshold_value)
    if not 0 <= threshold <= 1:
        raise ValueError(f"결과 JSON의 임계값이 0~1 범위를 벗어났습니다: {threshold}")
    return threshold, str(path.resolve())


def _first_mode(series: pd.Series) -> Any:
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else np.nan


def build_cohort_profiles(
    prepared: PreparedWeeklyData,
    feature_columns: list[str],
    categorical_features: list[str],
    start_week: int | None = None,
    end_week: int | None = None,
) -> pd.DataFrame:
    """08/11 CatBoost artifact와 같은 Feature 순서의 코호트 기준 CSV를 만든다."""
    missing_groups = [column for column in GROUP_COLUMNS if column not in feature_columns]
    if missing_groups:
        raise ValueError(f"코호트 프로필 그룹 Feature가 없습니다: {missing_groups}")
    missing_features = [
        column for column in feature_columns if column not in prepared.data.columns
    ]
    if missing_features:
        raise ValueError(f"학습 데이터에 모델 Feature가 없습니다: {missing_features}")

    profile_data = prepared.data
    if start_week is not None and end_week is not None:
        profile_data = profile_data.loc[
            pd.to_numeric(profile_data["prediction_week"], errors="coerce").between(
                start_week, end_week
            )
        ].copy()
        if profile_data.empty:
            raise ValueError(f"{start_week}~{end_week}주차 코호트 프로필 행이 없습니다.")

    numeric = [
        column
        for column in feature_columns
        if column not in categorical_features and column not in GROUP_COLUMNS
    ]
    numeric_profiles = (
        profile_data.groupby(GROUP_COLUMNS, observed=True, dropna=False)[numeric]
        .median(numeric_only=True)
        .reset_index()
    )
    non_key_categorical = [
        column for column in categorical_features if column not in GROUP_COLUMNS
    ]
    if non_key_categorical:
        categorical_profiles = (
            profile_data.groupby(GROUP_COLUMNS, observed=True, dropna=False)[
                non_key_categorical
            ]
            .agg(_first_mode)
            .reset_index()
        )
        profiles = numeric_profiles.merge(
            categorical_profiles,
            on=GROUP_COLUMNS,
            how="left",
            validate="one_to_one",
        )
    else:
        profiles = numeric_profiles

    for column in categorical_features:
        profiles[column] = profiles[column].fillna("미상").astype(str)
    profiles[numeric] = profiles[numeric].replace([np.inf, -np.inf], np.nan)
    profiles = profiles[feature_columns].sort_values(GROUP_COLUMNS).reset_index(drop=True)
    if profiles.duplicated(GROUP_COLUMNS).any():
        raise ValueError("생성된 코호트 프로필 복합키가 중복됩니다.")
    return profiles


def save_final_artifact(
    *,
    config: FinalArtifactConfig,
    model: Any,
    prepared: PreparedWeeklyData,
    data_path: Path,
    artifact_dir: Path,
    threshold: float,
    threshold_source: str,
    training_parameters: dict[str, Any],
    categorical_features: list[str],
    preprocessing: dict[str, Any],
) -> dict[str, str]:
    """08과 호환되는 joblib 딕셔너리 및 모델별 코호트 CSV를 저장한다."""
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / config.artifact_filename
    profiles_path = destination / config.profiles_filename
    feature_columns = prepared.features.columns.tolist()
    profiles = build_cohort_profiles(
        prepared,
        feature_columns,
        categorical_features,
        int(prepared.profile["operating_start_week"]),
        int(prepared.profile["operating_end_week"]),
    )

    artifact = {
        # 기존 08_train_final_catboost_joblib.py와 동일한 핵심 키
        "model_name": config.model_name,
        "model": model,
        "feature_columns": feature_columns,
        "categorical_features": list(categorical_features),
        "target_column": TARGET_COL,
        "id_column": ID_COL,
        "data_file": data_path.name,
        "training_rows": len(prepared.data),
        "feature_count": len(feature_columns),
        "target_rate": float(prepared.target.mean()),
        "threshold": float(threshold),
        "training_parameters": training_parameters,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        # Early 및 Pipeline 모델에 필요한 추가 메타데이터
        "artifact_version": 1,
        "training_scope": "full_existing_weekly_data",
        "training_start_week": int(prepared.profile["training_start_week"]),
        "training_end_week": int(prepared.profile["training_end_week"]),
        "start_week": int(prepared.profile["operating_start_week"]),
        "end_week": int(prepared.profile["operating_end_week"]),
        "positive_label": 1,
        "probability_column": config.probability_column,
        "threshold_source": threshold_source,
        "preprocessing": preprocessing,
        "cohort_profiles_file": profiles_path.name,
        "smoke_test": bool(prepared.profile.get("smoke_test", False)),
    }
    joblib.dump(artifact, model_path, compress=3)
    profiles.to_csv(profiles_path, index=False, encoding="utf-8-sig")

    # 저장 직후 재로딩해 손상 여부와 Streamlit 핵심 계약을 확인한다.
    loaded = joblib.load(model_path)
    required = {
        "model",
        "feature_columns",
        "categorical_features",
        "target_column",
        "threshold",
    }
    missing = sorted(required.difference(loaded))
    if missing:
        raise RuntimeError(f"저장 joblib의 필수 키가 없습니다: {missing}")
    if not hasattr(loaded["model"], "predict_proba"):
        raise RuntimeError("저장 모델이 predict_proba를 지원하지 않습니다.")
    if loaded["feature_columns"] != feature_columns:
        raise RuntimeError("저장 후 Feature 순서가 변경되었습니다.")

    return {
        "joblib": str(model_path.resolve()),
        "cohort_profiles_csv": str(profiles_path.resolve()),
    }


def print_artifact_result(
    paths: dict[str, str], prepared: PreparedWeeklyData, threshold: float
) -> None:
    model_path = Path(paths["joblib"])
    profiles_path = Path(paths["cohort_profiles_csv"])
    print(f"저장 모델: {model_path}")
    print(f"코호트 프로필: {profiles_path}")
    print(
        f"학습 행 수: {len(prepared.data):,}, Feature 수: {prepared.features.shape[1]}, "
        f"운영 임계값: {threshold:.9f}"
    )
    print(f"joblib 크기: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"CSV 크기: {profiles_path.stat().st_size / 1024 / 1024:.2f} MB")

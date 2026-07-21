"""최종 CatBoost와 Streamlit을 연결하는 공통 추론 인터페이스.

모델과 124개 Feature 순서는 ``catboost.joblib``에서 읽고, 1~10주차
조기경보 운영 구간과 임계값은 ``early_service_config.json``에서 읽는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "artifacts" / "catboost.joblib"
PROFILE_PATH = PROJECT_ROOT / "models" / "artifacts" / "catboost_cohort_profiles.csv"
SERVICE_CONFIG_PATH = PROJECT_ROOT / "models" / "artifacts" / "early_service_config.json"
REQUIRED_ARTIFACT_KEYS = {
    "model",
    "feature_columns",
    "categorical_features",
    "threshold",
}
PROFILE_KEY_COLUMNS = ["code_module", "code_presentation", "prediction_week"]


@st.cache_resource(show_spinner=False)
def load_model_artifact() -> dict | None:
    """모델 artifact를 읽고 추론에 필요한 메타데이터를 검증한다."""
    if not MODEL_PATH.exists():
        return None
    import joblib

    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict):
        raise TypeError("CatBoost artifact는 모델과 메타데이터를 담은 dict여야 합니다.")
    missing = REQUIRED_ARTIFACT_KEYS.difference(artifact)
    if missing:
        raise ValueError(f"CatBoost artifact 필수 항목이 없습니다: {sorted(missing)}")
    if len(artifact["feature_columns"]) != artifact.get("feature_count", 124):
        raise ValueError("CatBoost Feature 개수 메타데이터가 실제 순서와 다릅니다.")
    return artifact


@st.cache_data(show_spinner=False)
def load_service_config() -> dict:
    """1~10주차 Early 서비스 설정을 읽는다."""
    if not SERVICE_CONFIG_PATH.exists():
        return {"start_week": 1, "end_week": 10}
    return json.loads(SERVICE_CONFIG_PATH.read_text(encoding="utf-8"))


def load_model():
    """기존 화면 코드와의 호환을 위해 실제 CatBoost 모델 객체를 반환한다."""
    artifact = load_model_artifact()
    return None if artifact is None else artifact["model"]


def model_ready() -> bool:
    """최종 CatBoost artifact가 연결되어 있는지 반환한다."""
    return load_model_artifact() is not None


def service_week_range() -> tuple[int, int]:
    config = load_service_config()
    return int(config.get("start_week", 1)), int(config.get("end_week", 10))


def decision_threshold() -> float:
    """Early 운영용 다음 주 이탈 위험 분류 임계값을 반환한다."""
    config = load_service_config()
    if "decision_threshold" in config:
        return float(config["decision_threshold"])
    return float(_require_artifact()["threshold"])


def model_info() -> dict:
    """화면 표시용 모델·서비스 정보를 반환한다."""
    artifact = _require_artifact()
    start_week, end_week = service_week_range()
    return {
        "model_name": artifact.get("model_name", "CatBoost"),
        "feature_count": len(artifact["feature_columns"]),
        "training_rows": artifact.get("training_rows"),
        "target_rate": artifact.get("target_rate"),
        "threshold": decision_threshold(),
        "service_start_week": start_week,
        "service_end_week": end_week,
        "trained_at": artifact.get("trained_at"),
    }


def required_feature_columns() -> list[str]:
    """학습 시 저장한 124개 Feature 순서를 반환한다."""
    return list(_require_artifact()["feature_columns"])


def _require_artifact() -> dict:
    artifact = load_model_artifact()
    if artifact is None:
        raise FileNotFoundError(f"최종 CatBoost 모델이 없습니다: {MODEL_PATH}")
    return artifact


def feature_coverage(df: pd.DataFrame) -> dict:
    """입력 데이터가 최종 모델 Feature를 얼마나 포함하는지 진단한다."""
    required = required_feature_columns()
    missing = [column for column in required if column not in df.columns]
    return {
        "required_count": len(required),
        "available_count": len(required) - len(missing),
        "missing_columns": missing,
    }


def prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """입력을 학습 당시 Feature 순서와 자료형으로 맞춘다."""
    artifact = _require_artifact()
    feature_columns = list(artifact["feature_columns"])
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(f"CatBoost 입력 Feature {len(missing)}개가 없습니다: {preview}{suffix}")

    features = df.loc[:, feature_columns].copy()
    categorical = list(artifact["categorical_features"])
    numeric = [column for column in feature_columns if column not in categorical]
    for column in categorical:
        features[column] = features[column].fillna("미상").astype(str)
    features[numeric] = features[numeric].apply(pd.to_numeric, errors="coerce")
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)
    return features


def predict_probabilities(df: pd.DataFrame) -> np.ndarray:
    """각 행의 다음 주 중도이탈 확률을 0~1 범위로 반환한다."""
    if df.empty:
        return np.array([], dtype=float)
    artifact = _require_artifact()
    features = prepare_model_input(df)
    probabilities = np.asarray(artifact["model"].predict_proba(features)[:, 1], dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("CatBoost가 NaN 또는 무한대 확률을 반환했습니다.")
    return probabilities


def predict_risk(df: pd.DataFrame) -> np.ndarray:
    """기존 화면 호환용으로 다음 주 이탈확률을 0~100 점수로 반환한다."""
    return predict_probabilities(df) * 100.0


def predict_is_at_risk(df: pd.DataFrame) -> np.ndarray:
    """Early 운영 임계값 기준 위험 학생 여부를 반환한다."""
    return predict_probabilities(df) >= decision_threshold()


def prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """화면에서 바로 사용할 확률·위험 여부 결과를 반환한다."""
    probabilities = predict_probabilities(df)
    threshold = decision_threshold()
    return pd.DataFrame(
        {
            "next_week_withdrawal_probability": probabilities,
            "risk_score_pct": probabilities * 100.0,
            "is_at_risk": probabilities >= threshold,
            "decision_threshold": threshold,
        },
        index=df.index,
    )


@st.cache_data(show_spinner=False)
def load_cohort_profiles() -> pd.DataFrame:
    """과목·운영회차·주차별 124개 Feature 기준 프로필을 읽는다."""
    if not PROFILE_PATH.exists():
        return pd.DataFrame()
    profiles = pd.read_csv(PROFILE_PATH, low_memory=False)
    coverage = feature_coverage(profiles)
    if coverage["missing_columns"]:
        raise ValueError(
            "CatBoost 코호트 프로필의 Feature가 불완전합니다: "
            f"{coverage['missing_columns'][:10]}"
        )
    if profiles.duplicated(PROFILE_KEY_COLUMNS).any():
        raise ValueError("CatBoost 코호트 프로필의 복합키가 중복됩니다.")
    return profiles


def cohort_profiles_ready() -> bool:
    return PROFILE_PATH.exists() and not load_cohort_profiles().empty


def cohort_profile(
    profiles: pd.DataFrame,
    code_module: str,
    code_presentation: str,
    prediction_week: int,
) -> pd.Series:
    """선택한 과목·운영회차·예측주차의 모델 입력 기준 행을 반환한다."""
    selected = profiles.loc[
        profiles["code_module"].eq(code_module)
        & profiles["code_presentation"].eq(code_presentation)
        & profiles["prediction_week"].eq(prediction_week)
    ]
    if len(selected) != 1:
        raise ValueError("선택한 과목·운영회차·예측주차의 CatBoost 프로필을 하나로 찾지 못했습니다.")
    return selected.iloc[0].copy()


__all__ = [
    "MODEL_PATH",
    "PROFILE_PATH",
    "SERVICE_CONFIG_PATH",
    "cohort_profile",
    "cohort_profiles_ready",
    "decision_threshold",
    "feature_coverage",
    "load_cohort_profiles",
    "load_model",
    "load_model_artifact",
    "load_service_config",
    "model_info",
    "model_ready",
    "predict_is_at_risk",
    "predict_probabilities",
    "predict_risk",
    "prediction_frame",
    "prepare_model_input",
    "required_feature_columns",
    "service_week_range",
]

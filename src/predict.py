"""저장된 모델을 이용하는 공통 추론 함수."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "artifacts" / "feature_schema.json"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_features(frame: pd.DataFrame, expected_features: list[str]) -> None:
    missing = [feature for feature in expected_features if feature not in frame.columns]
    if missing:
        raise ValueError(f"추론 입력에 필요한 Feature가 없습니다: {missing}")


def risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "고위험"
    if probability >= 0.4:
        return "중위험"
    return "저위험"


def predict_dataframe(
    frame: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    schema = _load_json(schema_path)
    metadata = _load_json(metadata_path)
    features = schema.get("features", [])
    if not features:
        raise ValueError("feature_schema.json의 features가 비어 있습니다.")
    validate_features(frame, features)
    if not model_path.exists():
        raise FileNotFoundError(f"학습된 모델이 없습니다: {model_path}")

    pipeline = joblib.load(model_path)
    probabilities = pipeline.predict_proba(frame[features])[:, 1]
    threshold = float(metadata.get("threshold", 0.5))

    result = frame.copy()
    result["withdrawal_probability"] = probabilities
    result["withdrawal_prediction"] = (probabilities >= threshold).astype(int)
    result["risk_level"] = [risk_level(value) for value in probabilities]
    return result


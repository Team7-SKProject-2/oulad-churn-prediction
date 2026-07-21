"""전체 학습 데이터로 CatBoost를 재학습하고 Streamlit용 joblib 파일을 저장한다.

이 파일은 후보 모델 비교용 OOF 학습과 별개다. 최종 후보 CatBoost의 모델 객체,
Feature 순서, 범주형 Feature 목록을 하나의 joblib 파일로 저장해 추론 시에도 같은
입력 구조를 사용할 수 있게 한다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


MODEL_DIR = Path(__file__).resolve().parent
DATA_PATH = MODEL_DIR / "data" / "oulad_weekly_next_week.csv"
ARTIFACT_DIR = MODEL_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "catboost.joblib"
TARGET_COL = "target_next_week_withdrawn"
ID_COL = "id_student"
RANDOM_STATE = 42
# 팀 검증으로 확정한 운영용 이탈 위험군 분류 임계값이다.
DECISION_THRESHOLD = 0.065


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """저장 모델의 학습·추론에서 공통으로 사용할 Feature와 범주형 목록을 준비한다."""
    features = data.drop(columns=[ID_COL, TARGET_COL]).copy()
    categorical = [
        column
        for column in features.columns
        if pd.api.types.is_object_dtype(features[column])
        or pd.api.types.is_string_dtype(features[column])
        or isinstance(features[column].dtype, pd.CategoricalDtype)
    ]
    for column in categorical:
        features[column] = features[column].fillna("미상").astype(str)
    numeric = [column for column in features.columns if column not in categorical]
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)
    return features, categorical


def validate_data(data: pd.DataFrame) -> None:
    """최종 결과·등록 해지 등 미래 정보를 직접 알려 주는 누수 변수를 확인한다."""
    forbidden = [
        column
        for column in data.columns
        if column == "target"
        or any(term in column.lower() for term in ["final_result", "unregistration", "withdraw_week"])
    ]
    if forbidden:
        raise ValueError(f"누수 가능 변수가 포함되어 있습니다: {forbidden}")
    key_columns = ["code_module", "code_presentation", ID_COL, "prediction_week"]
    if data.duplicated(key_columns).any():
        raise ValueError("학생·과목·운영회차·예측주차 복합키에 중복 행이 있습니다.")


def main() -> None:
    """전체 데이터 학습 후, 모델·Feature 메타데이터를 joblib artifact로 저장한다."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    validate_data(data)
    features, categorical = prepare_features(data)
    target = data[TARGET_COL].astype(int).to_numpy()

    # OOF 성능 비교에서 사용한 CatBoost 기본 하이퍼파라미터와 동일하게 학습한다.
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=500,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=5,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(features, target, cat_features=categorical, verbose=False)

    artifact = {
        "model_name": "CatBoost",
        "model": model,
        "feature_columns": features.columns.tolist(),
        "categorical_features": categorical,
        "target_column": TARGET_COL,
        "id_column": ID_COL,
        "data_file": DATA_PATH.name,
        "training_rows": len(data),
        "feature_count": features.shape[1],
        "target_rate": float(target.mean()),
        "threshold": DECISION_THRESHOLD,
        "training_parameters": model.get_params(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)

    print(f"저장 모델: {MODEL_PATH}")
    print(f"학습 행 수: {len(data):,}, Feature 수: {features.shape[1]}, 범주형 Feature 수: {len(categorical)}")
    print(f"운영용 이탈 위험군 임계값: {DECISION_THRESHOLD:.3f}")
    print(f"파일 크기: {MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()

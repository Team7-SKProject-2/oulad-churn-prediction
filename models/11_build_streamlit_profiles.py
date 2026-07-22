"""Streamlit 수동 예측용 CatBoost 코호트 기준 프로필을 생성한다.

대용량 주간 학습 테이블은 Git에서 제외되어 있으므로, 과목·운영회차·예측주차별
수치형 중앙값과 범주형 최빈값만 작은 CSV로 저장한다. Streamlit은 이 프로필을
기본 입력으로 사용하고 사용자가 입력한 값만 덮어쓴 뒤 최종 CatBoost에 전달한다.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "catboost.joblib"
OUTPUT_PATH = ROOT / "artifacts" / "catboost_cohort_profiles.csv"
DATA_CANDIDATES = [
    ROOT / "models" / "data" / "oulad_weekly_next_week.csv",
    ROOT / "models" / "ML" / "used_data" / "weekly_next_week_with_vle_enhanced.csv",
]
GROUP_COLUMNS = ["code_module", "code_presentation", "prediction_week"]


def find_training_data() -> Path:
    """로컬에 존재하는 최종 CatBoost 학습 테이블을 찾는다."""
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n- ".join(str(path) for path in DATA_CANDIDATES)
    raise FileNotFoundError(f"CatBoost 학습 테이블을 찾지 못했습니다:\n- {candidates}")


def first_mode(series: pd.Series):
    """결측을 제외한 최빈값 하나를 반환한다."""
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else np.nan


def build_profiles(data: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """학습 데이터에서 CatBoost 입력 순서와 동일한 코호트 프로필을 만든다."""
    feature_columns = artifact["feature_columns"]
    categorical = artifact["categorical_features"]
    missing = [column for column in feature_columns if column not in data.columns]
    if missing:
        raise ValueError(f"학습 데이터에 CatBoost Feature가 없습니다: {missing}")

    numeric = [
        column
        for column in feature_columns
        if column not in categorical and column not in GROUP_COLUMNS
    ]
    numeric_profiles = (
        data.groupby(GROUP_COLUMNS, observed=True, dropna=False)[numeric]
        .median(numeric_only=True)
        .reset_index()
    )

    non_key_categorical = [column for column in categorical if column not in GROUP_COLUMNS]
    if non_key_categorical:
        categorical_profiles = (
            data.groupby(GROUP_COLUMNS, observed=True, dropna=False)[non_key_categorical]
            .agg(first_mode)
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

    for column in categorical:
        profiles[column] = profiles[column].fillna("미상").astype(str)
    profiles[numeric] = profiles[numeric].replace([np.inf, -np.inf], np.nan)

    # 최종 모델 입력과 정확히 같은 124개 Feature 순서로 저장한다.
    profiles = profiles[feature_columns].sort_values(GROUP_COLUMNS).reset_index(drop=True)
    if profiles.duplicated(GROUP_COLUMNS).any():
        raise ValueError("생성된 코호트 프로필의 복합키가 중복됩니다.")
    return profiles


def main() -> None:
    artifact = joblib.load(MODEL_PATH)
    data_path = find_training_data()
    usecols = artifact["feature_columns"]
    print(f"학습 데이터: {data_path}")
    print(f"불러올 Feature: {len(usecols)}개")
    data = pd.read_csv(data_path, usecols=usecols, low_memory=False)
    profiles = build_profiles(data, artifact)
    profiles.to_csv(OUTPUT_PATH, index=False)
    print(f"프로필 저장: {OUTPUT_PATH}")
    print(f"프로필 크기: {profiles.shape}")
    print(f"파일 크기: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()

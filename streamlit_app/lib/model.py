"""이탈 예측 모델 연동 인터페이스.

TODO(팀원 모델 전달 시):
- 학습된 모델(전처리 포함 sklearn Pipeline 권장)을 `models/dropout_model.pkl`에
  joblib.dump()로 저장해서 넣어주면 끝. 페이지 코드는 전혀 수정할 필요 없음.
- 모델 객체는 `.predict_proba(X)` 를 지원해야 하고, X는 model_snapshot_week_*.csv의
  컬럼 중 ID_COLUMNS를 제외한 나머지를 그대로 받는다고 가정한다
  (원본 dtype 그대로 전달 — 범주형 인코딩은 모델 파이프라인 내부에서 처리).
- 만약 전처리를 앱 쪽에서 별도로 해줘야 하는 모델이라면(예: 이미 원-핫 인코딩된
  피처를 기대하는 raw estimator), FEATURE_COLUMNS 정의와 predict_risk() 내부의
  `X = df[feature_columns(df)]` 부분을 함께 조정해야 한다. 이 경우 먼저 상의.

지금은 models/dropout_model.pkl이 없으므로 predict_risk()는 항상 placeholder_score()로
동작한다. 화면에는 "임시 규칙 기반 점수(모델 대기 중)"라는 배지를 반드시 노출해서
실제 모델 점수와 혼동되지 않게 한다.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dropout_model.pkl"

# 스냅샷 CSV에서 식별자/타깃 컬럼 (모델 입력에서 제외)
ID_COLUMNS = ["code_module", "code_presentation", "id_student", "cutoff_week", "target"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLUMNS]


@st.cache_resource(show_spinner=False)
def load_model():
    """학습된 모델을 로드한다. 파일이 없으면 None (placeholder 모드로 동작)."""
    if not MODEL_PATH.exists():
        return None
    import joblib
    return joblib.load(MODEL_PATH)


def model_ready() -> bool:
    """실제 모델이 연결되어 있는지 여부. 페이지에서 배지 표시용으로 사용."""
    return load_model() is not None


def placeholder_score(df: pd.DataFrame) -> np.ndarray:
    """TODO(임시): 실제 모델 도착 전까지 쓰는 규칙 기반 근사 점수(0~100).
    스냅샷의 몇 가지 대표 피처만 사용한다. 모델이 붙으면 predict_risk()가
    자동으로 이 함수 대신 실제 모델을 호출하므로, 이 함수 자체는 수정할
    필요 없이 그대로 둬도 된다."""
    def col(name, default=0.0):
        return pd.to_numeric(df[name], errors="coerce").fillna(default) if name in df.columns else pd.Series(default, index=df.index)

    click_drop = np.clip(-col("click_change_rate"), 0, 1)  # 클릭 급감 정도
    no_activity = col("current_no_activity")
    missed_rate = np.clip(col("assessment_missing_due_rate"), 0, 1)
    prev_attempts = np.minimum(col("num_of_prev_attempts"), 3)
    late_reg = col("registered_after_start")
    inactivity_weeks = np.clip(col("weeks_since_last_activity"), 0, 4)

    score = (
        15
        + click_drop * 20
        + no_activity * 15
        + missed_rate * 25
        + prev_attempts * 6
        + late_reg * 8
        + inactivity_weeks * 4
    )
    return np.clip(score, 0, 100).to_numpy()


def predict_risk(df: pd.DataFrame) -> np.ndarray:
    """0~100 위험 점수를 반환한다.
    실제 모델이 있으면 model.predict_proba(X)[:,1]*100, 없으면 placeholder_score()."""
    if df.empty:
        return np.array([])
    model = load_model()
    if model is None:
        return placeholder_score(df)
    X = df[feature_columns(df)]
    proba = model.predict_proba(X)[:, 1]
    return proba * 100

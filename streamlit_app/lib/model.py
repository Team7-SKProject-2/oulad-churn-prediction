"""이탈 예측 모델 연동 인터페이스.

모델: CatBoost, `models/artifacts/catboost.joblib`에 joblib으로 저장된 것을 로드한다.
모델 객체는 `.predict_proba(X)` 를 지원해야 하고, X는 model_snapshot_week_*.csv의
컬럼 중 ID_COLUMNS를 제외한 나머지를 그대로 받는다고 가정한다 (원본 dtype 그대로 전달 —
범주형 처리는 CatBoost가 학습 시점의 cat_features 정보로 내부에서 알아서 한다).

models/artifacts/catboost.joblib이 없거나 catboost 패키지가 설치되어 있지 않으면
predict_risk()는 placeholder_score()로 동작한다. 화면에는 "임시 규칙 기반 점수(모델 대기 중)"
배지를 노출해서 실제 모델 점수와 혼동되지 않게 한다.

점수 변환: 전체 이탈률이 0.63%로 워낙 낮아서, 모델이 뱉는 원본 확률(predict_proba)은
가장 위험한 학생도 몇 %대로만 나온다. 이걸 절대적인 수식(예: power 변환)으로 0~100에
펼치면 "위험한 학생은 높게, 안전한 학생은 낮게"를 동시에 만족시키기 어렵다.
그래서 전체 학생(스냅샷 데이터) 대비 백분위(percentile)로 점수를 매긴다 —
"이 학생보다 확률이 낮은 학생이 몇 %인가"를 점수로 쓴다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .data import PROJECT_ROOT

# data.py가 data/interim을 기준으로 이미 찾아둔 프로젝트 루트를 그대로 사용한다.
# (streamlit_app 폴더 기준이 아니라 실제 저장소 루트의 models/ 폴더를 가리켜야 하므로)
MODEL_PATH = PROJECT_ROOT / "models" / "artifacts" / "catboost.joblib"

# 스냅샷 CSV에서 식별자/타깃 컬럼 (모델 입력에서 제외)
ID_COLUMNS = ["code_module", "code_presentation", "id_student", "cutoff_week", "target"]

# weekly_next_week_with_vle_enhanced_sample.csv(모델 실제 학습 스키마 샘플) 기준으로 계산한
# 컬럼별 평균(수치형)/최빈값(범주형)을 lib/sample_defaults.json에서 불러온다.
# 지금 앱이 쓰는 vle_snapshot_week_*.csv엔 없는 컬럼을 모델에 넣을 때, 무작정 0/"unknown"으로
# 채우는 대신 이 값으로 채운다.
# ⚠️ 샘플(3,500행) 기준이라 실제 전체 데이터 통계와 다를 수 있다 — 전체 데이터를 구하면
#    lib/sample_defaults.json 파일만 다시 계산해서 교체하면 된다 (코드 수정 불필요).
SAMPLE_DEFAULTS_PATH = Path(__file__).resolve().parent / "sample_defaults.json"


def _load_sample_defaults() -> dict:
    if not SAMPLE_DEFAULTS_PATH.exists():
        return {}
    try:
        with open(SAMPLE_DEFAULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"lib/sample_defaults.json을 읽는 중 오류가 발생해 기본값(0/unknown)으로 대체합니다: {e}")
        return {}


SAMPLE_DEFAULTS = _load_sample_defaults()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLUMNS]


@st.cache_resource(show_spinner="CatBoost 모델 로딩 중…")
def load_model():
    """학습된 모델을 로드한다. 파일이 없거나 로드에 실패하면 None (placeholder 모드로 동작).

    joblib.dump()로 모델 객체 하나만 저장했다면 그대로 반환하고,
    {"model": ..., ...} 처럼 dict로 감싸서 저장했다면 그 안에서 predict_proba를
    가진 값을 자동으로 찾아 꺼낸다."""
    if not MODEL_PATH.exists():
        return None
    import joblib
    try:
        obj = joblib.load(MODEL_PATH)
    except ModuleNotFoundError:
        st.warning(
            "models/artifacts/catboost.joblib을 찾았지만 `catboost` 패키지가 설치되어 있지 않아 "
            "로드하지 못했습니다. `pip install catboost` 후 앱을 재시작하세요. "
            "그 전까지는 임시 규칙 기반 점수로 표시됩니다."
        )
        return None

    if hasattr(obj, "predict_proba"):
        return obj

    if isinstance(obj, dict):
        for key in ("model", "catboost", "catboost_model", "estimator", "clf", "classifier", "pipeline"):
            if key in obj and hasattr(obj[key], "predict_proba"):
                return obj[key]
        for v in obj.values():
            if hasattr(v, "predict_proba"):
                return v
        st.warning(
            "models/artifacts/catboost.joblib이 dict로 저장되어 있는데, 그 안에서 "
            f"predict_proba를 가진 모델을 못 찾았습니다. 실제 키: {list(obj.keys())} — "
            "이 키 이름을 알려주시면 코드에 반영해드릴게요. 그 전까지는 임시 규칙 기반 점수로 표시됩니다."
        )
        return None

    st.warning(
        f"models/artifacts/catboost.joblib을 로드했지만 예상한 모델 형태가 아닙니다 (타입: {type(obj)}). "
        "임시 규칙 기반 점수로 표시됩니다."
    )
    return None


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


def _prepare_X(df: pd.DataFrame, model):
    """predict_risk()와 기준 확률 분포 계산에서 공통으로 쓰는 입력 전처리.
    (컬럼 순서 맞추기, cutoff_week→prediction_week/cutoff_day 매핑, 없는 컬럼 기본값 채우기,
    범주형 컬럼 문자열 변환)"""
    X = df[feature_columns(df)]

    feature_names = getattr(model, "feature_names_", None)
    cat_idx = model.get_cat_feature_indices() if hasattr(model, "get_cat_feature_indices") else []
    cat_cols = set(feature_names[i] for i in cat_idx if feature_names and i < len(feature_names))

    missing = []
    if feature_names:
        missing = [c for c in feature_names if c not in X.columns]
        X = X.reindex(columns=feature_names)

        # ID_COLUMNS로 X에서 제외됐지만, 모델은 실제 피처로 쓰는 컬럼들은
        # 원본 df 값으로 명시적으로 다시 채워준다.
        if "cutoff_week" in df.columns:
            if "prediction_week" in X.columns:
                X["prediction_week"] = df["cutoff_week"].values
                missing = [c for c in missing if c != "prediction_week"]
            if "cutoff_day" in X.columns:
                X["cutoff_day"] = (df["cutoff_week"].values - 1) * 7
                missing = [c for c in missing if c != "cutoff_day"]
        for id_col in ("code_module", "code_presentation"):
            if id_col in df.columns and id_col in X.columns:
                X[id_col] = df[id_col].values
                missing = [c for c in missing if c != id_col]

        # 그래도 남는, 지금 화면 데이터에 아예 없는 컬럼은 무작정 0/"unknown"이 아니라
        # SAMPLE_DEFAULTS(학습 스키마 샘플 기준 평균/최빈값)로 채운다.
        for col in missing:
            if col in SAMPLE_DEFAULTS:
                X[col] = SAMPLE_DEFAULTS[col]
            else:
                X[col] = "unknown" if col in cat_cols else 0

    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].apply(lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))

    return X, missing


@st.cache_data(show_spinner="기준 확률 분포 계산 중…")
def _reference_probabilities() -> np.ndarray:
    """전체 스냅샷 데이터에 대해 모델 확률을 한 번 계산해서 캐싱해둔 기준 분포.
    새 예측값을 이 분포에서 백분위로 환산해 점수를 매기는 데 쓴다."""
    model = load_model()
    if model is None:
        return np.array([])
    from .data import load_model_snapshots
    snaps = load_model_snapshots()
    if snaps.empty:
        return np.array([])
    X, _ = _prepare_X(snaps, model)
    return model.predict_proba(X)[:, 1]


def predict_risk(df: pd.DataFrame) -> np.ndarray:
    """0~100 위험 점수를 반환한다.
    실제 모델이 있으면 전체 학생 대비 백분위로 점수를 매기고, 없으면 placeholder_score()."""
    if df.empty:
        return np.array([])
    model = load_model()
    if model is None:
        return placeholder_score(df)

    X, missing = _prepare_X(df, model)
    if missing:
        st.caption(f"⚠️ 모델이 기대하는 피처 {len(missing)}개는 현재 데이터에 없어 기본값으로 채워졌습니다 (예측 정확도에 영향 가능).")

    proba = model.predict_proba(X)[:, 1]

    ref = _reference_probabilities()
    if ref.size > 0:
        # 백분위 점수: 전체 학생 중 이 확률보다 낮은 확률을 가진 비율(%)
        score = np.array([(ref < p).mean() * 100 for p in proba])
    else:
        # 기준 분포를 못 구했을 때의 대비책 (예: 스냅샷이 비어있는 경우)
        score = np.power(proba, 0.15) * 100

    return np.clip(score, 0, 100)
import streamlit as st
import pandas as pd
import numpy as np

from lib.theme import inject_base_css, risk_badge_html, RISK_COLORS
from lib.risk import RISK_FACTOR_CATALOG, factors_for_snapshot, actions_for
from lib.model import (
    cohort_profile,
    cohort_profiles_ready,
    decision_threshold,
    load_cohort_profiles,
    model_info,
    model_ready,
    prediction_frame,
)
from utils.styles import load_css
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

load_css(ROOT_DIR / "styles.css")
st.set_page_config(page_title="이탈 예측", layout="wide")
inject_base_css()


st.title("🧠 모델 기반 차주 이탈 예측")
st.caption("현재 주차까지의 정보를 입력하면 CatBoost가 다음 주 중도이탈 위험을 예측합니다.")

if not model_ready():
    st.error("최종 CatBoost 모델(models/artifacts/catboost.joblib)이 연결되지 않았습니다.")
    st.stop()

if not cohort_profiles_ready():
    st.error(
        "CatBoost 입력 프로필(models/artifacts/catboost_cohort_profiles.csv)이 없습니다. "
        "models/11_build_streamlit_profiles.py를 먼저 실행해 주세요."
    )
    st.stop()

profiles = load_cohort_profiles()
info = model_info()
threshold = decision_threshold()
st.success(
    f"{info['model_name']} · {info['feature_count']}개 Feature · "
    f"위험 기준 확률 {threshold:.3f} 연결 완료"
)

st.info(
    "입력하지 않은 Feature는 최종 학습 데이터의 과목·운영회차·예측주차별 "
    "중앙값/최빈값으로 채웁니다. 이 화면은 입력값 변화에 따른 시나리오 확인용이며, "
    "실제 학생 예측에는 해당 학생의 124개 Feature를 그대로 전달해야 합니다.",
    icon="ℹ️",
)

with st.container(border=True):
    st.markdown("**과목 · 시점**")
    c1, c2, c3 = st.columns(3)
    modules = sorted(profiles["code_module"].unique().tolist())
    module = c1.selectbox("과목", modules)
    presentations = sorted(
        profiles.loc[profiles["code_module"] == module, "code_presentation"].unique().tolist()
    )
    presentation = c2.selectbox("운영 회차", presentations)
    weeks = sorted(
        profiles.loc[
            profiles["code_module"].eq(module)
            & profiles["code_presentation"].eq(presentation),
            "prediction_week",
        ].unique().tolist()
    )
    week = c3.selectbox("현재 예측 주차", weeks, index=0)

with st.container(border=True):
    st.markdown("**학생 정보**")
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("성별", ["F", "M"], format_func=lambda g: "여" if g == "F" else "남")
    age_options = sorted(profiles["age_band"].dropna().unique().tolist())
    age_band = c2.selectbox("연령대", age_options)
    disability = c3.selectbox("장애 여부", ["N", "Y"], format_func=lambda d: "있음" if d == "Y" else "없음")
    c4, c5 = st.columns(2)
    region_options = sorted(profiles["region"].dropna().unique().tolist())
    region = c4.selectbox("거주 지역", region_options)
    prev_attempts = c5.number_input("이전 수강 시도 횟수", min_value=0, max_value=6, value=0)

with st.container(border=True):
    st.markdown("**참여·과제 현황 (예측 주차 기준)**")
    c1, c2, c3 = st.columns(3)
    current_clicks = c1.number_input("이번 주차 클릭 수", min_value=0, value=25)
    registered_after_start = c2.selectbox("등록 시점", ["개강 전 등록", "개강 후 등록"]) == "개강 후 등록"
    missed_count = c3.number_input("미제출 과제 개수(누적)", min_value=0, max_value=10, value=0)

# 1) 최종 학습 데이터와 동일한 124개 Feature 코호트 프로필을 기본 행으로 사용한다
template = cohort_profile(profiles, module, presentation, int(week))

# 2) 사용자가 입력한 핵심 항목만 덮어쓴다
row = template.copy()
row["gender"] = gender
row["age_band"] = age_band
row["disability"] = disability
row["region"] = region
row["num_of_prev_attempts"] = prev_attempts
row["registered_after_start"] = int(registered_after_start)
row["current_total_clicks"] = current_clicks
row["current_no_activity"] = int(current_clicks == 0)
row["assessment_missing_due_count"] = missed_count

# 3) 입력값에 직접 연동되는 파생 Feature를 함께 갱신해 내부 일관성을 맞춘다
template_current_clicks = float(template.get("current_total_clicks", 0) or 0)
prev_clicks = float(template.get("previous_total_clicks", current_clicks) or 0)
row["click_change"] = current_clicks - prev_clicks
row["click_change_rate"] = (current_clicks - prev_clicks) / prev_clicks if prev_clicks else 0.0
row["log1p_current_total_clicks"] = np.log1p(current_clicks)
row["current_has_vle_record"] = int(current_clicks > 0)

click_delta = current_clicks - template_current_clicks
row["vle_cum_total_clicks"] = max(float(template.get("vle_cum_total_clicks", 0)) + click_delta, 0)
row["log1p_cum_total_clicks"] = np.log1p(row["vle_cum_total_clicks"])

activity_columns = [
    "current_forumng_clicks",
    "current_oucontent_clicks",
    "current_quiz_clicks",
    "current_resource_clicks",
    "current_other_clicks",
]
if template_current_clicks > 0:
    activity_scale = current_clicks / template_current_clicks
    for column in activity_columns:
        row[column] = float(template.get(column, 0)) * activity_scale
elif current_clicks == 0:
    for column in activity_columns:
        row[column] = 0.0

due_count = float(template.get("assessment_due_count", 0) or 0)
row["assessment_submitted_due_count"] = max(due_count - missed_count, 0)
row["assessment_missing_due_rate"] = missed_count / due_count if due_count else 0.0
row["assessment_submission_rate"] = (
    row["assessment_submitted_due_count"] / due_count if due_count else 0.0
)

input_df = pd.DataFrame([row])
result = prediction_frame(input_df).iloc[0]
score = float(result["risk_score_pct"])
is_at_risk = bool(result["is_at_risk"])
grade = "high" if is_at_risk else "low"
factor_keys = factors_for_snapshot(row)
grade_color = RISK_COLORS[grade]["dot"]

# 위험 신호 칩 HTML
if factor_keys:
    chips_html = " ".join(
        f'<span class="chip">{RISK_FACTOR_CATALOG[k]["label"]}</span>' for k in factor_keys
    )
else:
    chips_html = '<span style="font-size:12px;color:#6a7286;">감지된 위험 신호가 없습니다.</span>'

# 추천 행동 카드 HTML
actions = actions_for(factor_keys)
if actions:
    actions_html = "".join(
        f'<div class="action-card"><b>{a["title"]}</b><br>'
        f'<span style="font-size:12.5px;color:#5b6478;">{a["desc"]}</span></div>'
        for a in actions
    )
else:
    actions_html = '<span style="font-size:12px;color:#6a7286;">추천 행동이 없습니다.</span>'

# 화면에 항상 고정으로 떠있는 패널 — position: fixed
st.markdown(
    f"""
    <div style="
        position: fixed;
        bottom: 10px;
        right: 17px;
        width: 420px;
        max-height: 82vh;
        overflow-y: auto;
        background: #ffffff;
        border: 1px solid #e3e6ee;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 12px 32px rgba(28,35,51,0.18);
        z-index: 9999;
    ">
        <div style="font-size:13px;font-weight:700;color:#6a7286;margin-bottom:6px;">🔍 예측 결과</div>
        <div style="font-size:26px;font-weight:700;color:#1c2333;margin-bottom:8px;">{score:.2f}%</div>
        {risk_badge_html(grade)}
        <div style="background:#e3e6ee;border-radius:8px;height:10px;overflow:hidden;margin:12px 0 16px;">
            <div style="width:{score:.0f}%;background:{grade_color};height:100%;"></div>
        </div>
        <div style="font-size:13px;font-weight:700;color:#1c2333;margin-bottom:6px;">감지된 위험 신호 (원본 데이터 기반 근거)</div>
        <div style="margin-bottom:14px;">{chips_html}</div>
        <div style="font-size:13px;font-weight:700;color:#1c2333;margin-bottom:6px;">추천 행동</div>
        {actions_html}
        <div style="font-size:11px;color:#6a7286;margin-top:10px;">
            운영 위험 기준: {threshold * 100:.2f}% · CatBoost 124개 Feature<br>
            입력하지 않은 Feature는 같은 과목·운영회차·예측주차 코호트의 중앙값/최빈값으로 채워졌습니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

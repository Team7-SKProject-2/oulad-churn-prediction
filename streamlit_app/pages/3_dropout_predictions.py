import streamlit as st
import pandas as pd

from lib import data as D
from lib.theme import inject_base_css, page_header, risk_badge_html, risk_grade, RISK_COLORS
from lib.risk import RISK_FACTOR_CATALOG, factors_for_snapshot, actions_for
from lib.model import predict_risk, model_ready
from utils.styles import load_css
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

load_css(ROOT_DIR / "styles.css")
st.set_page_config(page_title="이탈 예측", layout="wide")
inject_base_css()


st.title("🧠 모델 기반 차주 이탈 예측")
st.caption("학생 정보를 입력하면 ML 모델이 이탈 위험을 예측합니다. ")

if not D.model_snapshots_available():
    st.error("data/interim/ 에 model_snapshot_week_*.csv 가 없습니다.")
    st.stop()

weeks = D.available_snapshot_weeks()
snapshots = D.load_model_snapshots()

if not model_ready():
    st.caption("⚠️ 아직 학습된 모델이 연결되지 않아, 임시 규칙 기반 점수로 표시됩니다.")

st.info(
    "입력하지 않은 나머지 피처는 선택한 과목·예측주차 학생들의 중앙값/최빈값으로 자동 채웁니다. "
    "정확한 예측이 필요하면 2번 페이지에서 실제 학생을 조회하세요.",
    icon="ℹ️",
)

with st.container(border=True):
    st.markdown("**과목 · 시점**")
    c1, c2, c3 = st.columns(3)
    modules = sorted(snapshots["code_module"].unique().tolist())
    module = c1.selectbox("과목", modules)
    presentations = sorted(snapshots.loc[snapshots["code_module"] == module, "code_presentation"].unique().tolist())
    presentation = c2.selectbox("운영 회차", presentations)
    week = c3.selectbox("예측 주차(cutoff_week)", weeks, index=len(weeks) - 1)

with st.container(border=True):
    st.markdown("**학생 정보**")
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("성별", ["F", "M"], format_func=lambda g: "여" if g == "F" else "남")
    age_options = sorted(snapshots["age_band"].dropna().unique().tolist())
    age_band = c2.selectbox("연령대", age_options)
    disability = c3.selectbox("장애 여부", ["N", "Y"], format_func=lambda d: "있음" if d == "Y" else "없음")
    c4, c5 = st.columns(2)
    region_options = sorted(snapshots["region"].dropna().unique().tolist())
    region = c4.selectbox("거주 지역", region_options)
    prev_attempts = c5.number_input("이전 수강 시도 횟수", min_value=0, max_value=6, value=0)

with st.container(border=True):
    st.markdown("**참여·과제 현황 (예측 주차 기준)**")
    c1, c2, c3 = st.columns(3)
    current_clicks = c1.number_input("이번 주차 클릭 수", min_value=0, value=25)
    registered_after_start = c2.selectbox("등록 시점", ["개강 전 등록", "개강 후 등록"]) == "개강 후 등록"
    missed_count = c3.number_input("미제출 과제 개수(누적)", min_value=0, max_value=10, value=0)

# 1) 같은 과목·운영회차·예측주차 코호트의 중앙값/최빈값으로 기본 행을 만든다
template = D.cohort_template(snapshots, module, presentation, week)

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

# 3) 입력값에 따라 달라지는 파생 피처를 최소한으로 재계산해 내부 일관성을 맞춘다
prev_clicks = template.get("previous_total_clicks", current_clicks) or 1
row["click_change"] = current_clicks - prev_clicks
row["click_change_rate"] = (current_clicks - prev_clicks) / prev_clicks if prev_clicks else 0.0

input_df = pd.DataFrame([row])
score = predict_risk(input_df)[0]
grade = risk_grade(score)
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
        <div style="font-size:26px;font-weight:700;color:#1c2333;margin-bottom:8px;">{score:.0f} / 100</div>
        {risk_badge_html(grade)}
        <div style="background:#e3e6ee;border-radius:8px;height:10px;overflow:hidden;margin:12px 0 16px;">
            <div style="width:{score:.0f}%;background:{grade_color};height:100%;"></div>
        </div>
        <div style="font-size:13px;font-weight:700;color:#1c2333;margin-bottom:6px;">감지된 위험 신호 (원본 데이터 기반 근거)</div>
        <div style="margin-bottom:14px;">{chips_html}</div>
        <div style="font-size:13px;font-weight:700;color:#1c2333;margin-bottom:6px;">추천 행동</div>
        {actions_html}
        <div style="font-size:11px;color:#6a7286;margin-top:10px;">입력하지 않은 나머지 피처는 같은 과목·예측주차 코호트의 중앙값/최빈값으로 채워졌습니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

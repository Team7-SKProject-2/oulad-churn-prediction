import streamlit as st
import pandas as pd
import numpy as np

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
    "catboost_feature_importance_all.csv 기준 중요도 상위 피처는 직접 입력받고, "
    "나머지는 선택한 과목·예측주차 학생들의 중앙값/최빈값으로 자동 채웁니다. "
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
    week_options = list(range(1, 21)) if model_ready() else weeks
    week = c3.selectbox("예측 주차(cutoff_week)", week_options, index=len(week_options) - 1)
    if model_ready() and week not in weeks:
        st.caption(
            f"ℹ️ {week}주차는 실제 스냅샷이 없어, 가장 가까운 체크포인트({min(weeks, key=lambda w: abs(w - week))}주차) 학생들의 값을 기준으로 나머지 피처를 채웁니다.")

with st.container(border=True):
    st.markdown("**학생 정보**")
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("성별", ["F", "M"], format_func=lambda g: "여" if g == "F" else "남")
    age_options = sorted(snapshots["age_band"].dropna().unique().tolist())
    age_band = c2.selectbox("연령대", age_options)
    disability = c3.selectbox("장애 여부", ["N", "Y"], format_func=lambda d: "있음" if d == "Y" else "없음")
    c4, c5, c6 = st.columns(3)
    region_options = sorted(snapshots["region"].dropna().unique().tolist())
    region = c4.selectbox("거주 지역", region_options)
    prev_attempts = c5.number_input("이전 수강 시도 횟수", min_value=0, max_value=6, value=0)
    # importance rank 23
    edu_options = sorted(snapshots["highest_education"].dropna().unique().tolist())
    highest_education = c6.selectbox("최종 학력", edu_options)

with st.container(border=True):
    st.markdown("**수강 정보**")
    c1, c2 = st.columns(2)
    # importance rank 3 — 상위권 피처인데 기존 폼엔 없었음
    studied_credits = c1.number_input("수강 학점", min_value=30, max_value=655, value=60, step=30)
    # importance rank 6
    current_active_days = c2.number_input("이번 주 활동한 일수 (0~7)", min_value=0, max_value=7, value=3)

with st.container(border=True):
    st.markdown("**참여 현황 (클릭)**")
    c1, c2 = st.columns(2)
    current_clicks = c1.number_input("이번 주차 클릭 수", min_value=0, value=25)
    # importance rank 12 — 기존엔 코호트 중앙값으로만 채워지던 값을 직접 입력받도록 변경
    previous_clicks = c2.number_input("직전 주차 클릭 수", min_value=0, value=25)
    c3, c4 = st.columns(2)
    # importance rank 8
    previous_active_days = c3.number_input("직전 주 활동한 일수 (0~7)", min_value=0, max_value=7, value=3)
    registered_after_start = c4.selectbox("등록 시점", ["개강 전 등록", "개강 후 등록"]) == "개강 후 등록"

with st.container(border=True):
    st.markdown("**과제 현황 (예측 주차 기준)**")
    c1, c2, c3 = st.columns(3)
    # importance rank 1(비율), 11(제출률)의 정확한 계산을 위해 마감 건수도 함께 받는다
    due_count = c1.number_input("마감된 과제 개수(누적)", min_value=0, max_value=20, value=3)
    missed_count = c2.number_input("미제출 과제 개수(누적)", min_value=0, max_value=20, value=0)
    # importance rank 2, 9, 16, 18, 25, 26, 28, 30을 한 번에 대표하는 값
    avg_score = c3.number_input("과제 평균 점수", min_value=0, max_value=100, value=70)

# 1) 같은 과목·운영회차·예측주차 코호트의 중앙값/최빈값으로 기본 행을 만든다
template = D.cohort_template(snapshots, module, presentation, week)

# 2) 사용자가 입력한 핵심 항목만 덮어쓴다
row = template.copy()
row["gender"] = gender
row["age_band"] = age_band
row["disability"] = disability
row["region"] = region
row["num_of_prev_attempts"] = prev_attempts
row["highest_education"] = highest_education
row["studied_credits"] = studied_credits
row["registered_after_start"] = int(registered_after_start)
row["current_total_clicks"] = current_clicks
row["current_no_activity"] = int(current_clicks == 0)
row["current_active_days"] = current_active_days
row["previous_total_clicks"] = previous_clicks
row["previous_active_days"] = previous_active_days
row["assessment_due_count"] = due_count
row["assessment_missing_due_count"] = missed_count

# 3) 입력값에 따라 달라지는 파생 피처를 최소한으로 재계산해 내부 일관성을 맞춘다
prev_clicks = previous_clicks or 1
row["click_change"] = current_clicks - prev_clicks
row["click_change_rate"] = (current_clicks - prev_clicks) / prev_clicks if prev_clicks else 0.0

# importance rank 15 — log1p 변환 피처는 현재 클릭수에서 바로 계산 가능
row["log1p_current_total_clicks"] = np.log1p(current_clicks)

# importance rank 17 — 이번 주 활동이 있으면 0, 없으면 코호트 기준값 유지
row["weeks_since_last_activity"] = 0 if current_clicks > 0 else (template.get("weeks_since_last_activity", 1) or 1)

# importance rank 7 — 지금까지 관측된 주차 수는 선택한 예측 주차로 근사
row["observed_weeks"] = week

# importance rank 1, 11 — 마감/미제출 개수로 비율을 직접 계산 (0으로 나누기 방지)
if due_count > 0:
    row["assessment_missing_due_rate"] = missed_count / due_count
    row["assessment_submission_rate"] = max(due_count - missed_count, 0) / due_count
else:
    row["assessment_missing_due_rate"] = 0.0
    row["assessment_submission_rate"] = template.get("assessment_submission_rate", 0.0)

# importance rank 2, 9, 16, 18, 25, 26, 28, 30 — 점수 관련 피처는 입력받은 평균 점수로 통일
for score_col in [
    "assessment_min_score", "assessment_max_score", "assessment_median_score",
    "assessment_mean_score", "assessment_weighted_mean_score",
    "any_known_mean_score", "any_known_median_score",
]:
    row[score_col] = avg_score

# importance rank 2 — weighted_score_sum은 입력받지 않으므로, 코호트의 채점된 배점 총량(scored_weight_sum,
# rank 4)에 입력받은 평균 점수 비율을 곱해 근사한다. (정확한 값이 아니라 근사치임에 주의)
scored_weight_sum_template = template.get("scored_weight_sum", 0) or 0
row["weighted_score_sum"] = (avg_score / 100) * scored_weight_sum_template

input_df = pd.DataFrame([row])
score = float(predict_risk(input_df)[0])  # predict_risk()가 이미 0~100으로 보정해서 반환한다 (lib/model.py 참고)
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

st.markdown('<div style="height:420px;"></div>', unsafe_allow_html=True)
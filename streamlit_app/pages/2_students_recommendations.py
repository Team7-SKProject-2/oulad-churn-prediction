import streamlit as st
import pandas as pd
import plotly.express as px

from lib import data as D
from lib.theme import inject_base_css, page_header, risk_badge_html, RISK_COLORS, risk_grade
from lib.risk import RISK_FACTOR_CATALOG, factors_for_snapshot, score_snapshot_row, actions_for
from utils.styles import load_css
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

load_css(ROOT_DIR / "styles.css")
st.set_page_config(page_title="학생별 행동추천", layout="wide")
inject_base_css()

if not D.model_snapshots_available():
    st.error("data/interim/ 에 model_snapshot_week_*.csv 가 없습니다.")
    st.stop()

weeks = D.available_snapshot_weeks()
snapshots = D.load_model_snapshots()
st.title("👨‍🎓 학생별 차주이탈 분석")
st.caption("학생과 예측 주차를 선택하면 수강 과목 정보 · 개인정보 · 과목별 위험도/참여도를 확인할 수 있습니다.")

with st.container(border=True):
    all_students = sorted(snapshots["id_student"].unique().tolist())
    default_idx = 0
    if "preselected_student" in st.session_state and st.session_state["preselected_student"] in all_students:
        default_idx = all_students.index(st.session_state["preselected_student"])

    f1, f2 = st.columns([1, 1])
    student_id = f1.selectbox("학생 선택", all_students, index=default_idx, format_func=lambda s: f"#{s}")

    student_rows = snapshots[snapshots["id_student"] == student_id]
    if student_rows.empty:
        st.info("해당 학생의 수강 기록이 없습니다.")
        st.stop()

    default_week_idx = len(weeks) - 1
    if "preselected_week" in st.session_state and st.session_state["preselected_week"] in weeks:
        default_week_idx = weeks.index(st.session_state["preselected_week"])
    week = f2.selectbox("예측 주차(cutoff_week)", weeks, index=default_week_idx)

info_row = student_rows.iloc[0]
enrollments = student_rows[["code_module", "code_presentation"]].drop_duplicates()

with st.container(border=True):
    st.markdown("**학생 개인정보**")
    c = st.columns(5)
    labels = ["성별", "지역", "연령대", "장애 여부", "수강 과목 수"]
    values = [
        "여" if info_row["gender"] == "F" else "남",
        info_row["region"],
        info_row["age_band"],
        "있음" if info_row["disability"] == "Y" else "없음",
        str(len(enrollments)),
    ]
    for col, label, value in zip(c, labels, values):
        col.markdown(f'<div class="kpi-label">{label}</div><div style="font-size:14px;font-weight:600;">{value}</div>', unsafe_allow_html=True)
    st.write("")
st.write("")
st.markdown("**수강 과목별 위험도 · 참여도**")

# 과목(수강)별로 예측 주차 스냅샷을 찾되, 없으면 '이탈 확정으로 관측 종료' 여부를 판정한다.
enroll_states = []
for _, e in enrollments.iterrows():
    sub = student_rows[
        (student_rows["code_module"] == e["code_module"]) & (student_rows["code_presentation"] == e["code_presentation"])
    ].sort_values("cutoff_week")
    at_week = sub[sub["cutoff_week"] == week]
    if not at_week.empty:
        enroll_states.append({"module": e["code_module"], "presentation": e["code_presentation"], "row": at_week.iloc[0], "status": "active", "trend": sub})
    else:
        prior = sub[sub["cutoff_week"] < week]
        if prior.empty:
            continue  # 예측 주차 이전에 아직 등록/관측 시작 전
        last = prior.iloc[-1]
        status = "churned" if last["target"] == 1 else "no_data"
        enroll_states.append({"module": e["code_module"], "presentation": e["code_presentation"], "row": last, "status": status, "trend": sub})

if not enroll_states:
    st.info("선택한 예측 주차까지의 관측 기록이 없습니다.")
    st.stop()

module_labels = [f'{s["module"]} · {s["presentation"]}' for s in enroll_states]
cols = st.columns(len(enroll_states))
for col, s, lbl in zip(cols, enroll_states, module_labels):
    clicks = s["row"].get("cum_total_clicks", s["row"].get("current_total_clicks", "-"))
    if s["status"] == "churned":
        badge = '<span class="risk-badge" style="background:#fbe4e4;color:#b23a3a;">이탈 확정</span>'
    else:
        score = score_snapshot_row(s["row"])
        badge = risk_badge_html(risk_grade(score))
    col.markdown(
        f'<div class="section-card" style="text-align:center;">'
        f'<div style="font-size:12px;color:#5b6478;">{lbl}</div>{badge}'
        f'<div style="font-size:11.5px;color:#6a7286;margin-top:6px;">참여도(누적 클릭) {clicks}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

active_idx = 0
if len(enroll_states) > 1:
    choice = st.radio("그래프/추천을 볼 과목", module_labels, horizontal=True, label_visibility="collapsed")
    active_idx = module_labels.index(choice)
active = enroll_states[active_idx]

if active["status"] == "churned":
    st.warning(f'이 학생은 {int(active["row"]["cutoff_week"])}주차 이후 관측 기록이 없습니다 — 이탈이 확정되어 더 이상 활동 데이터가 쌓이지 않는 것으로 보입니다. 아래 내용은 마지막 관측 시점 기준입니다.')
elif active["status"] == "no_data":
    st.info(f'{week}주차 스냅샷이 없어 마지막 관측 시점({int(active["row"]["cutoff_week"])}주차) 기준으로 표시합니다.')

st.write("")
with st.container(border=True):
    st.markdown(f"**{module_labels[active_idx]} cutoff_week별 누적 클릭 수(참여도) 추이**")

with st.container(border=True):

    trend = active["trend"]
    score_now = score_snapshot_row(active["row"])
    grade_now = risk_grade(score_now)
    fig = px.line(trend, x="cutoff_week", y="cum_total_clicks", markers=True)
    fig.update_traces(line_color=RISK_COLORS[grade_now]["dot"])
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220, xaxis_title="cutoff_week", yaxis_title="누적 클릭 수")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("점이 끊긴 구간은 해당 cutoff_week에 스냅샷이 없다는 뜻입니다 (예측 체크포인트: " + ", ".join(map(str, weeks)) + ").")


st.write("")
c1, c2 = st.columns(2)
factor_keys = factors_for_snapshot(active["row"])
with c1:
    with st.container(border=True,key="sep_card_danger"):
        st.markdown("**위험 요인 (원본 데이터 기반 근거)**")
        if factor_keys:
            for k in factor_keys:
                f = RISK_FACTOR_CATALOG[k]
                st.markdown(f"- **{f['label']}** — {f['detail']}")
        else:
            st.caption("특별한 위험 요인이 감지되지 않았습니다.")

with c2:
    with st.container(border=True,key="sep_card_recommd"):
        st.markdown("**과목별 맞춤 행동추천**")
        actions = actions_for(factor_keys)
        if actions:
            for a in actions:
                st.markdown(f'<div class="action-card"><b>{a["title"]}</b><br><span style="font-size:12.5px;color:#5b6478;">{a["desc"]}</span></div>', unsafe_allow_html=True)
        else:
            st.caption("추천할 행동이 없습니다.")

import streamlit as st
import pandas as pd

from lib import data as D
from lib.theme import inject_base_css, page_header, risk_badge_html, RISK_COLORS
from lib.data import build_master_table
from lib.risk import RISK_FACTOR_CATALOG, ACTION_CATALOG, factors_for
from utils.styles import load_css
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

load_css(f'{ROOT_DIR}\styles.css')

st.set_page_config(page_title="과목/주차별 행동제안", layout="wide")
inject_base_css()

if not D.data_available():
    st.error("data/interim/ 데이터가 없습니다. 메인 페이지 안내를 참고하세요.")
    st.stop()

raw = D.load_raw()
master = build_master_table(raw)

st.title("📋 과목/주차별 행동제안")
st.caption("과목과 주차를 선택하면 해당 시점의 위험군 학생과 공통 행동제안을 확인할 수 있습니다.")

with st.container(border=True):

    courses = raw["courses"]
    f1, f2, f3 = st.columns([1, 1, 1])
    module = f1.selectbox("과목", D.module_list(courses))
    presentation = f2.selectbox("운영 회차", D.presentation_list(courses, module))
    max_week = int(master.loc[(master.code_module == module) & (master.code_presentation == presentation), "week_index"].max() or 1)
    week = f3.selectbox("주차", list(range(1, max_week + 1)), index=min(5, max_week - 1))

scope = master[
    (master["code_module"] == module) & (master["code_presentation"] == presentation) & (master["week_index"] <= week)
]
if scope.empty:
    st.info("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

snapshot = scope.loc[scope.groupby("id_student")["week_index"].idxmax()].copy()
snapshot["factors"] = snapshot.apply(
    lambda r: factors_for(bool(r["engagement_drop"]), float(r["missed_this_week"]), float(r["num_of_prev_attempts"]), bool(r["late_registration"])),
    axis=1,
)
snapshot = snapshot.sort_values("risk_score", ascending=False)

high_count = (snapshot["risk_grade"] == "high").sum()
drop_pct = snapshot["engagement_drop"].mean() * 100
missed_pct = (snapshot["missed_this_week"] >= 1).mean() * 100

k1, k2, k3 = st.columns(3)
for col, label, value in [
    (k1, "위험군 학생 수(고위험)", f"{high_count:,}"),
    (k2, "참여도 급감 학생 비율", f"{drop_pct:.0f}%"),
    (k3, "과제 미제출 학생 비율", f"{missed_pct:.0f}%"),
]:
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


st.write("")
st.markdown("**과목 공통 행동제안**")
with st.container(border=True):  # streamlit 1.28+ 에서 border 파라미터 지원
    all_factors = [f for fl in snapshot["factors"] for f in fl]
    top_factors = pd.Series(all_factors).value_counts().index[:3].tolist() if all_factors else []
    cols = st.columns(max(len(top_factors), 1))

    if not top_factors:
        st.caption("현재 이 조건에서 감지된 공통 위험 요인이 없습니다.")
    for c, key in zip(cols, top_factors):
        a = ACTION_CATALOG[key]
        c.markdown(f'<div class="action-card"><b>{a["title"]}</b><br><span style="font-size:12.5px;color:#5b6478;">{a["desc"]}</span></div>', unsafe_allow_html=True)

    st.write("")
st.markdown(f'<div class="section-card">**위험 학생 리스트 · {len(snapshot)}명**', unsafe_allow_html=True)
top = snapshot.head(15)
table = pd.DataFrame({
    "학생 ID": top["id_student"].astype(str),
    "위험도": top["risk_grade"].map(lambda g: RISK_COLORS[g]["label"]),
    "위험 점수": top["risk_score"].round(0).astype(int),
    "주요 위험요인": top["factors"].map(lambda fl: ", ".join(RISK_FACTOR_CATALOG[f]["label"] for f in fl) or "-"),
    "누적 미제출": top["missed_this_week"].astype(int),
})
st.dataframe(table, use_container_width=True, hide_index=True)

selected = st.selectbox("상세 행동추천을 볼 학생 선택", top["id_student"].tolist(), format_func=lambda s: f"#{s}")
if selected:
    st.page_link("pages/2_students_recommendations.py", label=f"#{selected} 학생 상세 페이지로 이동 →")
    st.session_state["preselected_student"] = int(selected)
st.markdown("</div>", unsafe_allow_html=True)

import streamlit as st
import plotly.express as px
import pandas as pd

from lib import data as D
from lib.theme import inject_base_css, page_header, RESULT_COLORS, RISK_COLORS, MUTED
from lib.data import build_master_table, latest_snapshot_per_enrollment
from utils.styles import load_css
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

load_css(f'{ROOT_DIR}\styles.css')
inject_base_css()



if not D.data_available():
    st.error(
        "data/interim/ 아래에 정제 데이터 CSV가 없습니다. "
        "oulad_data_spec.md에 명시된 8개 파일을 data/interim/에 넣은 뒤 새로고침하세요."
    )
    st.stop()

raw = D.load_raw()
master = build_master_table(raw)
latest = latest_snapshot_per_enrollment(master)

page_header("대시보드", "OULAD 학습 활동 데이터 기반 이탈 위험 현황")

student_info = raw["student_info"]
total_enrollments = len(student_info)
result_counts = student_info["final_result"].value_counts(normalize=True)
churn_rate = result_counts.get("Withdrawn", 0) + result_counts.get("Fail", 0)
high_risk_count = latest.loc[latest["risk_grade"] == "high", "id_student"].nunique()
avg_clicks = raw["weekly"]["total_clicks"].mean()

k1, k2, k3, k4 = st.columns(4)
for col, label, value, sub, sub_color in [
    (k1, "전체 수강 등록 수", f"{total_enrollments:,}", f"{student_info['code_module'].nunique()}개 과목 운영 중", MUTED),
    (k2, "전체 이탈률 (Withdrawn+Fail)", f"{churn_rate*100:.1f}%",
     f"Pass {result_counts.get('Pass',0)*100:.1f}% · Distinction {result_counts.get('Distinction',0)*100:.1f}%", MUTED),
    (k3, "고위험 학생 수(최신 주차 기준)", f"{high_risk_count:,}", "즉시 조치 필요", RISK_COLORS["high"]["dot"]),
    (k4, "평균 주간 참여 클릭 수", f"{avg_clicks:.0f}", "전체 과목·주차 평균", MUTED),
]:
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub" style="color:{sub_color}">{sub}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")
c1, c2 = st.columns([1, 1.2])


with c1:
    with st.container(border=True):

        st.markdown("**전체 결과 분포**")
        dist_df = result_counts.rename_axis("final_result").reset_index(name="ratio")
        fig = px.pie(
            dist_df, names="final_result", values="ratio", hole=0.6,
            color="final_result", color_discrete_map=RESULT_COLORS,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with c2:
    with st.container(border=True):

        st.markdown("**과목별 이탈률 비교**")
        by_module = (
            student_info.groupby("code_module")["final_result"]
            .apply(lambda s: (s.isin(["Withdrawn", "Fail"])).mean())
            .rename("churn_rate").reset_index().sort_values("churn_rate", ascending=True)
        )
        by_module["risk_grade"] = by_module["churn_rate"].apply(
            lambda r: "high" if r >= 0.30 else ("mid" if r >= 0.20 else "low")
        )
        fig2 = px.bar(
            by_module, x="churn_rate", y="code_module", orientation="h",
            color="risk_grade", color_discrete_map={k: v["dot"] for k, v in RISK_COLORS.items()},
            text=by_module["churn_rate"].map(lambda v: f"{v*100:.0f}%"),
        )
        fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280, showlegend=False,
                            xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig2, use_container_width=True)

with st.container(border=True):

    st.markdown("**주차별 평균 참여도(클릭 수) 추이**")
    weekly_avg = raw["weekly"].groupby("week_index")["total_clicks"].mean().reset_index()
    weekly_avg = weekly_avg[weekly_avg["week_index"] <= 20]
    fig3 = px.area(weekly_avg, x="week_index", y="total_clicks")
    fig3.update_traces(line_color="#2f5bd7", fillcolor="rgba(47,91,215,0.12)")
    fig3.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260, xaxis_title="주차", yaxis_title="평균 클릭 수")
    st.plotly_chart(fig3, use_container_width=True)

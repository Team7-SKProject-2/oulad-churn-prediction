import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from lib import data as D
from lib.theme import inject_base_css, page_header, RESULT_COLORS, RISK_COLORS, MUTED, PRIMARY
from lib.data import build_master_table, latest_snapshot_per_enrollment
from utils.styles import load_css
from pathlib import Path

# ── 차트 스타일 설정 (여기 두 값만 바꾸면 대시보드 전체 바 차트에 반영됨) ──
BAR_WIDTH = 0.55  # 막대 너비, 0~1 사이 (기본 plotly는 약 0.8)
PALETTE = [PRIMARY, "#4c7bf0", "#8aa8f5", "#16224a", "#e0a83c", "#3f9142", "#d64545"]  # 카테고리 색상 팔레트


def style_bar(fig):
    """막대 끝 텍스트 라벨이 잘리거나 긴 축 라벨이 밀려서 잘리는 것을 방지하는 공통 스타일."""
    fig.update_traces(cliponaxis=False)  # 데이터 라벨이 플롯 경계에서 잘리지 않도록
    fig.update_xaxes(automargin=True)    # 긴 x축 라벨을 위한 여백 자동 확보
    fig.update_yaxes(automargin=True)    # 긴 y축 라벨(카테고리명)을 위한 여백 자동 확보
    return fig

ROOT_DIR = Path(__file__).resolve().parent.parent

load_css(ROOT_DIR / "styles.css")
inject_base_css()

if not D.data_available():
    st.error(
        "data/interim/ 아래에 정제 데이터 CSV가 없습니다. "
        "oulad_data_spec.md에 명시된 8개 파일을 data/interim/에 넣은 뒤 새로고침하세요."
    )
    st.stop()


def first_existing(df: pd.DataFrame, *candidates):
    """후보 컬럼명 중 실제로 존재하는 첫 번째 컬럼명을 반환. 없으면 None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


raw = D.load_raw()
master = build_master_table(raw)
latest = latest_snapshot_per_enrollment(master)

student_info = raw["student_info"]
registration = raw["registration"]
weekly = raw["weekly"]
assessments = raw["assessments"]
student_assessment = raw["student_assessment"]

page_header("대시보드", "OULAD 데이터 한눈에 보기 — 병합 데이터 EDA · VLE EDA 중간 결론 기반")

# ───────────────────────── KPI ─────────────────────────
total_enrollments = len(student_info)
result_counts = student_info["final_result"].value_counts(normalize=True)
withdrawn_rate = result_counts.get("Withdrawn", 0)
churn_rate = withdrawn_rate + result_counts.get("Fail", 0)
high_risk_count = latest.loc[latest["risk_grade"] == "high", "id_student"].nunique()
avg_clicks = weekly["total_clicks"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
for col, label, value, sub, sub_color in [
    (k1, "전체 수강 등록 수", f"{total_enrollments:,}", f"{student_info['code_module'].nunique()}개 과목 운영 중", MUTED),
    (k2, "Withdrawn 비율", f"{withdrawn_rate*100:.1f}%", "병합 EDA #1 — 중도 자퇴", MUTED),
    (k3, "전체 이탈률 (Withdrawn+Fail)", f"{churn_rate*100:.1f}%",
     f"Pass {result_counts.get('Pass',0)*100:.1f}% · Distinction {result_counts.get('Distinction',0)*100:.1f}%", MUTED),
    (k4, "고위험 학생 수(최신 주차 기준)", f"{high_risk_count:,}", "즉시 조치 필요", RISK_COLORS["high"]["dot"]),
    (k5, "평균 주간 참여 클릭 수", f"{avg_clicks:.0f}", "전체 과목·주차 평균", MUTED),
]:
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub" style="color:{sub_color}">{sub}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# ───────────────────────── 결과 분포 · 과목별 이탈률 ─────────────────────────
c1, c2 = st.columns([1, 1.2])

with c1:
    with st.container(border=True):
        st.markdown("**전체 결과 분포**")
        dist_df = result_counts.rename_axis("final_result").reset_index(name="ratio")
        fig = px.pie(
            dist_df, names="final_result", values="ratio", hole=0.6,
            color="final_result", color_discrete_map=RESULT_COLORS,
        )
        fig.update_traces(textinfo="percent+label", textposition="outside")
        fig.update_layout(margin=dict(t=30, b=30, l=40, r=40), height=300, showlegend=False,
                           uniformtext_minsize=10, uniformtext_mode="hide")
        st.plotly_chart(fig, use_container_width=True)

with c2:
    with st.container(border=True):
        # st.markdown("**과목별 이탈률 비교** — 병합 EDA #3")
        by_module = (
            student_info.groupby("code_module")["final_result"]
            .apply(lambda s: (s.isin(["Withdrawn", "Fail"])).mean())
            .rename("churn_rate").reset_index().sort_values("churn_rate", ascending=True)
        )
        by_module["risk_grade"] = by_module["churn_rate"].apply(
            lambda r: "high" if r >= 0.30 else ("mid" if r >= 0.20 else "low")
        )
        spread = by_module["churn_rate"].max() - by_module["churn_rate"].min()
        fig2 = px.bar(
            by_module, x="churn_rate", y="code_module", orientation="h",
            color="risk_grade", color_discrete_map={k: v["dot"] for k, v in RISK_COLORS.items()},
            text=by_module["churn_rate"].map(lambda v: f"{v*100:.0f}%"),
        )
        fig2.update_traces(width=BAR_WIDTH)
        fig2.update_layout(margin=dict(t=10, b=10, l=10, r=30), height=280, showlegend=False,
                            xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
        fig2 = style_bar(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            f"과목 간 이탈률 격차 {spread*100:.0f}%p — "
            + ("과목별 편차가 커서 강좌 단위 분리 또는 강좌 Feature 투입을 검토할 만합니다."
               if spread >= 0.15 else "과목 간 편차가 크지 않아 단일 모델 + 강좌 Feature로도 충분할 수 있습니다.")
        )

st.write("")

# ───────────────────────── 이탈 시점 분포 (골든타임 후보) ─────────────────────────
with st.container(border=True):
    st.markdown("**이탈 시점(주차) 분포 — 골든타임 후보** · 병합 EDA #2")
    unreg_col = first_existing(registration, "date_unregistration")
    if unreg_col is None:
        st.caption("⚠️ registration 데이터에 `date_unregistration` 컬럼이 없어 이 분석은 건너뜁니다. "
                   "이탈 골든타임은 반드시 이 컬럼(또는 `studentRegistration.date_unregistration`)으로 별도 확인이 필요합니다.")
    else:
        reg_u = registration.dropna(subset=[unreg_col]).copy()
        reg_u["unregister_week"] = (reg_u[unreg_col] // 7 + 1).astype(int)
        reg_u = reg_u[reg_u["unregister_week"] >= 1]
        if reg_u.empty:
            st.caption("이탈(자퇴) 기록이 있는 학생이 없어 분포를 그릴 수 없습니다.")
        else:
            med = reg_u["unregister_week"].median()
            p75 = reg_u["unregister_week"].quantile(0.75)
            p90 = reg_u["unregister_week"].quantile(0.90)
            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, label, value in [
                (sc1, "이탈 학생 수", f"{len(reg_u):,}명"),
                (sc2, "중앙값", f"{med:.0f}주차"),
                (sc3, "75% 분위수", f"{p75:.0f}주차"),
                (sc4, "90% 분위수", f"{p90:.0f}주차"),
            ]:
                col.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
                    unsafe_allow_html=True,
                )
            st.write("")
            fig_u = px.histogram(reg_u, x="unregister_week", nbins=39)
            fig_u.update_traces(marker_color=PALETTE[0])
            fig_u.add_vline(x=med, line_dash="dash", line_color=RISK_COLORS["high"]["dot"],
                             annotation_text="중앙값", annotation_position="top", annotation_yshift=10)
            fig_u.update_layout(margin=dict(t=45, b=40, l=55, r=20), height=260, bargap=1 - BAR_WIDTH,
                                 xaxis_title="이탈 주차", yaxis_title="학생 수")
            fig_u = style_bar(fig_u)
            st.plotly_chart(fig_u, use_container_width=True)
            st.caption("이탈이 몰리는 구간이 여기서 확인되면 '골든타임' 후보 시점으로 기록하고, "
                       "아래 VLE 활동 감소 구간(11-12주차, 25-26주차 등)과 겹치는지 함께 살펴보시면 좋습니다.")

st.write("")

# ───────────────────────── 학생 특성별 Withdrawn 비율 ─────────────────────────
with st.container(border=True):
    st.markdown("**학생 특성별 Withdrawn 비율** · 병합 EDA #6")

    def churn_bar(col_name, label, code_map=None, exclude=("?",)):
        d = student_info[~student_info[col_name].astype(str).isin(exclude)].copy()
        if code_map:
            d[col_name] = d[col_name].map(lambda v: code_map.get(v, v))
        g = (
            d.groupby(col_name)["final_result"].apply(lambda s: (s == "Withdrawn").mean())
            .rename("withdrawn_rate").reset_index().sort_values("withdrawn_rate", ascending=True)
        )
        fig = px.bar(g, x="withdrawn_rate", y=col_name, orientation="h", color=col_name,
                     color_discrete_sequence=PALETTE,
                     text=g["withdrawn_rate"].map(lambda v: f"{v*100:.0f}%"))
        fig.update_traces(width=BAR_WIDTH)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=35), height=200, xaxis_tickformat=".0%",
                           xaxis_title=None, yaxis_title=None, showlegend=False)
        fig = style_bar(fig)
        return fig, g

    dims = [
        (first_existing(student_info, "gender"), "성별", None),
        (first_existing(student_info, "age_band_cd", "age_band"), "연령대", {"Y": "0-35세", "M": "35-55세", "S": "55세 이상"}),
        (first_existing(student_info, "disability"), "장애 여부", {"N": "없음", "Y": "있음"}),
        (first_existing(student_info, "imd_band_cd", "imd_band"), "IMD(사회경제) 구간", None),
        (first_existing(student_info, "highest_education_cd", "highest_education"), "최종 학력", None),
    ]
    available = [d for d in dims if d[0] is not None]
    missing = [d for d in dims if d[0] is None]

    if available:
        row1 = st.columns(min(3, len(available)))
        row2_dims = available[3:]
        for col, (colname, label, cmap) in zip(row1, available[:3]):
            fig, g = churn_bar(colname, label, cmap)
            col.markdown(f"<div style='font-size:13px;font-weight:600;color:#1c2333;margin-bottom:4px;'>{label}</div>", unsafe_allow_html=True)
            col.plotly_chart(fig, use_container_width=True)
        if row2_dims:
            row2 = st.columns(len(row2_dims))
            for col, (colname, label, cmap) in zip(row2, row2_dims):
                fig, g = churn_bar(colname, label, cmap)
                col.markdown(f"<div style='font-size:13px;font-weight:600;color:#1c2333;margin-bottom:4px;'>{label}</div>", unsafe_allow_html=True)
                col.plotly_chart(fig, use_container_width=True)
    if missing:
        st.caption("⚠️ 다음 항목은 student_info에 해당 컬럼이 없어 생략됨: " + ", ".join(lbl for _, lbl, _ in missing))

st.write("")

# ───────────────────────── 과목별 주차 활동 학생 수 추이 ─────────────────────────
with st.container(border=True):
    st.markdown("**과목별 주차 활동 학생 수 추이** · VLE EDA #1·#2·#3·#4")
    active_by_week = (
        weekly.groupby(["code_module", "week_index"])["id_student"].nunique().reset_index(name="active_students")
    )
    active_by_week = active_by_week[active_by_week["week_index"] <= 39]
    fig_active = px.line(active_by_week, x="week_index", y="active_students", color="code_module")
    for wk in [11.5, 25.5, 34]:
        fig_active.add_vline(x=wk, line_dash="dot", line_color=MUTED, opacity=0.5)
    fig_active.update_layout(margin=dict(t=15, b=40, l=60, r=20), height=320, xaxis_title="주차", yaxis_title="활동 학생 수")
    fig_active.update_xaxes(automargin=True)
    fig_active.update_yaxes(automargin=True)
    st.plotly_chart(fig_active, use_container_width=True)
    st.caption(
        "전반적으로 주차가 진행될수록 활동 학생 수는 감소하지만, 과목별 편차가 커서 전체 평균만으로 특정 주차를 "
        "'이탈 골든타임'으로 단정하면 안 됩니다. 점선으로 표시한 11-12·25-26·34주차 부근의 감소는 과제 일정이나 "
        "일부 강좌의 종료 시점 등 운영 구조의 영향일 수 있어, 위 이탈 시점 분포와 교차 확인이 필요합니다."
    )

st.write("")

# ───────────────────────── 클릭 수 분포 ─────────────────────────
with st.container(border=True):
    st.markdown("**주차별 총 클릭 수 분포** · VLE EDA #6·#7")
    clicks = weekly["total_clicks"].dropna()
    q50, q90, q95, q99 = clicks.quantile([0.5, 0.9, 0.95, 0.99])
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    for col, label, value in [
        (qc1, "중앙값", f"{q50:.0f}"), (qc2, "90% 분위수", f"{q90:.0f}"), (qc3, "95% 분위수", f"{q95:.0f}"),
        (qc4, "99% 분위수", f"{q99:.0f}"), (qc5, "최댓값", f"{clicks.max():,.0f}"),
    ]:
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    log_clicks = pd.DataFrame({"log1p_total_clicks": np.log1p(clicks)})
    fig_hist = px.histogram(log_clicks, x="log1p_total_clicks", nbins=60)
    fig_hist.update_traces(marker_color=PALETTE[0])
    fig_hist.update_layout(margin=dict(t=20, b=40, l=55, r=20), height=260, bargap=1 - BAR_WIDTH,
                            xaxis_title="log1p(총 클릭 수)", yaxis_title="빈도")
    fig_hist = style_bar(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("오른쪽으로 크게 치우친 분포입니다 — 모델링 시 원본 클릭 수 대신 log1p 변환값 사용을 비교해볼 필요가 있습니다.")

st.write("")

# ───────────────────────── 과제 EDA ─────────────────────────
with st.container(border=True):
    st.markdown("**과제(assessment) EDA** · 병합 EDA #4·#5")
    a1, a2, a3 = st.columns(3)

    with a1:
        if "assessment_type" in assessments.columns and "score" in student_assessment.columns:
            sa_type = student_assessment.merge(
                assessments[["id_assessment", "assessment_type"]], on="id_assessment", how="left"
            )
            by_type = sa_type.groupby("assessment_type")["score"].mean().reset_index().sort_values("score")
            fig_t = px.bar(by_type, x="score", y="assessment_type", orientation="h", color="assessment_type",
                           color_discrete_sequence=PALETTE, text=by_type["score"].round(1))
            fig_t.update_traces(width=BAR_WIDTH)
            fig_t.update_layout(margin=dict(t=15, b=10, l=10, r=35), height=210,
                                 xaxis_title=None, yaxis_title=None, showlegend=False)
            fig_t = style_bar(fig_t)
            st.markdown("<div style='font-size:13px;font-weight:600;color:#1c2333;margin-bottom:4px;'>과제유형별 평균 점수</div>", unsafe_allow_html=True)
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.caption("assessment_type 또는 score 컬럼이 없어 생략됨")

    with a2:
        if "score" in student_assessment.columns:
            sa_res = student_assessment.merge(
                student_info[["id_student", "code_module", "code_presentation", "final_result"]],
                on="id_student", how="left",
            )
            by_res = sa_res.groupby("final_result")["score"].mean().reset_index()
            fig_r = px.bar(by_res, x="final_result", y="score", color="final_result",
                            color_discrete_map=RESULT_COLORS, text=by_res["score"].round(1))
            fig_r.update_traces(width=BAR_WIDTH)
            fig_r.update_layout(margin=dict(t=15, b=10, l=10, r=10), height=210,
                                 showlegend=False, xaxis_title=None, yaxis_title=None)
            fig_r = style_bar(fig_r)
            st.markdown("<div style='font-size:13px;font-weight:600;color:#1c2333;margin-bottom:4px;'>최종결과별 평균 점수</div>", unsafe_allow_html=True)
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.caption("score 컬럼이 없어 생략됨")

    with a3:
        date_submitted_col = first_existing(student_assessment, "date_submitted")
        due_date_col = first_existing(assessments, "date")
        if date_submitted_col and due_date_col:
            sa_gap = student_assessment.merge(
                assessments[["id_assessment", "code_module", "code_presentation", due_date_col]], on="id_assessment", how="inner"
            )
            if "is_banked" in sa_gap.columns:
                sa_gap = sa_gap[sa_gap["is_banked"] == 0]
            sa_gap["submission_gap"] = sa_gap[date_submitted_col] - sa_gap[due_date_col]
            sa_gap = sa_gap.merge(
                student_info[["id_student", "code_module", "code_presentation", "final_result"]],
                on=["id_student", "code_module", "code_presentation"], how="left",
            )
            late_rate = sa_gap.groupby("final_result")["submission_gap"].apply(lambda s: (s > 0).mean()).reset_index(name="late_rate")
            fig_l = px.bar(late_rate, x="final_result", y="late_rate", color="final_result",
                            color_discrete_map=RESULT_COLORS, text=late_rate["late_rate"].map(lambda v: f"{v*100:.0f}%"))
            fig_l.update_traces(width=BAR_WIDTH)
            fig_l.update_layout(margin=dict(t=15, b=10, l=10, r=10), height=210,
                                 showlegend=False, yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
            fig_l = style_bar(fig_l)
            st.markdown("<div style='font-size:13px;font-weight:600;color:#1c2333;margin-bottom:4px;'>최종결과별 지각 제출 비율</div>", unsafe_allow_html=True)
            st.plotly_chart(fig_l, use_container_width=True)
        else:
            st.caption("date_submitted 또는 due date 컬럼이 없어 생략됨")

    st.caption("지각 제출 비율이 Withdrawn/Fail 그룹에서 뚜렷이 높게 나오면, 지각 제출이 이탈의 선행 신호일 가능성을 뒷받침합니다.")

st.write("")

# ───────────────────────── 상관관계 ─────────────────────────
with st.container(border=True):
    st.markdown("**주차별 VLE 피처 상관관계** · VLE EDA #8·#9")
    corr_cols = [c for c in ["total_clicks", "interaction_rows", "unique_sites", "activity_type_count"] if c in weekly.columns]
    if len(corr_cols) >= 2:
        corr = weekly[corr_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues", zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        fig_corr.update_xaxes(automargin=True, tickangle=-30)
        fig_corr.update_yaxes(automargin=True)
        st.plotly_chart(fig_corr, use_container_width=True)
        high_pairs = [
            (corr_cols[i], corr_cols[j], corr.iloc[i, j])
            for i in range(len(corr_cols)) for j in range(i + 1, len(corr_cols))
            if abs(corr.iloc[i, j]) >= 0.7
        ]
        if high_pairs:
            pair_txt = ", ".join(f"{a}↔{b} ({v:.2f})" for a, b, v in high_pairs)
            st.caption(f"⚠️ 상관계수 0.7 이상 — 다중공선성 의심: {pair_txt}. 선형 모델에서는 중복 Feature 선택/규제가 필요합니다.")
        else:
            st.caption("0.7 이상의 뚜렷한 다중공선성 쌍은 확인되지 않았습니다.")
    else:
        st.caption(
            "⚠️ weekly 데이터에 " + ", ".join(["total_clicks", "interaction_rows", "unique_sites", "activity_type_count"])
            + " 중 일부 컬럼이 없어 상관관계 분석을 제한적으로만 수행했습니다."
        )

st.write("")


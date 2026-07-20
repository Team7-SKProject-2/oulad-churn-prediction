"""공통 컬러 팔레트 & 스타일 주입. 대시보드 전체 톤(블루 계열)을 여기서 관리한다."""
import streamlit as st

PRIMARY = "#2f5bd7"
PRIMARY_DARK = "#16224a"
PRIMARY_SOFT = "#e8edfb"
BG = "#f5f6fa"
CARD = "#ffffff"
BORDER = "#e3e6ee"
TEXT = "#1c2333"
MUTED = "#6a7286"

RISK_COLORS = {
    "high": {"bg": "#fbe4e4", "fg": "#b23a3a", "dot": "#d64545", "label": "고위험"},
    "mid":  {"bg": "#fbeed8", "fg": "#8a5b12", "dot": "#e0a83c", "label": "중위험"},
    "low":  {"bg": "#e2f2e3", "fg": "#2f6b34", "dot": "#3f9142", "label": "저위험"},
}

RESULT_COLORS = {
    "Pass": "#3f9142",
    "Withdrawn": "#d64545",
    "Fail": "#e0a83c",
    "Distinction": PRIMARY,
}


def risk_grade(score: float) -> str:
    if score >= 65:
        return "high"
    if score >= 35:
        return "mid"
    return "low"


def inject_base_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background:{BG}; }}
        [data-testid="stSidebar"] {{ background:{PRIMARY_DARK}; }}
        [data-testid="stSidebar"] * {{ color:#dbe1f5 !important; }}
        [data-testid="stSidebarNav"] a {{ border-radius:8px; }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{ background:rgba(255,255,255,0.12); }}
        .kpi-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:18px 20px; }}
        .kpi-label {{ font-size:12px; color:{MUTED}; }}
        .kpi-value {{ font-size:26px; font-weight:700; color:{TEXT}; margin-top:4px; }}
        .kpi-sub {{ font-size:11.5px; margin-top:4px; }}
        .section-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:20px; }}
        .risk-badge {{ display:inline-block; font-size:12px; font-weight:700; padding:3px 10px; border-radius:999px; }}
        .chip {{ display:inline-block; font-size:11px; padding:3px 8px; border-radius:6px; background:{PRIMARY_SOFT}; color:{PRIMARY}; margin:2px 4px 2px 0; }}
        .action-card {{ background:{BG}; border-radius:8px; padding:12px 14px; margin-bottom:8px; }}
        .app-title {{ font-size:20px; font-weight:700; color:{TEXT}; }}
        .app-subtitle {{ font-size:13px; color:{MUTED}; margin-top:2px; margin-bottom:18px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(grade: str) -> str:
    c = RISK_COLORS[grade]
    return f'<span class="risk-badge" style="background:{c["bg"]};color:{c["fg"]};">{c["label"]}</span>'


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)

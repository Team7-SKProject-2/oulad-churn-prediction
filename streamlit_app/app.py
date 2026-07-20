import streamlit as st

st.set_page_config(page_title="학생 이탈 예측 시스템", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background:#16224a; }
    [data-testid="stSidebar"] * { color:#dbe1f5 !important; }
    [data-testid="stSidebarNav"] a[aria-current="page"] { background:rgba(255,255,255,0.12); border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<div style='font-size:14px;font-weight:700;color:white;padding:8px 4px 0;'>학생 이탈 예측 시스템</div>"
    "<div style='font-size:11px;color:#9aa6cf;padding:0 4px 12px;'>관리자 콘솔</div>",
    unsafe_allow_html=True,
)

# 각 메뉴에 아이콘을 명시적으로 지정 (파일명만으로는 사이드바에 아이콘이 뜨지 않는다)
pages = [
    st.Page("pages/0_dashboard.py", title="대시보드", icon=":material/dashboard:", default=True),
    st.Page("pages/1_course_weekly_recommendations.py", title="과목별 행동제안", icon=":material/checklist:"),
    st.Page("pages/2_students_recommendations.py", title="학생별 행동추천", icon=":material/person:"),
    st.Page("pages/3_dropout_predictions.py", title="이탈 예측", icon=":material/query_stats:"),
]
nav = st.navigation(pages)
nav.run()

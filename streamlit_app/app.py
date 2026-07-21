import streamlit as st

st.set_page_config(page_title="1~10주차 다음 주 이탈 조기경보", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background:#16224a; }
    [data-testid="stSidebar"] * { color:#dbe1f5 !important; }
    [data-testid="stSidebarNav"] a[aria-current="page"] { background:rgba(255,255,255,0.12); border-radius:8px; }
    
    [data-testid="stSidebarNav"]::before {
    content: "🎓 1~10주차 다음 주 이탈 조기경보";
    display: block;
    color: white;
    font-size: 18px;
    font-weight: 700;
    padding: 8px 12px 12px 12px;
}
    </style>
    
    """,
    unsafe_allow_html=True,
)



# 각 메뉴에 아이콘을 명시적으로 지정 (파일명만으로는 사이드바에 아이콘이 뜨지 않는다)
pages = [
    st.Page(
        "pages/3_dropout_predictions.py",
        title="1~10주차 이탈 예측",
        icon=":material/query_stats:",
        default=True,
    ),
    st.Page("pages/0_dashboard.py", title="전체 EDA", icon=":material/dashboard:"),
    st.Page("pages/1_course_weekly_recommendations.py", title="과목별 행동제안", icon=":material/checklist:"),
    st.Page("pages/2_students_recommendations.py", title="학생별 행동추천", icon=":material/person:"),
]
nav = st.navigation(pages)
nav.run()

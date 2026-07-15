import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

st.title("프로젝트 현황")

cohort_path = PROJECT_ROOT / "data" / "interim" / "cohort_base.csv"
metadata_path = PROJECT_ROOT / "artifacts" / "model_metadata.json"

if cohort_path.exists():
    cohort = pd.read_csv(cohort_path)
    col1, col2 = st.columns(2)
    col1.metric("수강 사례", f"{len(cohort):,}")
    if "is_withdrawn" in cohort.columns:
        col2.metric("이탈 비율", f"{cohort['is_withdrawn'].mean():.1%}")
else:
    st.warning("아직 cohort_base.csv가 없습니다.")

if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    st.subheader("모델 메타데이터")
    st.json(metadata)


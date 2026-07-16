import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

st.title("주차·과목별 이탈 현황")

weekly_path = ARTIFACTS_DIR / "weekly_dropout_summary.csv"
module_week_path = ARTIFACTS_DIR / "module_week_dropout_summary.csv"
metadata_path = ARTIFACTS_DIR / "model_metadata.json"

if not weekly_path.exists() or not module_week_path.exists():
    st.warning("요약 파일이 없습니다. `python -m src.export_eda_artifacts`를 실행하세요.")
else:
    weekly = pd.read_csv(weekly_path)
    module_week = pd.read_csv(module_week_path)
    peak = weekly.loc[weekly["dropout_rate_pct"].idxmax()]

    col1, col2, col3 = st.columns(3)
    col1.metric("정상 범위 이탈", f"{int(weekly['dropout_count'].sum()):,}건")
    col2.metric("최고 이탈률 주차", f"{int(peak['week_index'])}주차")
    col3.metric("해당 주차 이탈률", f"{peak['dropout_rate_pct']:.2f}%")

    st.subheader("전체 주차별 이탈률")
    st.line_chart(weekly.set_index("week_index")[["dropout_rate_pct"]])
    st.dataframe(weekly, width="stretch", hide_index=True)

    st.subheader("과목 × 주차 이탈률 Heatmap")
    heatmap = module_week.pivot(
        index="code_module",
        columns="week_index",
        values="dropout_rate_pct",
    )
    st.dataframe(
        heatmap.style.background_gradient(cmap="OrRd").format("{:.2f}%"),
        width="stretch",
    )

if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_name"):
        st.subheader("선택된 모델")
        st.json(metadata)

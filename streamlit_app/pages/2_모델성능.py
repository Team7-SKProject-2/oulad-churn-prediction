from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.csv"

st.title("모델 성능")

if not METRICS_PATH.exists():
    st.warning("아직 metrics.csv가 없습니다.")
else:
    metrics = pd.read_csv(METRICS_PATH)
    if metrics.empty:
        st.info("모델 학습 후 이 페이지에 성능이 표시됩니다.")
    else:
        st.dataframe(metrics, use_container_width=True)
        chart_columns = [column for column in ["precision", "recall", "f1", "pr_auc", "roc_auc"] if column in metrics]
        if chart_columns:
            st.bar_chart(metrics.set_index("model")[chart_columns])


from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="OULAD 중도이탈 조기경보", page_icon="📘", layout="wide")

st.title("OULAD 학습 중도이탈 조기경보")
st.write(
    "1·2·4주차 학습행동을 비교해 개입 골든타임을 찾고, "
    "과목별 위험 특성에 맞는 유지 활동을 제안하는 프로젝트입니다."
)

st.info("왼쪽 메뉴에서 현황, 모델 성능, 이탈 예측 페이지를 선택하세요.")

st.subheader("운영 원칙")
st.markdown(
    """
- Streamlit에서 모델을 다시 학습하지 않습니다.
- `models/churn_pipeline.joblib`과 `src/predict.py`를 사용합니다.
- 모델이 없으면 먼저 ML 학습을 완료해야 합니다.
"""
)

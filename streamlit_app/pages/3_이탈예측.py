from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_dataframe  # noqa: E402


st.title("개별·일괄 이탈 예측")
st.write("모델 입력 Feature가 포함된 CSV를 업로드하세요.")

uploaded = st.file_uploader("예측 CSV", type=["csv"])
if uploaded is not None:
    input_frame = pd.read_csv(uploaded)
    st.subheader("입력 미리보기")
    st.dataframe(input_frame.head(), use_container_width=True)
    if st.button("이탈 위험 예측"):
        try:
            result = predict_dataframe(input_frame)
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
        else:
            st.subheader("예측 결과")
            st.dataframe(result, use_container_width=True)


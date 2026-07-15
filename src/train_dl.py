"""선택 딥러닝 실험 진입점.

기본 ML 파이프라인이 안정된 뒤 TabNet 등 선택 모델을 이 파일에 구현한다.
동일한 데이터 분할과 평가 함수를 반드시 재사용한다.
"""

from __future__ import annotations


def train_dl(*_args, **_kwargs):
    raise NotImplementedError(
        "딥러닝은 선택 작업입니다. 기본 ML·Streamlit 완성 후 구현하세요."
    )


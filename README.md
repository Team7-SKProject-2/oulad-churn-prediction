# OULAD 학습 중도이탈 예측 프로젝트

OULAD 학습행동 데이터를 이용해 1·2·3주차의 중도이탈 위험을 비교하고, 가장 빠르게 안정적인 예측이 가능한 시점을 유지 개입 골든타임으로 선정하는 프로젝트입니다.

## PyCharm 시작 방법

1. PyCharm에서 이 `project` 폴더를 프로젝트로 엽니다.
2. 프로젝트 전용 가상환경을 생성합니다.
3. `requirements.txt`를 설치합니다.
4. OULAD 원본 CSV를 `data/raw/`에 넣습니다.
5. 데이터 점검부터 실행합니다.

```bash
python -m pip install -r requirements.txt
python -m src.data --check
python -m src.data --build-cohort
python -m src.features --build-vle
```

Streamlit 실행:

```bash
streamlit run streamlit_app/app.py
```

테스트 실행:

```bash
python -m unittest discover -s tests
```

## 공통 기준

- 분석 단위: `id_student + code_module + code_presentation`
- Target: `final_result == "Withdrawn"`
- 비교 시점: 1주차(0~6일), 2주차(0~13일), 3주차(0~20일)
- 동일 학생은 Train·Validation·Test 중 하나에만 포함합니다.
- Test는 최종 모델과 임계값 결정 후 한 번만 평가합니다.

## 협업 결과물

- 공통 코호트: `data/interim/cohort_base.csv`
- 평가 Feature: `data/interim/assessment_weekly_features.csv`
- VLE Feature: `data/interim/vle_weekly_features.csv`
- 최종 모델링 데이터: `data/processed/modeling_week1.csv` 등
- 최종 모델: `models/churn_pipeline.joblib` — 학습 완료 후 생성
- 발표 자료: `presentation.pdf` — 발표 자료 확정 후 생성

## 운영 원칙

- `data/raw` 원본은 수정하지 않습니다.
- 코드에서는 개인 PC 절대경로를 사용하지 않습니다.
- 생성 데이터와 모델은 Git에 올리지 않습니다.
- 모델과 함께 Feature 순서, 전처리기, threshold, 메타데이터를 저장합니다.
- Streamlit은 저장된 모델과 `src/predict.py`만 이용해 추론합니다.

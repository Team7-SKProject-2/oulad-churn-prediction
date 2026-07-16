# OULAD 학습 중도이탈 예측 프로젝트

OULAD 학습행동 데이터를 이용해 1·2·4주차의 중도이탈 위험을 비교하고, 가장 빠르게 안정적인 예측이 가능한 시점을 유지 개입 골든타임으로 선정합니다. 과목별 이탈 시점과 행동 차이를 함께 분석해 유지 활동도 제안합니다.

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
- 비교 시점: 1주차(0~6일), 2주차(0~13일), 4주차(0~27일)
- 머신러닝 단계에서는 동일 학생을 Train·Validation·Test 중 하나에만 포함해야 합니다.
- Test는 머신러닝 담당자가 최종 모델과 임계값을 결정한 뒤 한 번만 평가합니다.

## 협업 결과물

- 정정 코호트: `data/interim/student_registration_merged_corrected.csv`
- VLE Feature: `data/interim/vle_weekly_features.csv`
- 주차별 최종 Snapshot: `data/processed/model_snapshot_week_1.csv` 등
- 최종 모델: `models/churn_pipeline.joblib` — 학습 완료 후 생성
- EDA 보고서: `reports/eda_report.md`
- 발표 자료: `presentation.pdf` — 발표 자료 확정 후 생성

## 머신러닝 담당자 빠른 시작

최종 1·2·4주차 Snapshot은 Git에서 함께 관리한다. 새로 clone하거나 pull한 뒤
노트북을 다시 실행할 필요 없이 아래 검증만 통과하면 바로 머신러닝 단계로
넘어갈 수 있다.

검증 환경은 Python 3.13.13이며 패키지 버전은 `requirements.txt`에 고정했다.

```bash
python -m pip install -r requirements.txt
python -m src.check_preprocessing_handoff
```

학습 입력 파일:

- `data/processed/model_snapshot_week_1.csv` — 29,018행, 99열
- `data/processed/model_snapshot_week_2.csv` — 27,984행, 99열
- `data/processed/model_snapshot_week_4.csv` — 27,449행, 99열

`target`은 정답 컬럼이며 `id_student`는 학생 단위 분할에만 사용하고 입력
Feature에서는 제외한다. 데이터 분할·인코딩·모델·임계값 결정은 ML 담당 범위다.

## 운영 원칙

- `data/raw` 원본은 수정하지 않습니다.
- 코드에서는 개인 PC 절대경로를 사용하지 않습니다.
- 원본·중간 생성 데이터와 모델은 Git에 올리지 않습니다.
- 인수인계용 최종 `model_snapshot_week_*.csv` 3개만 Git에서 관리합니다.
- 모델과 함께 Feature 순서, 전처리기, threshold, 메타데이터를 저장합니다.
- Streamlit은 저장된 모델과 `src/predict.py`만 이용해 추론합니다.

## 실행 순서

아래는 원본 OULAD 데이터부터 전처리와 EDA를 전부 재생성할 때만 사용한다.
ML 담당자는 위의 최종 Snapshot 검증만 실행하면 된다.

```bash
python -m src.data --check --build-cohort
python -m src.vle_features
python -m src.build_model_snapshots
python -m src.export_eda_artifacts
python -m src.check_preprocessing_handoff
```

여기까지가 전처리·EDA 담당 범위입니다. 생성된 1·2·4주차 `model_snapshot`을 머신러닝 담당자에게 전달하며, 데이터 분할·모델 학습·Validation·Test 평가는 머신러닝 단계에서 진행합니다.

# OULAD 학습 중도이탈 조기예측 및 과목별 유지전략 프로젝트

OULAD 학습 데이터를 활용하여 학습 중도이탈 위험 학생을 조기에 예측하고, 적절한 개입 시점과 과목 특성에 맞는 유지 활동을 제안하는 프로젝트입니다.

이번 프로젝트는 단순히 정확도가 높은 모델을 만드는 데 그치지 않고 다음 흐름이 자연스럽게 연결되는 것을 목표로 합니다.

> 데이터 분석 → 모델 학습 → 성능 평가 → 위험 학생 예측 → 유지 활동 제안

프로젝트는 주차별 분석과 과목별 분석이라는 두 가지 소주제로 구성됩니다.

- 주차별 분석: 언제 개입해야 하는가?
- 과목별 분석: 누구에게 어떤 방식으로 개입해야 하는가?

두 분석을 결합하여 적절한 골든타임에 위험 학생을 발견하고, 과목 특성에 맞는 유지 전략을 제안합니다.

---

## 1. 프로젝트 분석 주제

### 소주제 1. 주차별 분석

> 이탈은 언제 집중되며, 위험 학생을 예측하기 적절한 골든타임은 언제인가?

전체 수강 기간의 주차별 이탈 현황을 먼저 분석한 뒤, 조기예측 후보 시점인 1·2·4주차의 데이터를 비교합니다.

확인할 내용:

- 전체 수강 기간의 주차별 이탈 건수와 이탈률
- 이탈이 집중되거나 증가하는 시점
- 이탈·비이탈 학생의 주차별 VLE 학습행동 차이
- 1·2·4주차별 머신러닝 성능
- 조기성 및 예측 성능을 고려한 공통 골든타임 선정

주차별 분석 결과는 위험 학생에게 언제 개입할지를 결정하는 데 사용합니다.

### 소주제 2. 과목별 분석

> 과목마다 이탈 시점과 학습행동에 어떤 차이가 있으며, 개입 방법을 어떻게 다르게 해야 하는가?

과목별 이탈률과 이탈 시점, VLE 활동 차이를 분석하여 과목 특성에 맞는 유지 활동을 제안합니다.

확인할 내용:

- 과목별 수강 인원과 이탈률
- 과목별 이탈 주차 중앙값과 최빈값
- 과목별 클릭량, 활동일 수, 이용 콘텐츠 수
- 과목별 포럼·퀴즈·학습자료 이용 차이
- 과목별 또는 과목 그룹별 모델 성능
- 과목 특성에 맞는 유지 활동

과목별 분석 결과는 누구에게 어떤 방식으로 개입할지를 결정하는 데 사용합니다.

### 최종 결론

- 주차별 분석 → 언제 개입할지 결정
- 과목별 분석 → 누구에게 어떤 방식으로 개입할지 결정
- 머신러닝 예측 → 실제 위험 학생을 선별
- 비즈니스 활용 → 위험 원인과 과목 특성에 맞는 유지 활동 제안

---

## 2. 공통 분석 기준

- 데이터셋: OULAD(Open University Learning Analytics Dataset)
- 분석 단위: `id_student + code_module + code_presentation`
- 기본 이탈 기준: `final_result == "Withdrawn"`
- 이탈: `Withdrawn`
- 비이탈: `Pass`, `Distinction`, `Fail`
- 조기예측 후보 시점: 1주차, 2주차, 4주차
- 1주차 관측 범위: 개강 후 0~6일
- 2주차 관측 범위: 개강 후 0~13일
- 4주차 관측 범위: 개강 후 0~27일

머신러닝 Snapshot에서는 각 예측 시점 이전에 이미 이탈한 학생-강좌를 제외합니다.

따라서 Snapshot의 `target`은 다음과 같습니다.

- `target = 1`: 해당 예측 시점 이후 실제로 이탈한 학생
- `target = 0`: 최종적으로 이탈하지 않은 학생

예측 시점 이전에 이미 이탈한 학생은 조기경보 대상이 아니므로 학습 데이터에서 제외합니다.

---

## 3. 주요 데이터 및 Feature

### 학생·강좌 정보

- 학생 ID
- 과목 코드
- 강좌 개설 시기
- 연령대
- 성별
- 최종 학력
- 지역
- 이전 수강 횟수
- 학점 수
- 장애 여부
- 최종 수강 결과

### VLE 학습행동 Feature

- 주차별 전체 클릭 수
- 활동 기록 수
- 활동일 수
- 이용 콘텐츠 수
- 이용한 활동 유형 수
- 활동일당 평균 클릭 수
- 콘텐츠당 평균 클릭 수
- 포럼 클릭 수
- 퀴즈 클릭 수
- 학습자료 클릭 수
- 기타 활동 클릭 수
- 개강 전 클릭 수
- 누적 클릭 수
- 최근 활동 여부 및 무활동 여부

OULAD의 클릭 수는 정확한 학습시간이 아니라 온라인 학습시스템에 기록된 콘텐츠 접근 및 상호작용 횟수입니다. 따라서 학습 참여량을 나타내는 대리 지표로 해석합니다.

### 평가 관련 Feature

- 해당 시점까지 예정된 평가 수
- 제출 가능한 평가 수
- 평가 제출 수
- 평가 미제출 수
- 지각 제출 수
- 평균 및 중앙값 점수
- 평가 유형별 제출 현황
- TMA·CMA·Exam 관련 Feature

평가 일정이 아직 도래하지 않은 시점에는 평가 관련 값이 없을 수 있으므로, 평가 미실시와 평가 미제출을 구분합니다.

---

## 4. 전처리·EDA 주요 결과

전처리 및 EDA에서는 다음 내용을 확인했습니다.

- 전체 수강 기간의 주차별 이탈 현황
- 과목별 이탈률
- 과목별 이탈 주차 중앙값과 최빈값
- 주차별 VLE 활동 변화
- 이탈 예정 학생과 비이탈 학생의 행동 차이
- 과목별 클릭량과 무활동 비율 차이
- 평가 제출률, 미제출률, 지각 제출률 차이
- 1·2·4주차별 머신러닝 입력 데이터의 무결성

현재까지의 EDA에서는 이탈 예정 학생이 비이탈 학생보다 다음과 같은 특징을 보이는 경향을 확인했습니다.

- 주차별 클릭 수가 적음
- 이용하는 콘텐츠의 범위가 좁음
- VLE 활동이 없는 비율이 높음
- 평가 점수가 상대적으로 낮음
- 평가 미제출률과 지각 제출률이 높음

이러한 차이는 머신러닝 입력 Feature와 유지 활동 설계에 활용합니다.

---

## 5. 전처리·EDA 결과물

- 정정 코호트: `data/interim/student_registration_merged_corrected.csv`
- 주차별 VLE Feature: `data/interim/vle_weekly_features.csv`
- 개강 전 VLE Feature: `data/interim/vle_pre_course_features.csv`
- 정제된 VLE 메타데이터: `data/interim/vle_metadata_clean.csv`
- 1주차 Snapshot: `data/processed/model_snapshot_week_1.csv`
- 2주차 Snapshot: `data/processed/model_snapshot_week_2.csv`
- 4주차 Snapshot: `data/processed/model_snapshot_week_4.csv`
- EDA 보고서: `reports/eda_report.md`
- 전처리 보고서: `reports/preprocessing_report.md`
- 평가 Feature 보고서: `reports/assessment_preprocessing_report.md`
- 머신러닝 인계 문서: `reports/ml_handoff.md`

원본 및 중간 생성 데이터는 Git에서 관리하지 않으며, 머신러닝 인계용 최종 Snapshot 3개만 Git에서 관리합니다.

---

## 6. 머신러닝 담당자 빠른 시작

최종 1·2·4주차 Snapshot은 Git에서 함께 관리합니다.

새로 clone하거나 `main` 브랜치를 pull한 뒤, 노트북을 다시 실행하지 않고 아래 검증만 통과하면 바로 머신러닝 단계로 넘어갈 수 있습니다.

검증 환경은 Python 3.13.13이며 패키지 버전은 `requirements.txt`에 고정했습니다.

```bash
python -m pip install -r requirements.txt
python -m src.check_preprocessing_handoff
```

학습 입력 파일:

- `data/processed/model_snapshot_week_1.csv` — 29,018행, 99열
- `data/processed/model_snapshot_week_2.csv` — 27,984행, 99열
- `data/processed/model_snapshot_week_4.csv` — 27,449행, 99열

### 머신러닝 진행 시 주의사항

- `target`은 예측 정답 컬럼입니다.
- `id_student`는 입력 Feature에서 제외합니다.
- `id_student`는 동일 학생이 Train·Validation·Test에 중복되지 않도록 데이터 분할에만 사용합니다.
- 동일 학생은 Train·Validation·Test 중 하나에만 포함해야 합니다.
- 전처리기와 인코더는 Train 데이터에만 학습해야 합니다.
- Validation 데이터로 모델과 임계값을 결정합니다.
- Test 데이터는 최종 모델과 임계값을 확정한 뒤 한 번만 평가합니다.
- 모델 성능은 Recall을 우선으로 확인하되 Precision, F1-score, PR-AUC도 함께 비교합니다.
- 1·2·4주차 모델 성능을 비교하여 성능이 안정적인 가장 빠른 시점을 골든타임 후보로 선정합니다.
- 전체 성능뿐만 아니라 과목별 Recall과 예측 성능도 함께 확인합니다.

데이터 분할, 인코딩, 모델 학습, 임계값 결정 및 Test 평가는 머신러닝 담당 범위입니다.

---

## 7. Streamlit 구현 방향

Streamlit은 저장된 모델과 전처리 파이프라인을 이용해 위험 학생과 유지 활동을 보여주는 조기경보 화면으로 구현합니다.

예정 기능:

- 전체 및 과목별 이탈 현황
- 주차별 이탈 추이
- 최종 모델 성능
- 학생별 이탈 위험도
- 위험도 상·중·하 분류
- 주요 위험 요인
- 과목별 유지 활동 추천
- 개입 대상 학생 목록

Streamlit은 노트북을 직접 실행하지 않고 저장된 모델과 `src/predict.py`를 사용합니다.

실행 명령:

```bash
streamlit run streamlit_app/app.py
```

---

## 8. 전체 실행 순서

아래 명령은 원본 OULAD 데이터부터 전처리와 EDA 결과를 모두 재생성할 때 사용합니다.

머신러닝 담당자는 최종 Snapshot을 pull한 뒤 인계 검증만 실행하면 됩니다.

```bash
python -m src.data --check --build-cohort
python -m src.vle_features
python -m src.build_model_snapshots
python -m src.export_eda_artifacts
python -m src.check_preprocessing_handoff
```

테스트 실행:

```bash
python -m unittest discover -s tests -v
```

---

## 9. 프로젝트 운영 원칙

- `data/raw`의 원본 데이터는 수정하지 않습니다.
- 개인 PC의 절대경로를 사용하지 않습니다.
- 데이터 경로는 프로젝트 루트를 기준으로 작성합니다.
- 원본 데이터와 중간 생성 데이터는 GitHub에 올리지 않습니다.
- 머신러닝 인계용 최종 Snapshot 3개만 Git에서 관리합니다.
- API Key, 비밀번호, 가상환경 파일은 GitHub에 올리지 않습니다.
- 데이터 분할 전에 Target 누수 가능성이 있는 컬럼을 제거합니다.
- 모델, 전처리기, Feature 순서, 임계값 및 메타데이터를 함께 저장합니다.
- Streamlit은 저장된 모델과 공통 추론 함수를 이용합니다.

---

## 10. 현재 진행 상태

### 완료

- 데이터 구조 확인
- 공통 코호트 생성
- Target 정의
- VLE 데이터 정제 및 주차별 집계
- 평가 데이터 정제 및 Feature 생성
- 주차별 이탈 현황 EDA
- 과목별 이탈 현황 EDA
- 이탈·비이탈 학생의 행동 비교
- 1·2·4주차 최종 Snapshot 생성
- 결측치, 중복, 무한값 및 Target 누수 점검
- 머신러닝 담당자 인계 준비

### 다음 단계

- Train·Validation·Test 학생 단위 분할
- Logistic Regression·Random Forest·Boosting 비교
- 1·2·4주차별 모델 성능 비교
- 공통 골든타임 선정
- 과목별 모델 성능 확인
- 최종 모델 및 임계값 저장
- 과목별 유지 활동 구체화
- Streamlit 조기경보 화면 연결
- README 및 발표 자료 최종 정리

---

현재 전처리·EDA 담당 범위는 완료되었습니다.

생성된 1·2·4주차 `model_snapshot`을 머신러닝 담당자에게 전달하며, 이후 데이터 분할, 모델 학습, Validation 및 Test 평가는 머신러닝 단계에서 진행합니다.

# 전처리 보고서

## 1. 사용 데이터

- 학생·강좌: `studentInfo`, `studentRegistration`, `courses`
- 온라인 학습행동: `studentVle`, `vle`
- 평가 일정·결과: `assessments`, `studentAssessment`
- 원본 파일은 `data/raw`에서 수정하지 않는다.

## 2. 분석 단위와 Target

- 분석 단위: `id_student + code_module + code_presentation`
- Target: 최종 결과가 `Withdrawn`이면 1, 나머지는 0
- 각 예측 주차가 끝난 시점에 이미 이탈한 학생은 그 시점의 예측 대상에서 제외한다.
- 개강 전 이탈과 이탈일이 없는 Withdrawn은 이탈 시점 분석에서 별도로 제외한다.

## 3. 후보 시점

- 1주차: 개강 후 0~6일
- 2주차: 개강 후 0~13일
- 4주차: 개강 후 0~27일
- 전체 이탈률은 2주차가 3.68%로 가장 높아 초기 후보에 포함한다.
- 머신러닝 성능과 개입 가능 시점을 함께 고려해 최종 골든타임을 정한다.

## 4. VLE 병합과 집계

- `studentVle`와 `vle`를 과목·회차·콘텐츠 ID로 연결한다.
- 학생·강좌·7일 단위로 클릭, 활동일, 콘텐츠 수, 활동 유형별 클릭을 집계한다.
- 클릭 기록이 없는 학생·주차는 공통 학생 명단과 Left Join한 뒤 0으로 채운다.
- 개강 전 클릭은 별도 Feature로 보존한다.
- 누적량, 현재 주차량, 직전 주 대비 변화, 미활동 기간과 활동 유형 비중을 생성한다.

## 5. 평가 Feature

- 평가 일정은 `assessments`, 학생 제출은 `studentAssessment`에서 가져온다.
- 기준일 이후 제출·점수는 사용하지 않는다.
- 마감 평가 수, 제출률, 미제출률, 지각률, 확인된 점수와 평가 유형별 제출 수를 생성한다.
- 상세 기준은 `reports/assessment_preprocessing_report.md`에 기록한다.

## 6. 최종 Snapshot 품질

| 주차 | 행 | 컬럼 | 키 중복 | 전체 결측 |
|---:|---:|---:|---:|---:|
| 1 | 29,018 | 99 | 0 | 0 |
| 2 | 27,984 | 99 | 0 | 0 |
| 4 | 27,449 | 99 | 0 | 0 |

## 7. 머신러닝 전달 전 누수 점검

- 최종 Snapshot에는 머신러닝 담당자가 사용할 수 있도록 공통 키와 `target`을 보존한다.
- `final_result`, `date_unregistration`, 이탈 여부 파생 컬럼은 Feature에서 제거했다.
- 평가 제출·점수는 각 후보 주차 종료일까지 확인 가능한 정보만 사용했다.
- 학생 분할, 모델 학습, Validation, Test 평가는 아직 수행하지 않았다.
- 동일 학생의 여러 과목이 분할 사이에 섞이지 않도록 `id_student` 기준 Group 분할이 필요하다.
- 학습·검증 전체 분포를 미리 보게 되는 중앙값 대비·백분위 8개 컬럼은 최종 Snapshot에서 제거했다.
- 전처리 단계에서 `split`이나 Validation 결과는 만들지 않았다.

## 8. 최종 저장 파일

- `data/processed/model_snapshot_week_1.csv`
- `data/processed/model_snapshot_week_2.csv`
- `data/processed/model_snapshot_week_4.csv`

생성 순서:

```bash
python -m src.vle_features
python -m src.build_model_snapshots
python -m src.export_eda_artifacts
python -m src.check_preprocessing_handoff
```

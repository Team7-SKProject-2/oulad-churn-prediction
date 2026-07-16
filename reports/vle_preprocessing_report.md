# VLE 전처리 보고서

## 1. 작업 목적

OULAD의 VLE 학습활동 로그를 모델링에 사용할 수 있도록 학생·강좌·운영 회차·주차 단위의 행동 Feature로 변환한다.

이 단계에서는 예측 주차를 미리 고정하지 않고 전체 수강 기간을 7일 단위로 집계한다. 최종 관찰 기간과 예측 시점은 `date_unregistration`을 이용한 전체 이탈 시점 EDA 이후 결정한다.

---

## 2. 사용 데이터

### studentVle.csv

학생이 날짜별로 VLE 콘텐츠를 이용한 기록이다.

- 원본 행 수: 10,655,280
- 결측치: 없음
- 활동 날짜 범위: 개강일 기준 -25일~269일
- 클릭 수 범위: 1~6,977
- 0 이하 클릭 수: 없음

주요 컬럼:

- `code_module`: 강좌 코드
- `code_presentation`: 강좌 운영 회차
- `id_student`: 학생 ID
- `id_site`: VLE 콘텐츠 ID
- `date`: 개강일 기준 상대 일수
- `sum_click`: 해당 콘텐츠 클릭 수

### vle.csv

VLE 콘텐츠의 종류와 제공 주차에 대한 메타데이터다.

- 행 수: 6,364
- 활동 유형: 20종
- 기본 키: `code_module + code_presentation + id_site`

원본 파일의 첫 번째 컬럼명이 `d"id_site"`로 저장되어 있어 `id_site`로 변경했다.

`week_from`, `week_to`의 `?`는 결측치로 변환했다.

- `week_from` 결측치: 5,243개
- `week_to` 결측치: 5,243개

두 컬럼은 결측률이 높고 학생의 실제 활동 주차를 의미하지 않으므로, 현재 주차별 행동 Feature에는 사용하지 않았다.

---

## 3. 전처리 기준

### 3.1 원본 보존

`data/raw`의 원본 CSV는 수정하지 않는다. 전처리 결과는 `data/interim`에 별도로 저장한다.

### 3.2 개강 전 활동

`date < 0`인 기록은 오류가 아니라 개강 전 활동이다.

- 개강 전 원본 활동: 688,988행
- 개강 전 학생·강좌 집계: 23,809행

개강 후 주차별 활동과 섞지 않고 별도 Feature 파일로 저장했다.

### 3.3 주차 계산

개강 후 활동은 다음 기준으로 7일 단위 주차를 생성했다.

```python
week_index = (date // 7) + 1
```

주차 기준:

- 1주차: 0~6일
- 2주차: 7~13일
- 3주차: 14~20일
- 4주차: 21~27일
- 이후 동일한 방식으로 계산

전체 데이터에서는 1~39주차가 확인되었다.
모델 후보 Snapshot은 누적 1·2·4주차를 사용한다.

### 3.4 집계 단위

주차별 집계 키는 다음과 같다.

```text
code_module
+ code_presentation
+ id_student
+ week_index
```

따라서 최종 데이터의 한 행은 한 학생이 특정 강좌·운영 회차의 특정 주차에 보인 학습 행동을 나타낸다.

### 3.5 활동 유형 연결

`studentVle`과 `vle`는 다음 키로 병합했다.

```text
code_module
+ code_presentation
+ id_site
```

전체 로그에서 활동 유형 연결에 실패한 행은 0개였다.

활동 유형 20종 중 핵심 유형은 다음과 같이 별도 Feature로 생성했다.

- `forumng`
- `quiz`
- `oucontent`
- `resource`

나머지 16종은 `other`로 통합했다.

---

## 4. 생성 Feature

주차별 Feature는 총 12개다.

| Feature | 설명 |
|---|---|
| `total_clicks` | 해당 주차 총 클릭 수 |
| `interaction_rows` | 해당 주차 원본 활동 기록 행 수 |
| `active_days` | 해당 주차 실제 활동일 수 |
| `unique_sites` | 해당 주차 이용한 고유 콘텐츠 수 |
| `avg_clicks_per_active_day` | 활동일당 평균 클릭 수 |
| `avg_clicks_per_site` | 콘텐츠당 평균 클릭 수 |
| `activity_type_count` | 이용한 활동 유형 수 |
| `forumng_clicks` | 포럼 활동 클릭 수 |
| `quiz_clicks` | 퀴즈 활동 클릭 수 |
| `oucontent_clicks` | 강의 콘텐츠 클릭 수 |
| `resource_clicks` | 학습자료 클릭 수 |
| `other_clicks` | 기타 활동 클릭 수 |

별도 개강 전 Feature:

| Feature | 설명 |
|---|---|
| `pre_course_clicks` | 개강 전 총 클릭 수 |
| `pre_course_interaction_rows` | 개강 전 활동 기록 행 수 |

---

## 5. 검증 결과

### 원본 반영 검증

- 원본 행 수: 10,655,280
- 집계에 반영된 행 수: 10,655,280
- 누락된 원본 행: 0

### 주차별 결과 검증

- 주차별 집계 행 수: 579,438
- 집계 키 중복: 0
- 최종 Feature 결측치: 0
- `active_days` 범위: 1~7
- `unique_sites` 범위: 1~268
- 활동 유형 연결 실패: 0
- 활동 유형별 클릭 합계와 전체 클릭 합계 일치

---

## 6. 저장 결과

다음 파일을 `data/interim`에 저장했다.

| 파일 | 내용 | 크기 |
|---|---|---:|
| `vle_weekly_features.csv` | 학생·강좌·주차별 행동 Feature | 약 33.5MB |
| `vle_pre_course_features.csv` | 개강 전 활동 Feature | 약 0.5MB |
| `vle_metadata_clean.csv` | 정리된 VLE 메타데이터 | 약 0.2MB |

중간 결과 CSV는 GitHub에 업로드하지 않는다.

---

## 7. 실행 방법

프로젝트 최상위 폴더에서 다음 명령어를 실행한다.

```bash
python -m src.vle_features
```

실행 코드:

```text
src/vle_features.py
```

원본 데이터 경로:

```text
data/raw/studentVle.csv
data/raw/vle.csv
```

---

## 8. 후속 작업

현재 결과에는 Target이 연결되어 있지 않다.

다음 단계에서 진행할 작업:

1. `studentRegistration.date_unregistration`으로 전체 이탈 시점 EDA
2. 이탈이 집중되는 위험 구간 확인
3. 위험 구간 이전의 관찰 기간과 예측 시점 결정
4. 공통 학생·강좌 명단 및 Target과 병합
5. 활동 기록이 없는 학생·주차는 Left Join 후 0으로 처리
6. 이탈·비이탈 학생의 VLE 행동 차이 분석
7. 주차별 누적 Feature와 이전 주차 대비 변화량 생성

주의할 점은 현재 `vle_weekly_features.csv`에는 실제 활동 기록이 있는 학생·주차만 존재한다는 것이다.

활동하지 않은 학생을 삭제된 것으로 해석하면 안 되며, 최종 학습 대상 명단을 기준으로 병합한 뒤 해당 주차의 활동 Feature를 0으로 채워야 한다.

# 데이터 사전

## 공통 키와 Target

| 컬럼 | 자료형 | 단위 | 출처 | 설명 |
|---|---|---|---|---|
| `code_module` | string | 과목 | OULAD | 과목 코드 |
| `code_presentation` | string | 강좌 회차 | OULAD | 과목 개설 회차 |
| `id_student` | integer | 학생 | OULAD | 익명 학생 ID |
| `cutoff_week` | integer | 주 | 파생 | 누적 Feature 기준 주차 |
| `target` | integer | 0/1 | `studentInfo` | Withdrawn이면 1, 나머지는 0 |

## Feature 기록 양식

| Feature | 자료형 | 단위 | 원본 파일 | 생성 방법 | 사용 가능 시점 | 누수 위험 |
|---|---|---|---|---|---|---|
| `cum_total_clicks` | integer | 클릭 | `studentVle` | 기준 주차까지 `sum_click` 합계 | 주차별 | 낮음 |
| `cum_active_days` | integer | 일 | `studentVle` | 기준 주차까지 고유 활동일 수 합계 | 주차별 | 낮음 |
| `weeks_since_last_activity` | integer | 주 | `studentVle` | 기준 주차 - 마지막 활동 주차 | 주차별 | 낮음 |
| `current_no_activity` | integer | 0/1 | `studentVle` | 현재 주차 활동이 없으면 1 | 주차별 | 낮음 |
| `click_change_rate` | float | 비율 | `studentVle` | 직전 주 대비 클릭 변화율 | 주차별 | 낮음 |
| `assessment_missing_due_rate` | float | 비율 | 평가 데이터 | 마감 평가 중 미제출 비율 | 주차별 | 낮음 |
| `assessment_late_rate` | float | 비율 | 평가 데이터 | 확인 가능한 비이월 제출 중 지각 비율 | 주차별 | 낮음 |
| `any_known_mean_score` | float | 점수 | 평가 데이터 | 기준일까지 확인된 제출 점수 평균 | 주차별 | 낮음 |
Feature를 추가할 때 컬럼명, 단위, 집계 구간과 결측치 처리 기준을 반드시 기록한다.

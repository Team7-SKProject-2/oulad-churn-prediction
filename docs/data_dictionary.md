# 데이터 사전

## 공통 키와 Target

| 컬럼 | 자료형 | 단위 | 출처 | 설명 |
|---|---|---|---|---|
| `code_module` | string | 과목 | OULAD | 과목 코드 |
| `code_presentation` | string | 강좌 회차 | OULAD | 과목 개설 회차 |
| `id_student` | integer | 학생 | OULAD | 익명 학생 ID |
| `cutoff_week` | integer | 주 | 파생 | 누적 Feature 기준 주차 |
| `is_withdrawn` | integer | 0/1 | `studentInfo` | Withdrawn이면 1 |

## Feature 기록 양식

| Feature | 자료형 | 단위 | 원본 파일 | 생성 방법 | 사용 가능 시점 | 누수 위험 |
|---|---|---|---|---|---|---|
| `cumulative_clicks` | integer | 클릭 | `studentVle` | 기준일까지 `sum_click` 합계 | 주차별 | 낮음 |
| `cumulative_active_days` | integer | 일 | `studentVle` | 기준일까지 고유 활동일 수 | 주차별 | 낮음 |
| `recent_activity_gap` | integer | 일 | `studentVle` | 기준일 - 마지막 활동일 | 주차별 | 낮음 |
| | | | | | | |

Feature를 추가할 때 컬럼명, 단위, 집계 구간과 결측치 처리 기준을 반드시 기록한다.


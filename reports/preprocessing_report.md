# 7팀 인공지능 데이터 전처리 결과서

## 1. 전처리 결과 요약

- 프로젝트: OULAD 기반 **1~10주차 다음 주 중도이탈 조기경보**
- 분석 단위: `id_student + code_module + code_presentation + prediction_week`
- Target: `target_next_week_withdrawn`
  - 1: 현재 수강 중이며 다음 주에 Withdrawn
  - 0: 다음 주에는 이탈하지 않음
- 최종 주간 모델링 데이터: 895,005행 × 126열
- 모델 입력 Feature: 124개 (`id_student`, Target 제외)
- 복합키 중복: 0건
- 서비스 구간: 개강 후 1~10주차

이 문서는 단순 EDA가 아니라 원천 데이터가 최종 학습 데이터로 변환되는 과정, 처리 근거, 누수 통제와 재현 절차를 설명한다.

## 2. 데이터 소개

### 출처·라이선스·실제 여부

- 출처: Open University Learning Analytics Dataset(OULAD)
- 실제·합성 여부: Open University 2013·2014 익명화 실제 학습 데이터
- 라이선스: CC BY 4.0
- 원 논문: Kuzilek, J., Hlosta, M. & Zdrahal, Z. (2017), *Scientific Data* 4:170171
- DOI: <https://doi.org/10.1038/sdata.2017.171>

### 원천 데이터 크기

| 테이블 | 행 | 역할 |
|---|---:|---|
| `courses` | 22 | 강좌 운영 정보 |
| `assessments` | 206 | 평가 일정·유형·가중치 |
| `studentInfo` | 32,593 | 학생정보·최종 결과 |
| `studentRegistration` | 32,593 | 등록일·이탈일 |
| `studentAssessment` | 173,912 | 학생별 평가 제출·점수 |
| `vle` | 6,364 | 학습 콘텐츠 메타데이터 |
| `studentVle` | 10,655,280 | 일별 LMS 상호작용 |

원본 파일은 `data/raw`에서 수정하지 않는다.

## 3. 분석 기준과 Target

최종 결과가 `Withdrawn`인 수강 사례를 이탈자로 식별하고 `date_unregistration`으로 실제 이탈 주차를 계산한다.

```python
withdraw_week = (date_unregistration // 7) + 1
target_next_week_withdrawn = int(withdraw_week == prediction_week + 1)
```

- 현재 예측 주차 이전에 이탈한 수강 사례는 대상에서 제외한다.
- 개강 전 이탈, 이탈일 결측 Withdrawn, 강좌 종료 후 이탈은 주차 Target에서 제외한다.
- 음성은 최종 비이탈만 뜻하지 않는다. 나중에 이탈하더라도 **다음 주에 이탈하지 않으면 현재 행에서는 0**이다.
- 모델 Feature에는 `final_result`, `date_unregistration`, `withdraw_week`을 넣지 않는다.

## 4. 품질 점검과 데이터 정제

| 대상 | 점검 결과 | 처리 |
|---|---|---|
| `studentVle` | 결측 0, date -25~269, click 1~6,977 | `date < 0`은 개강 전 활동으로 별도 보존 |
| `vle` | `week_from`, `week_to`의 `?` 각각 5,243 | 실제 활동 주차가 아니므로 행동 Feature에서 제외 |
| `vle.id_site` | 배포본 컬럼명 `d"id_site"` | `id_site`로 정규화 |
| `studentInfo.imd_band` | `?` 1,111 | `Unknown` 범주와 결측 지표 생성 |
| 등록 데이터 | 등록일 `?` 45, 이탈일 `?` 22,521 | 숫자 변환 후 NaN, 결측 지표 생성 |
| `assessments.date` | `?` 11 | 마감일 미확정 Exam으로 주차 마감 계산에서 제외 |
| `studentAssessment.score` | `?` 173 | 삭제하지 않고 점수 결측 개수 Feature로 보존 |

공통 키 병합은 `one_to_one` 또는 `many_to_one`으로 검증한다. VLE 활동 유형 연결 실패 0건, 주차 집계와 최종 주간 테이블 복합키 중복 0건이다.

## 5. VLE 전처리와 Feature Engineering

`studentVle`과 `vle`를 `code_module + code_presentation + id_site`로 연결하고 7일 단위로 집계한다.

```python
week_index = (date // 7) + 1
```

- 원본 활동: 10,655,280행
- 개강 전 활동: 688,988행
- 주차별 집계: 579,438행
- 활동 유형 연결 실패: 0행
- 원본 클릭 합계와 집계 합계: 일치

주요 Feature:

- 활동량: `total_clicks`, `interaction_rows`
- 활동 지속성: `active_days`, `active_weeks`, `inactive_weeks`
- 콘텐츠 폭: `unique_sites`, `activity_type_count`
- 유형별 활동: `forumng`, `quiz`, `oucontent`, `resource`, `other`
- 현재·직전: `current_*`, `previous_*`
- 변화: `click_change`, `click_change_rate`, `active_days_change`
- 무활동: `current_no_activity`, `weeks_since_last_activity`
- 개강 전: `pre_course_clicks`, `pre_course_interaction_rows`
- 로그 변환: `log1p_cum_total_clicks`, `log1p_current_total_clicks`, `log1p_pre_course_clicks`

로그가 없는 학생·주차는 삭제하지 않는다. 공통 학생-주차 Grid에 Left Join한 후 활동값을 0으로 채운다.

## 6. 평가·학생·등록 Feature

- `assessments`와 `studentAssessment` 병합 전후 173,912행 일치
- 평가정보 연결 실패 0건
- 이월 평가 1,909건, 지각 제출 49,318건
- 기준일 이후 제출·점수는 제외
- 마감 평가 수·가중치, 제출·미제출, 지각·이월, 점수 통계, TMA/CMA/Exam별 제출 Feature 생성
- 학생 배경과 등록 선행·지각 일수, 결측 지표 생성

평가가 없거나 제출이 확인되지 않은 경우 점수·제출 간격은 0점이 아니므로 NaN을 유지한다. 가용성은 `due_count`, `known_submission_count`, 결측 지표로 함께 표현한다.

## 7. 변환·인코딩·스케일링·불균형 처리

| 항목 | 적용 | 근거 |
|---|---|---|
| 범주형 인코딩 | 전처리 단계 미적용 | CatBoost가 직접 처리. 전역 인코더 누수 방지 |
| 수치 스케일링 | 최종 CatBoost에는 미적용 | 트리 모델에 필수 아님. 비교 모델은 Fold 내부 처리 |
| 로그 변환 | 적용 | 클릭량의 긴 오른쪽 꼬리 완화 |
| 이상치 삭제·클리핑 | 미적용 | 실제 활동 로그를 보존하고 로그 Feature 병행 |
| SMOTE·언더샘플링 | 최종 전처리에 미적용 | 실제 양성률·확률 의미 보존. 별도 보조 실험으로만 비교 |
| Class Weight | 학습 단계 | 분할 후 모델·Fold 내부에서만 결정 |

전체 주간 양성률은 0.7455%, 1~10주차는 1.2206%다. 따라서 Accuracy 대신 PR-AUC, Recall, Precision, F1을 중심으로 평가한다.

## 8. EDA 근거

### 주차별 이탈

| 주차 | 수강 중 | 이탈 | 조건부 이탈률 | 누적 이탈 비율 |
|---:|---:|---:|---:|---:|
| 1 | 29,592 | 713 | 2.41% | 9.65% |
| 2 | 29,019 | 1,068 | 3.68% | 24.11% |
| 4 | 27,803 | 360 | 1.29% | 31.90% |
| 10 | 26,209 | 220 | 0.84% | 51.81% |

![전체 주차별 이탈률](figures/preprocessing/weekly_dropout_rate.png)

### 과목별 차이

- CCC 전체 이탈률 44.54%로 가장 높다.
- BBB·CCC·DDD·EEE·FFF의 최빈 이탈 주차는 2주차다.
- AAA 중앙 이탈 20주, GGG 15주로 중후반 모니터링도 필요하다.

![과목별 주차 이탈률](figures/preprocessing/module_week_dropout_heatmap.png)

### Target별 행동 차이

- 1주차 클릭 중앙값: 향후 이탈 18, 비이탈 37
- 1주차 무활동률: 향후 이탈 27.24%, 비이탈 17.61%
- 4주차 평가 미제출률: 향후 이탈 21.82%, 비이탈 8.66%
- 4주차 지각 제출률: 향후 이탈 35.68%, 비이탈 21.00%
- Spearman 상관: `total_clicks`-`interaction_rows` 약 0.94, `interaction_rows`-`unique_sites` 약 0.92

집단 차이는 연관성을 보여줄 뿐 이탈 원인이나 개입 효과를 증명하지 않는다.

## 9. 누수 방지와 데이터 분리

- 예측 주차 종료 후 VLE 활동 제외
- `date_submitted <= cutoff_day`인 평가만 사용
- 최종 결과·실제 이탈일 파생 Feature 제거
- 전체 분포를 미리 보는 중앙값 대비·백분위 Feature 8개 제거
- 모델 단계에서 `id_student` 기준 3-Fold GroupKFold OOF
- 동일 학생의 모든 과목·회차·주차는 하나의 Fold에만 배정
- 인코더·스케일러·샘플링이 필요한 비교 모델은 학습 Fold 안에서만 학습

전처리 단계에서는 임의 Train/Test 라벨을 고정하지 않았다. 독립 외부 Test가 없으므로 OOF 성능을 외부 Test 성능으로 표현하지 않는다.

## 10. 전처리 전후 결과

### 초기 Snapshot

| 파일 | 행 | 열 | 키 중복 | 결측 |
|---|---:|---:|---:|---:|
| `model_snapshot_week_1.csv` | 29,018 | 99 | 0 | 0 |
| `model_snapshot_week_2.csv` | 27,984 | 99 | 0 | 0 |
| `model_snapshot_week_4.csv` | 27,449 | 99 | 0 | 0 |

### 최종 주간 모델링 데이터

| 항목 | 전체 1~38주 | 운영 1~10주 |
|---|---:|---:|
| 행 | 895,005 | 271,663 |
| 학생 | 26,045 | 26,044 |
| 양성 | 6,672 | 3,316 |
| 양성률 | 0.7455% | 1.2206% |
| 열 / 입력 Feature | 126 / 124 | 동일 계약 |

최종 주간 테이블에는 16개 열, 2,232,074개의 구조적 결측 셀이 있다. 평가 미발생·제출 미확인과 같은 의미를 보존해 NaN으로 유지하며 CatBoost가 native missing으로 처리한다. `vle_cum_unique_sites`의 대체 누적 폭 Feature인 `cum_unique_site_week_count`는 결측이 없다.

## 11. 재현 방법

```bash
python -m pip install -r requirements.txt
python -m src.data --check --build-cohort
python -m src.vle_features
python -m src.build_model_snapshots
python -m src.export_eda_artifacts
python -m src.check_preprocessing_handoff
```

주요 분석 파일:

- `notebooks/01_*_data_check.ipynb`
- `notebooks/02_vle_eda.ipynb`
- `notebooks/03_dropout_timing_eda.ipynb`
- `notebooks/04_target_vle_eda.ipynb`
- `notebooks/05_assessment_features.ipynb`
- `notebooks/06_demo1_weekly_eda.ipynb`
- `src/data.py`
- `src/vle_features.py`
- `src/build_model_snapshots.py`
- `src/export_eda_artifacts.py`
- `models/common_weekly_metrics.py`

최종 데이터:

- `models/demo_1/used_data/weekly_next_week_with_vle_enhanced.csv`

## 12. 한계

- 클릭은 학습시간·이해도의 직접 측정값이 아니다.
- 과목별 학습 설계가 달라 동일한 클릭 기준의 의미가 다르다.
- 다음 주 양성이 매우 희소하므로 Accuracy는 부적절하다.
- 구조적 결측과 0을 구분해야 한다.
- 관찰 데이터의 연관성은 개입 효과의 인과성을 보장하지 않는다.
- 독립 외부 Test가 없어 OOF 결과와 최종 Test를 구분해야 한다.

## 부록 A. 최종 124개 Feature

1. `code_module`
2. `code_presentation`
3. `prediction_week`
4. `cutoff_day`
5. `module_presentation_length`
6. `gender`
7. `region`
8. `highest_education`
9. `imd_band`
10. `age_band`
11. `num_of_prev_attempts`
12. `studied_credits`
13. `disability`
14. `date_registration`
15. `registration_day`
16. `registered_after_start`
17. `registration_lead_days`
18. `late_registration_days`
19. `assessment_due_count`
20. `assessment_due_weight`
21. `assessment_submitted_due_count`
22. `assessment_late_count`
23. `assessment_missing_due_count`
24. `assessment_submission_rate`
25. `assessment_mean_score`
26. `vle_cum_total_clicks`
27. `vle_cum_interaction_rows`
28. `vle_last_active_day`
29. `vle_cum_active_days`
30. `vle_cum_unique_sites`
31. `vle_week_clicks_dataplus`
32. `vle_week_clicks_dualpane`
33. `vle_week_clicks_externalquiz`
34. `vle_week_clicks_folder`
35. `vle_week_clicks_forumng`
36. `vle_week_clicks_glossary`
37. `vle_week_clicks_homepage`
38. `vle_week_clicks_htmlactivity`
39. `vle_week_clicks_oucollaborate`
40. `vle_week_clicks_oucontent`
41. `vle_week_clicks_ouelluminate`
42. `vle_week_clicks_ouwiki`
43. `vle_week_clicks_page`
44. `vle_week_clicks_questionnaire`
45. `vle_week_clicks_quiz`
46. `vle_week_clicks_repeatactivity`
47. `vle_week_clicks_resource`
48. `vle_week_clicks_sharedsubpage`
49. `vle_week_clicks_subpage`
50. `vle_week_clicks_url`
51. `vle_has_record`
52. `imd_band_missing`
53. `date_registration_missing`
54. `current_total_clicks`
55. `current_interaction_rows`
56. `current_last_active_day`
57. `current_active_days`
58. `current_unique_sites`
59. `current_activity_type_count`
60. `current_forumng_clicks`
61. `current_oucontent_clicks`
62. `current_quiz_clicks`
63. `current_resource_clicks`
64. `pre_course_clicks`
65. `pre_course_interaction_rows`
66. `current_has_vle_record`
67. `observed_weeks`
68. `active_weeks`
69. `inactive_weeks`
70. `active_week_rate`
71. `cum_unique_site_week_count`
72. `cum_activity_type_week_count`
73. `cum_avg_clicks_per_active_day`
74. `cum_avg_clicks_per_site_week`
75. `previous_total_clicks`
76. `previous_active_days`
77. `previous_unique_sites`
78. `last_active_week`
79. `weeks_since_last_activity`
80. `current_no_activity`
81. `click_change`
82. `click_change_rate`
83. `active_days_change`
84. `unique_sites_change`
85. `cum_forumng_clicks`
86. `cum_forumng_share`
87. `cum_quiz_clicks`
88. `cum_quiz_share`
89. `cum_oucontent_clicks`
90. `cum_oucontent_share`
91. `cum_resource_clicks`
92. `cum_resource_share`
93. `current_other_clicks`
94. `cum_other_clicks`
95. `cum_other_share`
96. `log1p_cum_total_clicks`
97. `log1p_current_total_clicks`
98. `log1p_pre_course_clicks`
99. `assessment_due_cma_count`
100. `assessment_due_exam_count`
101. `assessment_due_tma_count`
102. `assessment_scored_due_count`
103. `assessment_missing_score_count`
104. `assessment_banked_due_count`
105. `assessment_nonbanked_submitted_count`
106. `weighted_score_sum`
107. `scored_weight_sum`
108. `assessment_submitted_cma_count`
109. `assessment_submitted_tma_count`
110. `assessment_submitted_exam_count`
111. `assessment_median_score`
112. `assessment_min_score`
113. `assessment_max_score`
114. `assessment_median_submission_gap`
115. `assessment_mean_submission_gap`
116. `assessment_missing_due_rate`
117. `assessment_late_rate`
118. `assessment_weighted_mean_score`
119. `any_known_submission_count`
120. `any_known_scored_count`
121. `any_known_score_missing_count`
122. `any_known_banked_count`
123. `any_known_mean_score`
124. `any_known_median_score`

# 학생 이탈 예측 시스템 (Streamlit)

## 실행
```
pip install -r requirements.txt
streamlit run app.py
```

## 데이터 배치
`uploads/oulad_data_spec.md`에 명시된 정제 데이터 8종을 아래 위치에 넣어야 합니다.
```
data/interim/
  student_info_processed.csv
  student_registration_processed.csv
  vle_weekly_features.csv
  vle_pre_course_features.csv
  courses_processed.csv
  assessments_processed.csv
  student_assessment_processed.csv
```
파일이 없으면 대시보드/1/2번 페이지는 안내 메시지만 표시합니다(3번 예측 페이지는 입력 폼만으로 동작).

## 구조
- `app.py` — 메인 대시보드(전체 이탈률, 결과 분포, 과목별 비교, 주차별 참여도 추이)
- `pages/1_과목_주차별_행동제안.py` — 과목/주차 선택 → 위험군 리스트 + 공통 행동제안
- `pages/2_학생별_행동추천.py` — 학생 선택(독립 폼) → 과목별 위험도 + 맞춤 행동추천
- `pages/3_이탈_예측.py` — 학생/과목/주차 정보 입력(독립 폼) → 예측 결과 팝업(`st.popover`)
- `lib/data.py` — CSV 로딩, 주차별 위험 스냅샷(`build_master_table`) 계산 (모두 `st.cache_data`)
- `lib/risk.py` — 위험 점수 산정(`compute_risk_score`, 규칙 기반) · 위험요인/행동추천 카탈로그
- `lib/theme.py` — 블루 톤 컬러 팔레트 · 공통 CSS

## 모델 연결 지점
`lib/risk.py`의 `compute_risk_score()` / `score_row()`가 유일한 위험도 산정 함수입니다.
feature가 확정되고 모델이 학습되면 이 함수 내부만 `model.predict(...)` 호출로 교체하면
대시보드/3개 메뉴 페이지 전부 자동으로 실제 모델 결과를 사용하게 됩니다.

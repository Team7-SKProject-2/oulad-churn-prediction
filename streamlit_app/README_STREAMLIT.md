# OULAD 조기이탈 경보 Streamlit

개강 후 1~10주차 학습 정보를 이용해 다음 주 이탈 위험을 예측하고 행동 제안을
제공하는 멀티페이지 앱입니다.

## 실행

프로젝트 루트에서 실행합니다.

```bash
python -m pip install -r streamlit_app/requirements.txt
python -m streamlit run streamlit_app/app.py
```

## 필수 데이터

`data/interim/`에 다음 파일이 필요합니다.

- `student_info_processed.csv`
- `student_registration_processed.csv`
- `vle_weekly_features.csv`
- `vle_pre_course_features.csv`
- `courses_processed.csv`
- `assessments_processed.csv`
- `student_assessment_processed.csv`
- `vle_snapshot_week_*.csv`

## 최종 모델 artifact

`models/artifacts/`에 다음 파일이 필요합니다.

- `catboost.joblib`: 최종 CatBoost와 124개 Feature 순서
- `catboost_cohort_profiles.csv`: 사용자 미입력 Feature를 채우는 코호트 프로필
- `early_service_config.json`: 1~10주차 운영 범위와 판정 임계값

## 페이지

| 페이지 | 역할 |
|---|---|
| 대시보드 | 전체 학습·이탈 현황 확인 |
| 과목별 행동제안 | 과목·주차별 위험 신호와 개입안 확인 |
| 학생별 행동추천 | 학생 단위 학습행동과 추천 확인 |
| 이탈 예측 | 최종 CatBoost로 다음 주 이탈 확률 계산 |

대시보드와 행동제안은 정제 데이터를 사용하고, 이탈 예측은 저장된 CatBoost와
코호트 프로필을 사용합니다. 최종 모델 파일이 없으면 예측 페이지는 실행되지
않으며 임시 점수로 대체하지 않습니다.

# 모델·Streamlit 인수인계

## 최종 기준

- 주제: 개강 후 1~10주차의 매주 다음 주 이탈 조기예측
- 최종 모델: CatBoost Enhanced 124 Feature
- 운영 임계값: `0.1100300614`
- 확률 보정: 미적용
- GRU·TCN·108 Feature CatBoost: 추가 비교 실험

## 바로 사용할 파일

- 모델: `artifacts/catboost.joblib`
- 124개 Feature 코호트 프로필: `artifacts/catboost_cohort_profiles.csv`
- Early 설정: `artifacts/early_service_config.json`
- 공통 추론: `streamlit_app/lib/model.py`
- Early 평가: `outputs/threshold_analysis/early_catboot/`

대용량 주차별 학습 테이블과 OOF 행 단위 예측은 Git에서 제외할 수 있으므로,
모델을 재학습하거나 Early 결과를 재생성할 때는 별도로 전달된 데이터를
지정해야 한다.

## 주의사항

- 모델은 전체 주차 학습 행으로 학습했고, 서비스만 1~10주차로 제한한다.
- `target_next_week_withdrawn`, `final_result`, `date_unregistration`, 이탈 파생 컬럼을 입력 Feature로 사용하지 않는다.
- `id_student`는 Feature가 아니며 학생 Group 분할에만 사용한다.
- 추론 시 124개 Feature의 이름·순서·자료형을 저장 모델과 동일하게 맞춘다.
- Early 임계값은 `early_service_config.json`을 단일 운영 기준으로 사용한다.
- Early 성능은 OOF 부분집합 결과이며 독립 외부 Test 결과로 표현하지 않는다.

## 확인 명령

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
streamlit run streamlit_app/app.py
```

`model_snapshot_week_1.csv`, `week_2.csv`, `week_4.csv`는 초기 시점별 전처리·EDA
검증 산출물이며, 현재 매주 다음 주 예측 메인 테이블을 대체하지 않는다.

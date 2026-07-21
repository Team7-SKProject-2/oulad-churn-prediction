# 1~10주차 다음 주 이탈 예측 최종 모델 보고서

## 1. 최종 서비스 방향

개강 후 1~10주 동안 매주 현재까지의 학습 정보로 다음 주 중도이탈
위험자를 예측한다. 위험 학생에게는 주요 행동 신호와 과목 특성에 맞는
유지 활동을 제안한다.

## 2. Early 평가 기준

- 평가 행: 271,663건
- 다음 주 이탈: 3,316건(1.2206%)
- 평가 구간: `prediction_week` 1~10
- 분할: `id_student` 기준 3-Fold OOF
- 임계값: 모델별 Early OOF F1 최대 지점

CatBoost는 전체 주차 학습 행으로 학습·OOF 검증했다. 따라서 본 결과는
1~10주차 전용 재학습 모델이 아니라, 전체 OOF 예측의 Early 운영 구간
부분집합 평가이다.

## 3. Early 모델 비교

| 모델 | 임계값 | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **CatBoost** | 0.110030 | **27.86%** | 20.14% | **23.38%** | **0.158890** | **0.843639** |
| XGBoost weighted | 0.878961 | 20.56% | 20.75% | 20.65% | 0.118739 | 0.837438 |
| Random Forest | 0.811896 | 27.61% | 15.41% | 19.78% | 0.141936 | 0.828475 |
| ElasticNet | 0.039806 | 7.20% | **28.44%** | 11.49% | 0.050780 | 0.804845 |

CatBoost는 Precision, F1, PR-AUC, ROC-AUC의 종합 균형이 가장 좋아 최종 서비스
모델로 선택했다. Recall만은 ElasticNet이 가장 높으므로, CatBoost가 모든
지표에서 최고라고 해석하지 않는다.

![Early CatBoost 평가 요약](../outputs/threshold_analysis/early_catboot/early_catboost_metrics_table.png)

## 4. 전체 주차·Feature·딥러닝 추가 실험

| 모델 | 입력 | 전체 주차 PR-AUC | Recall@Top20% | 역할 |
|---|---|---:|---:|---|
| CatBoost Enhanced | 124개 정형 Feature | **0.094775** | 71.13% | 최종 서비스 모델 |
| CatBoost Reduced | 108개 정형 Feature | 0.093502 | **71.24%** | Feature 축소 실험 |
| GRU | 최근 4주 × 행동 11개 | 0.027145 | 49.54% | 딥러닝 비교 |
| TCN형 1D-CNN | 최근 4주 × 행동 11개 | 0.027917 | 49.34% | 딥러닝 비교 |

108개 Feature 모델은 Recall@Top20%가 소폭 높지만 PR-AUC가 낮고, 기존
SHAP·Streamlit·저장 모델이 124개 기준으로 연결되어 있어 추가 실험으로
남겨두었다. GRU·TCN은 무작위 기준보다는 높았지만 CatBoost를 넘지
못했고, 앙상블도 성능을 개선하지 못해 실제 추론에 연결하지 않는다.

## 5. 최종 서비스 구성

1. 1~10주차 현재 시점까지의 124개 Feature로 다음 주 이탈확률을 계산한다.
2. 예측확률이 `0.1100300614` 이상이면 Early 위험 학생으로 분류한다.
3. Platt Scaling 등 별도 확률 보정은 적용하지 않는다.
4. 행동 신호와 과목 특성을 이용해 유지 활동을 제안한다.
5. GRU·TCN·108 Feature CatBoost는 비교·확장 실험으로 제시한다.

## 6. 해석 주의사항

- Early 구간의 양성 비율 1.2206%는 전체 주차 0.7455%보다 높다.
- Early 지표 상승은 모델 자체 변경뿐 아니라 운영 범위 한정의 영향도 받았다.
- 임계값은 Early OOF 부분집합에서 선택한 운영 기준이며, 독립 외부 Test 성능은 아니다.

## 7. 산출물

- 최종 모델: `models/artifacts/catboost.joblib`
- Early 운영 설정: `models/artifacts/early_service_config.json`
- 최종 모델 생성: `models/08_train_final_catboost_joblib.py`
- Early CatBoost 평가: `src/early_catboost_threshold_report.py`
- Early 모델별 결과: `outputs/threshold_analysis/early_*/`
- 딥러닝 비교: `reports/demo1_gru_comparison_report.md`
- Streamlit 추론: `streamlit_app/lib/model.py`

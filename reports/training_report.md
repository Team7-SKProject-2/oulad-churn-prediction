# 7팀 인공지능 학습 결과서

## 1. 요약

- 문제: 개강 후 1~10주 동안 현재까지의 학습행동으로 **다음 주 과목 이탈 여부**를 예측하는 이진 분류
- 분석 단위: 학생 × 과목 × 운영회차 × 예측주차
- Target: `target_next_week_withdrawn` (`1`=다음 주 이탈, `0`=다음 주까지 비이탈)
- 최종 서비스 모델: **CatBoost Enhanced 124 Feature**
- 운영 임계값: **0.110030061**
- Early 평가: 271,663행, 이탈 3,316건(1.2206%)
- 최종 Early Group OOF: Precision 27.86%, Recall 20.14%, F1 23.38%, PR-AUC 0.158890, ROC-AUC 0.843639

본 결과서는 모델 성능 점수만 나열하지 않고 데이터 분할, 후보 비교, 임계값 선택, 오류 분석, 해석, artifact와 재현 절차를 기록한다.

## 2. 문제 유형과 서비스 범위

전체 학습 테이블은 1~38주차를 포함하지만 실제 조기개입 서비스는 개강 후 **1~10주차**로 제한한다. 예측주차 `w`의 Feature는 `w`주 종료 시점까지 관측된 정보만 포함하며, 정답은 `w+1`주 이탈이다.

중요하게, 최종 CatBoost는 전체 주차 학습행으로 학습한 모델이고 1~10주차 OOF 부분집합에서 운영 성능과 임계값을 평가했다. 따라서 “1~10주차 전용 재학습 모델”이 아니라 **전체 주간 CatBoost를 1~10주차 조기개입 서비스에 적용한 구성**이다.

## 3. 데이터 분할과 누수 방지

| 항목 | 적용 기준 |
|---|---|
| 검증 방식 | `id_student` 기준 3-Fold `GroupKFold` |
| Fold별 비율 | 약 2/3 학습, 1/3 검증 |
| 전체 Fold 행 | 학습 596,670행 / 검증 298,335행 |
| OOF 원칙 | 모든 행은 정확히 한 번 검증 Fold에서 예측 |
| 학생 누수 방지 | 한 학생의 모든 과목·주차 행을 같은 Fold에 배정 |
| 전처리 누수 방지 | One-Hot·스케일링이 필요한 모델은 각 학습 Fold에서만 적합 |
| 제외 Feature | `id_student`, Target, `final_result`, `date_unregistration`, `withdraw_week` 등 |
| Random State | 모델 42, Fold 배정은 GroupKFold 규칙으로 고정 |

### 독립 Test 여부

현재 프로젝트는 고정된 외부 Test 세트를 별도로 두지 않고 학생 Group OOF를 최종 후보 비교에 사용했다. OOF는 각 학생을 학습에 사용하지 않은 Fold에서 평가하지만, 최종 독립 Test와 동일하지는 않다. 따라서 아래 수치를 **Test 성능이라고 표현하지 않는다**. 이는 결과 해석의 명시적 한계다.

## 4. 기준 모델과 비교 모델

| 모델 | 선정 이유 | 핵심 학습 조건 |
|---|---|---|
| Dummy prior | 무작위 수준 기준선 | Fold 학습 양성률을 확률로 출력 |
| ElasticNet Logistic | 선형·희소 기준선 | One-Hot + 수치 전처리, ElasticNet |
| Random Forest | 비선형 앙상블 비교 | 300 trees, entropy, balanced subsample |
| XGBoost | Boosting 비교 | depth 6, learning rate 0.05, 가중치 적용 실험 |
| CatBoost | 범주형·결측 native 처리 | iterations 500, depth 7, learning rate 0.05, L2 5 |
| GRU | 최근 4주 시계열 비교 | 4주 × 행동 11개 |
| TCN형 1D-CNN | 합성곱 시계열 비교 | 4주 × 행동 11개 |

전체 주차 Dummy의 PR-AUC는 0.007437로 양성률 0.7455% 수준이다. 모델이 이 값을 충분히 넘는지 먼저 확인했다.

## 5. 평가 지표

- **PR-AUC**: 극단적 불균형 데이터에서 양성 탐지 순위 품질을 평가하는 주 지표
- **Recall**: 실제 다음 주 이탈자 중 경보로 찾은 비율
- **Precision**: 경보 학생 중 실제 다음 주 이탈자의 비율
- **F1**: Precision과 Recall의 조화평균, 운영 임계값 선택 기준
- **ROC-AUC**: 전체 양성·음성 순위 분리 능력
- **Brier Score**: 확률 예측 오차, 낮을수록 좋음
- **Recall@Top20%**: 위험확률 상위 20%가 실제 이탈자를 얼마나 포함하는지 보는 비교 실험 지표

## 6. 1~10주차 모델 비교

| 모델 | F1 최적 임계값 | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.110030 | 27.86% | 20.14% | 23.38% | 0.158890 | 0.843639 |
| XGBoost | 0.878961 | 20.56% | 20.75% | 20.65% | 0.118739 | 0.837438 |
| Random Forest | 0.811896 | 27.61% | 15.41% | 19.78% | 0.141936 | 0.828475 |
| ElasticNet | 0.039806 | 7.20% | 28.44% | 11.49% | 0.050780 | 0.804845 |

![Early 모델 비교](figures/training/early_model_comparison.png)

CatBoost는 Precision, F1, PR-AUC, ROC-AUC의 균형이 가장 좋았다. ElasticNet은 Recall이 가장 높지만 Precision과 F1이 낮아 경보 피로가 커질 수 있다. 따라서 최종 서비스 모델은 CatBoost로 선정했다.

## 7. 최종 임계값과 Group OOF 평가

1~10주차의 모든 고유 CatBoost OOF 확률을 후보로 두고 F1이 최대가 되는 임계값 **0.110030061**를 선택했다. Platt Scaling 등 별도 확률 보정은 적용하지 않았다.

| 지표 | 결과 |
|---|---:|
| 평가 행 / 실제 이탈 | 271,663 / 3,316 |
| Accuracy / Specificity | 98.39% / 99.36% |
| Precision / Recall / F1 | 27.86% / 20.14% / 23.38% |
| ROC-AUC / PR-AUC | 0.843639 / 0.158890 |
| Brier Score | 0.011063 |
| TP / FP / TN / FN | 668 / 1,730 / 266,617 / 2,648 |
| 경보 대상 | 2,398건 (0.8827%) |

![Early ROC·PR](figures/training/early_catboost_roc_pr.png)

![Early 혼동행렬](figures/training/early_catboost_confusion_matrix.png)

## 8. 오류 분석과 비즈니스 해석

- TP 668건: 다음 주 이탈자를 사전에 찾은 사례
- FP 1,730건: 실제로는 남아 있지만 상담 대상으로 분류된 사례
- FN 2,648건: 다음 주 이탈했지만 찾지 못한 사례
- TN 266,617건: 비이탈을 정상적으로 낮은 위험으로 분류한 사례

현재 임계값에서는 약 100명의 경보 학생 중 28명이 실제 다음 주 이탈자이며, 전체 실제 이탈자의 약 20명을 찾는다. FP의 비용은 상담·알림 자원이고 FN의 비용은 개입 기회 상실이다. 따라서 확률을 자동 탈락 판정으로 사용하지 않고 상담 우선순위로 사용하며, 과목별 행동 근거와 함께 운영한다.

![주차별 성능](figures/training/early_catboost_weekly_performance.png)

주차별 양성률과 행동 가용성이 달라 같은 임계값에서도 성능이 변한다. 운영 후에는 주차별 Precision·Recall과 상담 수용량을 지속 모니터링해야 한다.

## 9. Feature 구성·딥러닝 추가 실험

| 모델 | Feature 수 | 전체 주차 PR-AUC | Recall@Top20% | Brier Score |
|---|---:|---:|---:|---:|
| CatBoost 124 | 124 | 0.094775 | 71.13% | 0.007045 |
| CatBoost 108 | 108 | 0.093502 | 71.24% | 0.007050 |
| GRU 4-week | 11 | 0.027145 | 49.54% | 0.210724 |
| TCN 4-week | 11 | 0.027917 | 49.34% | 0.207573 |

![전체 비교 실험](figures/training/full_model_comparison.png)

- 124개 Enhanced CatBoost는 51개 Base CatBoost보다 PR-AUC가 0.089320에서 0.094775로 상승했다.
- 108개 축소 모델은 Recall@Top20%가 71.24%로 소폭 높았지만 PR-AUC가 0.093502로 낮아 보조 실험으로 남겼다.
- GRU·TCN은 최근 4주 행동을 학습했으나 CatBoost를 넘지 못해 실제 추론에는 연결하지 않았다.
- 언더샘플링·오버샘플링·L1/L2 조정은 불균형 대응 보조 실험이며 실제 양성률과 확률 의미를 보존하기 위해 최종 모델에는 적용하지 않았다.

## 10. 모델 해석

| 순위 | Feature | mean abs SHAP | 중요도 비중 |
|---:|---|---:|---:|
| 1 | `assessment_missing_due_rate` | 0.211522 | 5.98% |
| 2 | `weighted_score_sum` | 0.152318 | 4.31% |
| 3 | `studied_credits` | 0.133171 | 3.77% |
| 4 | `scored_weight_sum` | 0.114304 | 3.23% |
| 5 | `code_module` | 0.102703 | 2.90% |
| 6 | `current_active_days` | 0.091743 | 2.59% |
| 7 | `observed_weeks` | 0.086459 | 2.45% |
| 8 | `previous_active_days` | 0.079588 | 2.25% |
| 9 | `assessment_min_score` | 0.078402 | 2.22% |
| 10 | `prediction_week` | 0.074141 | 2.10% |
| 11 | `assessment_submission_rate` | 0.072912 | 2.06% |
| 12 | `previous_total_clicks` | 0.071718 | 2.03% |
| 13 | `cum_quiz_share` | 0.067181 | 1.90% |
| 14 | `cutoff_day` | 0.065199 | 1.84% |
| 15 | `log1p_current_total_clicks` | 0.062970 | 1.78% |

![CatBoost Feature Importance](figures/training/catboost_top15_feature_importance.png)

평가 미제출률, 점수·가중치, 수강 학점, 현재·직전 활동일, 과목·예측주차가 주요 신호였다. SHAP 중요도는 인과효과가 아니라 예측 기여도이므로 “이 Feature가 이탈을 발생시킨다”고 해석하지 않는다.

## 11. 최종 모델과 artifact

| 항목 | 값 |
|---|---|
| 실제 모델 | `artifacts/catboost.joblib` |
| Early 패키지 | `artifacts/early_catboost.joblib` |
| 운영 설정 | `artifacts/early_service_config.json` |
| 모델 | Early CatBoost |
| 학습 범위 | 전체 사용 가능 주차 |
| 서비스 범위 | 1~10주차 |
| Feature | 124개, 범주형 8개 |
| 학습 행 | 895,005 |
| 임계값 | 0.110030061 |
| 저장 내용 | 모델 객체, Feature 순서, 범주형 목록, Target, 임계값, 학습 파라미터 |
| 재로딩 확인 | `predict_proba` 지원: True |

## 12. 재현 방법과 분석 파일

```bash
python -m pip install -r requirements.txt
python models/ML/02_catboost_weekly_next_week.py
python models/ML/08_train_final_catboost_joblib.py
python -m unittest discover -s tests -v
```

대용량 주간 학습 CSV는 Git에서 제외될 수 있다. 재학습 전에 동일한 126열·895,005행 학습 데이터를 `models/data/oulad_weekly_next_week.csv`에 배치하고 124개 Feature 계약을 확인한다.

주요 분석 파일:

- `models/ML/01_xgboost_weekly_next_week.py/.ipynb`
- `models/ML/02_catboost_weekly_next_week.py/.ipynb`
- `models/ML/03_dummy_weekly_next_week.py/.ipynb`
- `models/ML/04_elasticnet_logistic_weekly_next_week.py/.ipynb`
- `models/ML/05_randomforest_weekly_next_week.py/.ipynb`
- `models/DL/06_gru_weekly_next_week.py`
- `models/DL/09_tcn_weekly_next_week.py`
- `models/feature_importance/catboost_feature_importance.py/.ipynb`

결과 artifact:

- `artifacts/metrics.csv`
- `artifacts/model_metadata.json` — 최종 모델 ZIP 단계에서 확정
- `artifacts/early_service_config.json`
- `models/ML/*_metrics.csv`, `models/DL/*_metrics.csv`

## 13. 최종 선정 근거

CatBoost Enhanced 124 Feature를 최종 모델로 선정한 이유는 다음과 같다.

1. Early 1~10주차에서 F1·PR-AUC·ROC-AUC의 종합 균형이 가장 좋다.
2. 범주형과 구조적 결측을 native 처리해 전처리 누수와 배포 복잡도가 낮다.
3. 124개 Feature가 평가·VLE·등록·학생 배경을 함께 표현한다.
4. Feature 순서와 임계값이 joblib에 저장되어 Streamlit과 연결할 수 있다.
5. GRU·TCN·축소 Feature·샘플링 실험보다 현재 서비스 기준에 적합하다.

## 14. 한계와 개선 방향

- 독립 외부 Test가 없어 최종 성능은 학생 Group OOF 기준이다.
- Early 임계값도 같은 OOF 부분집합에서 선택했으므로 새 학기에서 재검증해야 한다.
- 모델은 OULAD의 익명화 대학 원격교육 데이터에 한정되어 다른 기관에 바로 일반화할 수 없다.
- 양성률이 매우 낮아 Accuracy보다 PR-AUC·Precision·Recall을 중심으로 봐야 한다.
- 확률 보정을 적용하지 않았으므로 확률의 절대값보다 위험 순위와 임계값 기준으로 운영한다.
- 상담 후 실제 유지 효과를 측정하는 개입 실험 데이터가 아직 없다.

---

작성일: 2026-07-22
프로젝트: OULAD 1~10주차 다음 주 이탈 조기경보

# 검증 계획

## 학생 단위 OOF

- `id_student`를 Group으로 사용한 3-Fold OOF 검증을 적용한다.
- 동일 학생의 모든 과목·운영 회차·예측 주차는 하나의 Fold에만 배정한다.
- 각 행의 OOF 확률은 해당 학생을 학습하지 않은 모델이 생성한다.

## 시점 누수 방지

- `final_result`, `date_unregistration`, `withdraw_week` 등 정답·이탈 파생 정보를 제외한다.
- 예측 주차 종료 후에 발생한 VLE 활동·평가 제출·점수를 제외한다.
- 결측치 대체, 인코딩, 스케일링, 샘플링, Feature Selection은 각 학습 Fold 안에서만 학습한다.

## 모델 비교

- Dummy, ElasticNet, Random Forest, XGBoost, CatBoost를 학생 단위 OOF로 비교한다.
- CatBoost 124 Feature를 최종 서비스 모델로 사용한다.
- CatBoost 108 Feature와 GRU·TCN은 추가 비교 실험으로 기록한다.
- 불균형 처리·규제 실험은 최종 모델을 대체하지 않고 보조 실험으로 보고한다.

## Early 운영 평가

- 전체 주차 OOF 예측 중 `prediction_week` 1~10을 Early 운영 구간으로 평가한다.
- 임계값 `0.1100300614`는 Early OOF 부분집합의 F1을 최대화한 값이다.
- 해당 결과는 OOF 운영 분석으로 표기하며, 독립 외부 Test 성능으로 표현하지 않는다.
- 전체 주차와 Early 구간의 양성 비율이 다르므로 지표를 단순 동일 모집단 향상으로 해석하지 않는다.

## 저장 항목

- 최종 모델 artifact
- 124개 Feature 이름·순서와 범주형 Feature 목록
- Early 운영 구간·분류 임계값
- 학습 데이터·코드 버전과 난수 시드
- Fold별·전체·Early 평가 지표

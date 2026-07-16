# 검증 계획

이 문서는 머신러닝 담당자의 이후 작업 기준이다. 현재 전처리·EDA 단계에서는
데이터 분할, Validation, 모델 학습, Test 평가를 실행하지 않았다.

## 데이터 분할

- Train 60~70%, Validation 15~20%, Test 15~20%를 기본 범위로 한다.
- `id_student` 기준 Group 분할을 사용한다.
- 동일 학생의 다른 과목·회차와 1·2·4주차 스냅샷은 같은 분할에 둔다.
- 난수 시드는 프로젝트에서 하나로 고정하고 기록한다.

## 전처리 누수 방지

다음 항목은 Train에서만 학습하고 Validation·Test에는 변환만 적용한다.

- 결측치 대체값
- 범주형 인코딩
- 스케일링
- 클래스 불균형 처리
- Feature Selection

`final_result`, `date_unregistration`과 예측 시점 이후에 발생한 정보는 입력 Feature에서 제외한다.

## 주차 선택

1·2·4주차 각각 동일한 분할과 평가 방법을 사용한다. Validation에서 Recall을 우선 확인하고, 성능이 비슷하면 더 빠른 주차를 선택한다.

## 모델 및 임계값 선택

- Dummy 모델을 기준선으로 둔다.
- Logistic Regression, Random Forest, Boosting 모델을 비교한다.
- 모델과 threshold는 Validation에서 결정한다.
- 선택이 끝난 뒤 Test를 한 번만 평가한다.

## 저장 항목

- 최종 전처리기와 모델이 결합된 Pipeline
- Feature 이름과 순서
- 분류 threshold
- 학습 데이터·코드 버전
- 모델별 평가 지표

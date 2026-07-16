# 머신러닝 인수인계

## 현재 상태

전처리와 EDA까지 완료했다. 데이터 분할, Validation, 모델 학습, 임계값 선택,
Test 평가는 수행하지 않았다.

## 바로 사용할 파일

| 후보 주차 | 파일 | 행 | 열 | Target 0 | Target 1 |
|---:|---|---:|---:|---:|---:|
| 1 | `data/processed/model_snapshot_week_1.csv` | 29,018 | 99 | 22,366 | 6,652 |
| 2 | `data/processed/model_snapshot_week_2.csv` | 27,984 | 99 | 22,395 | 5,589 |
| 4 | `data/processed/model_snapshot_week_4.csv` | 27,449 | 99 | 22,424 | 5,025 |

세 파일의 컬럼과 순서는 동일하다. 학생·과목·개설 회차 키 중복과 결측치는
0건이며 `target`은 0 또는 1이다.

## pull 후 확인

```bash
python -m pip install -r requirements.txt
python -m src.check_preprocessing_handoff
```

검증 결과와 SHA-256은 `artifacts/preprocessing_manifest.json`에 저장된다.
Manifest의 해시가 같으면 Validation 파일 삭제나 노트북 출력 삭제 여부와 관계없이
동일한 전처리 데이터다.

## ML 담당자가 지켜야 할 경계

- `target`: 정답 컬럼이므로 입력 Feature에서 제외
- `id_student`: 같은 학생이 분할 사이에 섞이지 않도록 Group 분할에만 사용
- `final_result`, `date_unregistration`, 이탈 파생 컬럼: Snapshot에서 이미 제거됨
- 범주형 인코딩, 결측 대체, 스케일링, Feature 선택: Train에서만 학습
- 1·2·4주차는 동일한 학생 Group 분할을 사용

전처리 단계에서 `split` 컬럼이나 Validation 결과는 생성하지 않았다. 따라서 이후
예측값은 ML 담당자가 선택하는 분할, 난수 시드, 전처리기, 모델, 임계값에 의해
결정되며 Validation 삭제 자체 때문에 달라지지는 않는다.

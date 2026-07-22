# OULAD 학생 이탈 예측 대시보드

OULAD(Open University Learning Analytics Dataset) 기반으로 학생의 중도 이탈(Withdrawn) 위험을
탐지하고, 행동추천을 제공하는 Streamlit 멀티페이지 앱입니다.

- **규칙 기반 페이지** (대시보드, 과목/주차별 행동제안): `vle_weekly_features.csv` 등 원본 정제
  데이터를 그대로 사용해 참여도 급감·미제출 등 룰(rule)로 위험도를 계산합니다.
- **모델 기반 페이지** (학생별 행동추천, 이탈 예측): `vle_snapshot_week_*.csv` 스냅샷과
  (선택) CatBoost 모델을 사용합니다. 모델이 없으면 임시 규칙 기반 점수로 자동 대체됩니다.

## 1. 폴더 구조 (필요한 파일 목록)

업로드해주신 파일들을 아래 구조로 배치하면 됩니다. `lib/`, `pages/` 등 폴더명은
코드의 `from lib import ...`, `pages/2_students_recommendations.py` 참조와
일치해야 하므로 반드시 이 이름을 지켜주세요.

```
project-root/
├── app.py                              # ← 0_dashboard.py 를 이 이름으로 두거나,
│                                          streamlit run 0_dashboard.py 로 직접 실행해도 됩니다.
├── pages/
│   ├── 1_course_weekly_recommendations.py
│   ├── 2_students_recommendations.py
│   └── 3_dropout_predictions.py
├── lib/
│   ├── __init__.py
│   ├── data.py
│   ├── theme.py
│   ├── risk.py
│   ├── model.py
│   └── sample_defaults.json
├── utils/
│   └── styles.py        
├── styles.css                    
├── data/
│   └── interim/
│       ├── student_info_processed.csv
│       ├── student_registration_processed.csv
│       ├── vle_weekly_features.csv
│       ├── vle_pre_course_features.csv
│       ├── courses_processed.csv
│       ├── assessments_processed.csv
│       ├── student_assessment_processed.csv
│       └── vle_snapshot_week_{1,2,4,...}.csv   # cutoff_week별 모델 스냅샷
├── artifacts/
│   └── early_catboost.joblib           # 선택 사항 (없으면 규칙 기반 점수로 대체)
├── requirements.txt
└── README.md
```

### 지금 가진 파일 → 위 구조 매핑

| 업로드된 파일 | 배치 위치 |
|---|---|
| `0_dashboard.py` | `app.py` (또는 그대로 두고 `streamlit run 0_dashboard.py`) |
| `1_course_weekly_recommendations.py` | `pages/1_course_weekly_recommendations.py` |
| `2_students_recommendations.py` | `pages/2_students_recommendations.py` |
| `3_dropout_predictions.py` | `pages/3_dropout_predictions.py` |
| `data.py`, `theme.py`, `risk.py`, `model.py`, `sample_defaults.json`, `__init__.py` | `lib/` |


## 2. 설치 및 실행

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py             # 또는 streamlit run 0_dashboard.py
```

## 3. 페이지 구성

| 파일 | 제목 | 데이터 소스 | 위험도 산정 |
|---|---|---|---|
| `app.py` (`0_dashboard.py`) | 📊 대시보드 | `data/interim/*.csv` 원본 | 룰 기반(`lib/risk.py`) |
| `pages/1_course_weekly_recommendations.py` | 📋 과목/주차별 행동제안 | 원본 + `build_master_table` | 룰 기반 |
| `pages/2_students_recommendations.py` | 👨‍🎓 학생별 차주이탈 분석 | `vle_snapshot_week_*.csv` | 룰 기반(스냅샷 피처) |
| `pages/3_dropout_predictions.py` | 🧠 모델 기반 이탈 예측 | 사용자 입력 + 코호트 템플릿 | CatBoost(있으면) / 규칙 기반(없으면) |

## 4. 참고 사항

- `lib/data.py`의 `_find_project_root()`가 `data/interim` 폴더를 자동으로 찾으므로,
  `lib/`가 저장소 루트 바로 아래(`project-root/lib/`)에 있기만 하면 `app.py` 위치가
  루트든 하위 폴더든 크게 상관없습니다.
- `lib/model.py`는 `artifacts/early_catboost.joblib`가 없거나 `catboost` 패키지가
  미설치 상태면 조용히 규칙 기반 점수로 폴백하고, 화면에 "임시 규칙 기반 점수" 안내를 띄웁니다.
- `sample_defaults.json`은 모델이 요구하지만 현재 스냅샷 CSV엔 없는 피처를 채우기 위한
  참조용 평균/최빈값 테이블입니다. 전체 데이터 통계가 갱신되면 이 파일만 재계산해서
  교체하면 됩니다.

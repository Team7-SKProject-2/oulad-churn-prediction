"""CatBoost 중도이탈 예측 모델의 상위 영향 Feature를 추출한다.

Feature 중요도는 검증 학생 표본에서 계산한 평균 절대 SHAP 값으로 정렬한다.
값이 클수록 해당 Feature가 개별 예측확률을 바꾸는 정도가 평균적으로 크다는 뜻이다.
인과관계를 뜻하지는 않으므로, 교수자 개입 원인으로 단정해서는 안 된다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold


FEATURE_IMPORTANCE_DIR = Path(__file__).resolve().parent
MODELS_DIR = FEATURE_IMPORTANCE_DIR.parent
DATA_PATH = MODELS_DIR / "data" / "oulad_weekly_next_week.csv"
RESULT_DIR = MODELS_DIR / "results" / "feature_importance"
TARGET_COL = "target_next_week_withdrawn"
ID_COL = "id_student"

# 전체 평균 절대 SHAP 영향력 중 상위 Feature들이 설명할 누적 비중 범위다.
# 이 범위 안에서 중앙값(기본 75%)에 가장 가까운 최소 상위 N개를 자동 선택한다.
CUMULATIVE_IMPORTANCE_MIN = 0.70
CUMULATIVE_IMPORTANCE_MAX = 0.80
CUMULATIVE_IMPORTANCE_TARGET = 0.75
# SHAP 계산량을 관리하기 위한 검증 표본 수다. 전체 검증 집합보다 많으면 전체를 사용한다.
SHAP_SAMPLE_SIZE = 20_000
RANDOM_STATE = 42

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """식별자·Target을 제거하고 CatBoost 입력 Feature와 범주형 Feature를 준비한다."""
    features = data.drop(columns=[ID_COL, TARGET_COL]).copy()
    categorical = [
        column
        for column in features.columns
        if pd.api.types.is_object_dtype(features[column])
        or pd.api.types.is_string_dtype(features[column])
        or isinstance(features[column].dtype, pd.CategoricalDtype)
    ]
    for column in categorical:
        features[column] = features[column].fillna("미상").astype(str)
    numeric = [column for column in features.columns if column not in categorical]
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)
    return features, categorical


def validate_data(data: pd.DataFrame) -> None:
    """최종 결과·등록 해지 정보처럼 Target을 직접 알려 주는 누수 변수를 차단한다."""
    forbidden = [
        column
        for column in data.columns
        if column == "target"
        or any(term in column.lower() for term in ["final_result", "unregistration", "withdraw_week"])
    ]
    if forbidden:
        raise ValueError(f"누수 가능 변수가 포함되어 있습니다: {forbidden}")


def train_validation_split(features: pd.DataFrame, target: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """동일 학생의 여러 주차 행이 Train·Validation에 섞이지 않도록 GroupKFold 1개 Fold를 사용한다."""
    splitter = GroupKFold(n_splits=3)
    train_index, validation_index = next(splitter.split(features, target, groups))
    return train_index, validation_index


def make_importance_table(
    model: CatBoostClassifier,
    validation_features: pd.DataFrame,
    categorical: list[str],
) -> pd.DataFrame:
    """기본 중요도와 평균 절대 SHAP 값을 결합해 Feature별 영향도 표를 만든다."""
    sample_size = min(SHAP_SAMPLE_SIZE, len(validation_features))
    shap_features = validation_features.sample(n=sample_size, random_state=RANDOM_STATE)
    shap_pool = Pool(shap_features, cat_features=categorical)

    # 마지막 열은 base value이므로 Feature 중요도 계산에서 제외한다.
    shap_values = model.get_feature_importance(shap_pool, type="ShapValues")[:, :-1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    prediction_values_change = model.get_feature_importance(type="PredictionValuesChange")

    importance = pd.DataFrame(
        {
            "feature": validation_features.columns,
            "mean_abs_shap": mean_abs_shap,
            "prediction_values_change": prediction_values_change,
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    total_importance = importance["mean_abs_shap"].sum()
    importance["importance_share"] = importance["mean_abs_shap"] / total_importance
    importance["cumulative_importance_share"] = importance["importance_share"].cumsum()
    return importance


def select_cumulative_features(importance: pd.DataFrame) -> pd.DataFrame:
    """누적 영향력 70~80% 구간에서 75%에 가장 가까운 최소 상위 Feature를 선택한다."""
    if not 0 < CUMULATIVE_IMPORTANCE_MIN <= CUMULATIVE_IMPORTANCE_TARGET <= CUMULATIVE_IMPORTANCE_MAX <= 1:
        raise ValueError("누적 영향력 설정은 0 < 최소 ≤ 목표 ≤ 최대 ≤ 1을 만족해야 합니다.")

    in_range = importance[
        importance["cumulative_importance_share"].between(
            CUMULATIVE_IMPORTANCE_MIN, CUMULATIVE_IMPORTANCE_MAX, inclusive="both"
        )
    ]
    if not in_range.empty:
        # 70~80% 후보 중 75%에 가장 가까운 지점을 선택한다.
        end_index = (in_range["cumulative_importance_share"] - CUMULATIVE_IMPORTANCE_TARGET).abs().idxmin()
    else:
        # 한 Feature의 비중이 커서 구간에 정확히 들어가지 않으면 75% 이상이 되는 첫 지점을 사용한다.
        reached = importance[importance["cumulative_importance_share"] >= CUMULATIVE_IMPORTANCE_TARGET]
        end_index = reached.index[0] if not reached.empty else importance.index[-1]
    return importance.loc[:end_index].copy()


def save_bar_chart(selected_features: pd.DataFrame, output_path: Path) -> None:
    """누적 영향력 목표를 만족하는 상위 Feature의 평균 절대 SHAP 막대그래프를 저장한다."""
    top_features = selected_features.sort_values("mean_abs_shap", ascending=True)
    figure, axis = plt.subplots(figsize=(10, max(6, len(top_features) * 0.38)))
    axis.barh(top_features["feature"], top_features["mean_abs_shap"], color="#3B82F6")
    achieved = selected_features["cumulative_importance_share"].iloc[-1]
    axis.set_title(
        f"CatBoost 영향 Feature 상위 {len(selected_features)}개 "
        f"(누적 영향력 {achieved:.1%})"
    )
    axis.set_xlabel("평균 절대 SHAP 값")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """데이터 로드 → GroupKFold 학습 → SHAP 중요도 계산 → CSV·그래프 저장을 수행한다."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    validate_data(data)
    features, categorical = prepare_features(data)
    target = data[TARGET_COL].astype(int).to_numpy()
    groups = data[ID_COL].to_numpy()
    train_index, validation_index = train_validation_split(features, target, groups)

    # 최종 후보 CatBoost와 동일한 기본 하이퍼파라미터를 사용한다.
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=500,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=5,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(features.iloc[train_index], target[train_index], cat_features=categorical, verbose=False)

    importance = make_importance_table(model, features.iloc[validation_index], categorical)
    selected_importance = select_cumulative_features(importance)
    achieved_share = selected_importance["cumulative_importance_share"].iloc[-1]
    target_label = (
        f"{int(CUMULATIVE_IMPORTANCE_MIN * 100)}to"
        f"{int(CUMULATIVE_IMPORTANCE_MAX * 100)}pct"
    )
    importance.to_csv(RESULT_DIR / "catboost_feature_importance_all.csv", index=False, encoding="utf-8-sig")
    selected_importance.to_csv(
        RESULT_DIR / f"catboost_feature_importance_cumulative_{target_label}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_bar_chart(
        selected_importance,
        RESULT_DIR / f"catboost_feature_importance_cumulative_{target_label}.png",
    )

    print(f"검증 학생 수: {len(np.unique(groups[validation_index])):,}명")
    print(f"SHAP 계산 표본 수: {min(SHAP_SAMPLE_SIZE, len(validation_index)):,}행")
    print(
        f"\n=== 누적 영향력 {CUMULATIVE_IMPORTANCE_MIN:.0%}~{CUMULATIVE_IMPORTANCE_MAX:.0%} "
        f"구간에서 선택한 상위 {len(selected_importance)}개 Feature (실제 {achieved_share:.2%}) ==="
    )
    print(selected_importance.to_string(index=False))
    print(f"\n결과 저장 위치: {RESULT_DIR}")


if __name__ == "__main__":
    main()

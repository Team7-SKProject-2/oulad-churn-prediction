"""기본 53열 데이터와 확장 126열 데이터의 공정한 모델 성능 비교 코드."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


NOTEBOOK_DIR = Path(__file__).resolve().parent
MODELS_DIR = NOTEBOOK_DIR.parent
BASE_PATH = MODELS_DIR / "data" / "oulad_weekly_next_week_base.csv"
ENHANCED_PATH = MODELS_DIR / "data" / "oulad_weekly_next_week.csv"
RESULT_DIR = MODELS_DIR / "results" / "feature_set_comparison"

TARGET_COL = "target_next_week_withdrawn"
ID_COL = "id_student"
KEY_COLUMNS = ["code_module", "code_presentation", ID_COL, "prediction_week"]
N_SPLITS = 3
TOP_FRACTION = 0.20

# 빠른 확인은 xgboost만, 최종 선택 비교는 두 모델을 모두 실행한다.
MODELS_TO_RUN = ["xgboost", "catboost"]


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    """동일 폭 10개 확률 구간의 ECE를 계산한다."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(ece)


def top_fraction_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """상위 20% 위험군의 Recall·Precision·최소 확률을 계산한다."""
    top_k = max(1, int(np.ceil(len(y_true) * TOP_FRACTION)))
    top_index = np.argsort(probability)[-top_k:]
    true_positive = int(y_true[top_index].sum())
    positive_count = int(y_true.sum())
    return {
        "recall_at_top_20pct": np.nan if positive_count == 0 else true_positive / positive_count,
        "precision_at_top_20pct": true_positive / top_k,
        "top_20pct_count": top_k,
        "top_20pct_threshold": float(probability[top_index].min()),
    }


def calculate_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """불균형 분류 성능과 확률 보정 품질을 함께 계산한다."""
    result = {
        "pr_auc_average_precision": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "ece_10bin": expected_calibration_error(y_true, probability),
    }
    result.update(top_fraction_metrics(y_true, probability))
    return result


def get_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """범주형 One-Hot Encoding 대상과 수치형 Feature를 분리한다."""
    categorical = [
        column
        for column in features.columns
        if pd.api.types.is_object_dtype(features[column])
        or pd.api.types.is_string_dtype(features[column])
        or isinstance(features[column].dtype, pd.CategoricalDtype)
    ]
    numeric = [column for column in features.columns if column not in categorical]
    return categorical, numeric


def validate_same_observations() -> None:
    """두 파일의 학생·과목·주차·Target이 완전히 같은지 검증한다."""
    usecols = KEY_COLUMNS + [TARGET_COL]
    base_keys = pd.read_csv(BASE_PATH, usecols=usecols).sort_values(KEY_COLUMNS).reset_index(drop=True)
    enhanced_keys = pd.read_csv(ENHANCED_PATH, usecols=usecols).sort_values(KEY_COLUMNS).reset_index(drop=True)
    if not base_keys.equals(enhanced_keys):
        raise ValueError("두 데이터의 복합키 또는 Target이 달라 공정 비교를 할 수 없습니다.")
    if base_keys.duplicated(KEY_COLUMNS).any():
        raise ValueError("복합키 중복이 있어 비교를 중단합니다.")
    print(f"공통 관측치 확인 완료: {len(base_keys):,}행")


def load_feature_set(data_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """누수 변수를 점검하고 Feature·Target·학생 그룹을 분리한다."""
    data = pd.read_csv(data_path)
    forbidden = [
        column
        for column in data.columns
        if column == "target"
        or any(term in column.lower() for term in ["final_result", "unregistration", "withdraw_week"])
    ]
    if forbidden:
        raise ValueError(f"누수 가능 변수가 포함되어 있습니다: {forbidden}")

    features = data.drop(columns=[ID_COL, TARGET_COL]).copy()
    categorical, numeric = get_feature_types(features)
    for column in categorical:
        features[column] = features[column].fillna("미상").astype(str)
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)
    return features, data[TARGET_COL].astype(int).to_numpy(), data[ID_COL].to_numpy()


def fit_predict_xgboost(
    features: pd.DataFrame,
    target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """기존 XGBoost 설정으로 공통 Fold의 OOF 예측확률을 만든다."""
    categorical, numeric = get_feature_types(features)
    probability = np.zeros(len(features), dtype=float)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_index, test_index) in enumerate(splits, start=1):
        preprocessor = ColumnTransformer(
            [
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("numeric", "passthrough", numeric),
            ],
            sparse_threshold=0.3,
        )
        x_train = preprocessor.fit_transform(features.iloc[train_index])
        x_test = preprocessor.transform(features.iloc[test_index])
        y_train, y_test = target[train_index], target[test_index]
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            scale_pos_weight=float(scale_pos_weight),
            tree_method="hist",
            early_stopping_rounds=40,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
        fold_probability = model.predict_proba(x_test)[:, 1]
        probability[test_index] = fold_probability
        row: dict[str, float | int] = {"fold": fold, "best_iteration": int(model.best_iteration)}
        row.update(calculate_metrics(y_test, fold_probability))
        fold_rows.append(row)
        print(f"  XGBoost Fold {fold}: PR-AUC={row['pr_auc_average_precision']:.4f}")
    return probability, fold_rows


def fit_predict_catboost(
    features: pd.DataFrame,
    target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """기존 CatBoost 설정으로 공통 Fold의 OOF 예측확률을 만든다."""
    categorical, _ = get_feature_types(features)
    probability = np.zeros(len(features), dtype=float)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_index, test_index) in enumerate(splits, start=1):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="PRAUC",
            iterations=500,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=5,
            random_seed=42,
            od_type="Iter",
            od_wait=40,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            features.iloc[train_index],
            target[train_index],
            cat_features=categorical,
            eval_set=(features.iloc[test_index], target[test_index]),
            use_best_model=True,
            verbose=False,
        )
        fold_probability = model.predict_proba(features.iloc[test_index])[:, 1]
        probability[test_index] = fold_probability
        row: dict[str, float | int] = {"fold": fold, "best_iteration": int(model.get_best_iteration())}
        row.update(calculate_metrics(target[test_index], fold_probability))
        fold_rows.append(row)
        print(f"  CatBoost Fold {fold}: PR-AUC={row['pr_auc_average_precision']:.4f}")
    return probability, fold_rows


def main() -> None:
    """두 Feature Set을 같은 Fold와 모델 설정으로 실행하고 결과를 저장한다."""
    if not BASE_PATH.exists() or not ENHANCED_PATH.exists():
        raise FileNotFoundError("models/data 아래의 기본·확장 데이터 파일을 확인하세요.")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    validate_same_observations()

    runners = {"xgboost": fit_predict_xgboost, "catboost": fit_predict_catboost}
    datasets = {
        "base_51_features": BASE_PATH,
        "enhanced_124_features": ENHANCED_PATH,
    }
    comparison_rows: list[dict[str, float | int | str]] = []
    fold_metric_rows: list[dict[str, float | int | str]] = []

    for feature_set, data_path in datasets.items():
        print(f"\n===== {feature_set} =====")
        features, target, groups = load_feature_set(data_path)
        splits = list(GroupKFold(n_splits=N_SPLITS).split(features, target, groups))

        for model_name in MODELS_TO_RUN:
            print(f"[{model_name}] 입력 Feature {features.shape[1]}개 학습 시작")
            oof_probability, fold_rows = runners[model_name](features, target, splits)
            row: dict[str, float | int | str] = {
                "feature_set": feature_set,
                "model": model_name,
                "rows": len(features),
                "feature_count": features.shape[1],
                "target_rate": float(target.mean()),
            }
            row.update(calculate_metrics(target, oof_probability))
            comparison_rows.append(row)
            for fold_row in fold_rows:
                fold_metric_rows.append({"feature_set": feature_set, "model": model_name, **fold_row})
            del oof_probability
            gc.collect()

        del features, target, groups
        gc.collect()

    comparison = pd.DataFrame(comparison_rows).sort_values(["model", "feature_set"])
    fold_metrics = pd.DataFrame(fold_metric_rows).sort_values(["model", "feature_set", "fold"])
    comparison.to_csv(RESULT_DIR / "base_vs_enhanced_metrics.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(RESULT_DIR / "base_vs_enhanced_fold_metrics.csv", index=False, encoding="utf-8-sig")
    print("\n=== 전체 OOF 비교 결과 ===")
    print(comparison.to_string(index=False))
    print(f"\n저장 위치: {RESULT_DIR}")


if __name__ == "__main__":
    main()

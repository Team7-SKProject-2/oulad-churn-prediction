"""demo_1과 같은 출력 서식으로 실행하는 demo2 CatBoost 비교 버전.

동일하게 맞춘 항목
- 파일명과 출력 CSV 열 순서
- id_student 기준 3-Fold GroupKFold
- CatBoost 하이퍼파라미터와 조기 종료 기준
- Recall@Top-20%, PR-AUC, Brier Score, ECE(10구간)

demo2 고유 항목
- 4·19·25주차 Snapshot에서 만든 98개 Feature
- 최종 이탈이 아닌 '다음 7일 이내 이탈' Target
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold


OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = OUTPUT_DIR / "used_data" / "weekly_next_week_with_vle.csv"
TARGET_COL = "target_next_week_withdrawn"
ID_COL = "id_student"

# -----------------------------------------------------------------------------
# 사용자가 직접 수정할 수 있는 구간
# demo_1과 동일한 비교 조건을 기본값으로 두었다.
# -----------------------------------------------------------------------------
N_SPLITS = 3
TOP_FRACTION = 0.20
EXPECTED_WEEKS = (4, 19, 25)
ITERATIONS = 500
LEARNING_RATE = 0.05
DEPTH = 7
L2_LEAF_REG = 5
RANDOM_SEED = 42
OD_WAIT = 40

# 확률을 직접 비교하기 위해 demo_1처럼 클래스 가중치를 사용하지 않는다.
# demo2 원본 조건을 시험하려면 아래 값을 "Balanced"로 바꿀 수 있다.
AUTO_CLASS_WEIGHTS: str | None = None


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    """확률 구간별 실제 이탈률과 평균 예측확률 차이의 가중합이다."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)
        if mask.any():
            ece += mask.mean() * abs(
                y_true[mask].mean() - probability[mask].mean()
            )
    return float(ece)


def recall_at_top_fraction(
    y_true: np.ndarray,
    probability: np.ndarray,
    fraction: float = TOP_FRACTION,
) -> float:
    """예측확률 상위 fraction 안에 포함된 실제 이탈자의 비율이다."""
    if y_true.sum() == 0:
        return np.nan
    top_k = max(1, int(np.ceil(len(y_true) * fraction)))
    top_index = np.argsort(probability)[-top_k:]
    return float(y_true[top_index].sum() / y_true.sum())


def calculate_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    """demo_1과 이름·순서가 같은 네 가지 평가지표를 계산한다."""
    return {
        "recall_at_top_20pct": recall_at_top_fraction(y_true, probability),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "ece_10bin": expected_calibration_error(y_true, probability),
    }


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """학생 ID와 정답을 제외하고 CatBoost Feature를 준비한다."""
    features = data.drop(columns=[ID_COL, TARGET_COL]).copy()
    categorical = [
        column
        for column in features.columns
        if pd.api.types.is_object_dtype(features[column])
        or pd.api.types.is_string_dtype(features[column])
        or isinstance(features[column].dtype, pd.CategoricalDtype)
    ]
    for column in categorical:
        features[column] = features[column].fillna("Unknown").astype(str)
    numeric = [column for column in features.columns if column not in categorical]
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)
    return features, categorical


def validate_input(data: pd.DataFrame) -> None:
    """다음 주 예측 비교를 망가뜨리는 누수와 중복을 실행 전에 차단한다."""
    required = {
        "code_module",
        "code_presentation",
        ID_COL,
        "prediction_week",
        "cutoff_day",
        TARGET_COL,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}")

    # 새 정답 이외의 과거 target 또는 이탈일이 Feature로 남으면 정답 누수다.
    forbidden = [
        column
        for column in data.columns
        if column == "target"
        or any(
            term in column.lower()
            for term in ["final_result", "date_unregistration", "unregister_week"]
        )
    ]
    if forbidden:
        raise ValueError(f"누수 가능 열이 포함되어 있습니다: {forbidden}")

    unique_key = [
        "code_module",
        "code_presentation",
        ID_COL,
        "prediction_week",
    ]
    if data.duplicated(unique_key).any():
        raise ValueError("학생·과목·운영·예측주차 중복 행이 있습니다.")
    if set(data[TARGET_COL].dropna().unique()) != {0, 1}:
        raise ValueError("Target에는 0과 1이 모두 있어야 합니다.")

    actual_weeks = tuple(sorted(data["prediction_week"].unique().tolist()))
    if actual_weeks != EXPECTED_WEEKS:
        raise ValueError(
            f"예측 주차가 다릅니다. 기대={EXPECTED_WEEKS}, 실제={actual_weeks}"
        )
    expected_cutoff = data["prediction_week"] * 7 - 1
    if not data["cutoff_day"].eq(expected_cutoff).all():
        raise ValueError("cutoff_day가 prediction_week 종료일과 일치하지 않습니다.")


def main() -> None:
    # 1. 데이터 로드 및 누수·중복 검증
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"입력 데이터가 없습니다: {DATA_PATH}\n"
            "먼저 00_build_weekly_next_week_comparison_data.py를 실행하세요."
        )
    data = pd.read_csv(DATA_PATH)
    validate_input(data)

    # 2. Target·그룹·Feature 분리
    target = data[TARGET_COL].astype(int).to_numpy()
    groups = data[ID_COL].to_numpy()
    features, categorical = prepare_features(data)

    # 3. demo_1과 같은 학생 기준 3-Fold 교차검증
    splitter = GroupKFold(n_splits=N_SPLITS)
    probabilities = np.zeros(len(data), dtype=float)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(features, target, groups), start=1
    ):
        model_params: dict[str, object] = {
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "iterations": ITERATIONS,
            "learning_rate": LEARNING_RATE,
            "depth": DEPTH,
            "l2_leaf_reg": L2_LEAF_REG,
            "random_seed": RANDOM_SEED,
            "od_type": "Iter",
            "od_wait": OD_WAIT,
            "verbose": False,
            "allow_writing_files": False,
        }
        if AUTO_CLASS_WEIGHTS is not None:
            model_params["auto_class_weights"] = AUTO_CLASS_WEIGHTS

        # demo_1의 코드 서식과 동일하게 현재 Test Fold를 조기 종료 평가셋으로 쓴다.
        model = CatBoostClassifier(**model_params)
        model.fit(
            features.iloc[train_index],
            target[train_index],
            cat_features=categorical,
            eval_set=(features.iloc[test_index], target[test_index]),
            use_best_model=True,
            verbose=False,
        )

        fold_probability = model.predict_proba(features.iloc[test_index])[:, 1]
        probabilities[test_index] = fold_probability
        row: dict[str, float | int] = {
            "fold": fold,
            "train_rows": len(train_index),
            "test_rows": len(test_index),
            "test_target_rate": float(target[test_index].mean()),
            "best_iteration": int(model.get_best_iteration()),
        }
        row.update(calculate_metrics(target[test_index], fold_probability))
        fold_rows.append(row)
        print(
            f"Fold {fold} 완료: PR-AUC={row['pr_auc']:.4f}, "
            f"Recall@Top-20%={row['recall_at_top_20pct']:.4f}"
        )

    # 4. demo_1과 같은 파일명·열 순서로 OOF 결과 저장
    oof = data[
        [
            "code_module",
            "code_presentation",
            ID_COL,
            "prediction_week",
            TARGET_COL,
        ]
    ].copy()
    oof["catboost_oof_probability"] = probabilities

    overall: dict[str, str | float | int] = {
        "model": "CatBoost (demo2 Feature, 4·19·25주차 다음 주 예측)",
        "rows": len(data),
        "target_count": int(target.sum()),
        "target_rate": float(target.mean()),
        "feature_count": int(features.shape[1]),
        "categorical_feature_count": len(categorical),
    }
    overall.update(calculate_metrics(target, probabilities))

    pd.DataFrame([overall]).to_csv(
        OUTPUT_DIR / "catboost_weekly_next_week_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fold_rows).to_csv(
        OUTPUT_DIR / "catboost_weekly_next_week_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    oof.to_csv(
        OUTPUT_DIR / "catboost_weekly_next_week_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== demo2 CatBoost 교차검증 완료 ===")
    print(pd.DataFrame([overall]).to_string(index=False))


if __name__ == "__main__":
    main()

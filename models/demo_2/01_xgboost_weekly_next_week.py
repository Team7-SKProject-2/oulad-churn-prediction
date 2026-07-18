"""demo_1과 같은 서식으로 실행하는 demo2 XGBoost 비교 모델.

처음 보는 사람을 위한 한 줄 요약
--------------------------------
4·19·25주차까지 관측한 학생 활동으로 '바로 다음 7일 안에 이탈할 위험'을
예측하고, 같은 학생이 학습과 평가에 동시에 들어가지 않도록 3번 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = OUTPUT_DIR / "used_data" / "weekly_next_week_with_vle.csv"
MODEL_DIR = OUTPUT_DIR / "models" / "xgboost_weekly_next_week"
TARGET_COL = "target_next_week_withdrawn"
ID_COL = "id_student"

# -----------------------------------------------------------------------------
# 사용자가 직접 바꿀 수 있는 설정 구간
# demo_1과 같은 조건으로 비교하기 위해 기본값을 동일하게 두었다.
# -----------------------------------------------------------------------------
N_SPLITS = 3
TOP_FRACTION = 0.20
EXPECTED_WEEKS = (4, 19, 25)

N_ESTIMATORS = 500
LEARNING_RATE = 0.05
MAX_DEPTH = 6
MIN_CHILD_WEIGHT = 5
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
REG_LAMBDA = 5.0
EARLY_STOPPING_ROUNDS = 40
RANDOM_STATE = 42


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    """예측확률과 실제 이탈률 차이를 10개 확률 구간에서 계산한다."""
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
    """위험확률 상위 20% 안에서 실제 이탈자를 얼마나 찾았는지 계산한다."""
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


def split_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """문자 열은 One-Hot 대상으로, 숫자 열은 그대로 사용할 대상으로 나눈다."""
    categorical = [
        column
        for column in features.columns
        if pd.api.types.is_object_dtype(features[column])
        or pd.api.types.is_string_dtype(features[column])
        or isinstance(features[column].dtype, pd.CategoricalDtype)
    ]
    numeric = [column for column in features.columns if column not in categorical]
    return categorical, numeric


def validate_input(data: pd.DataFrame) -> None:
    """중복·정답 누수·잘못된 주차가 있으면 학습 전에 즉시 중단한다."""
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
        raise ValueError(f"정답 누수 가능 열이 포함되어 있습니다: {forbidden}")

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
    if not data["cutoff_day"].eq(data["prediction_week"] * 7 - 1).all():
        raise ValueError("cutoff_day가 prediction_week 종료일과 일치하지 않습니다.")


def main() -> None:
    # 1. 데이터 로드와 기본 안전성 검사
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"입력 데이터가 없습니다: {DATA_PATH}\n"
            "먼저 00_build_weekly_next_week_comparison_data.py를 실행하세요."
        )
    data = pd.read_csv(DATA_PATH)
    validate_input(data)

    # 2. 정답·학생 그룹·모델 입력 Feature 분리
    target = data[TARGET_COL].astype(int).to_numpy()
    groups = data[ID_COL].to_numpy()
    features = data.drop(columns=[ID_COL, TARGET_COL]).copy()
    categorical, numeric = split_columns(features)
    for column in categorical:
        features[column] = features[column].fillna("Unknown").astype(str)
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)

    # 3. 같은 학생이 Train과 Test에 겹치지 않는 3-Fold 평가
    splitter = GroupKFold(n_splits=N_SPLITS)
    probabilities = np.zeros(len(data), dtype=float)
    fold_rows: list[dict[str, float | int]] = []
    fold_metadata: list[dict[str, float | int | str]] = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for fold, (train_index, test_index) in enumerate(
        splitter.split(features, target, groups), start=1
    ):
        # One-Hot Encoder는 Train Fold에만 학습한다. Test 정보를 미리 보지 않는다.
        preprocessor = ColumnTransformer(
            [
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical,
                ),
                ("numeric", "passthrough", numeric),
            ],
            sparse_threshold=0.3,
        )
        x_train = preprocessor.fit_transform(features.iloc[train_index])
        x_test = preprocessor.transform(features.iloc[test_index])
        y_train = target[train_index]
        y_test = target[test_index]

        # 이탈자가 매우 적으므로 Fold 안의 비이탈/이탈 비율을 학습 가중치로 사용한다.
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            min_child_weight=MIN_CHILD_WEIGHT,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE_BYTREE,
            reg_lambda=REG_LAMBDA,
            scale_pos_weight=float(scale_pos_weight),
            tree_method="hist",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

        fold_probability = model.predict_proba(x_test)[:, 1]
        probabilities[test_index] = fold_probability
        row: dict[str, float | int] = {
            "fold": fold,
            "train_rows": len(train_index),
            "test_rows": len(test_index),
            "test_target_rate": float(y_test.mean()),
            "best_iteration": int(model.best_iteration),
        }
        row.update(calculate_metrics(y_test, fold_probability))
        fold_rows.append(row)

        # 실제 서비스 재사용을 위해 각 Fold의 전처리기와 모델을 함께 저장한다.
        preprocessor_path = MODEL_DIR / f"xgboost_preprocessor_fold_{fold}.joblib"
        model_path = MODEL_DIR / f"xgboost_fold_{fold}.json"
        joblib.dump(preprocessor, preprocessor_path)
        model.save_model(model_path)
        fold_metadata.append(
            {
                "fold": fold,
                "preprocessor": preprocessor_path.name,
                "model": model_path.name,
                "scale_pos_weight": float(scale_pos_weight),
                "best_iteration": int(model.best_iteration),
            }
        )

        print(
            f"Fold {fold} 완료: PR-AUC={row['pr_auc']:.4f}, "
            f"Recall@Top-20%={row['recall_at_top_20pct']:.4f}"
        )

    # 4. demo_1과 같은 파일명·열 순서로 평가 결과 저장
    oof = data[
        [
            "code_module",
            "code_presentation",
            ID_COL,
            "prediction_week",
            TARGET_COL,
        ]
    ].copy()
    oof["xgboost_oof_probability"] = probabilities
    overall: dict[str, str | float | int] = {
        "model": "XGBoost (demo2 Feature, 4·19·25주차 다음 주 예측)",
        "rows": len(data),
        "target_count": int(target.sum()),
        "target_rate": float(target.mean()),
        "feature_count_before_onehot": int(features.shape[1]),
        "categorical_feature_count": len(categorical),
    }
    overall.update(calculate_metrics(target, probabilities))

    pd.DataFrame([overall]).to_csv(
        OUTPUT_DIR / "xgboost_weekly_next_week_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fold_rows).to_csv(
        OUTPUT_DIR / "xgboost_weekly_next_week_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    oof.to_csv(
        OUTPUT_DIR / "xgboost_weekly_next_week_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "data_path": str(DATA_PATH.relative_to(OUTPUT_DIR)),
        "target": TARGET_COL,
        "weeks": list(EXPECTED_WEEKS),
        "rows": len(data),
        "features_before_onehot": features.columns.tolist(),
        "categorical_features": categorical,
        "numeric_features": numeric,
        "split": "GroupKFold(n_splits=3, group=id_student)",
        "fold_models": fold_metadata,
        "probability_warning": (
            "scale_pos_weight를 사용한 원시 확률이므로 사용자 화면 표시 전 "
            "별도 확률 보정을 검증해야 합니다."
        ),
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== demo2 XGBoost 교차검증 완료 ===")
    print(pd.DataFrame([overall]).to_string(index=False))


if __name__ == "__main__":
    main()

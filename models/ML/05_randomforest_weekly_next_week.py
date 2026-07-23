"""확장 VLE Feature 기반 랜덤포레스트 차주 이탈 예측 모델.

학생 한 명이 여러 주차 행을 가질 수 있으므로, 동일 학생의 행이 학습과 검증에
동시에 포함되지 않도록 id_student 기준 3-Fold GroupKFold를 사용한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder


# 모델 코드와 결과 파일은 models에, 대용량 학습 데이터는 models/ML/used_data에 둔다.
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = OUTPUT_DIR / "used_data" / "weekly_next_week_with_vle_enhanced.csv"
TARGET_COL = "target_next_week_withdrawn"
ID_COL = "id_student"
N_SPLITS = 3
TOP_FRACTION = 0.20


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    """예측확률 구간별 실제 이탈률과 평균 예측확률의 차이인 ECE를 계산한다."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability <= upper) if upper == 1 else (
            (probability >= lower) & (probability < upper)
        )
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(ece)


def recall_at_top_fraction(
    y_true: np.ndarray,
    probability: np.ndarray,
    fraction: float = TOP_FRACTION,
) -> float:
    """예측확률 상위 위험군에 포함된 실제 이탈자의 비율을 계산한다."""
    if y_true.sum() == 0:
        return np.nan
    top_k = max(1, int(np.ceil(len(y_true) * fraction)))
    top_index = np.argsort(probability)[-top_k:]
    return float(y_true[top_index].sum() / y_true.sum())


def calculate_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """불균형 분류 성능과 확률 보정 품질을 함께 반환한다."""
    return {
        "recall_at_top_20pct": recall_at_top_fraction(y_true, probability),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "ece_10bin": expected_calibration_error(y_true, probability),
    }


def split_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """범주형 One-Hot Encoding 대상과 수치형 Feature 목록을 분리한다."""
    categorical = [
        column
        for column in features.columns
        if pd.api.types.is_object_dtype(features[column])
        or pd.api.types.is_string_dtype(features[column])
        or isinstance(features[column].dtype, pd.CategoricalDtype)
    ]
    numeric = [column for column in features.columns if column not in categorical]
    return categorical, numeric


def main() -> None:
    # 1. 데이터 로드 및 누수·복합키 중복 검증
    data = pd.read_csv(DATA_PATH)
    forbidden = [
        column
        for column in data.columns
        if any(term in column.lower() for term in ["final_result", "unregistration", "withdraw_week"])
    ]
    if forbidden:
        raise ValueError(f"누수 가능 변수가 포함되어 있습니다: {forbidden}")
    key_columns = ["code_module", "code_presentation", ID_COL, "prediction_week"]
    if data.duplicated(key_columns).any():
        raise ValueError("학생·과목·운영회차·예측주차 복합키 중복이 있습니다.")

    # 2. Target·그룹·모델 입력을 분리한다. id_student는 GroupKFold에만 사용한다.
    target = data[TARGET_COL].astype(int).to_numpy()
    groups = data[ID_COL].to_numpy()
    features = data.drop(columns=[ID_COL, TARGET_COL]).copy()
    categorical, numeric = split_columns(features)
    for column in categorical:
        features[column] = features[column].fillna("미상").astype(str)
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)

    # 3. 동일 학생의 모든 주차 행을 같은 Fold에 배정한다.
    splitter = GroupKFold(n_splits=N_SPLITS)
    probabilities = np.zeros(len(data), dtype=float)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(features, target, groups), start=1):
        # 4-1. One-Hot Encoder는 학습 Fold에만 적합해 검증 Fold 정보 누수를 막는다.
        preprocessor = ColumnTransformer(
            [
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("numeric", "passthrough", numeric),
            ],
            sparse_threshold=0.3,
        )
        x_train = preprocessor.fit_transform(features.iloc[train_index])
        x_test = preprocessor.transform(features.iloc[test_index])
        y_train = target[train_index]
        y_test = target[test_index]

        # 4-2. 불균형 이탈 클래스를 고려한 랜덤포레스트를 학습한다.
        model = RandomForestClassifier(
            n_estimators=300,
            criterion="entropy",
            max_depth=18,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features=0.7,
            bootstrap=True,
            max_samples=0.8,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)

        # 4-3. 현재 검증 Fold의 확률을 전체 OOF 배열의 원래 행 위치에 저장한다.
        fold_probability = model.predict_proba(x_test)[:, 1]
        probabilities[test_index] = fold_probability

        row: dict[str, float | int] = {
            "fold": fold,
            "train_rows": len(train_index),
            "test_rows": len(test_index),
            "test_target_rate": float(y_test.mean()),
        }
        row.update(calculate_metrics(y_test, fold_probability))
        fold_rows.append(row)
        print(
            f"Fold {fold} 완료: PR-AUC={row['pr_auc']:.4f}, "
            f"Recall@Top-20%={row['recall_at_top_20pct']:.4f}"
        )

    # 5. 모든 Fold의 OOF 확률을 합쳐 통합 성능을 계산하고 결과 파일을 저장한다.
    oof = data[["code_module", "code_presentation", ID_COL, "prediction_week", TARGET_COL]].copy()
    oof["randomforest_oof_probability"] = probabilities
    overall: dict[str, str | float | int] = {
        "model": "RandomForest (확장 Feature + 세부 VLE Feature)",
        "rows": len(data),
        "target_count": int(target.sum()),
        "target_rate": float(target.mean()),
        "feature_count_before_onehot": int(features.shape[1]),
        "categorical_feature_count": len(categorical),
    }
    overall.update(calculate_metrics(target, probabilities))

    pd.DataFrame([overall]).to_csv(
        OUTPUT_DIR / "randomforest_weekly_next_week_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_rows).to_csv(
        OUTPUT_DIR / "randomforest_weekly_next_week_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    oof.to_csv(
        OUTPUT_DIR / "randomforest_weekly_next_week_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    print("\n=== 랜덤포레스트 교차검증 완료 ===")
    print(pd.DataFrame([overall]).to_string(index=False))


if __name__ == "__main__":
    main()

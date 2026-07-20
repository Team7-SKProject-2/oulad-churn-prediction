"""VLE 누적 Feature 기반 CatBoost 다음 주 이탈 예측 후보 모델을 검증한다.

CatBoost는 code_module, code_presentation 등 범주형 변수를 One-Hot Encoding
없이 직접 처리한다. 동일 학생의 모든 주차 행은 같은 Fold에 둔다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold


OUTPUT_DIR = Path(__file__).resolve().parent / "demo_1"
DATA_PATH = OUTPUT_DIR / "used_data" / "weekly_next_week_with_vle_enhanced.csv"
TARGET_COL = "target_next_week_withdrawn"
ID_COL = "id_student"
N_SPLITS = 3
TOP_FRACTION = 0.20


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    """확률 구간별 실제 이탈비율과 예측확률의 차이(ECE)를 계산한다."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability <= upper) if upper == 1 else (
            (probability >= lower) & (probability < upper)
        )
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(ece)


def recall_at_top_fraction(y_true: np.ndarray, probability: np.ndarray, fraction: float = TOP_FRACTION) -> float:
    """상위 위험군 비율 안에 포함된 실제 이탈자의 비율을 반환한다."""
    if y_true.sum() == 0:
        return np.nan
    top_k = max(1, int(np.ceil(len(y_true) * fraction)))
    top_index = np.argsort(probability)[-top_k:]
    return float(y_true[top_index].sum() / y_true.sum())


def calculate_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "recall_at_top_20pct": recall_at_top_fraction(y_true, probability),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "ece_10bin": expected_calibration_error(y_true, probability),
    }


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """식별자·정답을 제외하고 CatBoost 입력 및 범주형 열 목록을 만든다."""
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


def main() -> None:
    # 1. 데이터 로드 및 누수·복합키 중복 검증
    data = pd.read_csv(DATA_PATH)
    forbidden = [
        column for column in data.columns
        if any(term in column.lower() for term in ["final_result", "unregistration", "withdraw_week"])
    ]
    if forbidden:
        raise ValueError(f"누수 가능 변수가 포함되어 있습니다: {forbidden}")
    if data.duplicated(["code_module", "code_presentation", ID_COL, "prediction_week"]).any():
        raise ValueError("학생·과목·운영회차·주차 복합키 중복이 있습니다.")

    # 2. Target·그룹·Feature 분리: id_student는 분할에만 사용하고 모델 입력에서는 제외
    target = data[TARGET_COL].astype(int).to_numpy()
    groups = data[ID_COL].to_numpy()
    features, categorical = prepare_features(data)
    # 3. Train/Test 분할: 동일 학생의 여러 주차 행이 서로 다른 Fold로 섞이지 않게 처리
    splitter = GroupKFold(n_splits=N_SPLITS)
    probabilities = np.zeros(len(data), dtype=float)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(features, target, groups), start=1):
        # 4-1. 모델 생성: CatBoost는 범주형 Feature를 One-Hot Encoding 없이 직접 처리한다.
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
        # 4-2. 모델 학습: 현재 Fold의 검증 데이터 PR-AUC를 기준으로 최적 반복 수를 선택한다.
        model.fit(
            features.iloc[train_index],
            target[train_index],
            cat_features=categorical,
            eval_set=(features.iloc[test_index], target[test_index]),
            use_best_model=True,
            verbose=False,
        )
        # 4-3. 검증 Fold의 다음 주 이탈확률을 예측해 OOF 배열의 원래 행 위치에 저장한다.
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
        print(f"Fold {fold} 완료: PR-AUC={row['pr_auc']:.4f}, Recall@Top-20%={row['recall_at_top_20pct']:.4f}")

    # 5. 모든 Fold의 OOF 확률로 최종 평가지표를 계산하고 재현 가능한 결과 파일을 저장한다.
    oof = data[["code_module", "code_presentation", ID_COL, "prediction_week", TARGET_COL]].copy()
    oof["catboost_oof_probability"] = probabilities
    overall = {
        "model": "CatBoost (확장 Feature + 세부 VLE Feature)",
        "rows": len(data),
        "target_count": int(target.sum()),
        "target_rate": float(target.mean()),
        "feature_count": int(features.shape[1]),
        "categorical_feature_count": len(categorical),
    }
    overall.update(calculate_metrics(target, probabilities))

    pd.DataFrame([overall]).to_csv(
        OUTPUT_DIR / "catboost_weekly_next_week_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_rows).to_csv(
        OUTPUT_DIR / "catboost_weekly_next_week_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    oof.to_csv(
        OUTPUT_DIR / "catboost_weekly_next_week_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    print("\n=== CatBoost 교차검증 완료 ===")
    print(pd.DataFrame([overall]).to_string(index=False))


if __name__ == "__main__":
    main()

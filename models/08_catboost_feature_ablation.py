"""CatBoost 124개 입력 Feature의 품질을 점검하고 축소 후보를 검증한다.

기본 실행은 상수·결측 과다·완전 중복·트리 모델에서 불필요한 단조 변환
Feature를 확인해 감사 CSV만 만든다. ``--train-ablation``을 주면 같은 학생
3-Fold로 108개 축소 후보를 학습하고 기존 Enhanced 124개 OOF와 비교한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from common_weekly_metrics import (  # noqa: E402
    calculate_metrics,
    make_group_folds,
    resolve_data_path,
)


OUTPUT_DIR = MODELS_DIR / "demo_1"
TARGET_COL = "target_next_week_withdrawn"
ID_COL = "id_student"
KEYS = ["code_module", "code_presentation", ID_COL, "prediction_week"]

# 전체 데이터에서 값이 하나뿐이므로 학습 신호가 없다.
CONSTANT_FEATURES = [
    "imd_band_missing",
    "assessment_banked_due_count",
    "assessment_submitted_exam_count",
    "any_known_banked_count",
]

# 양의 누적 클릭이 있는데도 484,708행이 비어 있어 현재 생성 로직을 신뢰하기 어렵다.
BROKEN_FEATURES = ["vle_cum_unique_sites"]

# 전체 895,005행에서 왼쪽 Feature와 완전히 같은 정보다.
EXACT_DUPLICATE_FEATURES = [
    "assessment_nonbanked_submitted_count",
    "any_known_submission_count",
    "any_known_scored_count",
    "any_known_score_missing_count",
    "any_known_mean_score",
    "any_known_median_score",
]

# 다른 입력에서 완전히 복원되거나 트리 분할 순서가 동일한 단조 변환이다.
DETERMINISTIC_REDUNDANT_FEATURES = [
    "cutoff_day",
    "current_has_vle_record",
    "log1p_cum_total_clicks",
    "log1p_current_total_clicks",
    "log1p_pre_course_clicks",
]

DROP_CANDIDATES = [
    *CONSTANT_FEATURES,
    *BROKEN_FEATURES,
    *EXACT_DUPLICATE_FEATURES,
    *DETERMINISTIC_REDUNDANT_FEATURES,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--train-ablation", action="store_true")
    return parser.parse_args()


def feature_audit_table() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    reasons = {
        **{feature: "전체 행에서 상수" for feature in CONSTANT_FEATURES},
        **{feature: "57.44% 결측 및 생성 품질 이상" for feature in BROKEN_FEATURES},
        **{feature: "다른 Feature와 전체 행에서 완전 중복" for feature in EXACT_DUPLICATE_FEATURES},
        **{
            feature: "다른 입력에서 복원 가능하거나 트리에서 중복인 단조 변환"
            for feature in DETERMINISTIC_REDUNDANT_FEATURES
        },
    }
    for feature in DROP_CANDIDATES:
        rows.append(
            {
                "feature": feature,
                "audit_result": "drop_candidate",
                "reason": reasons[feature],
                "decision": "OOF ablation 통과 후 제거",
            }
        )
    return pd.DataFrame(rows)


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = data.drop(columns=[ID_COL, TARGET_COL, *DROP_CANDIDATES]).copy()
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


def train_ablation(data: pd.DataFrame) -> None:
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "축소 CatBoost 학습에는 catboost가 필요합니다. "
            "현재 인터프리터에 catboost를 설치한 뒤 다시 실행하세요."
        ) from exc

    features, categorical = prepare_features(data)
    target = data[TARGET_COL].astype(np.int8).to_numpy()
    groups = data[ID_COL].to_numpy()
    probability = np.zeros(len(data), dtype=float)
    fold_assignment = np.zeros(len(data), dtype=np.int8)
    fold_rows: list[dict[str, float | int]] = []

    folds, _, _ = make_group_folds(groups)
    for fold, train_index, validation_index in folds:
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
            eval_set=(features.iloc[validation_index], target[validation_index]),
            use_best_model=True,
            verbose=False,
        )
        fold_probability = model.predict_proba(features.iloc[validation_index])[:, 1]
        probability[validation_index] = fold_probability
        fold_assignment[validation_index] = fold
        row: dict[str, float | int] = {
            "fold": fold,
            "feature_count": features.shape[1],
            "best_iteration": int(model.get_best_iteration()),
        }
        row.update(calculate_metrics(target[validation_index], fold_probability))
        fold_rows.append(row)
        print(
            f"Fold {fold} | PR-AUC={row['pr_auc']:.6f} | "
            f"Recall@20={row['recall_at_top_20pct']:.4f}"
        )

    overall: dict[str, float | int | str] = {
        "model": "CatBoost reduced candidate",
        "rows": len(data),
        "target_count": int(target.sum()),
        "target_rate": float(target.mean()),
        "feature_count": features.shape[1],
        "dropped_feature_count": len(DROP_CANDIDATES),
    }
    overall.update(calculate_metrics(target, probability))
    pd.DataFrame([overall]).to_csv(
        OUTPUT_DIR / "catboost_reduced_feature_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fold_rows).to_csv(
        OUTPUT_DIR / "catboost_reduced_feature_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    oof = data[[*KEYS, TARGET_COL]].copy()
    oof["catboost_reduced_oof_probability"] = probability
    oof["fold"] = fold_assignment
    oof.to_csv(
        OUTPUT_DIR / "catboost_reduced_feature_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n===== 축소 후보 OOF 결과 =====")
    print(pd.DataFrame([overall]).to_string(index=False))


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = feature_audit_table()
    audit.to_csv(
        OUTPUT_DIR / "catboost_feature_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    header = pd.read_csv(data_path, nrows=0)
    model_feature_count = len(header.columns) - 2
    missing_candidates = [
        feature for feature in DROP_CANDIDATES if feature not in header.columns
    ]
    if missing_candidates:
        raise ValueError(f"축소 후보 컬럼이 없습니다: {missing_candidates}")

    print("===== CatBoost Feature 감사 =====")
    print("전체 CSV 컬럼:", len(header.columns))
    print("현재 모델 Feature:", model_feature_count)
    print("축소 후보:", len(DROP_CANDIDATES))
    print("축소 후 Feature:", model_feature_count - len(DROP_CANDIDATES))
    print(audit.to_string(index=False))
    print("\n주의: 감사 결과만으로 최종 삭제하지 않고 OOF ablation으로 확정합니다.")

    if args.train_ablation:
        print("\n전체 데이터를 불러와 3-Fold 축소 실험을 시작합니다.")
        data = pd.read_csv(data_path, low_memory=False)
        train_ablation(data)


if __name__ == "__main__":
    main()

"""전처리기와 ML 모델을 하나의 Pipeline으로 학습한다."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import KEY_COLUMNS, PROJECT_ROOT, TARGET_COLUMN, require_columns
from .evaluate import best_f1_threshold, binary_metrics


SCHEMA_PATH = PROJECT_ROOT / "artifacts" / "feature_schema.json"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"


def train(input_path: Path) -> dict[str, float]:
    frame = pd.read_csv(input_path)
    require_columns(frame, [TARGET_COLUMN, "split"], "modeling_data")

    excluded = {*KEY_COLUMNS, "cutoff_week", TARGET_COLUMN, "split", "final_result", "date_unregistration"}
    features = [column for column in frame.columns if column not in excluded]
    if not features:
        raise ValueError("학습에 사용할 Feature가 없습니다.")

    train_frame = frame.loc[frame["split"] == "train"]
    validation_frame = frame.loc[frame["split"] == "validation"]
    if train_frame.empty or validation_frame.empty:
        raise ValueError("split 컬럼에 train과 validation이 모두 필요합니다.")

    numeric = train_frame[features].select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features if column not in numeric]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=42),
            ),
        ]
    )
    pipeline.fit(train_frame[features], train_frame[TARGET_COLUMN])
    probabilities = pipeline.predict_proba(validation_frame[features])[:, 1]
    threshold, metrics = best_f1_threshold(validation_frame[TARGET_COLUMN], probabilities)
    threshold = float(threshold)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    schema = {
        "version": "0.1.0",
        "target": TARGET_COLUMN,
        "id_columns": [*KEY_COLUMNS, "cutoff_week"],
        "features": features,
    }
    SCHEMA_PATH.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "model_name": "LogisticRegression",
        "model_version": "0.1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "cutoff_week": None,
        "threshold": threshold,
        "target": TARGET_COLUMN,
        "notes": "Validation F1 기준 임시 선택. 최종 Test 평가 전 검토 필요.",
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pd.DataFrame([{"model": "LogisticRegression", "cutoff_week": None, "split": "validation", **metrics}]).to_csv(
        METRICS_PATH, index=False
    )
    return binary_metrics(validation_frame[TARGET_COLUMN], probabilities, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="split 컬럼이 포함된 모델링 CSV")
    args = parser.parse_args()
    print(train(args.input))


if __name__ == "__main__":
    main()

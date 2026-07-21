"""1~10주차 전용 모델 공통 학습 계약을 가짜 데이터로 검증한다."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from models.common_weekly_metrics import PreparedWeeklyData
from models.early_final_artifact_common import (
    FinalArtifactConfig,
    resolve_decision_threshold,
    save_final_artifact,
)
from models.early_weekly_common import (
    EarlyModelConfig,
    finalize_early_oof,
    subset_prepared_data,
)


class _FakeProbabilityModel:
    def predict_proba(self, features):
        probability = np.full(len(features), 0.25, dtype=float)
        return np.column_stack([1 - probability, probability])


class EarlyModelTrainingTests(unittest.TestCase):
    def _prepared(self) -> PreparedWeeklyData:
        rows = []
        for index, week in enumerate([1, 2, 3, 10, 11, 12, 1, 5]):
            rows.append(
                {
                    "code_module": "AAA",
                    "code_presentation": "2014J",
                    "id_student": 100 + index,
                    "prediction_week": week,
                    "target_next_week_withdrawn": int(index in {1, 4, 7}),
                    "feature": float(index),
                }
            )
        data = pd.DataFrame(rows)
        features = data[["code_module", "code_presentation", "prediction_week", "feature"]]
        target = data["target_next_week_withdrawn"].to_numpy(dtype=np.int8)
        groups = data["id_student"].to_numpy()
        return PreparedWeeklyData(
            data=data,
            features=features,
            target=target,
            groups=groups,
            categorical=["code_module", "code_presentation"],
            numeric=["prediction_week", "feature"],
            profile={
                "rows": len(data),
                "feature_count": features.shape[1],
                "categorical_feature_count": 2,
                "numeric_feature_count": 2,
            },
        )

    def test_subset_is_applied_before_model_validation(self) -> None:
        early = subset_prepared_data(self._prepared(), 1, 10)
        self.assertEqual(early.profile["source_rows"], 8)
        self.assertEqual(early.profile["rows"], 6)
        self.assertEqual(early.profile["target_count"], 2)
        self.assertTrue(early.data["prediction_week"].between(1, 10).all())
        self.assertEqual(len(early.data), len(early.features))
        self.assertEqual(len(early.data), len(early.target))

    def test_oof_threshold_and_reproducibility_files_are_saved(self) -> None:
        early = subset_prepared_data(self._prepared(), 1, 10)
        probability = np.asarray([0.1, 0.8, 0.3, 0.7, 0.2, 0.6])
        assignment = np.asarray([1, 2, 3, 1, 2, 3], dtype=np.int8)
        config = EarlyModelConfig(
            model_name="Fake Early Model",
            file_prefix="fake_early",
            probability_column="fake_early_oof_probability",
            hyperparameters={"depth": 3},
            probability_interpretation="가짜 predict_proba 양성 확률",
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = finalize_early_oof(
                config=config,
                prepared=early,
                probabilities=probability,
                fold_assignment=assignment,
                fold_hash="fake-fold-hash",
                fold_rows=[{"fold": 1}, {"fold": 2}, {"fold": 3}],
                data_path=Path(temporary) / "fake.csv",
                output_dir=Path(temporary) / "model_outputs",
                eval_params_dir=Path(temporary) / "eval_params",
                eval_results_dir=Path(temporary) / "eval_results",
                auc_graphs_dir=Path(temporary) / "auc_grahps",
            )
            self.assertAlmostEqual(
                float(result["metrics"].iloc[0]["threshold"]), 0.6
            )
            self.assertTrue(
                np.isclose(result["threshold_metrics"]["threshold"], 0.5).any()
            )
            for path in result["files"].values():
                self.assertTrue(Path(path).is_file(), path)
            summary = json.loads(
                Path(result["files"]["summary_json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["training_data"]["training_scope"], "early_weeks_only")
            self.assertEqual(summary["model"]["hyperparameters"]["depth"], 3)
            self.assertIn("id_student", summary["documentation"]["validation_ko"])

            params = json.loads(
                Path(result["files"]["eval_params_json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(params["hyperparameters"]["depth"], 3)
            self.assertEqual(params["validation"]["method"], "GroupKFold")

            results = json.loads(
                Path(result["files"]["eval_results_json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(results["confusion_matrix"]["layout"], "[[TN, FP], [FN, TP]]")
            self.assertEqual(results["metrics"]["f1_score"], 0.8)

            curves = pd.read_parquet(result["files"]["auc_curves_parquet"])
            self.assertEqual(set(curves["curve_type"]), {"roc", "precision_recall"})
            self.assertTrue(
                {
                    "x",
                    "y",
                    "threshold",
                    "score_name",
                    "score_value",
                    "selected_threshold",
                }.issubset(curves.columns)
            )

    def test_final_joblib_and_cohort_csv_follow_existing_artifact_contract(self) -> None:
        early = subset_prepared_data(self._prepared(), 1, 10)
        early.profile.update(
            {
                "training_start_week": 1,
                "training_end_week": 12,
                "operating_start_week": 1,
                "operating_end_week": 10,
                "smoke_test": False,
            }
        )
        config = FinalArtifactConfig(
            model_name="Fake Early Model",
            artifact_filename="fake_early.joblib",
            profiles_filename="fake_early_cohort_profiles.csv",
            threshold_results_path=Path("fake_results.json"),
            probability_column="fake_probability",
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = save_final_artifact(
                config=config,
                model=_FakeProbabilityModel(),
                prepared=early,
                data_path=Path(temporary) / "fake.csv",
                artifact_dir=Path(temporary) / "artifacts",
                threshold=0.6,
                threshold_source="unit_test",
                training_parameters={"depth": 3},
                categorical_features=early.categorical,
                preprocessing={"type": "fake"},
            )
            artifact = joblib.load(paths["joblib"])
            required = {
                "model_name",
                "model",
                "feature_columns",
                "categorical_features",
                "target_column",
                "id_column",
                "threshold",
                "training_parameters",
                "trained_at",
            }
            self.assertTrue(required.issubset(artifact))
            self.assertEqual(artifact["threshold"], 0.6)
            self.assertEqual(artifact["training_scope"], "full_existing_weekly_data")
            profiles = pd.read_csv(paths["cohort_profiles_csv"])
            self.assertEqual(profiles.columns.tolist(), artifact["feature_columns"])
            self.assertFalse(
                profiles.duplicated(
                    ["code_module", "code_presentation", "prediction_week"]
                ).any()
            )

    def test_threshold_json_rejects_smoke_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "selected_threshold": 0.2,
                        "evaluation_data": {"smoke_test": True},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "스모크"):
                resolve_decision_threshold(None, result_path)
            threshold, source = resolve_decision_threshold(0.3, result_path)
            self.assertEqual(threshold, 0.3)
            self.assertEqual(source, "user_supplied_threshold")
            result_path.write_text(
                json.dumps({"selected": {"threshold": 0.25}}), encoding="utf-8"
            )
            threshold, source = resolve_decision_threshold(None, result_path)
            self.assertEqual(threshold, 0.25)
            self.assertEqual(source, str(result_path.resolve()))


if __name__ == "__main__":
    unittest.main()

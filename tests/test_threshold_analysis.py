import argparse
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analyze_thresholds import reconstruct_group_folds, run_analysis, verify_oof_source
from src.compare_threshold_results import compare_result_files, save_comparison
from src.evaluate import (
    compare_thresholds,
    generate_thresholds,
    select_best_threshold,
    select_thresholds,
    threshold_metrics,
)


class ThresholdMetricsTest(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 0, 1, 1])
        self.probabilities = np.array([0.1, 0.6, 0.4, 0.9])

    def test_confusion_matrix_and_metrics(self):
        result = threshold_metrics(self.labels, self.probabilities, threshold=0.5)
        self.assertEqual((result["TP"], result["FP"], result["TN"], result["FN"]), (1, 1, 1, 1))
        self.assertEqual(result["predicted_positive_count"], 2)
        self.assertAlmostEqual(result["predicted_positive_ratio"], 0.5)
        for metric in ("accuracy", "precision", "recall", "specificity", "f1_score"):
            self.assertAlmostEqual(result[metric], 0.5)
        self.assertAlmostEqual(result["roc_auc"], 0.75)

    def test_zero_division_is_zero(self):
        result = threshold_metrics(self.labels, self.probabilities, threshold=0.95)
        self.assertEqual(result["predicted_positive_count"], 0)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1_score"], 0.0)

    def test_default_threshold_grid_and_custom_grid(self):
        defaults = generate_thresholds()
        np.testing.assert_allclose(defaults, np.arange(0.05, 1.0, 0.05))
        custom = generate_thresholds(0.1, 0.3, 0.1)
        np.testing.assert_allclose(custom, [0.1, 0.2, 0.3, 0.5])

    def test_positive_count_is_monotonic(self):
        table = compare_thresholds(self.labels, self.probabilities, generate_thresholds())
        self.assertTrue((table["predicted_positive_count"].diff().fillna(0) <= 0).all())
        self.assertEqual(table["roc_auc"].nunique(), 1)
        self.assertEqual(table["pr_auc"].nunique(), 1)

    def test_tie_break_prefers_closest_to_half_then_higher(self):
        table = pd.DataFrame(
            {
                "threshold": [0.3, 0.4, 0.6, 0.7],
                "score": [1.0, 1.0, 1.0, 1.0],
            }
        )
        selected = select_best_threshold(table, "score")
        self.assertAlmostEqual(selected["threshold"], 0.6)

    def test_constrained_selections_respect_required_metric(self):
        table = compare_thresholds(self.labels, self.probabilities, generate_thresholds())
        selected = select_thresholds(table, min_recall=0.5, min_precision=0.5)
        recall_constrained = selected["best_precision_at_min_recall"]
        precision_constrained = selected["best_recall_at_min_precision"]
        self.assertGreaterEqual(recall_constrained["recall"], 0.5)
        self.assertGreaterEqual(precision_constrained["precision"], 0.5)

    def test_standard_selections_include_best_recall(self):
        table = compare_thresholds(self.labels, self.probabilities, generate_thresholds())
        selected = select_thresholds(table)
        self.assertEqual(selected["best_recall"]["recall"], table["recall"].max())
        self.assertIsInstance(selected["best_f1"]["TP"], int)
        self.assertIsInstance(selected["default_0_5"]["FN"], int)


class ThresholdAnalysisIntegrationTest(unittest.TestCase):
    def test_joblib_validation_analysis_creates_all_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train = pd.DataFrame(
                {
                    "feature_a": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                    "target": [0, 0, 0, 1, 1, 1],
                }
            )
            model = Pipeline(
                [("scale", StandardScaler()), ("model", LogisticRegression(random_state=42))]
            ).fit(train[["feature_a"]], train["target"])
            model_path = root / "model.joblib"
            schema_path = root / "schema.json"
            data_path = root / "modeling.csv"
            output_dir = root / "outputs"
            joblib.dump(model, model_path)
            schema_path.write_text(
                json.dumps({"target": "target", "features": ["feature_a"]}), encoding="utf-8"
            )
            pd.DataFrame(
                {
                    "feature_a": [-1.5, -0.25, 0.25, 1.5, 10.0],
                    "target": [0, 0, 1, 1, 1],
                    "split": ["validation", "validation", "validation", "validation", "test"],
                }
            ).to_csv(data_path, index=False)

            args = argparse.Namespace(
                model_path=model_path,
                data_path=data_path,
                schema_path=schema_path,
                output_dir=output_dir,
                batch_size=2,
                threshold_min=0.05,
                threshold_max=0.95,
                threshold_step=0.05,
                min_recall=0.5,
                min_precision=0.5,
                target_column=None,
                split_column="split",
                validation_value="validation",
                probability_column=None,
                fold_column="fold",
                source_data_path=None,
                seed=42,
            )
            table, summary = run_analysis(args)

            self.assertEqual(summary["evaluation"]["validation_rows"], 4)
            self.assertEqual(summary["evaluation"]["inference_method"], "predict_proba")
            self.assertIn(0.5, table["threshold"].tolist())
            for filename in (
                "threshold_metrics.csv",
                "selected_thresholds.json",
                "threshold_metrics.png",
                "precision_recall_curve.png",
                "roc_curve_annotated.png",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)
            self.assertIn("documentation", summary)
            self.assertIn("notes_ko", summary["documentation"])
            self.assertEqual(
                summary["documentation"]["prediction_rule"],
                "pred = (positive_probability >= threshold).astype(int)",
            )

    def test_oof_analysis_verifies_source_and_does_not_need_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.csv"
            oof_path = root / "oof.csv"
            output_dir = root / "outputs"
            common = {
                "code_module": ["AAA"] * 4,
                "code_presentation": ["2013J"] * 4,
                "id_student": [1, 2, 3, 4],
                "prediction_week": [1, 1, 1, 1],
                "target_next_week_withdrawn": [0, 0, 1, 1],
            }
            pd.DataFrame({**common, "feature_a": [1, 2, 3, 4]}).to_csv(source_path, index=False)
            pd.DataFrame(
                {
                    **common,
                    "model_oof_probability": [0.1, 0.6, 0.4, 0.9],
                }
            ).to_csv(oof_path, index=False)

            verification = verify_oof_source(
                source_path, oof_path, "target_next_week_withdrawn"
            )
            self.assertTrue(verification["key_and_target_match"])
            args = argparse.Namespace(
                model_path=root / "missing.joblib",
                data_path=oof_path,
                schema_path=root / "missing-schema.json",
                output_dir=output_dir,
                batch_size=2,
                threshold_min=0.05,
                threshold_max=0.95,
                threshold_step=0.05,
                min_recall=None,
                min_precision=None,
                target_column="target_next_week_withdrawn",
                split_column="split",
                validation_value="validation",
                probability_column="model_oof_probability",
                fold_column="fold",
                group_column="id_student",
                n_splits=2,
                source_data_path=source_path,
                seed=42,
            )
            _, summary = run_analysis(args)
            self.assertEqual(summary["evaluation"]["source_mode"], "precomputed_oof_probabilities")
            self.assertIsNone(summary["evaluation"]["model_path"])
            self.assertTrue(summary["evaluation"]["source_verification"]["key_and_target_match"])
            self.assertEqual(summary["evaluation"]["fold_count"], 2)
            self.assertIn("reconstructed", summary["evaluation"]["fold_assignment"])

            fold_metadata = reconstruct_group_folds(oof_path, n_splits=2)
            self.assertEqual(sum(fold_metadata["fold_rows"].values()), 4)

    def test_model_result_comparison_ranks_f1_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result_paths = []
            for model, f1_value in (("model_a", 0.2), ("model_b", 0.3)):
                path = root / f"{model}.json"
                metrics = {
                    "threshold": 0.5,
                    "accuracy": 0.8,
                    "precision": 0.3,
                    "recall": 0.4,
                    "specificity": 0.9,
                    "f1_score": f1_value,
                    "roc_auc": 0.7,
                    "pr_auc": 0.2,
                    "TP": 4,
                    "FP": 2,
                    "TN": 8,
                    "FN": 6,
                    "predicted_positive_count": 6,
                    "predicted_positive_ratio": 0.3,
                }
                path.write_text(
                    json.dumps(
                        {
                            "selected": {"best_f1": metrics},
                            "threshold_grid": {
                                "requested_min": 0.1,
                                "requested_max": 0.9,
                                "requested_step": 0.1,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                result_paths.append(f"{model}={path}")
            table = compare_result_files(result_paths)
            self.assertEqual(table.iloc[0]["model"], "model_b")
            csv_path, json_path, plot_path = save_comparison(table, root / "comparison")
            self.assertTrue(csv_path.is_file())
            self.assertTrue(json_path.is_file())
            self.assertTrue(plot_path.is_file())


if __name__ == "__main__":
    unittest.main()

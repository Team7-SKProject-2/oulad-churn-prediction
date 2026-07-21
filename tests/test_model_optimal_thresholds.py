import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.compare_model_optimal_thresholds import (
    ModelSpec,
    exact_f1_optimal_threshold,
    load_aligned_oof,
    model_slug,
    parse_model_spec,
)


class ModelOptimalThresholdTests(unittest.TestCase):
    def test_exact_f1_threshold_checks_every_unique_probability(self):
        labels = np.asarray([1, 0, 1, 0])
        probabilities = np.asarray([0.9, 0.8, 0.7, 0.1])

        threshold, f1_score, candidate_count = exact_f1_optimal_threshold(
            labels, probabilities
        )

        self.assertAlmostEqual(threshold, 0.7)
        self.assertAlmostEqual(f1_score, 0.8)
        self.assertEqual(candidate_count, 4)

    def test_model_spec_parser_supports_windows_paths(self):
        spec = parse_model_spec(
            r"XGBoost|C:\project\outputs\oof.csv|positive_probability"
        )

        self.assertEqual(spec.name, "XGBoost")
        self.assertEqual(spec.path, Path(r"C:\project\outputs\oof.csv"))
        self.assertEqual(spec.probability_column, "positive_probability")

    def test_model_slug_is_safe_for_output_directories(self):
        self.assertEqual(model_slug("XGBoost weighted"), "xgboost")
        self.assertEqual(model_slug("CatBoost"), "catboost")
        self.assertEqual(model_slug("ElasticNet"), "elasticnet")

    def test_oof_files_are_aligned_by_composite_key_not_row_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_path = root / "first.csv"
            second_path = root / "second.csv"
            common = pd.DataFrame(
                {
                    "code_module": ["AAA", "AAA", "BBB", "BBB"],
                    "code_presentation": ["2013J"] * 4,
                    "id_student": [1, 2, 3, 4],
                    "prediction_week": [1, 1, 2, 2],
                    "target_next_week_withdrawn": [0, 1, 0, 1],
                }
            )
            common.assign(model_a_probability=[0.1, 0.8, 0.2, 0.9]).to_csv(
                first_path, index=False
            )
            common.assign(model_b_probability=[0.2, 0.7, 0.3, 0.8]).iloc[::-1].to_csv(
                second_path, index=False
            )

            labels, probabilities, _ = load_aligned_oof(
                [
                    ModelSpec("A", first_path, "model_a_probability"),
                    ModelSpec("B", second_path, "model_b_probability"),
                ],
                "target_next_week_withdrawn",
            )

            np.testing.assert_array_equal(labels, np.asarray([0, 1, 0, 1]))
            np.testing.assert_allclose(probabilities["A"], [0.1, 0.8, 0.2, 0.9])
            np.testing.assert_allclose(probabilities["B"], [0.2, 0.7, 0.3, 0.8])


if __name__ == "__main__":
    unittest.main()

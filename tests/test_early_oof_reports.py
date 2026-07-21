import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.early_catboost_threshold_report import CONFIG as CATBOOST_CONFIG
from src.early_elasticnet_threshold_report import CONFIG as ELASTICNET_CONFIG
from src.early_oof_report import evaluate_early_oof_model, save_early_report
from src.early_randomforest_threshold_report import CONFIG as RANDOMFOREST_CONFIG
from src.early_xgboost_threshold_report import CONFIG as XGBOOST_CONFIG


class EarlyOOFReportTests(unittest.TestCase):
    def test_filters_weeks_and_saves_complete_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            oof_path = root / "catboost.csv"
            output_dir = root / "output"
            pd.DataFrame(
                {
                    "prediction_week": [1, 2, 3, 10, 11, 12],
                    "target_next_week_withdrawn": [0, 1, 0, 1, 1, 0],
                    "catboost_oof_probability": [0.1, 0.8, 0.7, 0.6, 0.9, 0.2],
                }
            ).to_csv(oof_path, index=False)

            report = evaluate_early_oof_model(CATBOOST_CONFIG, oof_path)
            self.assertEqual(report.source["source_rows"], 6)
            self.assertEqual(report.source["rows"], 4)
            self.assertEqual(report.source["positive_count"], 2)
            self.assertAlmostEqual(report.metrics["threshold"], 0.6)
            self.assertAlmostEqual(report.metrics["f1_score"], 0.8)
            self.assertTrue(np.isclose(report.threshold_frame["threshold"], 0.5).any())
            counts = report.threshold_frame["predicted_positive_count"].to_numpy()
            self.assertTrue((np.diff(counts) <= 0).all())

            files = save_early_report(
                report,
                output_dir,
                Path(__file__),
                "evaluate_early_catboost",
            )
            self.assertTrue(all(Path(path).is_file() for path in files.values()))
            payload = json.loads(
                (output_dir / "early_catboost_optimal_f1_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["source"]["start_week"], 1)
            self.assertEqual(payload["source"]["end_week"], 10)
            self.assertTrue(payload["threshold_grid"]["includes_default_0_5"])

    def test_requested_output_directories_are_stable(self):
        self.assertEqual(CATBOOST_CONFIG.output_dir.name, "early_catboot")
        self.assertEqual(ELASTICNET_CONFIG.output_dir.name, "early_elasticNet")
        self.assertEqual(RANDOMFOREST_CONFIG.output_dir.name, "early_randomforest")
        self.assertEqual(XGBOOST_CONFIG.output_dir.name, "early_xgboost")


if __name__ == "__main__":
    unittest.main()

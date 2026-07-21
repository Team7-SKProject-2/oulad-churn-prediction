import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.elasticnet_threshold_report import evaluate_elasticnet
from src.oof_streamlit_report import save_oof_report
from src.xgboost_threshold_report import evaluate_xgboost


class OOFStreamlitReportTests(unittest.TestCase):
    def _write_oof(self, path: Path, probability_column: str) -> None:
        pd.DataFrame(
            {
                "target_next_week_withdrawn": [0, 1, 0, 1],
                probability_column: [0.1, 0.8, 0.7, 0.6],
            }
        ).to_csv(path, index=False)

    def test_xgboost_report_matches_streamlit_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "xgboost.csv"
            self._write_oof(path, "xgboost_scaled_oof_probability")
            report = evaluate_xgboost(path)
            self.assertEqual(report.config.model_name, "XGBoost weighted")
            self.assertAlmostEqual(report.metrics["threshold"], 0.6)
            self.assertAlmostEqual(report.metrics["f1_score"], 0.8)
            self.assertEqual(report.display_table.columns.tolist(), ["평가 지표", "값"])

    def test_elasticnet_report_and_saved_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "elasticnet.csv"
            output_dir = root / "output"
            self._write_oof(path, "elasticnet_logistic_oof_probability")
            report = evaluate_elasticnet(path)
            files = save_oof_report(
                report,
                output_dir,
                Path(__file__),
                "evaluate_elasticnet",
            )
            self.assertTrue(all(Path(value).is_file() for value in files.values()))
            payload = json.loads(
                (output_dir / "elasticnet_optimal_f1_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["source"]["model"], "ElasticNet")
            self.assertAlmostEqual(payload["selected"]["f1_score"], 0.8)


if __name__ == "__main__":
    unittest.main()

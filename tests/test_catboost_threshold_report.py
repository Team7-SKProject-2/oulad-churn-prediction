import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from src.catboost_threshold_report import (
    create_roc_figure,
    evaluate_catboost,
    save_report,
)


class CatBoostThresholdReportTests(unittest.TestCase):
    def _write_oof(self, path: Path) -> None:
        pd.DataFrame(
            {
                "target_next_week_withdrawn": [0, 1, 0, 1],
                "catboost_oof_probability": [0.1, 0.8, 0.7, 0.6],
            }
        ).to_csv(path, index=False)

    def test_report_uses_exact_f1_threshold_and_streamlit_objects(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            oof_path = Path(temporary_directory) / "catboost_oof.csv"
            self._write_oof(oof_path)

            report = evaluate_catboost(oof_path)

            self.assertAlmostEqual(report.metrics["threshold"], 0.6)
            self.assertAlmostEqual(report.metrics["f1_score"], 0.8)
            self.assertEqual(report.metrics_frame.iloc[0]["search_mode"], "all_unique_oof_probabilities")
            self.assertEqual(report.display_table.columns.tolist(), ["평가 지표", "값"])
            figure = create_roc_figure(report)
            self.assertEqual(len(figure.axes), 1)
            plt.close(figure)

    def test_save_report_creates_connected_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            oof_path = root / "catboost_oof.csv"
            output_dir = root / "output"
            self._write_oof(oof_path)
            report = evaluate_catboost(oof_path)

            files = save_report(report, output_dir)

            self.assertTrue(all(Path(path).is_file() for path in files.values()))
            payload = json.loads(
                (output_dir / "catboost_optimal_f1_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["source"]["positive_label"], 1)
            self.assertAlmostEqual(payload["selected"]["f1_score"], 0.8)


if __name__ == "__main__":
    unittest.main()

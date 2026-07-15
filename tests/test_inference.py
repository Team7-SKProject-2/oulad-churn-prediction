import unittest

import pandas as pd

from src.predict import risk_level, validate_features
from src.features import build_vle_snapshots


class InferenceTest(unittest.TestCase):
    def test_risk_level_boundaries(self):
        self.assertEqual(risk_level(0.39), "저위험")
        self.assertEqual(risk_level(0.40), "중위험")
        self.assertEqual(risk_level(0.69), "중위험")
        self.assertEqual(risk_level(0.70), "고위험")

    def test_validate_features_accepts_complete_frame(self):
        frame = pd.DataFrame({"feature_a": [1], "feature_b": [2]})
        validate_features(frame, ["feature_a", "feature_b"])

    def test_validate_features_rejects_missing_columns(self):
        frame = pd.DataFrame({"feature_a": [1]})
        with self.assertRaisesRegex(ValueError, "feature_b"):
            validate_features(frame, ["feature_a", "feature_b"])

    def test_vle_snapshots_include_inactive_students(self):
        daily = pd.DataFrame(
            {
                "code_module": ["AAA", "AAA"],
                "code_presentation": ["2013J", "2013J"],
                "id_student": [1, 1],
                "date": [0, 7],
                "activity_type": ["resource", "resource"],
                "sum_click": [3, 5],
            }
        )
        cohort = pd.DataFrame(
            {
                "code_module": ["AAA", "AAA"],
                "code_presentation": ["2013J", "2013J"],
                "id_student": [1, 2],
            }
        )
        result = build_vle_snapshots(daily, cohort)
        self.assertEqual(len(result), 6)
        week_two = result.loc[
            (result["id_student"] == 1) & (result["cutoff_week"] == 2)
        ].iloc[0]
        self.assertEqual(week_two["cumulative_clicks"], 8)
        inactive = result.loc[
            (result["id_student"] == 2) & (result["cutoff_week"] == 1)
        ].iloc[0]
        self.assertEqual(inactive["cumulative_clicks"], 0)
        self.assertEqual(inactive["recent_activity_gap"], 7)


if __name__ == "__main__":
    unittest.main()

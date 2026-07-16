import unittest

import pandas as pd

from src.check_preprocessing_handoff import validate_handoff_snapshot


class PreprocessingHandoffTest(unittest.TestCase):
    def make_snapshot(self):
        return pd.DataFrame(
            {
                "code_module": ["AAA", "AAA"],
                "code_presentation": ["2013J", "2013J"],
                "id_student": [1, 2],
                "cutoff_week": [1, 1],
                "target": [0, 1],
                "cum_total_clicks": [10, 2],
                "current_total_clicks": [10, 2],
                "current_no_activity": [0, 0],
                "weeks_since_last_activity": [0, 0],
                "assessment_missing_due_rate": [0.0, 0.0],
                "assessment_late_rate": [0.0, 0.0],
                "any_known_mean_score": [0.0, 0.0],
            }
        )

    def test_valid_snapshot(self):
        validate_handoff_snapshot(self.make_snapshot(), cutoff_week=1)

    def test_rejects_full_cohort_distribution_feature(self):
        frame = self.make_snapshot()
        frame["cum_total_clicks_course_percentile"] = [0.5, 1.0]
        with self.assertRaisesRegex(ValueError, "누수 위험"):
            validate_handoff_snapshot(frame, cutoff_week=1)

    def test_rejects_validation_split_column(self):
        frame = self.make_snapshot()
        frame["split"] = ["train", "validation"]
        with self.assertRaisesRegex(ValueError, "누수 위험"):
            validate_handoff_snapshot(frame, cutoff_week=1)


if __name__ == "__main__":
    unittest.main()

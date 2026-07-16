import unittest

import pandas as pd

from src.assessment_features import (
    build_assessment_features,
    prepare_assessment_events,
)


class AssessmentFeatureTest(unittest.TestCase):
    def test_assessment_features_exclude_submissions_after_cutoff(self):
        assessments = pd.DataFrame(
            {
                "code_module": ["AAA", "AAA"],
                "code_presentation": ["2013J", "2013J"],
                "id_assessment": [10, 20],
                "assessment_type": ["TMA", "CMA"],
                "date": [5.0, 15.0],
                "weight": [10.0, 20.0],
            }
        )
        student_assessment = pd.DataFrame(
            {
                "id_assessment": [10, 20, 10],
                "id_student": [1, 1, 2],
                "date_submitted": [4, 5, 8],
                "is_banked": [0, 0, 0],
                "score": [80.0, 90.0, 70.0],
            }
        )
        snapshot = pd.DataFrame(
            {
                "code_module": ["AAA", "AAA"],
                "code_presentation": ["2013J", "2013J"],
                "id_student": [1, 2],
                "target": [0, 1],
            }
        )
        events = prepare_assessment_events(assessments, student_assessment)
        final, _ = build_assessment_features(
            snapshot,
            cutoff_week=1,
            assessments=assessments,
            assessment_events=events,
        )

        student_one = final.loc[final["id_student"].eq(1)].iloc[0]
        student_two = final.loc[final["id_student"].eq(2)].iloc[0]
        self.assertEqual(student_one["assessment_submitted_due_count"], 1)
        self.assertEqual(student_one["any_known_submission_count"], 2)
        self.assertEqual(student_two["assessment_submitted_due_count"], 0)
        self.assertEqual(student_two["any_known_submission_count"], 0)


if __name__ == "__main__":
    unittest.main()

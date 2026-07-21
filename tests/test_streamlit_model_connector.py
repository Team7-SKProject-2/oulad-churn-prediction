"""최종 CatBoost–Streamlit 연결부 통합 테스트."""

from __future__ import annotations

import unittest

import numpy as np

from streamlit_app.lib.model import (
    cohort_profiles_ready,
    decision_threshold,
    feature_coverage,
    load_cohort_profiles,
    model_info,
    model_ready,
    predict_is_at_risk,
    predict_probabilities,
    prepare_model_input,
    required_feature_columns,
    service_week_range,
)


class StreamlitModelConnectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = load_cohort_profiles()

    def test_final_model_and_profiles_are_ready(self):
        self.assertTrue(model_ready())
        self.assertTrue(cohort_profiles_ready())
        self.assertEqual(model_info()["feature_count"], 124)
        self.assertAlmostEqual(decision_threshold(), 0.1100300614138347)
        self.assertEqual(service_week_range(), (1, 10))

    def test_profiles_follow_saved_feature_order(self):
        expected = required_feature_columns()
        self.assertEqual(expected, self.profiles.columns.tolist())
        self.assertEqual(feature_coverage(self.profiles)["missing_columns"], [])
        prepared = prepare_model_input(self.profiles.head(2))
        self.assertEqual(prepared.columns.tolist(), expected)

    def test_real_catboost_prediction_is_finite_and_uses_threshold(self):
        sample = self.profiles.head(8)
        probabilities = predict_probabilities(sample)
        labels = predict_is_at_risk(sample)
        self.assertEqual(len(probabilities), len(sample))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_array_equal(labels, probabilities >= decision_threshold())

    def test_missing_feature_is_rejected_instead_of_silently_filled(self):
        incomplete = self.profiles.head(1).drop(columns=[required_feature_columns()[-1]])
        with self.assertRaisesRegex(ValueError, "입력 Feature"):
            prepare_model_input(incomplete)


if __name__ == "__main__":
    unittest.main()

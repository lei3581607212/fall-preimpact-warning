import unittest

import numpy as np

from scripts.evaluate_paper_event_policy import evaluate_events


class PaperEventMetricTests(unittest.TestCase):
    def _data(self, normal_fps):
        return {
            "source_dataset": np.asarray(["fallset", "fallset", "normalset", "normalset"]),
            "source_path": np.asarray(["fall.mp4", "fall.mp4", "normal.mp4", "normal.mp4"]),
            "end_frame": np.asarray([7, 8, 0, 29]),
            "impact_frame": np.asarray([10, 10, -1, -1]),
            "fps": np.asarray([10.0, 10.0, normal_fps, normal_fps]),
        }

    def test_reports_median_lead_and_complete_duration_rate(self):
        result = evaluate_events(
            self._data(10.0), np.asarray([0.9, 0.1, 0.8, 0.1]), np.ones(4, dtype=bool), 0.5, 1
        )
        self.assertAlmostEqual(result["mean_lead_seconds"], 0.3)
        self.assertAlmostEqual(result["median_lead_seconds"], 0.3)
        self.assertEqual(result["normal_events_with_valid_duration"], 1)
        self.assertAlmostEqual(result["normal_duration_minutes"], 0.05)
        self.assertAlmostEqual(result["false_alert_events_per_minute"], 20.0)

    def test_missing_normal_fps_marks_rate_unavailable(self):
        result = evaluate_events(
            self._data(0.0), np.asarray([0.9, 0.1, 0.8, 0.1]), np.ones(4, dtype=bool), 0.5, 1
        )
        self.assertEqual(result["normal_events_with_valid_duration"], 0)
        self.assertIsNone(result["normal_duration_minutes"])
        self.assertIsNone(result["false_alert_events_per_minute"])

    def test_duration_sidecar_enables_rate_without_npz_fps(self):
        result = evaluate_events(
            self._data(0.0),
            np.asarray([0.9, 0.1, 0.8, 0.1]),
            np.ones(4, dtype=bool),
            0.5,
            1,
            {"normalset::normal.mp4": 30.0},
        )
        self.assertEqual(result["normal_events_with_valid_duration"], 1)
        self.assertAlmostEqual(result["normal_duration_minutes"], 0.5)
        self.assertAlmostEqual(result["false_alert_events_per_minute"], 2.0)
        self.assertEqual(result["normal_duration_basis"], "sidecar_evaluated_windows")


if __name__ == "__main__":
    unittest.main()

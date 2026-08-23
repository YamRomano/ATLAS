from __future__ import annotations

import unittest

import numpy as np

from audit_sift_reextraction import comparison_payload


class SiftReextractionAuditTest(unittest.TestCase):
    def test_exact_feature_tables_are_reproducible(self) -> None:
        features = {
            "keypoints": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            "descriptors": np.arange(256, dtype=np.uint8).reshape(2, 128),
        }
        result = comparison_payload(features, features)
        self.assertTrue(result["reproducible"])
        self.assertEqual(result["max_keypoint_delta_px"], 0.0)
        self.assertEqual(result["descriptor_values_different"], 0)

    def test_changed_descriptor_is_reported(self) -> None:
        first = {
            "keypoints": np.array([[10.0, 20.0]], dtype=np.float32),
            "descriptors": np.zeros((1, 128), dtype=np.uint8),
        }
        second = {
            "keypoints": first["keypoints"].copy(),
            "descriptors": first["descriptors"].copy(),
        }
        second["descriptors"][0, 7] = 1
        result = comparison_payload(first, second)
        self.assertFalse(result["reproducible"])
        self.assertEqual(result["descriptor_values_different"], 1)


if __name__ == "__main__":
    unittest.main()

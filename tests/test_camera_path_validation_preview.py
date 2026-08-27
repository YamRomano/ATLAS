import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT_PATH = ROOT / "scripts" / "publish_camera_path_validation_preview.py"

spec = importlib.util.spec_from_file_location("camera_path_validation_preview_test", SCRIPT_PATH)
PREVIEW = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(PREVIEW)


class CameraPathValidationPreviewTests(unittest.TestCase):
    def test_pacing_smoothing_preserves_path_endpoints_and_reduces_catchup_step(self):
        centers = [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [1.03, 0.0, 0.0],
            [2.03, 0.0, 0.0],
            [2.53, 0.0, 0.0],
        ]
        poses = [{"rcenter": list(center)} for center in centers]

        summary = PREVIEW.smooth_path_pacing(poses, (0, 6), radius=2)

        self.assertEqual(poses[0]["rcenter"], centers[0])
        self.assertEqual(poses[-1]["rcenter"], centers[-1])
        self.assertEqual(summary["adjusted_pose_count"], 5)
        self.assertLess(summary["smoothed_max_step_m"], summary["original_max_step_m"])
        self.assertGreater(poses[3]["rcenter"][0], centers[3][0])
        self.assertTrue(all(pose["preview_pacing_smoothed"] for pose in poses))
        total = sum(
            math.dist(left["rcenter"], right["rcenter"])
            for left, right in zip(poses, poses[1:])
        )
        self.assertAlmostEqual(total, 2.53)

    def test_frame_range_parser_rejects_reversed_range(self):
        with self.assertRaises(Exception):
            PREVIEW.parse_frame_range("20:10")

    def test_uniform_blend_removes_stop_then_rush_pacing(self):
        centers = [[0.0, 0.0, 0.0]]
        for step in [0.001] * 20 + [0.4] * 4 + [0.001] * 20:
            centers.append([centers[-1][0] + step, 0.0, 0.0])
        poses = [{"rcenter": list(center)} for center in centers]

        summary = PREVIEW.smooth_path_pacing(
            poses,
            (0, len(poses) - 1),
            radius=4,
            uniform_blend=0.85,
        )

        steps = [
            math.dist(left["rcenter"], right["rcenter"])
            for left, right in zip(poses, poses[1:])
        ]
        self.assertEqual(summary["uniform_blend"], 0.85)
        self.assertLess(max(steps) / min(steps), 2.5)
        self.assertEqual(poses[0]["rcenter"], centers[0])
        self.assertEqual(poses[-1]["rcenter"], centers[-1])


if __name__ == "__main__":
    unittest.main()

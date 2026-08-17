import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_camera_path_offline_replay.py"


spec = importlib.util.spec_from_file_location("camera_path_offline_replay_test", SCRIPT_PATH)
OFFLINE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(OFFLINE)


class CameraPathOfflineReplayTests(unittest.TestCase):
    @staticmethod
    def _trajectory_pose(index, center, *, time_sec=None):
        pose = {
            "success": True,
            "image_name": f"query_{index:06d}.jpg",
            "center": list(center),
            "rcenter": list(center),
            "rheading": [1.0, 0.0, 0.0],
            "time_sec": index * 0.2 if time_sec is None else time_sec,
        }
        if index == 0:
            pose["colmap_reference"] = {
                "center": list(center),
                "registered_points": 200,
            }
        return pose

    def test_short_gap_is_filled_at_video_timestamp(self):
        left = {
            "success": True,
            "image_name": "query_000000.jpg",
            "center": [0.0, 0.0, 0.0],
            "rcenter": [0.0, 0.0, 0.0],
            "rheading": [1.0, 0.0, 0.0],
        }
        right = {
            "success": True,
            "image_name": "query_000002.jpg",
            "center": [2.0, 0.0, 0.0],
            "rcenter": [2.0, 0.0, 0.0],
            "rheading": [0.0, 0.0, 1.0],
        }
        rows = [
            {"filename": f"query_{index:06d}.jpg", "time_sec": str(index * 0.2)}
            for index in range(3)
        ]
        complete, interpolated = OFFLINE.fill_short_gaps({0: left, 2: right}, rows)
        self.assertEqual(interpolated, 1)
        self.assertEqual(complete[1]["rcenter"], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(complete[1]["time_sec"], 0.2)
        self.assertTrue(complete[1]["interpolated_pose"])

    def test_absolute_recovery_drift_is_distributed_before_gap_fill(self):
        poses = {
            0: self._trajectory_pose(0, [0.0, 0.0, 0.0]),
            1: self._trajectory_pose(1, [0.1, 0.0, 0.0]),
            2: self._trajectory_pose(2, [0.2, 0.0, 0.0]),
            4: self._trajectory_pose(4, [0.4, 0.0, 0.0]),
            5: self._trajectory_pose(5, [1.0, 0.0, 0.0]),
        }
        poses[5]["colmap_reference"] = {
            "center": [1.0, 0.0, 0.0],
            "registered_points": 200,
        }
        smoothed, adjusted = OFFLINE.smooth_absolute_anchor_drift(poses)
        rows = [
            {"filename": f"query_{index:06d}.jpg", "time_sec": str(index * 0.2)}
            for index in range(6)
        ]
        complete, interpolated = OFFLINE.fill_short_gaps(smoothed, rows)

        self.assertGreater(adjusted, 0)
        self.assertEqual(interpolated, 1)
        self.assertEqual(complete[0]["rcenter"], [0.0, 0.0, 0.0])
        self.assertEqual(complete[5]["rcenter"], [1.0, 0.0, 0.0])
        steps = [
            abs(complete[index]["rcenter"][0] - complete[index - 1]["rcenter"][0])
            for index in range(1, len(complete))
        ]
        self.assertLess(max(steps), 0.4)
        self.assertTrue(complete[3]["interpolated_pose"])

    def test_server_surfaces_newer_validated_offline_manifest(self):
        source = (ROOT / "scripts" / "atlas_app_server.py").read_text(encoding="utf-8")
        self.assertIn('latest_offline = CAMERA_PATH_LAB_DIR / "offline_latest.json"', source)
        self.assertIn('snapshot.get("status") not in ACTIVE_JOB_STATES', source)

    def test_current_extractor_image_name_csv_is_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            frames_csv = Path(temporary) / "frames.csv"
            frames_csv.write_text(
                "image_name,source_frame,time_sec,width,height\n"
                "query_000001.jpg,24,0.2,675,1200\n"
                "query_000000.jpg,0,0.0,675,1200\n",
                encoding="utf-8",
            )
            rows = OFFLINE.load_frame_rows(frames_csv)
        self.assertEqual([row["filename"] for row in rows], ["query_000000.jpg", "query_000001.jpg"])

    def test_room_length_recovery_requires_elapsed_video_time(self):
        map_entry = {
            "safety_barriers": [
                {"a": [-1.0, 0.0, -1.0]},
                {"a": [9.0, 0.0, -1.0]},
                {"a": [9.0, 0.0, 2.0]},
                {"a": [-1.0, 0.0, 2.0]},
            ]
        }
        poses = [self._trajectory_pose(index, [0.0, 0.0, 0.0]) for index in range(25)]
        poses.extend(
            self._trajectory_pose(index, [7.0, 0.0, 0.0])
            for index in range(50, 100)
        )

        validation, _ = OFFLINE.validate_raw_trajectory(poses, 100, map_entry, 25, 0.60)
        self.assertTrue(validation["valid"])
        self.assertLessEqual(validation["max_time_scaled_step_excess_m"], 0.0)

        poses[25]["time_sec"] = poses[24]["time_sec"] + 0.2
        validation, _ = OFFLINE.validate_raw_trajectory(poses, 100, map_entry, 25, 0.60)
        self.assertFalse(validation["valid"])
        self.assertGreater(validation["max_time_scaled_step_excess_m"], 0.0)


if __name__ == "__main__":
    unittest.main()

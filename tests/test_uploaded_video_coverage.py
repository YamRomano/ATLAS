import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


sys.modules.setdefault("cv2", types.ModuleType("cv2"))
ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "scripts" / "atlas_app_server.py"

SPEC = importlib.util.spec_from_file_location("atlas_app_server_video_coverage_test", SERVER_PATH)
assert SPEC and SPEC.loader
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


class UploadedVideoCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_extraction(self, *, saved_frames: int) -> Path:
        frame_dir = self.root / "frames"
        frame_dir.mkdir()
        (frame_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "frame_count": 100,
                    "duration_sec": 10.0,
                    "step_frames": 2,
                    "saved_frames": saved_frames,
                }
            ),
            encoding="utf-8",
        )
        with (frame_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_name", "source_frame", "time_sec"])
            writer.writeheader()
            for index in range(saved_frames):
                writer.writerow(
                    {
                        "image_name": f"query_{index:06d}.jpg",
                        "source_frame": index * 2,
                        "time_sec": index * 0.2,
                    }
                )
        return frame_dir

    def test_full_extraction_passes(self):
        report = server.validate_extracted_video_coverage(
            self.write_extraction(saved_frames=50),
            min_temporal_coverage=0.98,
        )
        self.assertEqual(report["saved_frames"], 50)
        self.assertEqual(report["temporal_coverage"], 1.0)

    def test_truncated_extraction_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "extraction is incomplete"):
            server.validate_extracted_video_coverage(
                self.write_extraction(saved_frames=20),
                min_temporal_coverage=0.98,
            )

    def write_localization(self, *, processed: int, accepted: int, last_time: float):
        summary = self.root / "summary.json"
        poses = self.root / "poses.json"
        summary.write_text(
            json.dumps(
                {
                    "query_frames": 50,
                    "processed_frames": processed,
                    "accepted_cases": accepted,
                }
            ),
            encoding="utf-8",
        )
        poses.write_text(
            json.dumps(
                {
                    "poses": [
                        {
                            "success": True,
                            "time_sec": last_time * index / max(1, accepted - 1),
                            "center": [0.001 * index, 0.0, 0.0],
                        }
                        for index in range(accepted)
                    ]
                }
            ),
            encoding="utf-8",
        )
        return summary, poses

    def test_complete_localization_passes(self):
        summary, poses = self.write_localization(processed=50, accepted=45, last_time=9.9)
        report = server.validate_simulated_live_localization(
            summary,
            poses,
            expected_count=50,
            video_duration_sec=10.0,
            min_temporal_coverage=0.98,
            min_acceptance_ratio=0.75,
        )
        self.assertAlmostEqual(report["acceptance_ratio"], 0.9)
        self.assertAlmostEqual(report["temporal_coverage"], 0.99)

    def test_early_localization_is_rejected(self):
        summary, poses = self.write_localization(processed=20, accepted=18, last_time=3.8)
        with self.assertRaisesRegex(RuntimeError, "stopped early"):
            server.validate_simulated_live_localization(
                summary,
                poses,
                expected_count=50,
                video_duration_sec=10.0,
                min_temporal_coverage=0.98,
                min_acceptance_ratio=0.75,
            )

    def test_out_of_order_pose_stream_is_rejected(self):
        summary, poses = self.write_localization(processed=50, accepted=45, last_time=9.9)
        payload = json.loads(poses.read_text(encoding="utf-8"))
        payload["poses"][20], payload["poses"][21] = payload["poses"][21], payload["poses"][20]
        poses.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "strictly increasing timestamp order"):
            server.validate_simulated_live_localization(
                summary,
                poses,
                expected_count=50,
                video_duration_sec=10.0,
                min_temporal_coverage=0.98,
                min_acceptance_ratio=0.75,
            )


if __name__ == "__main__":
    unittest.main()

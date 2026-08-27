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

    def write_patrol_frames(self, name: str, indices: list[int]) -> Path:
        frame_dir = self.root / name
        frame_dir.mkdir()
        with (frame_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_name", "source_frame", "time_sec", "width", "height"],
            )
            writer.writeheader()
            for offset, index in enumerate(indices):
                image_name = f"query_{index:06d}.jpg"
                (frame_dir / image_name).write_bytes(f"frame-{name}-{index}".encode())
                writer.writerow(
                    {
                        "image_name": image_name,
                        "source_frame": index,
                        "time_sec": offset * 0.1,
                        "width": 1200,
                        "height": 675,
                    }
                )
        return frame_dir

    def test_two_lap_builder_uses_an_independent_second_recording_and_final_leg(self):
        baseline = self.write_patrol_frames("baseline", list(range(5)))
        alternate = self.write_patrol_frames("alternate", [10, 11, 12])
        output = self.root / "composite"
        boundaries = {
            "point2_arrival": 1,
            "point2_departure": 2,
            "point3_arrival": 3,
            "point3_departure": 3,
            "point4_arrival": 4,
            "point4_departure": 4,
            "point1_return": 4,
        }
        plan = server.build_recorded_patrol_lap_sequence(
            source_frame_dir=baseline,
            output_frame_dir=output,
            start_frame=0,
            loop_return_frame=3,
            next_departure_frame=4,
            laps=2,
            phase_boundaries=boundaries,
            source_replay_id="baseline-replay",
            second_lap_segments=[
                {
                    "source_frame_dir": alternate,
                    "source_replay_id": "independent-replay",
                    "start_frame": 10,
                    "end_frame": 12,
                    "phase_boundaries": {
                        "point2_arrival": 11,
                        "point2_departure": 12,
                        "point3_arrival": 99,
                    },
                },
                {
                    "source_frame_dir": baseline,
                    "source_replay_id": "baseline-replay",
                    "start_frame": 2,
                    "end_frame": 3,
                    "phase_boundaries": boundaries,
                },
            ],
        )
        self.assertEqual(len(plan), 10)
        self.assertEqual([row["lap"] for row in plan[:5]], [1] * 5)
        self.assertEqual([row["lap"] for row in plan[5:]], [2] * 5)
        self.assertEqual(plan[5]["recorded_source_replay_id"], "independent-replay")
        self.assertEqual(plan[8]["recorded_source_replay_id"], "baseline-replay")
        self.assertEqual(plan[5]["recorded_source_frame"], 10)
        self.assertEqual(plan[-1]["recorded_source_frame"], 3)
        self.assertTrue((output / "query_000005.jpg").is_file())
        frame_plan = json.loads((output / "frame_plan.json").read_text(encoding="utf-8"))
        self.assertTrue(frame_plan["composite_second_lap"])
        self.assertEqual(
            frame_plan["source_replay_ids"],
            ["baseline-replay", "independent-replay"],
        )

    def test_full_extraction_passes(self):
        report = server.validate_extracted_video_coverage(
            self.write_extraction(saved_frames=50),
            min_temporal_coverage=0.98,
        )
        self.assertEqual(report["saved_frames"], 50)
        self.assertEqual(report["temporal_coverage"], 1.0)

    def test_small_decoder_tail_shortfall_passes_coverage_threshold(self):
        report = server.validate_extracted_video_coverage(
            self.write_extraction(saved_frames=49),
            min_temporal_coverage=0.98,
        )
        self.assertEqual(report["saved_frames"], 49)
        self.assertEqual(report["missing_tail_frames"], 1)
        self.assertAlmostEqual(report["frame_coverage"], 0.98)
        self.assertAlmostEqual(report["temporal_coverage"], 0.98)

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

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from atlas_app_server import live_stage_diagnostics
from run_bounded_tsolve_video_stream import (
    POSE_STREAM_STATE,
    POSE_STREAM_WINDOW,
    append_manifest_row,
    append_stage,
    periodic_feature_refresh_due,
    write_partial_pose_stream,
    write_stage_header,
)


class LiveLocalizationDiagnosticsTest(unittest.TestCase):
    def test_periodic_feature_refresh_uses_first_elapsed_limit(self) -> None:
        # At high FPS the tenth frame arrives before one second.
        self.assertTrue(
            periodic_feature_refresh_due(
                frame_index=10,
                frame_time=0.33,
                last_frame_index=0,
                last_frame_time=0.0,
                max_frame_interval=10,
                max_time_interval_seconds=1.0,
            )
        )
        # At low FPS one second arrives before ten frames.
        self.assertTrue(
            periodic_feature_refresh_due(
                frame_index=2,
                frame_time=1.0,
                last_frame_index=0,
                last_frame_time=0.0,
                max_frame_interval=10,
                max_time_interval_seconds=1.0,
            )
        )
        self.assertFalse(
            periodic_feature_refresh_due(
                frame_index=9,
                frame_time=0.9,
                last_frame_index=0,
                last_frame_time=0.0,
                max_frame_interval=10,
                max_time_interval_seconds=1.0,
            )
        )

    def test_stage_csv_keeps_feature_counts_and_computes_pruned(self) -> None:
        with tempfile.TemporaryDirectory(prefix="atlas-stage-diagnostics-") as temporary:
            path = Path(temporary) / "stages.csv"
            write_stage_header(path)
            append_stage(
                path,
                {
                    "frame_index": 42,
                    "method": "optical_flow",
                    "accepted": True,
                    "tracked_points": 70,
                    "selected_points": 40,
                    "total_frame_ms": 80.0,
                },
                {
                    "flow_input_points": 100,
                    "optical_flow_ms": 4.0,
                },
            )
            with path.open(encoding="utf-8") as handle:
                row = list(csv.DictReader(handle))[0]
            self.assertEqual(row["flow_input_points"], "100")
            self.assertEqual(row["tracked_points"], "70")
            self.assertEqual(row["pruned_features"], "30")

    def test_api_payload_exposes_timing_remainder(self) -> None:
        payload = live_stage_diagnostics(
            {
                "frame_index": "2447",
                "method": "optical_flow",
                "accepted": "False",
                "extracted_features": "1369",
                "matched_features": "20",
                "tracked_points": "0",
                "selected_points": "0",
                "pruned_features": "1349",
                "frame_load_ms": "20",
                "heading_flow_ms": "10",
                "feature_extract_ms": "100",
                "match_ms": "50",
                "register_ms": "20",
                "optical_flow_ms": "10",
                "case_build_ms": "30",
                "case_output_ms": "10",
                "visual_route_ms": "40",
                "visual_heading_ms": "30",
                "route_logic_ms": "15",
                "local_recovery_ms": "25",
                "background_apply_ms": "5",
                "pose_update_ms": "20",
                "stream_publish_ms": "10",
                "tsolve_ms": "20",
                "pace_wait_ms": "0",
                "total_frame_ms": "500",
                "reason": "faiss_too_few_unique_2d3d",
            }
        )
        self.assertEqual(payload["features"]["extracted"], 1369)
        self.assertEqual(payload["features"]["pruned"], 1349)
        self.assertEqual(payload["timings_ms"]["frame_load_ms"], 20.0)
        self.assertEqual(payload["timings_ms"]["case_build_ms"], 30.0)
        self.assertEqual(payload["timings_ms"]["other_ms"], 85.0)
        self.assertFalse(payload["accepted"])

    def test_async_worker_latency_is_visible_but_not_foreground_time(self) -> None:
        payload = live_stage_diagnostics(
            {
                "method": "global_colmap_background_recovery_catchup",
                "feature_extract_ms": "100",
                "match_ms": "50",
                "register_ms": "20",
                "optical_flow_ms": "10",
                "frame_load_ms": "20",
                "background_worker_ms": "1200",
                "total_frame_ms": "300",
            }
        )
        self.assertTrue(payload["async_background"])
        self.assertEqual(payload["timings_ms"]["background_worker_ms"], 1200.0)
        self.assertEqual(payload["timings_ms"]["accounted_ms"], 30.0)
        self.assertEqual(payload["timings_ms"]["other_ms"], 270.0)

    def test_live_pose_stream_appends_events_and_bounds_snapshot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="atlas-pose-stream-") as temporary:
            path = Path(temporary) / "partial.json"
            state_key = str(path.resolve())
            POSE_STREAM_STATE.pop(state_key, None)
            poses = [
                {
                    "instance_id": f"instance_{index:06d}",
                    "image_name": f"query_{index:06d}.jpg",
                    "success": True,
                    "center": [float(index), 0.0, 0.0],
                }
                for index in range(POSE_STREAM_WINDOW + 25)
            ]
            write_partial_pose_stream(
                path=path,
                replay_id="diagnostic-test",
                drone_video=None,
                expected_count=len(poses) + 1,
                poses=poses,
                complete=False,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            event_path = path.with_name(payload["pose_events_file"])
            self.assertEqual(payload["processed_count"], len(poses))
            self.assertEqual(len(payload["poses"]), POSE_STREAM_WINDOW)
            self.assertEqual(payload["pose_start_index"], 25)
            self.assertEqual(
                len(event_path.read_text(encoding="utf-8").splitlines()),
                len(poses),
            )

            poses.append(
                {
                    "instance_id": f"instance_{len(poses):06d}",
                    "image_name": f"query_{len(poses):06d}.jpg",
                    "success": True,
                    "center": [float(len(poses)), 0.0, 0.0],
                }
            )
            write_partial_pose_stream(
                path=path,
                replay_id="diagnostic-test",
                drone_video=None,
                expected_count=len(poses),
                poses=poses,
                complete=False,
            )
            self.assertEqual(
                len(event_path.read_text(encoding="utf-8").splitlines()),
                len(poses),
            )
            POSE_STREAM_STATE.pop(state_key, None)

    def test_manifest_is_appended_without_rewriting_previous_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="atlas-manifest-") as temporary:
            output = Path(temporary)
            base = {
                "experiment": "test",
                "p3d_csv": "p3d.csv",
                "p2d_csv": "p2d.csv",
                "input_json": "input.json",
                "points": 20,
                "image_name": "query.jpg",
                "time_sec": 0.0,
                "localization_attempt": "optical_flow",
            }
            append_manifest_row(output, {**base, "case_id": "instance_000001"})
            append_manifest_row(output, {**base, "case_id": "instance_000002"})
            with (output / "manifest.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["case_id"] for row in rows],
                ["instance_000001", "instance_000002"],
            )


if __name__ == "__main__":
    unittest.main()

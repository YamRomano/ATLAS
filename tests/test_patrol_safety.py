import importlib.util
import itertools
import json
import math
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path


sys.modules.setdefault("cv2", types.ModuleType("cv2"))
BRIDGE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "atlas_dji_live_bridge.py"
SPEC = importlib.util.spec_from_file_location("atlas_dji_live_bridge", BRIDGE_PATH)
assert SPEC and SPEC.loader
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)

LOCALIZER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_bounded_tsolve_video_stream.py"
OVERLAY_PATH = Path(__file__).resolve().parents[1] / "viewer" / "drone_glb_overlay.js"
APP_PATH = Path(__file__).resolve().parents[1] / "viewer" / "app.js"
SERVER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "atlas_app_server.py"
SETUP_RUNTIME_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "setup_tsolve_runtime.py"
)
SETUP_RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "setup_tsolve_runtime", SETUP_RUNTIME_PATH
)
assert SETUP_RUNTIME_SPEC and SETUP_RUNTIME_SPEC.loader
setup_tsolve_runtime = importlib.util.module_from_spec(SETUP_RUNTIME_SPEC)
SETUP_RUNTIME_SPEC.loader.exec_module(setup_tsolve_runtime)
GEOMETRY_LOCK_PATH = (
    Path(__file__).resolve().parents[1]
    / "viewer"
    / "public"
    / "maps"
    / "map_copy_20260730_114851_cfefdc"
    / "taught_patrols"
    / "patrol_ms4br5xr_4xclts"
    / "geometry_lock.json"
)
LIVE_ATLAS_141750_POSES = (
    Path(__file__).resolve().parents[1]
    / "viewer"
    / "public"
    / "maps"
    / "map_copy_20260730_114851_cfefdc"
    / "replays"
    / "dji_live_20260809_141750_769971"
    / "poses.json"
)
FULL_PATROL_BASELINE_REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "viewer"
    / "public"
    / "maps"
    / "map_copy_20260730_114851_cfefdc"
    / "replays"
    / "patrol_baseline_20260809_154714_d26c33"
    / "reference_candidate.json"
)
FULL_RECORDED_PATROL_RESULT = (
    Path(__file__).resolve().parents[1]
    / "viewer"
    / "public"
    / "maps"
    / "map_copy_20260730_114851_cfefdc"
    / "replays"
    / "live_frames_20260812_130233_3afdbf"
    / "poses.json"
)


class PatrolSafetyTests(unittest.TestCase):
    def test_partial_pose_writer_imports_regex_dependency(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("\nimport re\n", source[:1000])
        self.assertIn("match = re.search(", source)

    def test_verified_point4_epoch_rebases_only_yaw_hover_position(self):
        original = {
            "ok": True,
            "processed_count": 4120,
            "pose": {
                "instance_id": "instance_004119",
                "received_unix": 100.0,
                "rcenter": [-3.22, -0.08, 0.98],
                "rotation_raw_rcenter": [-3.20, -0.08, 0.99],
                "rheading": [0.0, 0.0, -1.0],
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
        }
        point_four = [-3.0736291183, -0.0802621970, 1.1113942968]

        anchored = bridge.verified_route_anchor_pose_gate(
            original,
            point_four,
            epoch=1,
            epoch_unix=101.0,
            reason="verified_point4_handoff",
        )

        self.assertEqual(anchored["pose"]["rcenter"], point_four)
        self.assertEqual(anchored["pose"]["rotation_position_anchor"], point_four)
        self.assertTrue(anchored["pose"]["rotation_position_locked"])
        self.assertFalse(anchored["pose"]["translation_allowed"])
        self.assertEqual(anchored["pose"]["rheading"], [0.0, 0.0, -1.0])
        self.assertEqual(
            anchored["pose"]["route_pose_epoch_prior_rcenter"],
            [-3.22, -0.08, 0.98],
        )
        self.assertEqual(original["pose"]["rcenter"], [-3.22, -0.08, 0.98])
        self.assertTrue(bridge.pose_gate_predates_route_epoch(original, 101.0))
        self.assertFalse(
            bridge.pose_gate_predates_route_epoch(
                {"pose": {"received_unix": 101.01}},
                101.0,
            )
        )

    def test_verified_endpoint_gate_commits_route_progress_without_translation(self):
        original = {
            "ok": True,
            "processed_count": 2216,
            "pose": {
                "instance_id": "instance_002215",
                "received_unix": 100.0,
                "rcenter": [-3.1674, -0.0803, 0.2618],
                "rheading": [0.0, 0.0, -1.0],
            },
        }
        point_one = [-3.2330, -0.0803, -0.3324]

        committed = bridge.verified_route_endpoint_pose_gate(
            original,
            point_one,
            epoch=2,
            epoch_unix=101.0,
            reason="verified_point1_handoff",
        )

        self.assertEqual(bridge.pose_gate_position(committed), point_one)
        self.assertEqual(committed["route_progress"], 1.0)
        self.assertTrue(committed["verified_route_endpoint_gate"])
        self.assertTrue(committed["pose"]["route_verified_endpoint_committed"])
        self.assertTrue(committed["pose"]["route_verified_endpoint_anchor_only"])
        self.assertTrue(committed["pose"]["metric_rebootstrap_required"])
        self.assertFalse(committed["metric_position_valid"])
        self.assertFalse(bridge.pose_gate_has_fresh_metric_position(committed))
        self.assertTrue(committed["pose"]["rotation_position_locked"])
        self.assertFalse(committed["pose"]["translation_allowed"])
        self.assertEqual(original["pose"]["rcenter"], [-3.1674, -0.0803, 0.2618])

    def test_latest_point1_endpoint_commit_removes_lap2_cross_track_regression(self):
        poses_path = (
            BRIDGE_PATH.parents[1]
            / "viewer"
            / "public"
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "dji_live_20260823_134548_3c0104"
            / "poses_partial.json"
        )
        if not poses_path.exists():
            self.skipTest("13:45 Point-1 endpoint regression replay is unavailable")
        payload = json.loads(poses_path.read_text(encoding="utf-8"))
        endpoint_poses = [
            pose
            for pose in payload.get("poses", [])
            if pose.get("route_visual_endpoint_verified") is True
            and int(pose.get("route_visual_monitor_leg_index") or 0) == 4
        ]
        self.assertTrue(endpoint_poses)
        source_pose = endpoint_poses[-1]
        point_one = [-3.2329557447702215, -0.0802621969998909, -0.33236579860361815]
        point_two = [-0.6480244338911889, -0.0802621969998909, -0.48774887233005093]
        stale_cross_track = bridge.horizontal_xz_segment_distance(
            bridge.vector3(source_pose.get("rcenter")),
            point_one,
            point_two,
        )
        self.assertGreater(stale_cross_track, 0.55)

        committed = bridge.verified_route_endpoint_pose_gate(
            {"ok": True, "pose": source_pose},
            point_one,
            epoch=2,
            epoch_unix=200.0,
            reason="verified_point1_handoff",
        )
        committed_cross_track = bridge.horizontal_xz_segment_distance(
            bridge.pose_gate_position(committed),
            point_one,
            point_two,
        )

        self.assertAlmostEqual(committed_cross_track, 0.0, places=9)
        self.assertEqual(committed["route_progress"], 1.0)

    def test_return_pose_behind_point1_is_on_outgoing_route_line(self):
        point_one = [-3.2329557448, -0.0802621970, -0.3323657986]
        point_two = [-0.6480244339, -0.0802621970, -0.4877488723]
        measured_return = [-4.4987, -0.1775, -0.2074]

        progress = bridge.route_segment_progress_xz(
            measured_return,
            point_one,
            point_two,
        )
        cross_track = bridge.route_line_cross_track_xz(
            measured_return,
            point_one,
            point_two,
        )
        leg_length = bridge.horizontal_xz_distance(point_one, point_two)

        self.assertIsNotNone(progress)
        self.assertIsNotNone(cross_track)
        self.assertIsNotNone(leg_length)
        self.assertLess(progress, 0.0)
        self.assertGreaterEqual(progress, -0.60)
        self.assertLess(cross_track, 0.06)
        self.assertLess(-progress * leg_length, 1.55)

        lateral_mismatch = [-4.4987, -0.1775, 0.45]
        self.assertGreater(
            bridge.route_line_cross_track_xz(
                lateral_mismatch,
                point_one,
                point_two,
            ),
            0.30,
        )

    def test_decoded_frame_listener_distinguishes_new_frames_from_cached_reads(self):
        listener = bridge.DecodedFrameListener()
        sequence, frame, received_unix = listener.latest()
        self.assertEqual(sequence, 0)
        self.assertIsNone(frame)
        self.assertIsNone(received_unix)
        self.assertIsNone(listener.age_seconds())

        first = object()
        listener.onValue(first)
        sequence, frame, received_unix = listener.latest()
        self.assertEqual(sequence, 1)
        self.assertIs(frame, first)
        self.assertIsNotNone(received_unix)
        self.assertLess(listener.age_seconds(), 0.1)

        # Reading the cached value does not invent another decoded frame.
        cached_sequence, cached_frame, _ = listener.latest()
        self.assertEqual(cached_sequence, 1)
        self.assertIs(cached_frame, first)

        listener.onValue(object())
        self.assertEqual(listener.latest()[0], 2)

    def test_frozen_dji_video_refuses_takeoff_or_patrol_motion(self):
        self.assertIn("no fresh decoded", bridge.live_video_motion_safety_issue(None))
        self.assertIn("frozen", bridge.live_video_motion_safety_issue(2.5))
        self.assertIsNone(bridge.live_video_motion_safety_issue(0.2))

    def test_live_atlas_waits_for_a_real_decoded_frame_sequence(self):
        server = SERVER_PATH.read_text(encoding="utf-8")
        bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("required_fresh_live_frames = 8", server)
        self.assertIn("count_frame_images(query_frames) < required_fresh_live_frames", server)
        self.assertIn("decoded_sequence == last_seen_decoded_sequence", bridge_source)
        self.assertIn('"status": "video_stale"', bridge_source)
        self.assertIn('command_name in {"takeoff", "mission"}', bridge_source)

    def test_rotation_locked_pose_is_distinguished_from_bad_localization(self):
        self.assertTrue(
            bridge.pose_gate_rotation_locked(
                {
                    "ok": True,
                    "pose": {
                        "rotation_position_locked": True,
                        "translation_allowed": False,
                    },
                }
            )
        )
        self.assertFalse(
            bridge.pose_gate_rotation_locked(
                {
                    "ok": True,
                    "pose": {
                        "rotation_position_locked": False,
                        "translation_allowed": True,
                    },
                }
            )
        )

    def test_metric_pose_detection_excludes_route_only_pose(self):
        metric_gate = {
            "ok": True,
            "pose": {
                "instance_id": "instance_000123",
                "rcenter": [-3.23, -0.08, -0.33],
                "center": [1.0, 2.0, 3.0],
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [-1.0, -2.0, -3.0],
            },
        }
        self.assertTrue(bridge.pose_gate_has_fresh_metric_position(metric_gate))
        self.assertFalse(
            bridge.pose_gate_has_fresh_metric_position(
                {
                    **metric_gate,
                    "pose": {
                        **metric_gate["pose"],
                        "R": None,
                        "t": None,
                        "center": None,
                        "pose_source": "patrol_visual_route_recovery",
                    },
                }
            )
        )
        self.assertFalse(
            bridge.pose_gate_has_fresh_metric_position(
                {**metric_gate, "recent_hold_fallback": True}
            )
        )
        self.assertFalse(
            bridge.pose_gate_has_fresh_metric_position(
                {
                    **metric_gate,
                    "pose": {
                        **metric_gate["pose"],
                        "route_verified_endpoint_committed": True,
                    },
                }
            )
        )

    def test_point1_star_metric_consensus_accepts_stable_global_pose_far_from_point1(self):
        state = {}

        def gate(instance, x, z):
            return {
                "ok": True,
                "pose": {
                    "instance_id": f"instance_{instance:06d}",
                    "received_unix": 100.0 + instance,
                    "rcenter": [x, -0.08, z],
                    "center": [1.0, 2.0, 3.0],
                    "R": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "t": [-1.0, -2.0, -3.0],
                },
            }

        # The recovered Point 1* is intentionally about two metres from the
        # saved Point 1. Consensus is measured between current global poses,
        # not against the patrol waypoint.
        self.assertIsNone(
            bridge.lap_transition_metric_consensus_gate(
                state, gate(1, -5.11, 0.26), minimum_received_unix=100.0
            )
        )
        self.assertIsNone(
            bridge.lap_transition_metric_consensus_gate(
                state, gate(2, -5.09, 0.27), minimum_received_unix=100.0
            )
        )
        accepted = bridge.lap_transition_metric_consensus_gate(
            state, gate(3, -5.12, 0.25), minimum_received_unix=100.0
        )
        self.assertIsNotNone(accepted)
        self.assertTrue(accepted["lap_transition_metric_consensus"])
        self.assertEqual(accepted["lap_transition_metric_consensus_hits"], 3)

    def test_point1_star_metric_consensus_resets_on_inconsistent_pose(self):
        state = {}

        def gate(instance, x):
            return {
                "ok": True,
                "pose": {
                    "instance_id": f"instance_{instance:06d}",
                    "received_unix": 200.0 + instance,
                    "rcenter": [x, 0.0, 0.0],
                    "center": [x, 0.0, 0.0],
                    "R": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "t": [-x, 0.0, 0.0],
                },
            }

        bridge.lap_transition_metric_consensus_gate(
            state, gate(1, -5.10), minimum_received_unix=200.0
        )
        bridge.lap_transition_metric_consensus_gate(
            state, gate(2, -5.12), minimum_received_unix=200.0
        )
        self.assertIsNone(
            bridge.lap_transition_metric_consensus_gate(
                state, gate(3, -3.20), minimum_received_unix=200.0
            )
        )
        self.assertEqual(state["hits"], 1)

    def test_repeated_lap_checkpoint_accepts_metric_or_verified_endpoint(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("lap_metric_checkpoint_pending = current_lap_number > 1", source)
        lap_start = source.index(
            "if lap_metric_checkpoint_pending and not dynamic_lap_reentry:"
        )
        lap_end = source.index("turn_direction_override =", lap_start)
        checkpoint = source[lap_start:lap_end]
        self.assertIn("checkpoint_metric_ready", checkpoint)
        self.assertIn("checkpoint_visual_ready", checkpoint)
        self.assertIn("prior_point_one_visual_ready", checkpoint)
        self.assertIn("prior_verified_endpoint_arrival_record(", checkpoint)
        self.assertIn("expected_leg_index=4", checkpoint)
        self.assertIn("require_endpoint_verified=True", checkpoint)
        self.assertIn("endpoint_leg_index=4", checkpoint)
        self.assertIn('"lap_checkpoint_verified"', checkpoint)
        self.assertIn("checkpoint_anchor_gate = verified_route_anchor_pose_gate(", checkpoint)
        self.assertIn(
            "verified_endpoint_turn_source_gate = checkpoint_anchor_gate",
            checkpoint,
        )
        self.assertIn("lap_point_one_handoff = True", checkpoint)
        self.assertNotIn("require_metric_pose=True,", checkpoint)

        departure_start = source.index(
            "if rotation_position_untrusted:", lap_end
        )
        departure_end = source.index(
            "elif alignment_angle is not None", departure_start
        )
        departure = source[departure_start:departure_end]
        self.assertIn("if lap_point_one_handoff:", departure)
        self.assertIn("endpoint_departure_gate = None", departure)
        self.assertIn(
            "require_metric_pose=lap_point_one_handoff", departure
        )
        self.assertIn(
            "lap_start_metric_rebootstrap=(", departure
        )

        endpoint_start = source.index("if visual_checkpoint_arrival:")
        endpoint_end = source.index(
            "if (\n                            tight_early_leg_candidate",
            endpoint_start,
        )
        endpoint_commit = source[endpoint_start:endpoint_end]
        self.assertIn("commit_verified_point_one_handoff(", endpoint_commit)
        self.assertIn('"verified_endpoint_pose_committed"', endpoint_commit)

        localizer = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            '"require_metric_pose": progress.get("require_metric_pose") is True',
            localizer,
        )
        self.assertIn('route_context.get("require_metric_pose") is True', localizer)
        self.assertIn("lap_start_metric_center", localizer)
        self.assertIn("lap_checkpoint_faiss_current_frame_tsolve", localizer)
        self.assertIn("localize_faiss_current_frame(", localizer)
        self.assertIn("LAP CHECKPOINT CURRENT-FRAME METRIC RECOVERY", localizer)
        self.assertIn("superseded_by_new_lap_metric_checkpoint", localizer)
        self.assertIn(
            "superseded_by_lap_start_metric_rebootstrap", localizer
        )
        self.assertIn(
            'global_reason = "lap_start_metric_rebootstrap"', localizer
        )
        self.assertIn(
            'method = "lap_start_metric_rebootstrap_retry"', localizer
        )
        self.assertIn('"phase": "lap_start_metric_reentry"', source)
        self.assertIn("route_line_cross_track_xz(", departure)
        self.assertIn("segment_start = list(recovered_position)", departure)

        viewer = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("opticalHeadingTracks >= 16", viewer)
        self.assertIn("pose.rotation_position_locked", viewer)
        self.assertIn("!absoluteRouteHeading", viewer)
        self.assertIn('"recorded_departure_image_alignment"', viewer)
        self.assertIn('"recorded_departure_image_tracking_consensus"', viewer)
        self.assertNotIn(
            'pose.pose_source === "patrol_visual_route_recovery" ||',
            viewer,
        )

    def test_live_jobs_enable_metric_checkpoint_recovery(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        live_start = source.index("def dji_live_atlas_job(")
        live_end = source.index("def simulated_patrol_baseline_job(", live_start)
        fleet_start = source.index("def fleet_live_atlas_job(")
        fleet_end = source.index("def drone_video_job(", fleet_start)
        self.assertIn(
            'live_stream_cmd.append("--wait-for-metric-checkpoint-recovery")',
            source[live_start:live_end],
        )
        self.assertIn(
            'live_cmd.append("--wait-for-metric-checkpoint-recovery")',
            source[fleet_start:fleet_end],
        )

    def test_repeated_lap_relocalizes_point1_star_before_guarded_point_one_entry(self):
        bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
        localizer_source = LOCALIZER_PATH.read_text(encoding="utf-8")

        sequence_start = bridge_source.index("def mission_step_sequence(")
        sequence_end = bridge_source.index(
            "def patrol_loop_start_command_index(", sequence_start
        )
        sequence_source = bridge_source[sequence_start:sequence_end]
        self.assertIn('"type": "lap_relocalize_entry"', sequence_source)
        self.assertIn('"_atlas_lap_reentry": True', sequence_source)
        self.assertIn("Point 1* to Point 1", sequence_source)
        marker_start = bridge_source.index('if kind == "lap_relocalize_entry":')
        marker_end = bridge_source.index("gate_attempt =", marker_start)
        marker = bridge_source[marker_start:marker_end]
        self.assertIn("allow_metric_discontinuity_consensus=True", marker)
        self.assertIn('"lap_start_global_relocalization"', marker)
        self.assertIn("lap_reentry_metric_ready = True", marker)
        cruise_start = bridge_source.index(
            'dynamic_lap_reentry = step.get("_atlas_lap_reentry") is True'
        )
        cruise_end = bridge_source.index(
            "planned_duration = clamp_float", cruise_start
        )
        connector_cruise = bridge_source[cruise_start:cruise_end]
        self.assertIn(
            "strict_metric_arrival = bool(\n                    precise_arrival or dynamic_lap_reentry",
            connector_cruise,
        )
        self.assertIn("strict_target=strict_metric_arrival", connector_cruise)
        self.assertNotIn(
            "taught_leg_requires_precise_arrival(taught_leg)\n                    or dynamic_lap_reentry",
            connector_cruise,
        )
        self.assertIn(
            '"lap_start_global_relocalization",',
            localizer_source,
        )

    def test_tsolve_runtime_restores_required_harness_dependency(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            yam_code = root / "base" / "yam_code"
            source_harness = root / "base" / "harness"
            output_harness = root / "output" / "harness"
            yam_code.mkdir(parents=True)
            source_harness.mkdir(parents=True)
            dependency = source_harness / "ysolve_template_core.py"
            dependency.write_text("RUNTIME_DEPENDENCY = True\n", encoding="utf-8")

            setup_tsolve_runtime.ensure_harness_dependencies(
                yam_code, output_harness
            )

            restored = output_harness / "ysolve_template_core.py"
            self.assertEqual(restored.read_text(encoding="utf-8"), dependency.read_text(encoding="utf-8"))

    def test_entry_arrival_finishes_inside_loop_start_gate_with_margin(self):
        hard_radius, soft_radius = bridge.mission_arrival_radii(
            0.24,
            0.14,
            patrol_stage="entry",
        )
        self.assertEqual(hard_radius, 0.20)
        self.assertEqual(soft_radius, 0.20)
        self.assertLess(soft_radius, 0.24)

    def test_loop_points_keep_indoor_soft_arrival_deadband(self):
        hard_radius, soft_radius = bridge.mission_arrival_radii(
            0.24,
            0.14,
            patrol_stage="loop",
        )
        self.assertEqual(hard_radius, 0.24)
        self.assertAlmostEqual(soft_radius, 0.38)

    def test_recorded_endpoint_arrival_is_strict_on_the_complete_loop(self):
        hard_radius, soft_radius = bridge.mission_arrival_radii(
            0.24,
            0.14,
            patrol_stage="loop",
            strict_target=True,
        )
        self.assertEqual(hard_radius, 0.15)
        self.assertEqual(soft_radius, 0.15)
        for from_point, to_point in ((1, 2), (2, 3), (3, 4), (4, 1)):
            with self.subTest(from_point=from_point, to_point=to_point):
                self.assertTrue(
                    bridge.taught_leg_requires_precise_arrival(
                        {"from_point": from_point, "to_point": to_point}
                    )
                )
        self.assertFalse(bridge.taught_leg_requires_precise_arrival(None))

    def test_taught_endpoint_requires_independent_consensus_for_correct_leg(self):
        verified = {
            "route_visual_endpoint_verified": True,
            "route_visual_endpoint_view_geometry_verified": True,
            "route_visual_monitor_leg_index": 3,
            "route_visual_endpoint_hits": 3,
            "route_visual_endpoint_required_hits": 3,
            "route_visual_endpoint_best_progress": 0.967,
            "route_visual_endpoint_best_inliers": 257,
            "route_visual_endpoint_minimum_inliers": 75,
            "route_visual_monitor_minimum_inliers": 120,
        }
        self.assertTrue(
            bridge.taught_endpoint_arrival_verified(
                verified, expected_leg_index=3
            )
        )
        self.assertFalse(
            bridge.taught_endpoint_arrival_verified(
                verified, expected_leg_index=2
            )
        )
        self.assertTrue(
            bridge.taught_endpoint_arrival_verified(
                {
                    **verified,
                    "route_visual_endpoint_best_inliers": 99,
                },
                expected_leg_index=3,
            )
        )
        self.assertFalse(
            bridge.taught_endpoint_arrival_verified(
                {**verified, "route_visual_endpoint_hits": 2},
                expected_leg_index=3,
            )
        )
        self.assertFalse(
            bridge.taught_endpoint_arrival_verified(
                {
                    **verified,
                    "route_visual_endpoint_view_geometry_verified": False,
                },
                expected_leg_index=3,
            )
        )
        self.assertFalse(
            bridge.taught_endpoint_arrival_verified(
                {
                    **verified,
                    "route_visual_endpoint_best_progress": 0.062,
                },
                expected_leg_index=3,
            )
        )

    def test_strong_current_endpoint_can_end_a_leg_with_stale_metric_translation(self):
        # Exact evidence shape from the landed 09:39 run approaching Point 3:
        # TSolve stayed at 0.511 while the independent endpoint matcher saw
        # the current image at 0.908 and repeated a 0.975 endpoint match.
        verified = {
            "route_visual_monitor_verified": True,
            "route_visual_monitor_leg_index": 2,
            "route_visual_endpoint_checked": True,
            "route_visual_endpoint_verified": True,
            "route_visual_endpoint_view_geometry_verified": True,
            "route_visual_endpoint_hits": 3,
            "route_visual_endpoint_required_hits": 3,
            "route_visual_endpoint_candidate_progress": 0.907843,
            "route_visual_endpoint_best_progress": 0.974610,
            "route_visual_endpoint_best_inliers": 96,
            "route_visual_endpoint_minimum_inliers": 75,
            "route_visual_inliers": 96,
            "route_visual_monitor_inliers": 96,
            "rotation_position_locked": False,
            "translation_allowed": True,
        }
        self.assertTrue(
            bridge.taught_endpoint_stale_translation_arrival_verified(
                verified,
                expected_leg_index=2,
            )
        )
        for field, unsafe_value in (
            ("route_visual_monitor_verified", False),
            ("route_visual_endpoint_checked", False),
            ("route_visual_endpoint_candidate_progress", 0.899),
            ("rotation_position_locked", True),
            ("translation_allowed", False),
        ):
            with self.subTest(field=field):
                self.assertFalse(
                    bridge.taught_endpoint_stale_translation_arrival_verified(
                        {**verified, field: unsafe_value},
                        expected_leg_index=2,
                    )
                )
        self.assertFalse(
            bridge.taught_endpoint_stale_translation_arrival_verified(
                {
                    **verified,
                    "route_visual_endpoint_minimum_inliers": 50,
                    "route_visual_inliers": 0,
                    "route_visual_monitor_inliers": 49,
                },
                expected_leg_index=2,
            )
        )
        self.assertTrue(
            bridge.taught_endpoint_stale_translation_arrival_verified(
                {
                    **verified,
                    "route_visual_endpoint_minimum_inliers": 50,
                    "route_visual_inliers": 0,
                    "route_visual_monitor_inliers": 50,
                },
                expected_leg_index=2,
            )
        )
        self.assertFalse(
            bridge.taught_endpoint_stale_translation_arrival_verified(
                verified,
                expected_leg_index=3,
            )
        )

    def test_current_point_four_candidate_is_not_vetoed_by_earlier_best_anchor(self):
        # Exact evidence from visual_route_002226 in the landed 24-Aug lap.
        # The endpoint was independently verified in 0.58 s, but the bridge
        # waited 116 s because a stronger whole-leg alias sentinel remained at
        # progress 0.772.  Current candidate evidence is the arrival authority;
        # the earlier best anchor remains useful only as negative audit data.
        verified = {
            "route_visual_monitor_verified": True,
            "route_visual_monitor_leg_index": 3,
            "route_visual_monitor_inliers": 244,
            "route_visual_endpoint_checked": True,
            "route_visual_endpoint_verified": True,
            "route_visual_endpoint_match_consensus_verified": True,
            "route_visual_endpoint_view_geometry_verified": True,
            "route_visual_endpoint_hits": 3,
            "route_visual_endpoint_required_hits": 3,
            "route_visual_endpoint_candidate_progress": 0.9509254644,
            "route_visual_endpoint_best_progress": 0.7715599590,
            "route_visual_endpoint_best_inliers": 303,
            "route_visual_endpoint_minimum_inliers": 75,
            "route_visual_inliers": 244,
            "rotation_position_locked": False,
            "translation_allowed": True,
        }
        self.assertTrue(
            bridge.taught_endpoint_arrival_verified(
                verified,
                expected_leg_index=3,
            )
        )
        self.assertTrue(
            bridge.taught_endpoint_stale_translation_arrival_verified(
                verified,
                expected_leg_index=3,
            )
        )
        self.assertFalse(
            bridge.taught_endpoint_arrival_verified(
                {
                    **verified,
                    "route_visual_endpoint_candidate_progress": 0.899,
                },
                expected_leg_index=3,
            )
        )

    def test_tight_metric_arrival_uses_tsolve_on_early_legs(self):
        target = [-0.4886978074, -0.0802621970, 0.9560112231]
        pose = {
            "instance_id": "instance_002504",
            "rcenter": [-0.4936314403, -0.0802621970, 0.9113044318],
            "center": [0.7634820341, -0.0138464323, 1.2704317574],
            "t": [1.4093397291, -0.1488922061, -0.4343800516],
            "R": [
                [-0.2352711550, 0.0990392164, -0.9668705794],
                [0.0733577206, 0.9937666036, 0.0839439236],
                [0.9691574322, -0.0511778379, -0.2410699078],
            ],
            "pose_source": None,
            "rotation_position_locked": False,
            "route_visual_monitor_verified": True,
            "route_visual_monitor_leg_index": 2,
            "route_visual_monitor_inliers": 113,
            "route_visual_monitor_progress": 0.9690344736,
            "route_visual_monitor_tsolve_progress": 0.9690344736,
            "route_visual_monitor_disagreement_m": 0.0,
            "route_visual_endpoint_checked": True,
            "route_visual_endpoint_view_geometry_verified": True,
            # This is intentionally the repeated-whiteboard mismatch from the
            # landed run.  A mid-leg best image may identify the active leg,
            # but the fresh metric pose remains the endpoint authority.
            "route_visual_endpoint_best_progress": 0.4267656164,
            "route_visual_endpoint_best_inliers": 223,
        }
        gate = {"ok": True, "pose": pose, "processed_count": 1168}
        self.assertTrue(
            bridge.tight_metric_tsolve_endpoint_arrival_candidate(
                gate,
                target=target,
                expected_leg_index=2,
                command_progress_ceiling=1.0,
            )
        )
        # This reproduces the Point-2 live deadlock: TSolve is 2.8 cm from the
        # target while ORB fluctuates below 50 inliers. ORB is diagnostic at
        # the endpoint and may not veto the strict metric arrival.
        point_two_target = [-0.6480244339, -0.0802621970, -0.4877488723]
        point_two_pose = {
            **pose,
            "rcenter": [-0.6200092003, -0.0802621970, -0.4894328989],
            "route_visual_monitor_verified": False,
            "route_visual_monitor_leg_index": 1,
            "route_visual_monitor_inliers": 44,
            "route_visual_monitor_progress": None,
        }
        self.assertTrue(
            bridge.tight_metric_tsolve_endpoint_arrival_candidate(
                {**gate, "pose": point_two_pose},
                target=point_two_target,
                expected_leg_index=1,
                command_progress_ceiling=1.0,
            )
        )
        unsafe_cases = (
            ("held pose", {"recent_hold_fallback": True}),
            ("rotation locked", {"pose": {**pose, "rotation_position_locked": True}}),
            ("translation unsafe", {"pose": {**pose, "translation_allowed": False}}),
        )
        for label, changes in unsafe_cases:
            changed_gate = {**gate, **changes}
            with self.subTest(label=label):
                self.assertFalse(
                    bridge.tight_metric_tsolve_endpoint_arrival_candidate(
                        changed_gate,
                        target=target,
                        expected_leg_index=2,
                        command_progress_ceiling=1.0,
                    )
                )
        self.assertFalse(
            bridge.tight_metric_tsolve_endpoint_arrival_candidate(
                gate,
                target=target,
                expected_leg_index=2,
                command_progress_ceiling=0.97,
            )
        )
        self.assertFalse(
            bridge.tight_metric_tsolve_endpoint_arrival_candidate(
                gate,
                target=[-0.4886978074, -0.0802621970, 1.10],
                expected_leg_index=2,
                command_progress_ceiling=1.0,
            )
        )

    def test_tight_metric_visual_endpoint_consensus_counts_distinct_frames(self):
        state = {}
        gate = {"pose": {"instance_id": "instance_1"}}
        self.assertFalse(
            bridge.update_tight_endpoint_consensus(state, gate, candidate=True)
        )
        self.assertFalse(
            bridge.update_tight_endpoint_consensus(state, gate, candidate=True)
        )
        self.assertEqual(state["hits"], 1)
        gate["pose"]["instance_id"] = "instance_2"
        self.assertFalse(
            bridge.update_tight_endpoint_consensus(state, gate, candidate=True)
        )
        gate["pose"]["instance_id"] = "instance_3"
        self.assertTrue(
            bridge.update_tight_endpoint_consensus(state, gate, candidate=True)
        )
        self.assertFalse(
            bridge.update_tight_endpoint_consensus(state, gate, candidate=False)
        )
        self.assertEqual(state, {})

    def test_tight_metric_consensus_survives_two_missing_monitor_frames(self):
        state = {}
        for instance_id in ("instance_1", "instance_2"):
            self.assertFalse(
                bridge.update_tight_endpoint_consensus(
                    state,
                    {"pose": {"instance_id": instance_id}},
                    candidate=True,
                )
            )
        for _ in range(2):
            self.assertFalse(
                bridge.update_tight_endpoint_consensus(
                    state,
                    {"pose": {"instance_id": "missing"}},
                    candidate=False,
                    observation_missing=True,
                )
            )
        self.assertTrue(
            bridge.update_tight_endpoint_consensus(
                state,
                {"pose": {"instance_id": "instance_3"}},
                candidate=True,
            )
        )
        self.assertEqual(state["hits"], 3)

    def test_point_three_metric_consensus_is_evaluated_inside_endpoint_recovery(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        start = source.index("def wait_for_pose_recovery(")
        end = source.index("def continuity_guarded_pose_gate", start)
        recovery = source[start:end]
        self.assertIn("tight_metric_tsolve_endpoint_arrival_candidate(", recovery)
        self.assertIn('recovered_gate["tight_metric_endpoint_arrival"]', recovery)
        self.assertIn("maximum_metric_error=min(\n                            0.08", recovery)

    def test_early_leg_metric_consensus_preempts_ambiguous_endpoint_wait(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        candidate = source.index("tight_early_leg_candidate = bool(")
        consensus_wait = source.index(
            '"phase": "early_leg_metric_endpoint_consensus"', candidate
        )
        arrival = source.index('"strict_radius_metric_tsolve"', consensus_wait)
        old_endpoint_wait = source.index(
            '"phase": "taught_endpoint_verification"', arrival
        )
        self.assertLess(candidate, consensus_wait)
        self.assertLess(consensus_wait, arrival)
        self.assertLess(arrival, old_endpoint_wait)
        self.assertIn("or tight_early_leg_ready", source[candidate:consensus_wait])
        self.assertIn("neutral_hover(drone, 0.12)", source[consensus_wait:arrival])

    def test_live_pose_gate_preserves_taught_endpoint_consensus(self):
        now = time.time()
        endpoint_fields = {
            "route_visual_endpoint_guarded": True,
            "route_visual_endpoint_guard_progress": 0.84,
            "route_visual_endpoint_checked": True,
            "route_visual_endpoint_verified": True,
            "route_visual_endpoint_view_geometry_verified": True,
            "route_visual_endpoint_view_scale_min": 0.96,
            "route_visual_endpoint_view_scale_max": 1.04,
            "route_visual_endpoint_hits": 57,
            "route_visual_endpoint_required_hits": 3,
            "route_visual_endpoint_minimum_inliers": 75,
            "route_visual_endpoint_candidate_progress": 0.953,
            "route_visual_endpoint_best_progress": 0.953,
            "route_visual_endpoint_best_inliers": 146,
            "route_visual_endpoint_best_anchor": "query_001351.jpg",
        }
        payload = {
            "updated_at": now,
            "processed_count": 1,
            "poses": [{
                "instance_id": "instance_000001",
                "success": True,
                "held_pose": False,
                "rcenter": [-0.648, -0.080, -0.488],
                "rheading": [1.0, 0.0, 0.0],
                "received_unix": now,
                "route_visual_monitor_leg_index": 1,
                **endpoint_fields,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses_partial.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            gate = bridge.latest_tsolve_pose_gate(path, max_age_seconds=1.0)

        self.assertTrue(gate["ok"])
        for key, value in endpoint_fields.items():
            self.assertEqual(gate["pose"][key], value)
        self.assertTrue(
            bridge.taught_endpoint_arrival_verified(
                gate["pose"], expected_leg_index=1
            )
        )

    def test_verified_route_lock_covers_the_complete_loop(self):
        lock = bridge.load_verified_route_follow_lock(
            {
                "map_id": "map_copy_20260730_114851_cfefdc",
                "patrol_id": "patrol_ms4br5xr_4xclts",
            }
        )
        self.assertIsNotNone(lock)
        self.assertEqual(lock["source_replay_id"], "dji_live_20260811_115736_2b91ca")
        for from_point, to_point in ((1, 2), (2, 3), (3, 4), (4, 1)):
            with self.subTest(from_point=from_point, to_point=to_point):
                self.assertIsNotNone(
                    bridge.verified_route_follow_leg(
                        lock,
                        {"from_point": from_point, "to_point": to_point},
                    )
                )
        final_leg = bridge.verified_route_follow_leg(
            lock,
            {"from_point": 4, "to_point": 1},
        )
        self.assertAlmostEqual(final_leg["golden_camera_heading_deg"], -96.29741340752297)

    def test_latest_failed_endpoint_yaws_back_to_1157_view(self):
        lock = bridge.load_verified_route_follow_lock(
            {
                "map_id": "map_copy_20260730_114851_cfefdc",
                "patrol_id": "patrol_ms4br5xr_4xclts",
            }
        )
        leg = bridge.verified_route_follow_leg(
            lock,
            {"from_point": 1, "to_point": 2},
        )
        start = [-3.2329557447702215, -0.0802621969998909, -0.33236579860361815]
        end = [-0.6480244338911889, -0.0802621969998909, -0.48774887233005093]
        current = [-0.7647518594033587, -0.0802621969998909, -0.48073225796667496]
        desired = bridge.verified_route_desired_camera_heading(
            leg,
            current_position=current,
            segment_start=start,
            segment_end=end,
        )
        failed_heading = [0.979507408401155, 0.0, -0.2014081351069339]
        successful_heading = [0.998033003171988, 0.0, 0.06269070568674937]
        failed_error = bridge.signed_angle_xz(failed_heading, desired)
        successful_error = bridge.signed_angle_xz(successful_heading, desired)
        self.assertGreater(abs(math.degrees(failed_error)), 14.0)
        self.assertLess(abs(math.degrees(successful_error)), 1.0)

    def test_verified_route_lookahead_correction_is_bounded(self):
        leg = {
            "golden_camera_heading_deg": 3.683252203582368,
            "lookahead_m": 0.35,
            "max_route_correction_deg": 7.0,
        }
        start = [-3.2329557447702215, 0.0, -0.33236579860361815]
        end = [-0.6480244338911889, 0.0, -0.48774887233005093]
        desired = bridge.verified_route_desired_camera_heading(
            leg,
            current_position=[-2.0, 0.0, -0.75],
            segment_start=start,
            segment_end=end,
        )
        golden = [
            math.cos(math.radians(3.683252203582368)),
            0.0,
            math.sin(math.radians(3.683252203582368)),
        ]
        correction = bridge.signed_angle_xz(golden, desired)
        self.assertLessEqual(abs(math.degrees(correction)), 7.000001)

    def test_visual_route_pose_uses_fresh_optical_heading_for_1157_lock(self):
        gate = {
            "pose": {
                "rheading": [0.9981982090, 0.0, -0.0600027959],
                "rheading_raw": [0.560, 0.0, 0.828],
                "rheading_source": "recorded_patrol_leg_heading",
                "pose_source": "patrol_visual_route_recovery",
                "rotation_heading": [0.9986807377, 0.0, 0.0513496259],
                "rotation_heading_tracks": 462,
            }
        }
        heading = bridge.pose_gate_camera_heading(gate)
        desired = [
            math.cos(math.radians(3.683252203582368)),
            0.0,
            math.sin(math.radians(3.683252203582368)),
        ]
        error = bridge.signed_angle_xz(heading, desired)
        self.assertLess(abs(math.degrees(error)), 0.8)

    def test_point_three_recorded_view_rebases_optical_heading_for_cruise(self):
        optical_degrees = 143.1
        recorded_current_degrees = 175.3
        optical = [
            math.cos(math.radians(optical_degrees)),
            0.0,
            math.sin(math.radians(optical_degrees)),
        ]
        recorded_current = [
            math.cos(math.radians(recorded_current_degrees)),
            0.0,
            math.sin(math.radians(recorded_current_degrees)),
        ]
        bias = bridge.signed_angle_xz(optical, recorded_current)
        gate = {
            "pose": {
                "pose_source": "patrol_visual_route_recovery",
                "rotation_heading": optical,
                "rotation_heading_tracks": 197,
                "rheading": recorded_current,
            }
        }

        corrected = bridge.pose_gate_camera_heading(
            gate,
            optical_heading_bias_rad=bias,
        )
        error = bridge.signed_angle_xz(corrected, recorded_current)
        self.assertLess(abs(math.degrees(error)), 0.01)

    def test_physical_translation_authority_is_published_after_final_gate(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        final_gate = source.index("if translation_issue is not None:")
        active_phase = source.index(
            '"phase": "translation_command_active"',
            final_gate,
        )
        command = source.index("execute_guarded_cruise_window(", active_phase)
        self.assertLess(final_gate, active_phase)
        self.assertLess(active_phase, command)
        active_block = source[active_phase:command]
        self.assertIn('"physical_translation_active": True', active_block)

    def test_visual_route_pose_falls_back_to_recorded_heading_without_optical_tracks(self):
        gate = {
            "pose": {
                "rheading": [-0.998, 0.0, 0.060],
                "rheading_raw": [0.560, 0.0, 0.828],
                "rheading_source": "recorded_patrol_leg_heading",
                "pose_source": "patrol_visual_route_recovery",
                "rotation_heading_tracks": 0,
            }
        }
        heading = bridge.pose_gate_camera_heading(gate)
        self.assertLess(heading[0], -0.99)
        self.assertLess(abs(heading[2]), 0.07)

    def test_visual_route_control_preserves_fused_orb_heading_over_stale_optical(self):
        fused = [-0.1701144442, 0.0, -0.9854243126]
        gate = {
            "pose": {
                "rheading": fused,
                "rheading_source": "recorded_departure_image_alignment",
                "pose_source": "patrol_visual_route_recovery",
                "rotation_heading": [-0.8852610617, 0.0, -0.4650944557],
                "rotation_heading_tracks": 363,
            }
        }

        heading = bridge.pose_gate_camera_heading(gate)

        expected = bridge.normalize_xz(fused)
        self.assertTrue(
            all(
                abs(float(actual) - float(wanted)) <= 1.0e-9
                for actual, wanted in zip(heading, expected)
            )
        )

    def test_pose_stream_preserves_optical_heading_for_visual_route_control(self):
        now = time.time()
        payload = {
            "updated_at": now,
            "processed_count": 17,
            "poses": [
                {
                    "instance_id": "visual_route_000017",
                    "success": True,
                    "held_pose": False,
                    "received_unix": now,
                    "rcenter": [-2.0, 0.0, -0.4],
                    "rheading": [0.998, 0.0, -0.060],
                    "rheading_source": "recorded_patrol_leg_heading",
                    "pose_source": "patrol_visual_route_recovery",
                    "route_visual_verified": True,
                    "route_visual_progress": 0.2,
                    "route_visual_inliers": 180,
                    "route_visual_minimum_inliers": 120,
                    "route_visual_acquisition_hits": 2,
                    "rotation_heading": [0.9987, 0.0, 0.0513],
                    "rotation_heading_source": "optical_flow_yaw",
                    "rotation_heading_tracks": 462,
                    "rotation_heading_delta_deg": 0.31,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses_partial.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            gate = bridge.latest_tsolve_pose_gate(path, max_age_seconds=1.0)

        self.assertTrue(gate["ok"])
        self.assertEqual(gate["pose"]["rotation_heading_tracks"], 462)
        self.assertEqual(gate["pose"]["rotation_heading_source"], "optical_flow_yaw")
        self.assertGreater(bridge.pose_gate_camera_heading(gate)[2], 0.05)

    def test_all_taught_arrival_paths_require_endpoint_verification(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn('"strict_radius_endpoint_verified"', source)
        self.assertIn('"visual_checkpoint_endpoint_verified"', source)
        self.assertIn("and final_endpoint_ready", source)
        self.assertIn("require_endpoint_verified=True", source)
        self.assertIn("progress-independent whole-leg image matches", source)
        self.assertIn("another forward pulse", source)
        self.assertIn("or visual_checkpoint_arrived", source)
        self.assertGreaterEqual(
            source.count("endpoint_leg_index=endpoint_leg_index"),
            4,
        )

    def test_failed_run_cannot_leave_leg_one_at_0862_progress(self):
        leg_length = 2.589597223809809
        failed_run_progress = 0.8616640950860521
        remaining_distance = leg_length * (1.0 - failed_run_progress)
        _hard_radius, strict_radius = bridge.mission_arrival_radii(
            0.24,
            0.14,
            patrol_stage="loop",
            strict_target=bridge.taught_leg_requires_precise_arrival(
                {"from_point": 1, "to_point": 2}
            ),
        )

        self.assertGreater(remaining_distance, strict_radius)
        self.assertGreater(remaining_distance, 0.35)

    def test_zero_motion_probe_never_divides_by_zero(self):
        axis, distance = bridge.observed_motion_axis(
            [-5.320870510965717, -0.046353890460460206, 0.0015417021526944463],
            [-5.320870510965717, -0.046353890460460206, 0.0015417021526944463],
            min_delta=0.0,
        )
        self.assertIsNone(axis)
        self.assertEqual(distance, 0.0)

    def test_taught_baseline_live_playback_handles_room_only_poses(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("function lerpOptionalVec(a, b, t, fallback = null)", source)
        self.assertIn("center: lerpOptionalVec(a.center, b.center, u, nearest.center)", source)
        self.assertIn("formatVector(cur.rcenter || cur.center)", source)
        self.assertIn("Captured DJI frames drive a live-timed corrected-pose simulation", source)

    def test_new_live_path_simulation_is_incremental_and_locked(self):
        source = APP_PATH.read_text(encoding="utf-8")
        server = SERVER_PATH.read_text(encoding="utf-8")
        html = (APP_PATH.parent / "index.html").read_text(encoding="utf-8")
        self.assertIn("Create New Live Path", html)
        self.assertIn('value="10" selected>10 FPS (maximum)', html)
        self.assertIn("/api/drone/simulate-patrol-baseline", source)
        self.assertIn("def localized_recorded_patrol_job(", server)
        self.assertIn('baseline_replay_id != lock["baseline_replay_id"]', server)
        self.assertIn('"genuine_localization": True', server)
        self.assertIn('"--query-frames", query_frames', server)
        self.assertIn('"--partial-pose-out", partial_pose_path', server)
        self.assertIn('cmd.extend(["--taught-patrol-recovery-bank", recovery_bank])', server)
        self.assertIn('recorded_timing: true', source)
        self.assertIn('laps: 2', source)
        self.assertIn("selectedLivePatrolProfile()", source)
        self.assertIn("patrolProfile?.baseline_replay_id", source)
        self.assertIn(
            'item?.kind === "route_constrained_taught_baseline"', source
        )
        self.assertIn("def build_recorded_patrol_lap_sequence(", server)
        self.assertIn('"simulated_sequence_frame": last_frame', server)
        self.assertIn('"timing_mode": "recorded" if recorded_timing else "fixed_fps"', server)
        self.assertIn("lambda: localized_recorded_patrol_job(", server)
        self.assertIn('query_frames.glob("query_*.jpg")', server)
        self.assertIn('phase_boundaries.get("point1_return")', server)
        self.assertIn("return 0, True", server)
        self.assertIn("lap_two_metric_ready = laps < 2", server)
        self.assertIn('"require_metric_pose": metric_checkpoint_active', server)
        self.assertIn("composite_point1_seam_verified = False", server)
        self.assertIn("and not composite_point1_seam_verified", server)
        self.assertIn('"simulation_metric_checkpoint_bypassed": bool(', server)
        self.assertIn(
            'cmd.append("--wait-for-metric-checkpoint-recovery")', server
        )
        self.assertIn("Live-clock simulation:", source)
        self.assertIn("Simulated live input:", source)

    def test_live_renderer_reuses_map_geometry_and_displays_every_exact_frame_pose(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('launchParams.get("patrol-view") === "1"', source)
        self.assertIn('showDemo({ push: false, resetVideo: false })', source)
        self.assertIn("function updateLiveRoomPoseStream(sourcePoses)", source)
        self.assertIn("if (!updateLiveRoomPoseStream(poses)) room = buildRoomFrame();", source)
        self.assertIn("function enqueueLiveFrameLockedPoses(displayPoses, payload, stream = null)", source)
        self.assertIn("function updateLiveFrameViewForPose(pose, stream = null)", source)
        self.assertIn("LIVE_FRAME_MAX_BUFFER = 12", source)
        self.assertIn("LIVE_FRAME_TARGET_BUFFER = 1", source)
        self.assertIn("SIMULATED_LIVE_FRAME_MAX_BUFFER = 180", source)
        self.assertIn("SIMULATED_LIVE_FRAME_TARGET_BUFFER = 24", source)
        self.assertIn("function liveFramePlaybackIntervalMs(a, b)", source)
        self.assertIn('recordedLiveFlight = replay?.kind === "simulated_live_tsolve_recorded_frames"', source)
        self.assertIn("applyLanding: completedReplayDisplay && !recordedLiveFlight", source)
        self.assertIn("async function pollLivePoseStream()", source)
        self.assertIn("pollLivePoseStream();\n}, 100);", source)
        self.assertIn('setInterval(() => pollStatus(true), 2000);', source)
        self.assertIn('compact ? "/api/status?compact=1" : "/api/status"', source)
        self.assertIn("/api/live-replay?after=${Math.max(0, livePoseStreamCount)}", source)
        self.assertIn("poses = canAppendDelta ? poses.concat(incomingPoses) : incomingPoses", source)
        self.assertIn("liveFrameLockedQueue.splice(0, dropCount)", source)
        self.assertIn("liveFrameLockedNextAtMs = performance.now()", source)
        self.assertIn("preloadLiveFrame(pose, liveFrameLockedStream)", source)
        self.assertIn("const canReuseLoadedMap = Boolean(", source)
        self.assertIn("room.poses = []", source)
        self.assertIn("if (liveRouteRenderingActive()) advanceLiveFrameLockedPlayback();", source)
        self.assertIn("getCurrentPose: () => liveRouteRenderingActive()", source)
        self.assertIn("replayFrameHoldTimeSec = Number(completedLiveFramePose.time_sec)", source)
        self.assertIn("currentLiveDisplayPose(trustedCur)", source)
        self.assertIn("liveCurrentPoseOverride = correctedLatestPose", source)
        self.assertIn("pose.rotation_position_locked", source)
        self.assertIn("corrected.rheading = opticalHeading", source)
        self.assertIn("useDroneYawSmoothing: () => false", source)
        self.assertIn('liveRouteRenderingActive() ? "live-route"', source)
        live_update = source[source.index("async function loadLiveReplayPartial"):source.index("async function pollStatus")]
        self.assertNotIn("invalidateStaticLayer();", live_update)
        self.assertNotIn("renderStartPreview();", live_update)

    def test_frozen_rotation_position_is_never_accepted_for_translation(self):
        issue = bridge.stabilized_pose_safety_issue(
            {
                "rcenter": [-1.608, -0.224, -0.202],
                "rotation_raw_rcenter": [-0.977, 0.175, -0.279],
                "rotation_position_locked": True,
                "translation_allowed": False,
            }
        )
        self.assertIn("rotation-frozen", issue)

    def test_large_raw_published_pose_disagreement_is_rejected(self):
        issue = bridge.stabilized_pose_safety_issue(
            {
                "rcenter": [-3.211, -0.562, -0.464],
                "rotation_raw_rcenter": [-2.571, -0.168, -0.536],
            }
        )
        self.assertIn("disagree", issue)

    def test_controller_proven_post_yaw_bias_can_translate_after_release(self):
        issue = bridge.stabilized_pose_safety_issue(
            {
                "rcenter": [-0.7546984849, -0.0802621970, -0.4813365774],
                "rotation_raw_rcenter": [-0.7546984849, -0.0802621970, 0.4976634226],
                "rotation_position_locked": False,
                "translation_allowed": True,
                "rotation_position_source": "post_yaw_room_bias",
                "rotation_anchor_is_position_truth": True,
                "rotation_anchor_commanded": True,
            }
        )
        self.assertIsNone(issue)

    def test_uncommanded_post_yaw_bias_cannot_bypass_disagreement_guard(self):
        issue = bridge.stabilized_pose_safety_issue(
            {
                "rcenter": [0.0, 0.0, 0.0],
                "rotation_raw_rcenter": [0.0, 0.0, 0.979],
                "rotation_position_source": "post_yaw_room_bias",
                "rotation_anchor_is_position_truth": True,
                "rotation_anchor_commanded": False,
            }
        )
        self.assertIn("disagree", issue)

    def test_verified_visual_route_pose_can_translate_after_two_strong_hits(self):
        issue = bridge.stabilized_pose_safety_issue(
            {
                "pose_source": "patrol_visual_route_recovery",
                "route_visual_verified": True,
                "route_visual_progress": 0.42,
                "route_visual_inliers": 240,
                "route_visual_minimum_inliers": 120,
                "route_visual_acquisition_hits": 2,
                "rcenter": [-3.1, 0.0, 0.5],
                "rotation_raw_rcenter": [-3.1, 0.0, 0.5],
                "translation_allowed": True,
            }
        )
        self.assertIsNone(issue)

    def test_command_bounded_temporal_recovery_passes_final_motion_gate(self):
        issue = bridge.stabilized_pose_safety_issue(
            {
                "pose_source": "patrol_visual_route_recovery",
                "route_visual_verified": True,
                "route_visual_progress": 0.3717664674482947,
                "route_visual_inliers": 96,
                "route_visual_minimum_inliers": 90,
                "route_visual_acquisition_hits": 5,
                "route_visual_temporal_recovery": True,
                "route_visual_temporal_recovery_required_hits": 5,
                "route_visual_monitor_required": True,
                "route_visual_monitor_verified": True,
                "route_visual_monitor_inliers": 96,
                "route_visual_monitor_minimum_inliers": 90,
                "route_visual_monitor_temporal_recovery": True,
                "rcenter": [-0.5888, -0.0803, 0.0490],
                "rotation_raw_rcenter": [-0.5888, -0.0803, 0.0490],
                "translation_allowed": True,
            }
        )
        self.assertIsNone(issue)

    def test_point_four_temporal_recovery_uses_only_its_audited_50_inlier_gate(self):
        pose = {
            "pose_source": "patrol_visual_route_recovery",
            "route_visual_verified": True,
            "route_visual_progress": 0.10,
            "route_visual_inliers": 55,
            "route_visual_minimum_inliers": 50,
            "route_visual_acquisition_hits": 5,
            "route_visual_temporal_recovery": True,
            "route_visual_temporal_recovery_required_hits": 5,
            "route_visual_monitor_required": True,
            "route_visual_monitor_verified": True,
            "route_visual_monitor_leg_index": 4,
            "route_visual_monitor_inliers": 55,
            "route_visual_monitor_minimum_inliers": 50,
            "route_visual_monitor_temporal_recovery": True,
            "rcenter": [-3.09, -0.08, 0.96],
            "rotation_raw_rcenter": [-3.09, -0.08, 0.96],
            "translation_allowed": True,
        }
        self.assertIsNone(bridge.stabilized_pose_safety_issue(pose))
        self.assertIn(
            "below the 90 gate",
            bridge.stabilized_pose_safety_issue(
                {**pose, "route_visual_monitor_leg_index": 3}
            ),
        )

    def test_unconfirmed_visual_route_pose_never_authorizes_translation(self):
        issue = bridge.stabilized_pose_safety_issue(
            {
                "pose_source": "patrol_visual_route_recovery",
                "route_visual_verified": True,
                "route_visual_progress": 0.42,
                "route_visual_inliers": 240,
                "route_visual_minimum_inliers": 120,
                "route_visual_acquisition_hits": 1,
                "rcenter": [-3.1, 0.0, 0.5],
                "rotation_raw_rcenter": [-3.1, 0.0, 0.5],
            }
        )
        self.assertIn("two consistent frames", issue)

    def test_healthy_tsolve_does_not_freeze_on_weak_visual_supervision(self):
        issue = bridge.stabilized_pose_safety_issue(
            {
                "pose_source": None,
                "route_visual_monitor_required": True,
                "route_visual_monitor_leg_index": 2,
                "route_visual_monitor_verified": False,
                "route_visual_monitor_reason": "visual_route_inliers_below_threshold",
                "route_visual_monitor_inliers": 70,
                "route_visual_monitor_minimum_inliers": 120,
                "translation_allowed": True,
            }
        )
        self.assertIsNone(issue)

    def test_visual_route_reference_authority_uses_guarded_leg_four_fallback(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        start = source.index("        # Keep the established 1->2->3 metric path unchanged")
        end = source.index("            route_key = LivePatrolRouteGate._key", start)
        gate = source[start:end]
        self.assertIn("reference_frames_enabled", gate)
        self.assertIn("visual_route_position_recovery_needed(", gate)
        self.assertIn("and should_match_route_position", gate)
        self.assertNotIn("or current_pool is None", gate)
        self.assertNotIn("< required_tracking_points(", gate)

        reference_gate = source[
            source.index("def patrol_reference_frames_enabled("):
            source.index("def visual_recovery_supersedes_stalled_metric_pose(")
        ]
        self.assertIn("{1, 2, 3, 4}", reference_gate)
        self.assertIn("def visual_route_position_recovery_needed(", reference_gate)
        self.assertIn('route_context.get("recovery_hover") is True', reference_gate)
        self.assertIn("or force_route_taught_recovery", reference_gate)
        self.assertIn('int(route_context.get("leg_index") or 0) == 3', reference_gate)
        self.assertIn(
            'route_context.get("controller_translation_locked") is not True',
            reference_gate,
        )

    def test_controller_proven_post_yaw_offset_reaches_position_reanchor(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        start = source.index("def route_guarded_output_rejection(")
        end = source.index("def mark_global_recovery_pool(", start)
        guard = source[start:end]
        self.assertIn("post_yaw_reanchor_cap: float = 0.0", guard)
        self.assertIn('previous_pose.get("rotation_position_locked") is True', guard)
        self.assertIn('previous_pose.get("rotation_anchor_commanded") is True', guard)
        self.assertIn('"post_yaw_controller_reanchor"', guard)
        call = source[source.index("output_rejection_reason, last_output_center"):]
        self.assertIn(
            "post_yaw_reanchor_cap=args.patrol_turn_max_position_drift",
            call,
        )

    def test_yaw_only_recovery_is_throttled_without_delaying_translation_recovery(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn('"--rotation-recovery-cooldown-frames"', source)
        self.assertIn(
            'route_context.get("controller_translation_locked") is True',
            source,
        )
        self.assertIn(
            "frame_idx - last_rotation_taught_recovery_attempt_frame",
            source,
        )
        self.assertIn('taught_stage = {"reason": "rotation_recovery_cooldown"}', source)
        # The cooldown predicate is controller-lock scoped, so the first
        # translation frame bypasses it and retries metric recovery at once.
        cooldown_start = source.index("                rotation_recovery_cooling_down = bool(")
        cooldown_end = source.index("                    for taught_recovery in recovery_banks:", cooldown_start)
        cooldown = source[cooldown_start:cooldown_end]
        self.assertIn('controller_translation_locked") is True', cooldown)

    def test_repeated_route_rejection_quarantines_aliased_optical_pool(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn('"--route-rejection-recovery-cooldown-frames"', source)
        self.assertEqual(source.count("force_route_taught_recovery = True"), 2)
        recovery_start = source.index("                route_recovery_due = bool(")
        recovery_end = source.index(
            "                proactive_cooldown = int(", recovery_start
        )
        recovery = source[recovery_start:recovery_end]
        self.assertIn('if "full_loop" in recovery_bank.bank_path.name', recovery)
        self.assertIn("*(full_loop_route_banks or taught_recoveries[1:])", recovery)
        self.assertIn("online_recovery", recovery)
        self.assertIn("pool = route_recovered_pool", recovery)
        self.assertNotIn("merge_verified_tracking_pool", recovery)
        self.assertIn("stable_solve_point3d_ids = None", recovery)
        self.assertIn('pending_global["ignored"] = True', recovery)
        self.assertIn(
            'method = "route_rejection_taught_consensus_recovery"', recovery
        )

    def test_live_tsolve_miss_fails_fast_and_preserves_optical_chain(self):
        localizer = LOCALIZER_PATH.read_text(encoding="utf-8")
        runner = (
            LOCALIZER_PATH.parents[0] / "run_live_tsolve_existing_map_stream.py"
        ).read_text(encoding="utf-8")
        self.assertIn("fork_on_miss: bool = True", runner)
        self.assertEqual(runner.count("fork_on_miss=bool(fork_on_miss)"), 3)
        self.assertEqual(
            localizer.count("fork_on_miss=not interactive_recovery"),
            2,
        )
        self.assertIn('reference_update_reason = "tsolve_miss_flow_anchor_reselect"', localizer)
        miss_start = localizer.index("            # An algebraic TSolve miss does not invalidate")
        miss_end = localizer.index("\n        if (", miss_start)
        miss = localizer[miss_start:miss_end]
        self.assertIn("current_pool = cap_tracking_pool(", miss)
        self.assertIn("prev_gray = curr_gray", miss)
        self.assertIn("stable_solve_reset = True", miss)
        self.assertIn("stable_solve_point3d_ids = None", miss)

    def test_recorded_simulation_uses_same_fixed_altitude_as_real_patrol(self):
        server = SERVER_PATH.read_text(encoding="utf-8")
        builder = (
            LOCALIZER_PATH.parents[0] / "build_route_constrained_patrol_baseline.py"
        ).read_text(encoding="utf-8")
        self.assertIn("route_points = [", builder)
        self.assertIn("[float(point[0]), float(cruise_y), float(point[2])]", builder)
        self.assertIn('"from": route_points[point_index]', builder)
        self.assertIn("anchor[1] = baseline_cruise_y", server)
        self.assertIn("target[1] = baseline_cruise_y", server)

        reference = json.loads(FULL_PATROL_BASELINE_REFERENCE.read_text(encoding="utf-8"))
        cruise_y = 0.010052834697950513
        endpoint_y = [
            endpoint[1]
            for leg in reference["legs"]
            for endpoint in (leg["from"], leg["to"])
        ]
        self.assertTrue(endpoint_y)
        self.assertTrue(all(abs(value - cruise_y) < 1e-9 for value in endpoint_y))

    def test_conservative_visual_progress_does_not_delay_metric_tsolve_position(self):
        issue = bridge.stabilized_pose_safety_issue(
            {
                "pose_source": None,
                "route_visual_monitor_required": True,
                "route_visual_monitor_leg_index": 1,
                "route_visual_monitor_verified": True,
                "route_visual_monitor_inliers": 173,
                "route_visual_monitor_minimum_inliers": 120,
                "route_visual_monitor_progress": 0.285,
                "route_visual_monitor_tsolve_progress": 0.729,
                "route_visual_monitor_disagreement_m": 1.151,
                "rcenter": [-1.35, -0.08, -0.45],
                "translation_allowed": True,
            }
        )
        self.assertIsNone(issue)

    def test_verified_baseline_blocks_only_large_skipped_frame_catchup(self):
        pose = {
            "route_visual_monitor_required": True,
            "route_visual_monitor_verified": True,
            "route_visual_monitor_inliers": 178,
            "route_visual_monitor_minimum_inliers": 120,
            "route_visual_monitor_disagreement_m": 0.654,
            "route_visual_monitor_leg_index": 1,
        }
        self.assertIsNone(
            bridge.baseline_supervised_pose_jump_issue(
                pose,
                step=0.18,
                base_step_limit=0.30,
            )
        )
        issue = bridge.baseline_supervised_pose_jump_issue(
            pose,
            step=0.454,
            base_step_limit=0.30,
        )
        self.assertIn("baseline contradicts", issue)
        self.assertIn("refusing translation", issue)

    def test_pose_gate_preserves_baseline_supervision_for_controller(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        payload = source[source.index("    def pose_payload("):source.index("    if not (latest_pose.get", source.index("    def pose_payload("))]
        self.assertIn('"route_visual_monitor_verified": pose.get(', payload)
        self.assertIn('"route_visual_monitor_disagreement_m": pose.get(', payload)
        self.assertIn('"route_visual_temporal_recovery": pose.get(', payload)
        self.assertIn(
            '"route_visual_monitor_temporal_recovery": pose.get(',
            payload,
        )

    def test_verified_patrol_leg_supervision_authorizes_translation(self):
        issue = bridge.stabilized_pose_safety_issue(
            {
                "pose_source": "patrol_visual_route_recovery",
                "route_visual_verified": True,
                "route_visual_progress": 0.62,
                "route_visual_inliers": 180,
                "route_visual_minimum_inliers": 120,
                "route_visual_acquisition_hits": 2,
                "route_visual_monitor_required": True,
                "route_visual_monitor_leg_index": 2,
                "route_visual_monitor_verified": True,
                "route_visual_monitor_inliers": 180,
                "route_visual_monitor_minimum_inliers": 120,
                "route_visual_monitor_disagreement_m": 0.65,
                "rcenter": [-2.1, 0.0, 1.05],
                "rotation_raw_rcenter": [-2.1, 0.0, 1.05],
                "translation_allowed": True,
            }
        )
        self.assertIsNone(issue)

    def test_rotation_locked_pose_authorizes_yaw_but_never_forward(self):
        gate = {
            "ok": True,
            "pose": {
                "rcenter": [-1.608, -0.224, -0.202],
                "rotation_raw_rcenter": [-0.977, 0.175, -0.279],
                "rotation_position_locked": True,
                "translation_allowed": False,
            },
        }
        self.assertIsNone(bridge.guided_command_pose_safety_issue(gate, yaw=0.0325))
        self.assertIn(
            "rotation-frozen",
            bridge.guided_command_pose_safety_issue(gate, bf=0.035),
        )

    def test_reference_141750_locked_frames_are_yaw_only_eligible(self):
        payload = json.loads(LIVE_ATLAS_141750_POSES.read_text(encoding="utf-8"))
        locked = [pose for pose in payload["poses"] if pose.get("rotation_position_locked")]
        self.assertGreater(len(locked), 100)
        max_gap = 0.0
        for pose in locked:
            gate = {"ok": True, "pose": pose}
            self.assertIsNone(bridge.guided_command_pose_safety_issue(gate, yaw=0.0325))
            self.assertIsNotNone(bridge.guided_command_pose_safety_issue(gate, bf=0.035))
            raw = pose.get("rotation_raw_rcenter")
            published = pose.get("rcenter")
            if raw and published:
                max_gap = max(max_gap, math.hypot(raw[0] - published[0], raw[2] - published[2]))
        self.assertGreater(max_gap, 0.60)

    def test_patrol_gate_preserves_rotation_diagnostics_for_command_boundary(self):
        now = time.time()
        payload = {
            "updated_at": now,
            "processed_count": 1,
            "poses": [{
                "instance_id": "instance_000001",
                "success": True,
                "held_pose": False,
                "rcenter": [1.0, 0.0, 2.0],
                "rheading": [0.0, 0.0, 1.0],
                "received_unix": now,
                "rotation_raw_rcenter": [1.6, 0.0, 2.2],
                "rotation_position_locked": True,
                "translation_allowed": False,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses_partial.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            default_gate = bridge.latest_tsolve_pose_gate(path, max_age_seconds=1.0)
            patrol_gate = bridge.latest_tsolve_pose_gate(
                path,
                max_age_seconds=1.0,
                allow_rotation_frozen=True,
            )
        self.assertFalse(default_gate["ok"])
        self.assertTrue(patrol_gate["ok"])
        self.assertTrue(patrol_gate["pose"]["rotation_position_locked"])
        self.assertEqual(patrol_gate["pose"]["rotation_raw_rcenter"], [1.6, 0.0, 2.2])

    def test_fresh_locked_rotation_handoff_can_yaw_but_never_translate(self):
        now = time.time()
        trusted = {
            "instance_id": "visual_route_001190",
            "success": True,
            "held_pose": False,
            "rcenter": [-0.648, -0.080, -0.488],
            "rheading": [1.0, 0.0, 0.0],
            "received_unix": now - 0.5,
        }
        holds = []
        for index in range(20):
            holds.append({
                **trusted,
                "instance_id": f"hold_{index:06d}",
                "success": False,
                "held_pose": True,
                "received_unix": now - 0.1,
                "rotation_heading": [0.98, 0.0, 0.20],
                "rotation_heading_tracks": 500,
                "rotation_position_locked": True,
                "translation_allowed": False,
                "hold_reason": (
                    "missing_tracking_anchor_after_rejection_"
                    "holding_trusted_position_with_fresh_rotation_heading"
                ),
            })
        payload = {
            "updated_at": now,
            "processed_count": len(holds) + 1,
            "poses": [trusted, *holds],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses_partial.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            default_gate = bridge.latest_tsolve_pose_gate(
                path,
                max_age_seconds=1.0,
            )
            patrol_gate = bridge.latest_tsolve_pose_gate(
                path,
                max_age_seconds=1.0,
                allow_rotation_frozen=True,
            )
        self.assertFalse(default_gate["ok"])
        self.assertTrue(patrol_gate["ok"])
        self.assertTrue(patrol_gate["rotation_handoff_hold"])
        self.assertEqual(patrol_gate["trailing_hold_frames"], 20)
        self.assertIsNone(
            bridge.guided_command_pose_safety_issue(patrol_gate, yaw=0.03)
        )
        self.assertIn(
            "rotation-frozen",
            bridge.guided_command_pose_safety_issue(patrol_gate, bf=0.03),
        )

    def test_rotation_handoff_requires_strong_optical_heading(self):
        now = time.time()
        payload = {
            "updated_at": now,
            "processed_count": 1,
            "poses": [{
                "instance_id": "hold_000001",
                "success": False,
                "held_pose": True,
                "rcenter": [-0.648, -0.080, -0.488],
                "received_unix": now,
                "rotation_heading": [1.0, 0.0, 0.0],
                "rotation_heading_tracks": 15,
                "rotation_position_locked": True,
                "translation_allowed": False,
                "hold_reason": (
                    "missing_tracking_anchor_after_rejection_"
                    "holding_trusted_position_with_fresh_rotation_heading"
                ),
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses_partial.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            gate = bridge.latest_tsolve_pose_gate(
                path,
                max_age_seconds=1.0,
                allow_rotation_frozen=True,
            )
        self.assertFalse(gate["ok"])

    def test_loop_keeps_exact_command_order(self):
        commands = [{"type": "gate"}, {"type": "cruise", "to": [1, 0, 0]}, {"type": "hover"}]
        observed = list(itertools.islice(bridge.mission_step_sequence(commands, True), 6))
        self.assertEqual([index for index, _step in observed], [0, 1, 2, 0, 1, 2])

    def test_non_loop_stops_after_one_pass(self):
        commands = [{"type": "gate"}, {"type": "cruise"}]
        self.assertEqual(list(bridge.mission_step_sequence(commands, False)), list(enumerate(commands)))

    def test_two_laps_run_entry_once_and_add_point1_star_connector(self):
        commands = [
            {"type": "gate"},
            {
                "type": "cruise",
                "title": "Entry to point 1",
                "from": [-1.0, 0.0, 0.0],
                "to": [0.0, 0.0, 0.0],
            },
            {"type": "hover"},
            {"type": "yaw"},
            {
                "type": "cruise",
                "title": "Point 1 to point 2",
                "from": [0.0, 0.0, 0.0],
                "to": [2.0, 0.0, 0.0],
            },
            {"type": "hover"},
        ]
        observed = list(bridge.mission_step_sequence(commands, True, 2, 4))
        self.assertEqual(
            [index for index, _step in observed],
            [0, 1, 2, 3, 4, 5, 4, 4, 4, 5],
        )
        self.assertEqual(
            sum(
                step.get("title") == "Entry to point 1"
                for _index, step in observed
            ),
            1,
        )
        marker = observed[6][1]
        connector = observed[7][1]
        second_lap_cruise = observed[8][1]
        self.assertEqual(marker["type"], "lap_relocalize_entry")
        self.assertTrue(marker["_atlas_lap_start"])
        self.assertTrue(connector["_atlas_lap_reentry"])
        self.assertEqual(connector["to"], [0.0, 0.0, 0.0])
        self.assertEqual(second_lap_cruise["_atlas_lap_number"], 2)
        self.assertNotIn("_atlas_lap_start", second_lap_cruise)

    def test_connected_patrol_relocalizes_point1_star_then_replays_guarded_point1_entry(self):
        reference = json.loads(FULL_PATROL_BASELINE_REFERENCE.read_text(encoding="utf-8"))
        commands = [
            {"type": "gate", "title": "Patrol gate"},
            {"type": "yaw", "title": "Entry yaw"},
            {
                "type": "cruise",
                "title": "Entry cruise",
                "from": [-4.0, 0.0, -0.5],
                "to": reference["legs"][0]["from"],
                "patrol_stage": "entry",
            },
            {"type": "hover", "title": "Verify Point 1"},
        ]
        for leg in reference["legs"]:
            commands.extend(
                [
                    {
                        "type": "yaw",
                        "title": f"Loop yaw {leg['from_point']}->{leg['to_point']}",
                        "patrol_stage": "loop",
                    },
                    {
                        "type": "cruise",
                        "title": f"Loop cruise {leg['from_point']}->{leg['to_point']}",
                        "from": leg["from"],
                        "to": leg["to"],
                        "from_point": leg["from_point"],
                        "to_point": leg["to_point"],
                        "patrol_stage": "loop",
                    },
                    {"type": "hover", "title": "Relocalize", "patrol_stage": "loop"},
                ]
            )
        loop_start = bridge.patrol_loop_start_command_index(commands, reference)
        self.assertEqual(commands[loop_start]["title"], "Loop yaw 1->2")
        observed = list(bridge.mission_step_sequence(commands, True, 2, loop_start))
        self.assertEqual(sum(step["title"] == "Entry cruise" for _index, step in observed), 1)
        self.assertEqual(
            sum(step.get("type") == "lap_relocalize_entry" for _index, step in observed),
            1,
        )
        self.assertEqual(
            sum(
                step.get("_atlas_lap_reentry") is True
                and step.get("type") == "cruise"
                for _index, step in observed
            ),
            1,
        )
        self.assertEqual(sum(step["title"] == "Loop yaw 1->2" for _index, step in observed), 2)
        self.assertEqual(
            [
                (step["from_point"], step["to_point"])
                for _index, step in observed
                if step.get("from_point") is not None
            ],
            [(1, 2), (2, 3), (3, 4), (4, 1)] * 2,
        )

    def test_pinned_patrol_executes_all_four_legs_in_each_of_exactly_two_laps(self):
        reference = json.loads(FULL_PATROL_BASELINE_REFERENCE.read_text(encoding="utf-8"))
        commands = [
            {
                "type": "cruise",
                "from_point": leg["from_point"],
                "to_point": leg["to_point"],
                "from": leg["from"],
                "to": leg["to"],
            }
            for leg in reference["legs"]
        ]
        observed = list(bridge.mission_step_sequence(commands, True, 2, 0))
        self.assertEqual(
            [
                (step["from_point"], step["to_point"])
                for _index, step in observed
                if step.get("from_point") is not None
            ],
            [(1, 2), (2, 3), (3, 4), (4, 1)] * 2,
        )
        self.assertEqual(
            sum(step.get("type") == "lap_relocalize_entry" for _index, step in observed),
            1,
        )
        self.assertEqual(
            sum(step.get("_atlas_lap_reentry") is True for _index, step in observed),
            1,
        )
        loop_steps = [
            step
            for _index, step in observed
            if step.get("from_point") is not None
        ]
        lap_one_coordinates = [
            (step.get("from"), step.get("to"))
            for step in loop_steps[:4]
        ]
        lap_two_coordinates = [
            (step.get("from"), step.get("to"))
            for step in loop_steps[4:8]
        ]
        self.assertEqual(lap_two_coordinates, lap_one_coordinates)
        self.assertEqual(
            lap_two_coordinates,
            [(leg["from"], leg["to"]) for leg in reference["legs"]],
        )

    def test_loop_start_is_found_from_recorded_point_one_to_two_leg(self):
        reference = {
            "legs": [
                {
                    "from_point": 1,
                    "to_point": 2,
                    "from": [0.0, 0.0, 0.0],
                    "to": [2.0, 0.0, 0.0],
                }
            ]
        }
        commands = [
            {"type": "gate"},
            {"type": "cruise", "from": [-1.0, 0.0, 0.0], "to": [0.0, 0.0, 0.0]},
            {"type": "hover"},
            {"type": "yaw"},
            {"type": "cruise", "from": [0.0, 0.0, 0.0], "to": [2.0, 0.0, 0.0]},
        ]
        self.assertEqual(bridge.patrol_loop_start_command_index(commands, reference), 3)

    def test_matching_entry_leg_is_not_mistaken_for_the_closed_circle(self):
        reference = {
            "legs": [
                {"from_point": 1, "to_point": 2, "from": [0, 0, 0], "to": [2, 0, 0]},
                {"from_point": 2, "to_point": 3, "from": [2, 0, 0], "to": [2, 0, 2]},
                {"from_point": 3, "to_point": 4, "from": [2, 0, 2], "to": [0, 0, 2]},
                {"from_point": 4, "to_point": 1, "from": [0, 0, 2], "to": [0, 0, 0]},
            ]
        }
        commands = [
            {"type": "cruise", "title": "Patrol cruise 1", "from": [0, 0, 2], "to": [0, 0, 0]},
            {"type": "cruise", "title": "Patrol cruise 2", "from": [0, 0, 0], "to": [2, 0, 0]},
            {"type": "cruise", "title": "Patrol cruise 3", "from": [2, 0, 0], "to": [2, 0, 2]},
            {"type": "cruise", "title": "Patrol cruise 4", "from": [2, 0, 2], "to": [0, 0, 2]},
            {"type": "cruise", "title": "Patrol cruise 5", "from": [0, 0, 2], "to": [0, 0, 0]},
        ]
        self.assertEqual(bridge.patrol_loop_start_command_index(commands, reference), 1)

    def test_command_side_route_gate_rejects_backward_and_turn_translation(self):
        reason, progress = bridge.patrol_route_pose_rejection(
            [0.7, 0.0, 0.0],
            segment_start=[0.0, 0.0, 0.0],
            segment_end=[2.0, 0.0, 0.0],
            previous_progress=0.50,
            translation_locked=False,
            position_anchor=None,
            max_cross_track=0.55,
        )
        self.assertIn("moved backward", reason)
        self.assertEqual(progress, 0.50)
        reason, _progress = bridge.patrol_route_pose_rejection(
            [0.30, 0.0, 0.0],
            segment_start=[0.0, 0.0, 0.0],
            segment_end=[2.0, 0.0, 0.0],
            previous_progress=0.0,
            translation_locked=True,
            position_anchor=[0.0, 0.0, 0.0],
            max_cross_track=0.55,
        )
        self.assertIn("turn position drift", reason)

    def test_every_patrol_translation_pulse_requires_new_published_progress(self):
        self.assertTrue(
            bridge.published_position_advanced_toward_target(
                [0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            )
        )
        self.assertFalse(
            bridge.published_position_advanced_toward_target(
                [0.0, 0.0, 0.0],
                [0.005, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            )
        )
        self.assertFalse(
            bridge.published_position_advanced_toward_target(
                [0.0, 0.0, 0.0],
                [0.007, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            )
        )
        self.assertTrue(
            bridge.published_position_advanced_toward_target(
                [0.0, 0.0, 0.0],
                [0.021, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            )
        )
        self.assertIsNone(
            bridge.patrol_translation_pulse_progress_issue(
                [0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                got_new_pose=True,
                maximum_pose_step=0.30,
            )
        )
        frozen_issue = bridge.patrol_translation_pulse_progress_issue(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            got_new_pose=True,
            maximum_pose_step=0.30,
        )
        self.assertIn("after one translation pulse", frozen_issue)
        stale_issue = bridge.patrol_translation_pulse_progress_issue(
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            got_new_pose=False,
            maximum_pose_step=0.30,
        )
        self.assertIn("no newly processed", stale_issue)
        jump_issue = bridge.patrol_translation_pulse_progress_issue(
            [0.0, 0.0, 0.0],
            [0.50, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            got_new_pose=True,
            maximum_pose_step=0.30,
        )
        self.assertIn("pose jump", jump_issue)
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("horizontal_pulse\n                            and one_pulse_pose_confirmation", source)
        self.assertIn("require_observed_translation_progress=True", source)
        self.assertIn("abort_on_timeout=not one_pulse_pose_confirmation", source)
        self.assertIn('"translation_progress"', source)

    def test_point_two_overshoot_uses_bounded_reverse_not_permanent_hold(self):
        start = [-3.2329557447702215, -0.0802621969998909, -0.33236579860361815]
        end = [-0.6480244338911889, -0.0802621969998909, -0.48774887233005093]
        failed_position = [-0.37185170854332217, -0.017334588638940357, -0.4238969308288887]
        overshoot = bridge.patrol_endpoint_overshoot_distance(
            failed_position,
            start,
            end,
        )
        self.assertAlmostEqual(overshoot, 0.27184, places=3)

        rejected, _ = bridge.patrol_route_pose_rejection(
            end,
            segment_start=start,
            segment_end=end,
            previous_progress=1.1068575203173658,
            translation_locked=False,
            position_anchor=None,
            max_cross_track=0.55,
        )
        accepted, progress = bridge.patrol_route_pose_rejection(
            end,
            segment_start=start,
            segment_end=end,
            previous_progress=1.1068575203173658,
            translation_locked=False,
            position_anchor=None,
            max_cross_track=0.55,
            endpoint_overshoot_correction=True,
        )
        self.assertIn("moved backward", rejected)
        self.assertIsNone(accepted)
        self.assertAlmostEqual(progress, 1.0)
        intermediate = [
            start[index] + 1.09 * (end[index] - start[index])
            for index in range(3)
        ]
        accepted, progress = bridge.patrol_route_pose_rejection(
            intermediate,
            segment_start=start,
            segment_end=end,
            previous_progress=1.1068575203173658,
            translation_locked=False,
            position_anchor=None,
            max_cross_track=0.55,
            endpoint_overshoot_correction=True,
        )
        self.assertIsNone(accepted)
        self.assertAlmostEqual(progress, 1.09)

    def test_endpoint_correction_commands_negative_body_forward(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        start = source.index("if active_endpoint_overshoot_correction:")
        block = source[start:source.index("forward_gain =", start)]
        self.assertIn("bf_rc = -max_forward_rc * reverse_scale", block)
        self.assertIn('"bounded_endpoint_reverse_correction"', block)
        self.assertIn("endpoint_overshoot_distance > 0.40", source)

    def test_endpoint_undershoot_accepts_three_cm_and_allows_one_eight_cm_retry(self):
        self.assertFalse(
            bridge.patrol_endpoint_undershoot_correction_allowed(
                0.03,
                0.03,
                0.15,
                endpoint_arrived=False,
                retry_used=False,
            )
        )
        self.assertFalse(
            bridge.patrol_endpoint_undershoot_correction_allowed(
                0.12,
                0.12,
                0.15,
                endpoint_arrived=True,
                retry_used=False,
            )
        )
        self.assertTrue(
            bridge.patrol_endpoint_undershoot_correction_allowed(
                0.081,
                0.081,
                0.15,
                endpoint_arrived=False,
                retry_used=False,
            )
        )
        self.assertFalse(
            bridge.patrol_endpoint_undershoot_correction_allowed(
                0.12,
                0.12,
                0.15,
                endpoint_arrived=False,
                retry_used=True,
            )
        )
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        arrival = source.index("# Arrival always wins over correction.")
        correction = source.index(
            "patrol_endpoint_undershoot_correction_allowed(", arrival
        )
        self.assertLess(arrival, correction)
        self.assertIn("endpoint_undershoot_retry_used = True", source)
        smooth_gate = source[source.index("used_smooth_cruise = bool("):]
        self.assertIn("and not active_endpoint_undershoot_correction", smooth_gate)

    def test_patrol_recovery_never_authorizes_stationary_test_pulses(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        recovery_start = source.index("def wait_for_pose_recovery(")
        recovery_end = source.index("def continuity_guarded_pose_gate()", recovery_start)
        recovery = source[recovery_start:recovery_end]
        self.assertIn("require_observed_translation_progress: bool = False", recovery)
        self.assertNotIn("allow_stationary_metric_resume", source)
        self.assertNotIn("stationary_metric_bounded_resume", source)
        self.assertNotIn("stationary_metric_retry_count", source)
        server = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn('"one_pulse_pose_confirmation": True', server)
        self.assertIn('"max_unverified_translation_m": 0.18', server)

    def test_smooth_cruise_is_low_stick_and_stops_on_pose_or_progress_loss(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        helper_start = source.index("def execute_guarded_cruise_window(")
        helper_end = source.index("def observe_motion_after_probe(", helper_start)
        helper = source[helper_start:helper_end]
        self.assertIn("next_stick_refresh = now + 0.10", helper)
        self.assertIn("_send_rc_with_bounded_ack(", helper)
        self.assertIn("smooth-cruise acknowledgement was lost", helper)
        self.assertIn("neutral_result = neutral_hover(drone, 0.0)", helper)
        self.assertIn("control link closed while ending smooth cruise", helper)
        self.assertNotIn("drone.move(", helper)
        self.assertIn("neutral_hover(drone, 0.0)", helper)
        self.assertIn("now - last_safe_observation > cruise_pose_watchdog_seconds", helper)
        self.assertIn("now - last_progress_observation > cruise_pose_watchdog_seconds", helper)
        self.assertIn("fresh frames arrived but the published ATLAS position", helper)
        self.assertIn("maximum_pose_step=max_pose_step", helper)
        self.assertIn("command_seconds = max(0.45, min(0.55", helper)
        self.assertIn("minimum_improvement=0.010", helper)
        self.assertIn('sent["observed_model_progress"]', helper)
        self.assertIn("and not used_smooth_cruise", source)

        server = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn('"smooth_continuous_cruise": True', server)
        self.assertIn('"cruise_window_seconds": 0.55', server)
        self.assertIn('"cruise_pose_watchdog_seconds": 0.65', server)
        self.assertIn('"max_forward_rc": 0.022', server)
        self.assertIn("bounded_cruise_window = min(", source)
        self.assertIn("max(requested_cruise_window, 0.45)", source)
        self.assertIn("0.55,", source)
        self.assertNotIn(
            'mission["cruise_window_seconds"] = max(',
            source,
        )

    def test_waypoint_zero_vector_uses_verified_endpoint_heading_for_yaw_only(self):
        point_two = [-0.648024, -0.080263, -0.487748]
        endpoint_heading = [0.996195, 0.0, 0.087156]

        direction = bridge.patrol_navigation_direction_xz(
            point_two,
            list(point_two),
            endpoint_heading=endpoint_heading,
        )

        self.assertIsNotNone(direction)
        self.assertAlmostEqual(direction[0], endpoint_heading[0], places=5)
        self.assertAlmostEqual(direction[2], endpoint_heading[2], places=5)
        self.assertIsNone(
            bridge.patrol_navigation_direction_xz(
                point_two,
                list(point_two),
            )
        )
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("endpoint_heading=(", source)
        self.assertIn("if endpoint_alignment_pending", source)

    def test_zero_target_vector_is_arrival_only_after_required_endpoint_evidence(self):
        self.assertTrue(
            bridge.patrol_zero_direction_arrival_allowed(
                0.0,
                0.15,
                precise_arrival=True,
                endpoint_ready=True,
            )
        )
        self.assertFalse(
            bridge.patrol_zero_direction_arrival_allowed(
                0.0,
                0.15,
                precise_arrival=True,
                endpoint_ready=False,
            )
        )
        self.assertTrue(
            bridge.patrol_zero_direction_arrival_allowed(
                0.0,
                0.15,
                precise_arrival=False,
                endpoint_ready=False,
            )
        )
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("exact_position_endpoint_verified", source)
        self.assertIn(
            "accepting the zero remaining direction as arrival",
            source,
        )

    def test_unverified_single_pulse_margin_preserves_all_saved_patrol_points(self):
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "viewer"
            / "public"
            / "maps"
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        map_entry = next(
            item
            for item in manifest["maps"]
            if item["id"] == "map_copy_20260730_114851_cfefdc"
        )
        patrol = next(
            item
            for item in map_entry["patrols"]
            if item["id"] == "patrol_ms4br5xr_4xclts"
        )
        for index, point in enumerate(patrol["points"], start=1):
            room_position = [point["rxyz"][0], 0.010052834697950513, point["rxyz"][2]]
            issue = bridge.pursuit_geofence_issue(
                room_position,
                map_entry["safety_barriers"],
                map_entry.get("safety_obstacles") or [],
                motion_buffer_m=0.30 + 0.18,
            )
            self.assertIsNone(issue, f"Point {index}: {issue}")

    def test_saved_full_recorded_patrol_result_remains_the_regression_baseline(self):
        self.assertTrue(FULL_RECORDED_PATROL_RESULT.exists())
        payload = json.loads(FULL_RECORDED_PATROL_RESULT.read_text(encoding="utf-8"))
        self.assertTrue(payload["complete"])
        self.assertFalse(payload["cancelled"])
        self.assertEqual(payload["expected_count"], 3668)
        self.assertEqual(payload["processed_count"], 3668)
        self.assertEqual(payload["accepted_count"], 3004)

        poses = payload["poses"]
        by_frame = {
            int(str(pose["instance_id"]).rsplit("_", 1)[-1]): pose
            for pose in poses
        }
        point_one = [-3.2329557447702215, -0.33236579860361815]
        point_three = [-0.4886978074319452, 0.9560112230666532]
        point_four = [-3.0736291183109774, 1.1113942967930859]

        def xz(pose):
            return [float(pose["rcenter"][0]), float(pose["rcenter"][2])]

        for frame in range(2180, 2901):
            self.assertLess(math.dist(xz(by_frame[frame]), point_three), 1e-9)
        self.assertTrue(by_frame[2900]["rotation_position_locked"])
        self.assertFalse(by_frame[2900]["translation_allowed"])
        self.assertGreater(math.dist(xz(by_frame[2901]), point_three), 0.04)

        self.assertLess(math.dist(xz(by_frame[3899]), point_four), 1e-9)
        self.assertEqual(
            by_frame[3899]["pose_source"],
            "patrol_visual_route_recovery",
        )
        for frame in range(3901, 4261):
            self.assertLess(math.dist(xz(by_frame[frame]), point_four), 1e-9)
            self.assertTrue(by_frame[frame]["rotation_position_locked"])
            self.assertFalse(by_frame[frame]["translation_allowed"])

        self.assertLess(math.dist(xz(by_frame[4400]), point_one), 1e-9)
        self.assertEqual(by_frame[4400]["route_visual_progress"], 1.0)
        for frame in range(4401, 4659):
            self.assertLess(math.dist(xz(by_frame[frame]), point_one), 1e-9)
            self.assertTrue(by_frame[frame]["rotation_position_locked"])
            self.assertFalse(by_frame[frame]["translation_allowed"])

    def test_strong_visual_route_match_can_resume_only_bounded_translation(self):
        pose = {
            "pose_source": "patrol_visual_route_recovery",
            "route_visual_verified": True,
            "route_visual_monitor_verified": True,
            "route_visual_translation_safe": True,
            "translation_allowed": True,
            "rotation_position_locked": False,
            "route_visual_weak_endpoint_recovery": False,
            "route_visual_progress": 0.06885811991417097,
            "route_visual_monitor_progress": 0.06885811991417097,
            "route_visual_inliers": 221,
            "route_visual_monitor_inliers": 221,
            "route_visual_minimum_inliers": 120,
            "route_visual_monitor_minimum_inliers": 120,
            "route_visual_acquisition_hits": 2,
        }
        self.assertTrue(bridge.patrol_visual_translation_resume_ready(pose))
        self.assertFalse(
            bridge.patrol_visual_translation_resume_ready(
                {**pose, "route_visual_monitor_inliers": 119}
            )
        )
        self.assertFalse(
            bridge.patrol_visual_translation_resume_ready(
                {**pose, "route_visual_weak_endpoint_recovery": True}
            )
        )
        self.assertFalse(
            bridge.patrol_visual_translation_resume_ready(
                {**pose, "rotation_position_locked": True}
            )
        )
        self.assertTrue(
            bridge.patrol_visual_yaw_anchor_ready(
                {"ok": True, "pose": {**pose, "rcenter": [0.0, 0.0, 0.0]}}
            )
        )
        self.assertFalse(
            bridge.patrol_visual_yaw_anchor_ready(
                {
                    "ok": True,
                    "recent_hold_fallback": True,
                    "pose": {**pose, "rcenter": [0.0, 0.0, 0.0]},
                }
            )
        )
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn('"visual_route_bounded_resume": True', source)

    def test_point_three_stationary_visual_recovery_allows_exactly_one_bounded_retry(self):
        pose = {
            "pose_source": "patrol_visual_route_recovery",
            "route_visual_verified": True,
            "route_visual_monitor_verified": True,
            "route_visual_translation_safe": True,
            "translation_allowed": True,
            "rotation_position_locked": False,
            "route_visual_weak_endpoint_recovery": False,
            "route_visual_progress": 0.0015513705171383346,
            "route_visual_monitor_progress": 0.0015513705171383346,
            "route_visual_inliers": 402,
            "route_visual_monitor_inliers": 402,
            "route_visual_minimum_inliers": 120,
            "route_visual_monitor_minimum_inliers": 120,
            "route_visual_acquisition_hits": 2,
        }
        self.assertTrue(
            bridge.patrol_visual_stationary_retry_ready(
                pose,
                retry_available=True,
                observed_translation_progress=False,
            )
        )
        self.assertFalse(
            bridge.patrol_visual_stationary_retry_ready(
                pose,
                retry_available=False,
                observed_translation_progress=False,
            )
        )
        self.assertFalse(
            bridge.patrol_visual_stationary_retry_ready(
                pose,
                retry_available=True,
                observed_translation_progress=True,
            )
        )
        self.assertFalse(
            bridge.patrol_visual_stationary_retry_ready(
                {**pose, "route_visual_monitor_inliers": 119},
                retry_available=True,
                observed_translation_progress=False,
            )
        )

        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("allow_visual_stationary_retry", source)
        self.assertIn('"visual_route_stationary_retry": True', source)
        self.assertIn("visual_stationary_retry_reference_distance", source)
        self.assertIn("<= visual_stationary_retry_reference_distance - 0.015", source)
        self.assertIn("bounded retry was already used", source)

    def test_route_command_ceiling_accumulates_unobserved_live_motion(self):
        """Regress Live ATLAS 13:22:38's Point-2 -> Point-3 deadlock.

        Three real 18 cm commands were sent while the published route pose
        lagged at 0.0, 0.00535, and 0.07377.  Recomputing from each lagging
        pose advertised only 28.7 cm of possible travel.  The physical
        envelope must retain all three commands: 54 cm on this 1.4525 m leg.
        """
        leg_length = 1.4525247629416798
        budget = 0.18 / leg_length
        ceiling = bridge.advance_route_command_progress_ceiling(None, 0.0, budget)
        self.assertAlmostEqual(ceiling, budget)

        ceiling = bridge.advance_route_command_progress_ceiling(
            ceiling,
            0.005352769679300302,
            budget,
        )
        self.assertAlmostEqual(ceiling, 2.0 * budget)

        ceiling = bridge.advance_route_command_progress_ceiling(
            ceiling,
            0.073765395233164,
            budget,
        )
        self.assertAlmostEqual(ceiling, 3.0 * budget)
        self.assertAlmostEqual(ceiling * leg_length, 0.54)

    def test_route_command_ceiling_never_rewinds_or_exceeds_the_leg(self):
        self.assertAlmostEqual(
            bridge.advance_route_command_progress_ceiling(0.62, 0.41, 0.08),
            0.70,
        )
        self.assertEqual(
            bridge.advance_route_command_progress_ceiling(0.96, 0.99, 0.08),
            1.0,
        )
        self.assertAlmostEqual(
            bridge.advance_route_command_progress_ceiling(0.45, 0.40, -0.2),
            0.45,
        )

    def test_command_progress_ceiling_is_not_a_patrol_motion_gate(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("route_command_budget_exhausted", source)
        self.assertNotIn("route_command_progress_budget_exhausted", source)
        self.assertNotIn("acknowledge_visual_route_command_progress", source)
        self.assertNotIn("acknowledge_metric_route_command_progress", source)
        self.assertIn("active_route_command_progress_ceiling = (", source)
        self.assertIn("advance_route_command_progress_ceiling(", source)
        self.assertIn("guided_command_pose_safety_issue(", source)
        self.assertIn("enforce_patrol_geofence(", source)
        self.assertIn("patrol_translation_pulse_progress_issue(", source)

    def test_neutral_metric_consensus_can_undo_observed_false_forward_ratchet(self):
        def gate(instance, progress):
            center = [float(progress), 0.0, 0.0]
            return {
                "ok": True,
                "route_progress": float(progress),
                "pose": {
                    "instance_id": f"instance_{instance:06d}",
                    "rcenter": center,
                    "center": center,
                    "t": [-center[0], 0.0, 0.0],
                    "R": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            }

        state = {}
        ready = False
        # Exact shape of Live ATLAS 10:17: the trusted floor ratcheted to
        # 0.901, while repeated current-frame metric solves returned to ~0.74.
        for instance, progress in enumerate(
            (0.745, 0.743, 0.746, 0.744, 0.745),
            start=100,
        ):
            ready = bridge.patrol_metric_recovery_reconciliation_ready(
                gate(instance, progress),
                previous_progress=0.9009062981635819,
                candidate_progress=progress,
                state=state,
            )
        self.assertTrue(ready)

        # A single rollback, or a cluster spread over inconsistent positions,
        # cannot lower the monotonic floor.
        state = {}
        self.assertFalse(
            bridge.patrol_metric_recovery_reconciliation_ready(
                gate(200, 0.74),
                previous_progress=0.90,
                candidate_progress=0.74,
                state=state,
            )
        )
        for instance, progress in enumerate((0.74, 0.78, 0.72, 0.77), start=201):
            ready = bridge.patrol_metric_recovery_reconciliation_ready(
                gate(instance, progress),
                previous_progress=0.90,
                candidate_progress=progress,
                state=state,
            )
        self.assertFalse(ready)

    def test_five_frame_temporal_recovery_can_resume_one_bounded_pulse(self):
        pose = {
            "pose_source": "patrol_visual_route_recovery",
            "route_visual_verified": True,
            "route_visual_monitor_verified": True,
            "route_visual_translation_safe": True,
            "translation_allowed": True,
            "rotation_position_locked": False,
            "route_visual_weak_endpoint_recovery": False,
            "route_visual_temporal_recovery": True,
            "route_visual_progress": 0.3717664674482947,
            "route_visual_monitor_progress": 0.3717664674482947,
            "route_visual_inliers": 96,
            "route_visual_monitor_inliers": 96,
            "route_visual_minimum_inliers": 90,
            "route_visual_monitor_minimum_inliers": 90,
            "route_visual_acquisition_hits": 5,
            "route_visual_temporal_recovery_required_hits": 5,
        }
        self.assertTrue(bridge.patrol_visual_translation_resume_ready(pose))
        self.assertFalse(
            bridge.patrol_visual_translation_resume_ready(
                {**pose, "route_visual_acquisition_hits": 4}
            )
        )
        self.assertFalse(
            bridge.patrol_visual_translation_resume_ready(
                {**pose, "route_visual_monitor_inliers": 89}
            )
        )

    def test_point_four_five_frame_recovery_can_resume_at_50_inliers(self):
        pose = {
            "pose_source": "patrol_visual_route_recovery",
            "route_visual_verified": True,
            "route_visual_monitor_verified": True,
            "route_visual_monitor_leg_index": 4,
            "route_visual_translation_safe": True,
            "translation_allowed": True,
            "rotation_position_locked": False,
            "route_visual_weak_endpoint_recovery": False,
            "route_visual_temporal_recovery": True,
            "route_visual_progress": 0.1042,
            "route_visual_monitor_progress": 0.1042,
            "route_visual_inliers": 55,
            "route_visual_monitor_inliers": 55,
            "route_visual_minimum_inliers": 50,
            "route_visual_monitor_minimum_inliers": 50,
            "route_visual_acquisition_hits": 5,
            "route_visual_temporal_recovery_required_hits": 5,
        }
        self.assertTrue(bridge.patrol_visual_translation_resume_ready(pose))
        self.assertFalse(
            bridge.patrol_visual_translation_resume_ready(
                {**pose, "route_visual_monitor_leg_index": 3}
            )
        )

    def test_point_four_to_one_never_uses_an_unconfirmed_stationary_retry(self):
        pose = {
            "pose_source": "patrol_visual_route_recovery",
            "route_visual_verified": True,
            "route_visual_monitor_verified": True,
            "route_visual_monitor_leg_index": 4,
            "route_visual_translation_safe": True,
            "translation_allowed": True,
            "rotation_position_locked": False,
            "route_visual_weak_endpoint_recovery": False,
            "route_visual_temporal_recovery": True,
            "route_visual_progress": 0.12,
            "route_visual_monitor_progress": 0.12,
            "route_visual_inliers": 80,
            "route_visual_monitor_inliers": 80,
            "route_visual_minimum_inliers": 50,
            "route_visual_monitor_minimum_inliers": 50,
            "route_visual_acquisition_hits": 5,
            "route_visual_temporal_recovery_required_hits": 5,
        }
        self.assertFalse(
            bridge.patrol_visual_stationary_retry_ready(
                pose,
                retry_available=True,
                observed_translation_progress=False,
                expected_leg_index=4,
            )
        )

    def test_exact_august_11_visual_recovery_reconciles_small_stale_progress_lead(self):
        # The stopped 10:24:20 run held the drone forever because the command
        # checkpoint was 0.544995 while hundreds of strong visual-route poses
        # agreed on 0.489437. Three fresh observations may reconcile that
        # bounded 14.4 cm discrepancy during neutral recovery.
        state = {}
        base_pose = {
            "pose_source": "patrol_visual_route_recovery",
            "route_visual_verified": True,
            "route_visual_monitor_verified": True,
            "translation_allowed": True,
            "rotation_position_locked": False,
            "route_visual_progress": 0.4894373182159811,
            "route_visual_monitor_progress": 0.4894373182159811,
            "route_visual_inliers": 235,
            "route_visual_monitor_inliers": 235,
            "route_visual_minimum_inliers": 120,
            "route_visual_monitor_minimum_inliers": 120,
            "route_visual_acquisition_hits": 2,
        }
        ready = []
        for index in range(3):
            ready.append(
                bridge.patrol_visual_recovery_reconciliation_ready(
                    {**base_pose, "instance_id": f"visual_route_{2135 + index:06d}"},
                    previous_progress=0.5449946501939766,
                    candidate_progress=0.4894373182159811,
                    state=state,
                )
            )
        self.assertEqual(ready, [False, False, True])

    def test_visual_recovery_never_reconciles_a_large_or_unverified_jump(self):
        strong_pose = {
            "instance_id": "visual_route_000001",
            "pose_source": "patrol_visual_route_recovery",
            "route_visual_verified": True,
            "route_visual_monitor_verified": True,
            "translation_allowed": True,
            "rotation_position_locked": False,
            "route_visual_progress": 0.30,
            "route_visual_monitor_progress": 0.30,
            "route_visual_inliers": 240,
            "route_visual_monitor_inliers": 240,
            "route_visual_minimum_inliers": 120,
            "route_visual_monitor_minimum_inliers": 120,
            "route_visual_acquisition_hits": 2,
        }
        state = {}
        for index in range(5):
            pose = {**strong_pose, "instance_id": f"visual_route_{index:06d}"}
            self.assertFalse(
                bridge.patrol_visual_recovery_reconciliation_ready(
                    pose,
                    previous_progress=0.55,
                    candidate_progress=0.30,
                    state=state,
                )
            )
        self.assertFalse(
            bridge.patrol_visual_recovery_reconciliation_ready(
                {
                    "instance_id": "tsolve_000010",
                    "pose_source": "tsolve",
                    "route_visual_verified": True,
                },
                previous_progress=0.55,
                candidate_progress=0.49,
                state=state,
            )
        )

    def test_latest_neutral_recovery_pose_is_valid_only_after_stale_turn_lock_is_cleared(self):
        # Live ATLAS 13:24:18 reached 90.8% of Point 1 -> 2, then a stale
        # pre-yaw anchor made the stable recovered pose look like 36.1 cm of
        # impossible turn translation.  Neutral recovery must evaluate the
        # pose as ordinary monotonic route progress, not as an active yaw.
        segment_start = [-3.2329557447702215, -0.0802621969998909, -0.33236579860361815]
        segment_end = [-0.6480244338911889, -0.0802621969998909, -0.48774887233005093]
        stale_turn_anchor = [-0.9115137262614181, 0.08513394829906745, -0.9119815492907253]
        recovered_pose = [-0.8710135677681412, 0.059781960592258473, -0.5582597804536191]

        locked_reason, locked_progress = bridge.patrol_route_pose_rejection(
            recovered_pose,
            segment_start=segment_start,
            segment_end=segment_end,
            previous_progress=0.9082639605813042,
            translation_locked=True,
            position_anchor=stale_turn_anchor,
            max_cross_track=0.55,
        )
        self.assertIn("turn position drift", locked_reason)
        self.assertAlmostEqual(locked_progress, 0.9082639605813042)

        recovered_reason, recovered_progress = bridge.patrol_route_pose_rejection(
            recovered_pose,
            segment_start=segment_start,
            segment_end=segment_end,
            previous_progress=0.9082639605813042,
            translation_locked=False,
            position_anchor=None,
            max_cross_track=0.55,
        )
        self.assertIsNone(recovered_reason)
        self.assertGreater(recovered_progress, 0.9082639605813042)

    def test_exact_startup_yaw_regression_uses_locked_anchor_not_raw_backward_pose(self):
        # Live ATLAS 10:32:29 aborted after a valid yaw because raw route
        # progress appeared to move from 0.129 back to 0.039.  While yaw is
        # locked, progress must remain at the controller's 0.129 anchor.
        segment_start = [0.0, 0.0, 0.0]
        segment_end = [1.0, 0.0, 0.0]
        anchor = [0.12853315376002158, 0.0, 0.0]
        raw_yaw_pose = [0.039, 0.0, 0.0]
        reason, progress = bridge.patrol_route_pose_rejection(
            raw_yaw_pose,
            segment_start=segment_start,
            segment_end=segment_end,
            previous_progress=0.12853315376002158,
            translation_locked=True,
            position_anchor=anchor,
            max_cross_track=0.55,
        )
        self.assertIsNone(reason)
        self.assertAlmostEqual(progress, 0.12853315376002158)

        unlocked_reason, unlocked_progress = bridge.patrol_route_pose_rejection(
            raw_yaw_pose,
            segment_start=segment_start,
            segment_end=segment_end,
            previous_progress=0.12853315376002158,
            translation_locked=False,
            position_anchor=None,
            max_cross_track=0.55,
        )
        self.assertIn("moved backward", unlocked_reason)
        self.assertAlmostEqual(unlocked_progress, 0.12853315376002158)

    def test_pinned_full_patrol_baseline_passes_route_gate_for_two_laps(self):
        reference = json.loads(FULL_PATROL_BASELINE_REFERENCE.read_text(encoding="utf-8"))
        self.assertTrue(reference["complete_loop"])
        self.assertTrue(reference["enabled_for_live_route_gate"])
        self.assertTrue(reference["enabled_for_visual_route_recovery"])
        audit = json.loads(
            FULL_PATROL_BASELINE_REFERENCE.with_name(
                reference["visual_route_recovery_audit"]
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["wrong_leg_accepts"], [])
        self.assertEqual(
            audit["accepted_after_acquisition"],
            audit["expected_after_acquisition"],
        )
        self.assertGreaterEqual(audit["accepted_after_acquisition"], 398)
        self.assertEqual(
            [(leg["from_point"], leg["to_point"]) for leg in reference["legs"]],
            [(1, 2), (2, 3), (3, 4), (4, 1)],
        )
        checked = 0
        checked_turn_handoffs = 0
        for _lap in range(2):
            for leg in reference["legs"]:
                dx = float(leg["to"][0]) - float(leg["from"][0])
                dz = float(leg["to"][2]) - float(leg["from"][2])
                leg_length = math.hypot(dx, dz)
                self.assertGreater(leg_length, 0.13)
                raw_turn_drift = [
                    float(leg["from"][0]) - 0.13 * dx / leg_length,
                    float(leg["from"][1]),
                    float(leg["from"][2]) - 0.13 * dz / leg_length,
                ]
                turn_reason, turn_progress = bridge.patrol_route_pose_rejection(
                    raw_turn_drift,
                    segment_start=leg["from"],
                    segment_end=leg["to"],
                    previous_progress=0.0,
                    translation_locked=True,
                    position_anchor=leg["from"],
                    max_cross_track=0.55,
                )
                self.assertIsNone(turn_reason)
                self.assertAlmostEqual(turn_progress, 0.0)
                checked_turn_handoffs += 1
                progress = turn_progress
                for sample in leg["samples"]:
                    reason, progress = bridge.patrol_route_pose_rejection(
                        sample["rcenter"],
                        segment_start=leg["from"],
                        segment_end=leg["to"],
                        previous_progress=progress,
                        translation_locked=False,
                        position_anchor=None,
                        max_cross_track=0.55,
                    )
                    self.assertIsNone(reason)
                    checked += 1
        self.assertEqual(
            checked,
            2 * sum(len(leg["samples"]) for leg in reference["legs"]),
        )
        self.assertGreaterEqual(checked, 414)
        self.assertEqual(checked_turn_handoffs, 8)

    def test_route_corridor_distance_uses_finite_segment(self):
        self.assertAlmostEqual(
            bridge.horizontal_xz_segment_distance([1, 0, 0.4], [0, 0, 0], [2, 0, 0]),
            0.4,
        )
        self.assertAlmostEqual(
            bridge.horizontal_xz_segment_distance([3, 0, 0], [0, 0, 0], [2, 0, 0]),
            1.0,
        )

    def test_corridor_recovery_slows_before_hard_stop(self):
        self.assertEqual(
            bridge.corridor_recovery_speed_scale(0.2, recovery_start=0.3, hard_limit=0.8),
            1.0,
        )
        recovery_scale = bridge.corridor_recovery_speed_scale(
            0.55,
            recovery_start=0.3,
            hard_limit=0.8,
        )
        self.assertAlmostEqual(recovery_scale, 0.5)
        self.assertEqual(
            bridge.corridor_recovery_speed_scale(0.8, recovery_start=0.3, hard_limit=0.8),
            0.3,
        )

    def test_localizer_jump_limit_is_time_aware_but_hard_capped(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("hard_cap = float(max_step)", source)
        self.assertIn("hard_cap=hard_cap", source)
        self.assertIn("0.18 + max(0.0, float(max_speed)) * dt", source)

    def test_observed_skipped_frame_motion_is_allowed(self):
        limit = bridge.bounded_pose_step_limit(100.0, 100.416)
        self.assertGreater(limit, 0.318)

    def test_large_jump_remains_rejected_after_skipped_frames(self):
        limit = bridge.bounded_pose_step_limit(100.0, 100.416)
        self.assertLess(limit, 0.932)
        self.assertLessEqual(bridge.bounded_pose_step_limit(100.0, 110.0), 0.55)

    def test_issued_route_commands_allow_the_delayed_point_two_to_three_pose(self):
        pose = {
            "held_pose": False,
            "output_rejected": False,
            "translation_allowed": True,
            "rotation_position_locked": False,
            "route_cross_track_m": 0.03,
        }
        self.assertTrue(
            bridge.command_bounded_pose_catchup_ready(
                pose,
                [0.61, 0.0, 0.0],
                [0.15, 0.0, 0.0],
                segment_start=[0.0, 0.0, 0.0],
                segment_end=[1.0, 0.0, 0.0],
                trusted_progress=0.15,
                command_progress_ceiling=0.69,
                command_sequence=3,
                step=0.46,
            )
        )

    def test_command_catchup_rejects_uncommanded_backward_and_cross_track_jumps(self):
        base = {
            "held_pose": False,
            "output_rejected": False,
            "translation_allowed": True,
            "rotation_position_locked": False,
            "route_cross_track_m": 0.03,
        }
        common = {
            "segment_start": [0.0, 0.0, 0.0],
            "segment_end": [1.0, 0.0, 0.0],
            "trusted_progress": 0.45,
            "command_progress_ceiling": 0.70,
            "command_sequence": 2,
            "step": 0.46,
        }
        self.assertFalse(
            bridge.command_bounded_pose_catchup_ready(
                base,
                [-0.01, 0.0, 0.0],
                [0.45, 0.0, 0.0],
                **common,
            )
        )
        self.assertFalse(
            bridge.command_bounded_pose_catchup_ready(
                {**base, "route_cross_track_m": 0.31},
                [0.69, 0.0, 0.0],
                [0.23, 0.0, 0.0],
                **common,
            )
        )

    def test_glb_overlay_applies_visual_heading_trim(self):
        source = OVERLAY_PATH.read_text(encoding="utf-8")
        self.assertIn("api.getDroneHeadingTrimRad?.()", source)
        self.assertIn("Math.atan2(heading[0], heading[2]) + visualTrim", source)

    def test_operator_model_alignment_is_sent_with_bridge_sign(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("initial_body_heading_offset_deg: -selectedDroneHeadingTrimDeg()", source)
        self.assertIn("initial_pose_offset_room: initialPoseOffsetRoom.slice(0, 3)", source)
        self.assertIn("operator_heading_calibrated:", source)
        heading_function = source[source.index("function headingForPose"):source.index("function rotateHorizontalHeading")]
        self.assertLess(heading_function.index("cur?.rheading"), heading_function.index("cur?.rotationHeading"))

    def test_manual_pose_offset_is_applied_in_bridge_room_frame(self):
        gate = {"pose": {"rcenter": [1.0, 2.0, 3.0]}, "pose_offset_room": [0.2, 0.0, -0.4]}
        self.assertEqual(bridge.pose_gate_position(gate), [1.2, 2.0, 2.6])

    def test_point_four_profile_uses_aligned_heading_without_translation_probe(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        server_source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn("calibrated_heading_offset_rad: float | None = None", source)
        self.assertIn("elif calibrate_forward_heading(gate):", source)
        profile_start = server_source.index('"reference_profile": "live-atlas-141441"')
        profile_end = server_source.index("return safe", profile_start)
        profile = server_source[profile_start:profile_end]
        self.assertIn('"operator_heading_calibrated": True', profile)
        self.assertIn('"require_physical_forward_probe": False', profile)

    def test_forward_probe_is_never_advertised_as_rotation_only(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        start = source.index("def calibrate_body_axis(")
        end = source.index("def calibrate_body_axes(", start)
        calibration = source[start:end]
        self.assertIn("active_route_translation_locked = False", calibration)
        self.assertIn('"translation_locked": False', calibration)
        self.assertIn('"body_forward_gain": bf / axis_probe_rc', calibration)
        self.assertIn('"body_lateral_gain": lr / axis_probe_rc', calibration)

    def test_impossible_forward_heading_calibration_is_rejected(self):
        camera = [math.cos(math.radians(-12.7)), 0.0, math.sin(math.radians(-12.7))]
        false_forward = [math.cos(math.radians(75.8)), 0.0, math.sin(math.radians(75.8))]
        error = bridge.heading_calibration_error_degrees(camera, false_forward, 0.0)
        self.assertIsNotNone(error)
        self.assertAlmostEqual(error, 88.5, places=1)
        self.assertGreater(error, 45.0)
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn('"phase": "heading_calibration_rejected"', source)

    def test_point_four_profile_does_not_accept_august_11_false_heading_probe(self):
        server_source = SERVER_PATH.read_text(encoding="utf-8")
        profile_start = server_source.index('"reference_profile": "live-atlas-141441"')
        profile_end = server_source.index("return safe", profile_start)
        profile = server_source[profile_start:profile_end]
        self.assertIn('"require_physical_forward_probe": False', profile)
        self.assertIn('"max_heading_calibration_error_deg": 25.0', profile)
        self.assertGreater(37.162, 25.0)

    def test_stale_browser_patrol_packet_is_blocked_before_motion(self):
        mission = {
            "guided_enabled": True,
            "commands": [{"type": "cruise", "distance": 1.0, "to": [1, 0, 0]}],
        }
        with self.assertRaisesRegex(RuntimeError, "browser safety code is stale"):
            bridge.execute_guarded_mission_packet(object(), mission)

    def test_final_replay_preserves_tsolve_heading(self):
        source = APP_PATH.read_text(encoding="utf-8")
        closest_pose = source[source.index("function closestPose()"):source.index("function canConnectPath")]
        self.assertIn("const interpolatedRawHeading = a.rheading && b.rheading", closest_pose)
        self.assertIn("rheading: last.rheading || last.rotationHeading", closest_pose)

    def test_yaw_validation_waits_for_post_pulse_camera_capture(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("def wait_for_pose_captured_after(", source)
        self.assertIn("timeout: float = 5.0", source)
        self.assertIn("received_unix >= capture_cutoff_unix", source)
        self.assertIn("yaw_response_reversed = yaw_target_error_response(", source)
        self.assertIn("sign_action = yaw_sign_recovery_action(", source)
        self.assertIn("GUIDED_YAW_SIGN_CONFIRMATION_PULSES = 3", source)
        self.assertIn('"phase": "yaw_feedback_recovery"', source)
        self.assertIn('"yaw_sign_preserved": yaw_sign', source)

    def test_raw_yaw_heading_response_classifier_remains_available_for_diagnostics(self):
        heading = lambda degrees: [
            math.cos(math.radians(degrees)),
            0.0,
            math.sin(math.radians(degrees)),
        ]
        self.assertTrue(
            bridge.yaw_response_is_reversed(1.0, heading(-12.7), heading(-16.4))
        )
        self.assertFalse(
            bridge.yaw_response_is_reversed(1.0, heading(-12.7), heading(-9.0))
        )
        self.assertIsNone(
            bridge.yaw_response_is_reversed(1.0, heading(-12.7), heading(-12.2))
        )

    def test_point_three_delayed_yaw_feedback_does_not_reverse_verified_dji_sign(self):
        # Exact target-error sequence from atlas_dji_live_20260817_150451_caeecd.
        # These sub-degree changes were asynchronous/noisy, not proof that the
        # fixed DJI yaw polarity had reversed in the middle of the mission.
        errors_deg = [12.05697, 12.04850, 11.71537]
        votes = [
            bridge.yaw_target_error_response(
                math.radians(before),
                math.radians(after),
            )
            for before, after in zip(errors_deg, errors_deg[1:])
        ]
        self.assertEqual(votes, [None, None])
        self.assertEqual(
            bridge.yaw_sign_recovery_action(
                yaw_sign_verified=True,
                wrong_yaw_pulses=3,
                yaw_flip_count=0,
            ),
            "recover",
        )

    def test_yaw_sign_can_flip_once_only_during_initial_polarity_verification(self):
        self.assertFalse(
            bridge.yaw_target_error_response(
                math.radians(12.0),
                math.radians(10.0),
            )
        )
        self.assertTrue(
            bridge.yaw_target_error_response(
                math.radians(10.0),
                math.radians(12.0),
            )
        )
        self.assertEqual(
            bridge.yaw_sign_recovery_action(
                yaw_sign_verified=False,
                wrong_yaw_pulses=3,
                yaw_flip_count=0,
            ),
            "flip",
        )
        self.assertEqual(
            bridge.yaw_sign_recovery_action(
                yaw_sign_verified=False,
                wrong_yaw_pulses=5,
                yaw_flip_count=1,
            ),
            "abort",
        )

    def test_route_yaw_before_and_after_use_the_same_camera_heading_basis(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        start = source.index("if route_follow_desired_heading is not None:", source.index("raw_after_position"))
        end = source.index("horizontal_pulse =", start)
        post_yaw = source[start:end]
        self.assertIn("pose_gate_camera_heading(", post_yaw)
        self.assertIn("route_follow_desired_heading", post_yaw)
        self.assertIn("yaw_target_error_response(\n                                angle,", post_yaw)

    def test_hover_now_preempts_an_active_patrol(self):
        server_source = SERVER_PATH.read_text(encoding="utf-8")
        bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn('command_payload["emergency_stop"] = True', server_source)
        self.assertIn('command_name == "hover" and command_payload.get("emergency_stop")', bridge_source)
        self.assertIn('mission_cancel.set()', bridge_source)
        self.assertIn('"event": (', bridge_source)
        self.assertIn('"emergency_stop"', bridge_source)

    def test_navigation_yaw_is_faster_than_scan_yaw(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("max_yaw_rc: 0.050", source)
        self.assertIn("max_scan_yaw_rc: 0.025", source)
        bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("alignment_grace_seconds + max(4.0, planned_duration * 2.2)", bridge_source)

    def test_saved_patrol_connects_entry_to_exactly_two_circle_loop(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertNotIn(">Go to Start</button>", source)
        self.assertIn(">Start ${escapeHtml(patrolTitle(patrol))} · 2 Circles</button>", source)
        self.assertIn('patrol_stage: options.stage || patrol.patrol_stage || "combined"', source)
        self.assertIn('buildPatrolReturnToStartPlan()', source)
        self.assertIn('buildPatrolLoopPlan()', source)
        self.assertIn('buildConnectedPatrolPlan()', source)
        self.assertIn('runUi(() => executeSavedPatrol(patrol.id, "combined"))', source)
        self.assertIn('continuous_relocalization: true', source)

    def test_patrol_pose_recovery_hovers_until_online_relocalization_or_operator_stop(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("bounded_recovery = timeout is not None", source)
        self.assertIn(
            "(continuous_relocalization and not bounded_recovery)",
            source,
        )
        self.assertIn("or time.monotonic() < deadline", source)
        self.assertIn('"continuous_relocalization": continuous_relocalization', source)
        self.assertIn(
            '"metric_position_recovery_allowed": bool(\n'
            '                            require_metric_pose\n'
            '                        )',
            source,
        )
        self.assertIn('total_pose_recovery_pause_seconds +=', source)
        self.assertIn('return time.monotonic() - total_pose_recovery_pause_seconds', source)
        recovery_start = source.index("def wait_for_pose_recovery(")
        recovery_end = source.index("def continuity_guarded_pose_gate()", recovery_start)
        recovery = source[recovery_start:recovery_end]
        self.assertIn("active_route_translation_locked = False", recovery)
        self.assertIn("active_route_position_anchor = None", recovery)
        self.assertIn("yaw_position_anchor = None", recovery)
        self.assertIn('"translation_locked": True', recovery)
        self.assertIn('"route_visual_recovery_allowed": True', recovery)
        self.assertIn('"rotation_release_requested": True', recovery)
        self.assertIn('"body_forward_gain": 0.0', recovery)
        self.assertIn('"body_lateral_gain": 0.0', recovery)
        self.assertLess(
            recovery.index("active_route_translation_locked = False"),
            recovery.index("gate = continuity_guarded_pose_gate()"),
        )
        self.assertLess(
            recovery.index("publish_progress("),
            recovery.index("gate = continuity_guarded_pose_gate()"),
        )
        app_source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("continuous_relocalization: true", app_source)

    def test_travel_budget_advances_only_with_commanded_motion(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("alignment_deadline = motion_clock()", source)
        self.assertIn("segment_safety_deadline = motion_clock() + max_cruise_seconds", source)
        self.assertIn("absolute_segment_deadline = motion_clock() + min(240.0, max_cruise_seconds * 2.0)", source)
        self.assertIn("alignment_progress_deadline = alignment_deadline", source)
        self.assertIn("now + 20.0", source)
        self.assertIn("forward_command_seconds = 0.0", source)
        self.assertIn("travel_yaw_command_seconds = 0.0", source)
        self.assertIn("forward_command_seconds += command_seconds", source)
        self.assertIn("travel_yaw_command_seconds += pulse_seconds", source)
        self.assertNotIn("travel_deadline =", source)

    def test_heading_hysteresis_prevents_yaw_forward_chatter(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("forward_alignment_locked = False", source)
        self.assertIn(
            "alignment_limit_deg = 14.0 if forward_alignment_locked else 10.0",
            source,
        )
        self.assertIn(
            "yaw_scale = max(0.65, min(1.0, angle_abs / math.radians(70.0)))",
            source,
        )

    def test_invalid_pose_during_established_turn_allows_bounded_yaw_only(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("max_blind_yaw_seconds = 12.0", source)
        self.assertIn('"phase": "bounded_blind_yaw_recovery"', source)
        self.assertIn("yaw=recovery_yaw_rc", source)
        self.assertIn("lr=0.0", source)
        self.assertIn("bf=0.0", source)
        self.assertIn("blind_yaw_seconds += pulse_seconds", source)
        self.assertIn("blind_turn_active = (", source)
        self.assertIn("and not blind_turn_active", source)
        self.assertIn("frozen_heading = pose_gate_heading(", source)
        self.assertIn("frozen_angle = signed_angle_xz(frozen_heading, frozen_direction)", source)
        self.assertIn(
            'if kind == "cruise" and not gate_attempt.get("ok") and isinstance(last_pose_gate, dict):',
            source,
        )
        self.assertIn("forward_alignment_locked = True", source)

    def test_rotation_only_heading_can_accompany_but_never_validate_a_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "poses.json"
            stream.write_text(
                json.dumps(
                    {
                        "updated_at": time.time(),
                        "poses": [
                            {
                                # This reproduces the important case: the
                                # localizer had a candidate, while the bridge
                                # independently rejected its translation.
                                "success": True,
                                "held_pose": False,
                                "rcenter": [99.0, 0.0, 99.0],
                                "received_unix": time.time(),
                                "rotation_heading": [0.0, 0.0, 1.0],
                                "rotation_heading_source": "optical_flow_yaw",
                                "rotation_heading_tracks": 44,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            observation = bridge.latest_rotation_only_heading(stream, 2.5)
        self.assertTrue(observation["ok"])
        self.assertEqual(observation["heading"], [0.0, 0.0, 1.0])
        self.assertNotIn("rcenter", observation)

    def test_rotation_only_heading_rejects_discontinuous_frame_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "poses.json"
            stream.write_text(
                json.dumps(
                    {
                        "poses": [
                            {
                                "instance_id": "hold_001509",
                                "received_unix": time.time(),
                                "rotation_heading": [0.0, 0.0, 1.0],
                                "rotation_heading_source": "optical_flow_yaw",
                                "rotation_heading_tracks": 500,
                                "rotation_heading_timing_valid": False,
                                "rotation_heading_frame_gap_seconds": 1.775,
                                "rotation_heading_max_frame_gap_seconds": 0.30,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            observation = bridge.latest_rotation_only_heading(stream, 2.5)

        self.assertFalse(observation["ok"])
        self.assertTrue(observation["unsafe_for_yaw"])
        self.assertAlmostEqual(observation["frame_gap_seconds"], 1.775)

    def test_latest_point_four_recorded_heading_accepts_last_run_inlier_level(self):
        """The 10:51 run had 50 such frames but its old process required 120."""
        now = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "poses.json"
            stream.write_text(
                json.dumps(
                    {
                        "poses": [
                            {
                                "instance_id": "instance_001901",
                                "received_unix": now,
                                "route_visual_heading_required": True,
                                "route_visual_heading_verified": True,
                                "route_visual_heading_leg_index": 4,
                                "route_visual_heading_correction_deg": 1.7398239685,
                                "route_visual_heading_current": [-0.99, 0.0, 0.14],
                                "route_visual_heading_recorded": [-1.0, 0.0, 0.0],
                                "route_visual_heading_inliers": 60,
                                "route_visual_heading_minimum_inliers": 50,
                                "route_visual_heading_map_id": "map_a",
                                "route_visual_heading_patrol_id": "patrol_a",
                                "route_visual_heading_baseline_replay_id": "baseline_a",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            observation = bridge.latest_recorded_departure_heading(
                stream,
                2.5,
                map_id="map_a",
                patrol_id="patrol_a",
                baseline_replay_id="baseline_a",
                expected_leg_index=4,
            )

        self.assertTrue(observation["ok"])
        self.assertEqual(observation["minimum_inliers"], 50)
        self.assertEqual(observation["inliers"], 60)

    def test_fresh_recorded_heading_survives_newer_plain_solver_publication(self):
        """A solver pose without ORB metadata must not erase fresh leg-4 evidence."""
        now = time.time()
        verified_heading = {
            "instance_id": "hold_002280",
            "received_unix": now,
            "route_visual_heading_required": True,
            "route_visual_heading_verified": True,
            "route_visual_heading_leg_index": 4,
            "route_visual_heading_correction_deg": 16.3,
            "route_visual_heading_current": [-0.96, 0.0, 0.28],
            "route_visual_heading_recorded": [-1.0, 0.0, 0.0],
            "route_visual_heading_inliers": 59,
            "route_visual_heading_minimum_inliers": 50,
            "route_visual_heading_map_id": "map_a",
            "route_visual_heading_patrol_id": "patrol_a",
            "route_visual_heading_baseline_replay_id": "baseline_a",
        }
        plain_solver_pose = {
            "instance_id": "instance_002281",
            "received_unix": now + 0.01,
            "success": True,
            "rcenter": [-3.0, -0.08, 1.1],
            "rheading": [-0.95, 0.0, 0.31],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "poses.json"
            stream.write_text(
                json.dumps({"poses": [verified_heading, plain_solver_pose]}),
                encoding="utf-8",
            )
            observation = bridge.latest_recorded_departure_heading(
                stream,
                2.5,
                map_id="map_a",
                patrol_id="patrol_a",
                baseline_replay_id="baseline_a",
                expected_leg_index=4,
            )

        self.assertTrue(observation["ok"])
        self.assertEqual(observation["instance_id"], "hold_002280")
        self.assertAlmostEqual(observation["correction_deg"], 16.3)

    def test_newer_explicit_heading_failure_supersedes_older_verified_heading(self):
        """Retaining ORB evidence must not hide a newer failure for the same leg."""
        now = time.time()
        common = {
            "route_visual_heading_required": True,
            "route_visual_heading_leg_index": 4,
            "route_visual_heading_map_id": "map_a",
            "route_visual_heading_patrol_id": "patrol_a",
            "route_visual_heading_baseline_replay_id": "baseline_a",
        }
        verified_heading = {
            **common,
            "instance_id": "hold_002280",
            "received_unix": now,
            "route_visual_heading_verified": True,
            "route_visual_heading_correction_deg": 3.0,
            "route_visual_heading_current": [-1.0, 0.0, 0.0],
            "route_visual_heading_recorded": [-1.0, 0.0, 0.0],
            "route_visual_heading_inliers": 60,
            "route_visual_heading_minimum_inliers": 50,
        }
        failed_heading = {
            **common,
            "instance_id": "hold_002281",
            "received_unix": now + 0.01,
            "route_visual_heading_verified": False,
            "route_visual_heading_reason": "recorded departure view has too few inliers",
        }
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "poses.json"
            stream.write_text(
                json.dumps({"poses": [verified_heading, failed_heading]}),
                encoding="utf-8",
            )
            observation = bridge.latest_recorded_departure_heading(
                stream,
                2.5,
                map_id="map_a",
                patrol_id="patrol_a",
                baseline_replay_id="baseline_a",
                expected_leg_index=4,
            )

        self.assertFalse(observation["ok"])
        self.assertIn("too few inliers", observation["reason"])

    def test_completed_orb_heading_triplet_survives_next_weak_frame(self):
        """A fast 3030-3032 consensus must not be lost between bridge polls."""
        now = time.time()
        common = {
            "route_visual_heading_required": True,
            "route_visual_heading_leg_index": 4,
            "route_visual_heading_map_id": "map_a",
            "route_visual_heading_patrol_id": "patrol_a",
            "route_visual_heading_baseline_replay_id": "baseline_a",
            "route_visual_heading_minimum_inliers": 50,
        }
        poses = []
        for frame, inliers, correction in (
            (3030, 58, 1.218),
            (3031, 56, 1.159),
            (3032, 52, 1.143),
        ):
            poses.append(
                {
                    **common,
                    "instance_id": f"hold_{frame:06d}",
                    "image_name": f"query/query_{frame:06d}.jpg",
                    "received_unix": now + (frame - 3030) * 0.01,
                    "route_visual_heading_verified": True,
                    "route_visual_heading_correction_deg": correction,
                    "route_visual_heading_current": [-0.99, 0.0, -0.14],
                    "route_visual_heading_recorded": [-1.0, 0.0, 0.0],
                    "route_visual_heading_inliers": inliers,
                }
            )
        poses.append(
            {
                **common,
                "instance_id": "hold_003033",
                "image_name": "query/query_003033.jpg",
                "received_unix": now + 0.03,
                "route_visual_heading_verified": False,
                "route_visual_heading_reason": "48 inliers below the 50 gate",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "poses.json"
            stream.write_text(json.dumps({"poses": poses}), encoding="utf-8")
            observation = bridge.latest_recorded_departure_heading(
                stream,
                2.5,
                map_id="map_a",
                patrol_id="patrol_a",
                baseline_replay_id="baseline_a",
                expected_leg_index=4,
            )

        self.assertTrue(observation["ok"])
        self.assertTrue(observation["localizer_heading_consensus_verified"])
        self.assertGreaterEqual(observation["localizer_heading_consensus_count"], 3)
        self.assertEqual(observation["instance_id"], "hold_003032")

    def test_recorded_departure_heading_requires_matching_verified_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "poses.json"
            pose = {
                "instance_id": "heading_12",
                "received_unix": time.time(),
                "route_visual_heading_required": True,
                "route_visual_heading_verified": True,
                "route_visual_heading_correction_deg": 25.8,
                "route_visual_heading_current": [-0.80, 0.0, 0.60],
                "route_visual_heading_recorded": [-0.998, 0.0, 0.060],
                "route_visual_heading_inliers": 197,
                "route_visual_heading_minimum_inliers": 120,
                "route_visual_heading_anchor": "query_002901.jpg",
                "route_visual_heading_source_frame": 2901,
                "route_visual_heading_map_id": "map_a",
                "route_visual_heading_patrol_id": "patrol_a",
                "route_visual_heading_baseline_replay_id": "baseline_a",
            }
            stream.write_text(json.dumps({"poses": [pose]}), encoding="utf-8")
            observation = bridge.latest_recorded_departure_heading(
                stream,
                2.5,
                map_id="map_a",
                patrol_id="patrol_a",
                baseline_replay_id="baseline_a",
            )
            wrong_identity = bridge.latest_recorded_departure_heading(
                stream,
                2.5,
                map_id="map_a",
                patrol_id="patrol_a",
                baseline_replay_id="other_baseline",
            )

        self.assertTrue(observation["ok"])
        self.assertAlmostEqual(observation["correction_deg"], 25.8)
        self.assertEqual(observation["inliers"], 197)
        self.assertAlmostEqual(
            math.hypot(
                observation["current_heading"][0],
                observation["current_heading"][2],
            ),
            1.0,
        )
        self.assertFalse(wrong_identity["ok"])
        self.assertIn("identity mismatch", wrong_identity["reason"])

    def test_point_four_departure_heading_uses_verified_leg_specific_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "poses.json"
            pose = {
                "instance_id": "heading_point4",
                "received_unix": time.time(),
                "route_visual_heading_required": True,
                "route_visual_heading_leg_index": 4,
                "route_visual_heading_verified": True,
                "route_visual_heading_correction_deg": 2.1,
                "route_visual_heading_current": [0.04, 0.0, -0.999],
                "route_visual_heading_recorded": [0.0, 0.0, -1.0],
                "route_visual_heading_inliers": 50,
                "route_visual_heading_minimum_inliers": 50,
                "route_visual_heading_anchor": "query_004094.jpg",
                "route_visual_heading_source_frame": 4094,
                "route_visual_heading_map_id": "map_a",
                "route_visual_heading_patrol_id": "patrol_a",
                "route_visual_heading_baseline_replay_id": "baseline_a",
            }
            stream.write_text(json.dumps({"poses": [pose]}), encoding="utf-8")
            accepted = bridge.latest_recorded_departure_heading(
                stream,
                2.5,
                map_id="map_a",
                patrol_id="patrol_a",
                baseline_replay_id="baseline_a",
                expected_leg_index=4,
            )
            wrong_leg = bridge.latest_recorded_departure_heading(
                stream,
                2.5,
                map_id="map_a",
                patrol_id="patrol_a",
                baseline_replay_id="baseline_a",
                expected_leg_index=3,
            )
            pose["route_visual_heading_inliers"] = 49
            stream.write_text(json.dumps({"poses": [pose]}), encoding="utf-8")
            too_weak = bridge.latest_recorded_departure_heading(
                stream,
                2.5,
                map_id="map_a",
                patrol_id="patrol_a",
                baseline_replay_id="baseline_a",
                expected_leg_index=4,
            )
            pose["route_visual_heading_leg_index"] = 1
            pose["route_visual_heading_inliers"] = 50
            pose["route_visual_heading_minimum_inliers"] = 50
            stream.write_text(json.dumps({"poses": [pose]}), encoding="utf-8")
            point_one_live_gate = bridge.latest_recorded_departure_heading(
                stream,
                2.5,
                map_id="map_a",
                patrol_id="patrol_a",
                baseline_replay_id="baseline_a",
                expected_leg_index=1,
            )

        self.assertTrue(accepted["ok"])
        self.assertEqual(accepted["minimum_inliers"], 50)
        self.assertFalse(wrong_leg["ok"])
        self.assertIn("patrol-leg mismatch", wrong_leg["reason"])
        self.assertFalse(too_weak["ok"])
        self.assertIn("below the 50 gate", too_weak["reason"])
        self.assertTrue(point_one_live_gate["ok"])
        self.assertEqual(point_one_live_gate["minimum_inliers"], 50)

    def test_point_one_departure_heading_accepts_measured_live_inlier_floor(self):
        now = time.time()
        pose = {
            "instance_id": "heading_point1_live",
            "received_unix": now,
            "route_visual_heading_required": True,
            "route_visual_heading_leg_index": 1,
            "route_visual_heading_verified": True,
            "route_visual_heading_correction_deg": -3.35,
            "route_visual_heading_current": [0.998, 0.0, 0.063],
            "route_visual_heading_recorded": [1.0, 0.0, 0.0],
            "route_visual_heading_inliers": 48,
            "route_visual_heading_minimum_inliers": 48,
            "route_visual_heading_map_id": "map_a",
            "route_visual_heading_patrol_id": "patrol_a",
            "route_visual_heading_baseline_replay_id": "baseline_a",
        }
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "poses.json"
            stream.write_text(json.dumps({"poses": [pose]}), encoding="utf-8")
            observation = bridge.latest_recorded_departure_heading(
                stream,
                2.5,
                map_id="map_a",
                patrol_id="patrol_a",
                baseline_replay_id="baseline_a",
                expected_leg_index=1,
            )

        self.assertTrue(observation["ok"])
        self.assertEqual(observation["minimum_inliers"], 48)
        self.assertEqual(observation["inliers"], 48)

    def test_pose_readers_ignore_stale_recovery_appended_after_newer_frame(self):
        """Regress the live Point-4 hold_002151 appended after hold_002390."""
        now = time.time()
        fresh = {
            "instance_id": "hold_002390",
            "image_name": "query/query_002390.jpg",
            "time_sec": 291.191378,
            "received_unix": now,
            "success": True,
            "held_pose": False,
            "rcenter": [-3.0736, -0.0803, 1.1114],
            "rheading": [-0.3624, 0.0, -0.9320],
            "rotation_heading": [-0.7576, 0.0, -0.6527],
            "rotation_heading_source": "optical_flow_yaw",
            "rotation_heading_tracks": 499,
            "route_visual_heading_required": True,
            "route_visual_heading_leg_index": 4,
            "route_visual_heading_verified": True,
            "route_visual_heading_correction_deg": 14.95,
            "route_visual_heading_current": [-0.3624, 0.0, -0.9320],
            "route_visual_heading_recorded": [-0.1097, 0.0, -0.9940],
            "route_visual_heading_inliers": 213,
            "route_visual_heading_minimum_inliers": 50,
            "route_visual_heading_map_id": "map_a",
            "route_visual_heading_patrol_id": "patrol_a",
            "route_visual_heading_baseline_replay_id": "baseline_a",
        }
        stale_background_result = {
            **fresh,
            "instance_id": "hold_002151",
            "image_name": "query/query_002151.jpg",
            "time_sec": None,
            "received_unix": None,
            "success": False,
            "held_pose": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "poses.json"
            stream.write_text(
                json.dumps(
                    {
                        "updated_at": now,
                        "processed_count": 2391,
                        # The stale background result is intentionally last.
                        "poses": [fresh, stale_background_result],
                    }
                ),
                encoding="utf-8",
            )
            gate = bridge.latest_tsolve_pose_gate(stream, 2.5)
            rotation = bridge.latest_rotation_only_heading(stream, 2.5)
            recorded = bridge.latest_recorded_departure_heading(
                stream,
                2.5,
                map_id="map_a",
                patrol_id="patrol_a",
                baseline_replay_id="baseline_a",
                expected_leg_index=4,
            )

        self.assertTrue(gate["ok"])
        self.assertEqual(gate["pose"]["instance_id"], "hold_002390")
        self.assertTrue(rotation["ok"])
        self.assertEqual(rotation["instance_id"], "hold_002390")
        self.assertTrue(recorded["ok"])
        self.assertEqual(recorded["instance_id"], "hold_002390")
        self.assertAlmostEqual(recorded["correction_deg"], 14.95)

    def test_verified_point_four_turn_handoff_authorizes_one_bounded_departure(self):
        anchor = [-3.0736291183109774, -0.0802621969998909, 1.1113942967930859]
        source_gate = {
            "ok": True,
            "processed_count": 2755,
            "pose_offset_room": [0.4, 0.0, -0.2],
            "recent_hold_fallback": True,
            "pose": {
                "instance_id": "visual_route_002755",
                "rcenter": [-2.9, -0.08, 1.08],
                "rotation_raw_rcenter": [-2.7, -0.08, 1.0],
                "rotation_position_locked": True,
                "translation_allowed": False,
                "pose_source": "patrol_visual_route_recovery",
            },
        }
        heading_observation = {
            "ok": True,
            "instance_id": "heading_002755",
            "received_unix": time.time(),
            "correction_deg": 3.13,
            "inliers": 50,
            "minimum_inliers": 50,
            "current_heading": [-0.998, 0.0, 0.063],
        }

        gate = bridge.verified_endpoint_turn_departure_gate(
            source_gate,
            position_anchor=anchor,
            heading_observation=heading_observation,
            expected_leg_index=4,
            endpoint_handoff_verified=True,
            endpoint_handoff_source="verified_visual_endpoint",
            stable_heading_frames=3,
        )

        self.assertIsNotNone(gate)
        self.assertTrue(gate["verified_endpoint_turn_departure"])
        self.assertEqual(bridge.pose_gate_position(gate), anchor)
        self.assertFalse(gate["recent_hold_fallback"])
        self.assertFalse(bridge.pose_gate_rotation_locked(gate))
        self.assertIsNone(
            bridge.guided_command_pose_safety_issue(gate, bf=0.022)
        )
        # This command prior must never masquerade as a new metric solution.
        self.assertFalse(bridge.pose_gate_has_fresh_metric_position(gate))

    def test_verified_lap_point_one_turn_handoff_authorizes_one_bounded_departure(self):
        """Regression for the 15:02 full-lap abort before lap-2 motion."""
        anchor = [-3.2329557447702215, -0.0802621969998909, -0.33236579860361815]
        source_gate = {
            "ok": True,
            "processed_count": 2085,
            "pose": {
                "instance_id": "instance_002085",
                "received_unix": time.time() - 1.0,
                "rcenter": list(anchor),
                "rotation_position_locked": True,
                "translation_allowed": False,
                "route_pose_epoch": 2,
                "route_pose_epoch_reason": "verified_point1_handoff",
            },
        }
        # These values mirror the first successful physical lap boundary: the
        # Point-1 departure image was aligned to 0.85 degrees with 76 inliers,
        # but the old controller waited for another translation solution and
        # aborted after 219 held frames without sending a forward command.
        heading_observation = {
            "ok": True,
            "instance_id": "hold_002320",
            "received_unix": time.time(),
            "correction_deg": -0.849,
            "inliers": 76,
            "minimum_inliers": 48,
            "current_heading": [0.999887, 0.0, -0.015032],
        }

        gate = bridge.verified_endpoint_turn_departure_gate(
            source_gate,
            position_anchor=anchor,
            heading_observation=heading_observation,
            expected_leg_index=1,
            endpoint_handoff_verified=True,
            endpoint_handoff_source="metric_tsolve",
            stable_heading_frames=6,
        )

        self.assertIsNotNone(gate)
        self.assertTrue(gate["verified_endpoint_turn_departure"])
        self.assertEqual(bridge.pose_gate_position(gate), anchor)
        self.assertEqual(
            gate["pose"]["verified_endpoint_turn_leg_index"],
            1,
        )
        self.assertEqual(
            gate["pose"]["rotation_position_source"],
            "verified_lap_point1_turn_handoff",
        )
        self.assertFalse(bridge.pose_gate_rotation_locked(gate))
        self.assertIsNone(bridge.guided_command_pose_safety_issue(gate, bf=0.015))
        self.assertFalse(bridge.pose_gate_has_fresh_metric_position(gate))

    def test_verified_point_four_turn_handoff_fails_closed(self):
        source_gate = {"ok": True, "pose": {"rcenter": [-3.0, -0.08, 1.1]}}
        observation = {
            "ok": True,
            "received_unix": time.time(),
            "correction_deg": 3.0,
            "inliers": 50,
            "minimum_inliers": 50,
            "current_heading": [-1.0, 0.0, 0.0],
        }

        def build(**changes):
            params = {
                "source_gate": source_gate,
                "position_anchor": [-3.07, -0.08, 1.11],
                "heading_observation": observation,
                "expected_leg_index": 4,
                "endpoint_handoff_verified": True,
                "endpoint_handoff_source": "verified_visual_endpoint",
                "stable_heading_frames": 3,
            }
            params.update(changes)
            return bridge.verified_endpoint_turn_departure_gate(**params)

        self.assertIsNone(build(expected_leg_index=3))
        self.assertIsNone(build(endpoint_handoff_verified=False))
        self.assertIsNone(build(endpoint_handoff_source="unverified"))
        self.assertIsNone(build(stable_heading_frames=2))
        self.assertIsNone(
            build(heading_observation={**observation, "correction_deg": 4.01})
        )
        self.assertIsNone(
            build(heading_observation={**observation, "inliers": 49})
        )

    def test_verified_point_three_turn_handoff_authorizes_one_bounded_departure(self):
        anchor = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        source_gate = {
            "ok": True,
            "processed_count": 1359,
            "recent_hold_fallback": True,
            "pose": {
                "instance_id": "visual_route_001359",
                "rcenter": [-0.488857083951007, -0.0802621969998909, 0.9545679170263157],
                "rotation_position_locked": True,
                "translation_allowed": False,
                "pose_source": "patrol_visual_route_recovery",
            },
        }
        heading_observation = {
            "ok": True,
            "instance_id": "hold_001428",
            "received_unix": time.time(),
            "heading": [-0.999, 0.0, -0.045],
            "tracks": 500,
        }

        gate = bridge.verified_optical_endpoint_turn_departure_gate(
            source_gate,
            position_anchor=anchor,
            heading_observation=heading_observation,
            heading_error_rad=math.radians(4.0),
            expected_leg_index=3,
            endpoint_handoff_verified=True,
            stable_heading_frames=3,
        )

        self.assertIsNotNone(gate)
        self.assertTrue(gate["verified_endpoint_turn_departure"])
        self.assertEqual(bridge.pose_gate_position(gate), anchor)
        self.assertFalse(gate["recent_hold_fallback"])
        self.assertFalse(bridge.pose_gate_rotation_locked(gate))
        self.assertIsNone(
            bridge.guided_command_pose_safety_issue(gate, bf=0.022)
        )
        self.assertFalse(bridge.pose_gate_has_fresh_metric_position(gate))

    def test_recorded_point_three_view_can_stop_lagging_optical_yaw(self):
        anchor = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        source_gate = {
            "ok": True,
            "recent_hold_fallback": True,
            "pose": {
                "instance_id": "hold_001428",
                "rcenter": anchor,
                "rotation_position_locked": True,
                "translation_allowed": False,
            },
        }
        gate = bridge.verified_recorded_point_three_departure_gate(
            source_gate,
            position_anchor=anchor,
            heading_observation={
                "ok": True,
                "received_unix": time.time(),
                "correction_deg": 2.1,
                "inliers": 130,
                "minimum_inliers": 120,
                "current_heading": [-1.0, 0.0, 0.0],
            },
            endpoint_handoff_verified=True,
            endpoint_handoff_source="verified_visual_endpoint",
            stable_heading_frames=3,
        )

        self.assertIsNotNone(gate)
        self.assertEqual(bridge.pose_gate_position(gate), anchor)
        self.assertEqual(gate["pose"]["verified_endpoint_turn_leg_index"], 3)
        self.assertEqual(
            gate["pose"]["pose_source"],
            "verified_recorded_point3_turn_departure",
        )
        self.assertFalse(bridge.pose_gate_rotation_locked(gate))

    def test_point_three_command_effort_remains_telemetry_not_a_fixed_turn_cap(self):
        self.assertAlmostEqual(
            bridge.normalized_yaw_command_effort(0.05, 0.05, 0.30),
            0.30,
        )
        self.assertAlmostEqual(
            bridge.normalized_yaw_command_effort(0.0225, 0.05, 0.30),
            0.135,
        )
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("point_three_yaw_effort_limit_seconds", source)
        self.assertNotIn('"phase": "point3_yaw_command_limit_hold"', source)
        # Rotation still requires fresh visual feedback and retains the
        # independent overall rotation-only safety limit.
        self.assertIn('"phase": "point3_yaw_feedback_timing_hold"', source)
        self.assertIn("max_taught_rotation_yaw_seconds", source)

    def test_point_three_recorded_view_gets_only_a_bounded_fine_recovery(self):
        # The 11:04 live run stopped with a fresh 120+ inlier recorded-view
        # correction of about 18.8 degrees. It needs a fine recovery budget,
        # not an unlimited reopening of the coarse optical turn.
        self.assertAlmostEqual(
            bridge.point_three_recorded_recovery_effort_limit_seconds(18.8),
            2.85,
        )
        self.assertEqual(
            bridge.point_three_recorded_recovery_effort_limit_seconds(5.9),
            0.0,
        )
        self.assertEqual(
            bridge.point_three_recorded_recovery_effort_limit_seconds(30.1),
            0.0,
        )
        self.assertEqual(
            bridge.point_three_recorded_recovery_effort_limit_seconds(float("nan")),
            0.0,
        )

    def test_verified_point_three_turn_handoff_accepts_metric_arrival_source(self):
        anchor = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        source_gate = {
            "ok": True,
            "pose": {
                "instance_id": "instance_001359",
                "rcenter": anchor,
                "center": [0.77, -0.02, 1.27],
                "t": [1.41, -0.14, -0.43],
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
        }
        gate = bridge.verified_optical_endpoint_turn_departure_gate(
            source_gate,
            position_anchor=anchor,
            heading_observation={
                "ok": True,
                "received_unix": time.time(),
                "heading": [-1.0, 0.0, 0.0],
                "tracks": 500,
            },
            heading_error_rad=math.radians(2.0),
            expected_leg_index=3,
            endpoint_handoff_verified=True,
            stable_heading_frames=3,
            endpoint_handoff_source="metric_tsolve",
        )

        self.assertIsNotNone(gate)
        self.assertEqual(
            gate["pose"]["verified_endpoint_turn_handoff_source"],
            "metric_tsolve",
        )

    def test_verified_point_three_turn_handoff_fails_closed(self):
        source_gate = {
            "ok": True,
            "pose": {"rcenter": [-0.49, -0.08, 0.956]},
        }
        observation = {
            "ok": True,
            "received_unix": time.time(),
            "heading": [-1.0, 0.0, 0.0],
            "tracks": 500,
        }

        def build(**changes):
            params = {
                "source_gate": source_gate,
                "position_anchor": [-0.49, -0.08, 0.956],
                "heading_observation": observation,
                "heading_error_rad": math.radians(4.0),
                "expected_leg_index": 3,
                "endpoint_handoff_verified": True,
                "stable_heading_frames": 3,
            }
            params.update(changes)
            return bridge.verified_optical_endpoint_turn_departure_gate(
                **params
            )

        self.assertIsNone(build(expected_leg_index=4))
        self.assertIsNone(build(endpoint_handoff_verified=False))
        self.assertIsNone(build(stable_heading_frames=2))
        self.assertIsNone(
            build(heading_error_rad=math.radians(4.01))
        )
        self.assertIsNone(
            build(heading_error_rad=math.radians(8.607))
        )
        self.assertIsNone(
            build(heading_observation={**observation, "tracks": 59})
        )

    def test_verified_point_four_turn_handoff_releases_only_one_precise_probe(self):
        anchor = [-3.0736291183109774, -0.0802621969998909, 1.1113942967930859]
        source_gate = {
            "ok": True,
            "processed_count": 1496,
            "recent_hold_fallback": True,
            "rotation_handoff_hold": True,
            "pose": {
                "instance_id": "hold_002121",
                "rcenter": anchor,
                "rotation_position_locked": True,
                "translation_allowed": False,
                "rotation_position_source": "post_yaw_anchor_hold",
                "pose_source": "patrol_visual_route_recovery",
            },
        }
        heading_observation = {
            "ok": True,
            "instance_id": "hold_002124",
            "received_unix": time.time(),
            "heading": [-0.13, 0.0, -0.9915],
            "tracks": 495,
        }

        gate = bridge.verified_optical_endpoint_turn_departure_gate(
            source_gate,
            position_anchor=anchor,
            heading_observation=heading_observation,
            heading_error_rad=math.radians(1.8),
            expected_leg_index=4,
            endpoint_handoff_verified=True,
            stable_heading_frames=3,
            endpoint_handoff_source="verified_visual_endpoint",
        )

        self.assertIsNotNone(gate)
        self.assertTrue(gate["verified_endpoint_turn_departure"])
        self.assertEqual(bridge.pose_gate_position(gate), anchor)
        self.assertFalse(gate["recent_hold_fallback"])
        self.assertFalse(gate["rotation_handoff_hold"])
        self.assertFalse(bridge.pose_gate_rotation_locked(gate))
        self.assertEqual(
            gate["pose"]["rotation_position_source"],
            "verified_point4_turn_handoff",
        )
        self.assertEqual(
            gate["pose"]["verified_endpoint_turn_handoff_source"],
            "verified_visual_endpoint",
        )
        self.assertIsNone(
            bridge.guided_command_pose_safety_issue(gate, bf=0.012)
        )

    def test_point_four_visual_arrival_proof_survives_hovers_and_route_change(self):
        # Exact handoff shape from Live ATLAS 15:30:00: 3->4 ended with
        # progress-independent endpoint verification, then two neutral hover
        # steps changed the live route context before 4->1 began.
        point_four = [
            -3.0736291183109774,
            -0.0802621969998909,
            1.1113942967930859,
        ]
        executed = [
            {
                "index": 14,
                "type": "cruise",
                "title": "Patrol cruise 3",
                "closed_loop": True,
                "target": point_four,
                "endpoint_leg_index": 3,
                "arrival_mode": "visual_checkpoint_endpoint_verified",
                "reached": True,
            },
            {"index": 15, "type": "hover", "title": "Hover and re-localize"},
            {"index": 16, "type": "hover", "title": "Patrol point 3"},
        ]

        record = bridge.prior_verified_endpoint_arrival_record(
            executed,
            segment_start=point_four,
            expected_leg_index=3,
        )

        self.assertIsNotNone(record)
        self.assertEqual(record["arrival_mode"], "visual_checkpoint_endpoint_verified")

    def test_point_four_prior_arrival_proof_fails_closed_for_wrong_or_weak_arrival(self):
        point_four = [-3.0736, -0.0802, 1.1114]

        def record(**changes):
            value = {
                "type": "cruise",
                "closed_loop": True,
                "target": point_four,
                "endpoint_leg_index": 3,
                "arrival_mode": "visual_checkpoint_endpoint_verified",
                "reached": True,
            }
            value.update(changes)
            return [value]

        self.assertIsNone(
            bridge.prior_verified_endpoint_arrival_record(
                record(reached=False),
                segment_start=point_four,
                expected_leg_index=3,
            )
        )
        self.assertIsNone(
            bridge.prior_verified_endpoint_arrival_record(
                record(arrival_mode="closest_soft_deadband"),
                segment_start=point_four,
                expected_leg_index=3,
            )
        )
        self.assertIsNone(
            bridge.prior_verified_endpoint_arrival_record(
                record(endpoint_leg_index=2),
                segment_start=point_four,
                expected_leg_index=3,
            )
        )
        self.assertIsNone(
            bridge.prior_verified_endpoint_arrival_record(
                record(target=[-2.7, -0.0802, 1.1114]),
                segment_start=point_four,
                expected_leg_index=3,
            )
        )

    def test_point_four_handoff_rotates_while_locked_before_translation_release(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        start = source.index("if point_four_handoff:")
        end = source.index(
            "if lap_metric_checkpoint_pending and not dynamic_lap_reentry:",
            start,
        )
        handoff = source[start:end]
        self.assertIn("prior_verified_endpoint_arrival_record(", handoff)
        self.assertIn("prior_point_four_visual_ready", handoff)
        self.assertIn("require_translation_safe=False", handoff)
        self.assertIn(
            '"the rotation-only 4->1 alignment may begin."',
            handoff,
        )

    def test_verified_point_four_turn_handoff_fails_closed(self):
        source_gate = {
            "ok": True,
            "pose": {
                "rcenter": [-3.0736, -0.0802, 1.1114],
                "rotation_position_locked": True,
                "translation_allowed": False,
            },
        }
        observation = {
            "ok": True,
            "received_unix": time.time(),
            "heading": [-0.11, 0.0, -0.994],
            "tracks": 495,
        }

        def build(**changes):
            params = {
                "source_gate": source_gate,
                "position_anchor": [-3.0736, -0.0802, 1.1114],
                "heading_observation": observation,
                "heading_error_rad": math.radians(2.0),
                "expected_leg_index": 4,
                "endpoint_handoff_verified": True,
                "stable_heading_frames": 3,
                "endpoint_handoff_source": "verified_visual_endpoint",
            }
            params.update(changes)
            return bridge.verified_optical_endpoint_turn_departure_gate(**params)

        self.assertIsNotNone(build(endpoint_handoff_source="metric_tsolve"))
        self.assertIsNone(build(endpoint_handoff_source="unverified"))
        self.assertIsNone(build(stable_heading_frames=2))
        self.assertIsNone(build(heading_error_rad=math.radians(2.01)))
        # The exact final error from Live ATLAS 14:21:30 must continue yaw,
        # never release forward motion or enter the old 25-second deadlock.
        self.assertIsNone(build(heading_error_rad=math.radians(4.488)))
        self.assertIsNone(
            build(heading_observation={**observation, "tracks": 59})
        )

    def test_point_four_optical_fine_handoff_is_short_strong_and_absolute_anchored(self):
        absolute = {
            "ok": True,
            "instance_id": "hold_002492",
            "received_unix": 100.0,
            "correction_deg": -4.95,
            "inliers": 34,
            "minimum_inliers": 30,
            "current_heading": [-0.02347, 0.0, -0.99972],
            "recorded_heading": [-0.10969, 0.0, -0.99397],
        }
        optical_at_anchor = [-0.28802, 0.0, -0.95762]
        optical = {
            "ok": True,
            "instance_id": "hold_002500",
            "received_unix": 101.0,
            "heading": [-0.31014, 0.0, -0.95069],
            "tracks": 498,
        }
        bias = bridge.signed_angle_xz(
            optical_at_anchor,
            absolute["current_heading"],
        )

        fused = bridge.recorded_heading_optical_fine_handoff(
            absolute,
            optical,
            optical_heading_bias_rad=bias,
        )

        self.assertIsNotNone(fused)
        self.assertTrue(fused["optical_fine_handoff"])
        self.assertLess(abs(fused["correction_deg"]), 4.0)
        self.assertEqual(fused["inliers"], 34)
        self.assertEqual(fused["optical_tracks"], 498)
        self.assertIsNone(
            bridge.recorded_heading_optical_fine_handoff(
                absolute,
                {**optical, "tracks": 59},
                optical_heading_bias_rad=bias,
            )
        )
        self.assertIsNone(
            bridge.recorded_heading_optical_fine_handoff(
                absolute,
                {**optical, "received_unix": 102.51},
                optical_heading_bias_rad=bias,
            )
        )

    def test_previous_fighting_run_fuses_heading_but_stays_below_new_departure_floor(self):
        poses_path = (
            BRIDGE_PATH.parents[1]
            / "viewer"
            / "public"
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "dji_live_20260817_093250_4c4119"
            / "poses_partial.json"
        )
        if not poses_path.exists():
            self.skipTest("09:32 fighting-turn regression poses are unavailable")
        poses = json.loads(poses_path.read_text(encoding="utf-8"))["poses"]
        by_instance = {
            str(pose.get("instance_id") or ""): pose
            for pose in poses
        }
        absolute_pose = by_instance["hold_002492"]
        absolute = {
            "ok": True,
            "instance_id": absolute_pose["instance_id"],
            "received_unix": absolute_pose["received_unix"],
            "correction_deg": absolute_pose[
                "route_visual_heading_correction_deg"
            ],
            "inliers": absolute_pose["route_visual_heading_inliers"],
            "minimum_inliers": absolute_pose[
                "route_visual_heading_minimum_inliers"
            ],
            "current_heading": absolute_pose[
                "route_visual_heading_current"
            ],
            "recorded_heading": absolute_pose[
                "route_visual_heading_recorded"
            ],
        }
        bias = bridge.signed_angle_xz(
            absolute_pose["rotation_heading"],
            absolute["current_heading"],
        )
        fused = []
        for frame in (2498, 2499, 2500):
            pose = by_instance[f"hold_{frame:06d}"]
            fused.append(
                bridge.recorded_heading_optical_fine_handoff(
                    absolute,
                    {
                        "ok": True,
                        "instance_id": pose["instance_id"],
                        "received_unix": pose["received_unix"],
                        "heading": pose["rotation_heading"],
                        "tracks": pose["rotation_heading_tracks"],
                    },
                    optical_heading_bias_rad=bias,
                )
            )

        self.assertTrue(all(item is not None for item in fused))
        self.assertTrue(all(abs(item["correction_deg"]) <= 4.0 for item in fused))
        departure_gate = bridge.verified_endpoint_turn_departure_gate(
            {
                "ok": True,
                "pose": {
                    "rcenter": [-3.0736, -0.0803, 1.1114],
                    "rotation_position_locked": True,
                    "translation_allowed": False,
                },
            },
            position_anchor=[-3.0736, -0.0803, 1.1114],
            heading_observation=fused[-1],
            expected_leg_index=4,
            endpoint_handoff_verified=True,
            endpoint_handoff_source="verified_visual_endpoint",
            stable_heading_frames=3,
        )
        # The old run acquired only 34 absolute-view inliers. Optical flow can
        # still propagate that diagnostic heading, but the new 50-inlier
        # Point-4 experiment must not authorize physical translation from it.
        self.assertIsNone(departure_gate)

    def test_taught_leg_requires_exact_patrol_leg_endpoints(self):
        reference = {
            "legs": [
                {
                    "from_point": 3,
                    "to_point": 4,
                    "from": [0.4, 1.0, 1.2],
                    "to": [-1.9, 1.0, 1.5],
                }
            ]
        }
        self.assertEqual(
            bridge.taught_leg_for_step(reference, {"from": [0.4, 1.0, 1.2], "to": [-1.9, 1.0, 1.5]}),
            reference["legs"][0],
        )
        self.assertIsNone(
            bridge.taught_leg_for_step(reference, {"from": [0.6, 1.0, 1.2], "to": [-1.9, 1.0, 1.5]})
        )

    def test_adjusted_patrol_points_never_reuse_stale_taught_leg_by_command_order(self):
        reference = {
            "legs": [
                {"from_point": 1, "to_point": 2},
                {"from_point": 2, "to_point": 3},
                {"from_point": 3, "to_point": 4},
                {"from_point": 4, "to_point": 1},
            ]
        }
        adjusted_step = {
            "title": "Patrol cruise 4",
            "from": [-0.46, 0.0, 0.90],
            "to": [-2.04, 0.0, 0.76],
        }
        self.assertIsNone(bridge.taught_leg_for_step(reference, adjusted_step))

    def test_adjusted_point_three_to_four_reuses_only_heading_compatible_turn_guard(self):
        guarded_leg = {
            "from_point": 3,
            "to_point": 4,
            "from": [0.40895883196438965, 1.7599605504441844, 1.1828101894072862],
            "to": [-1.929244345701856, -0.20678855225328296, 1.4676049473028439],
            "expected_heading_deg": 173.0555410638647,
        }
        reference = {"legs": [guarded_leg]}
        corrected_step = {
            "title": "Patrol cruise 4",
            "from": [-0.4886978074319452, 1.7599605504441844, 0.9560112230666532],
            "to": [-3.0736291183109774, -0.20678855225328296, 1.1113942967930859],
        }
        self.assertEqual(
            bridge.taught_leg_for_step(reference, corrected_step),
            guarded_leg,
        )

    def test_heading_compatible_guard_does_not_restore_global_command_order_fallback(self):
        reference = {
            "legs": [
                {
                    "from_point": 3,
                    "to_point": 4,
                    "expected_heading_deg": 173.0555410638647,
                },
                {
                    "from_point": 4,
                    "to_point": 1,
                    "expected_heading_deg": -133.7715977923365,
                },
            ]
        }
        wrong_heading = {
            "title": "Patrol cruise 4",
            "from": [-0.48, 0.0, 0.95],
            "to": [-0.48, 0.0, -1.50],
        }
        renamed_leg = {
            "title": "Patrol cruise 5",
            "from": [-0.48, 0.0, 0.95],
            "to": [-3.07, 0.0, 1.11],
        }
        self.assertIsNone(bridge.taught_leg_for_step(reference, wrong_heading))
        self.assertIsNone(bridge.taught_leg_for_step(reference, renamed_leg))

    def test_point_three_to_four_uses_operator_requested_right_turn(self):
        leg = {"from_point": 3, "to_point": 4}
        self.assertTrue(bridge.is_guarded_point_three_to_four_turn(leg))
        # Restore the turn path that physically reached Point 4 in Live ATLAS
        # 11:57:36.  The right-turn direction remains fixed, but mandatory
        # recorded-image acquisition is reserved for the unfinished 4->1 leg.
        self.assertFalse(bridge.taught_turn_requires_recorded_departure_view(leg))
        self.assertTrue(
            bridge.taught_turn_requires_recorded_departure_view(
                {"from_point": 4, "to_point": 1}
            )
        )
        self.assertFalse(
            bridge.is_guarded_point_three_to_four_turn(
                {"from_point": 2, "to_point": 3}
            )
        )
        self.assertTrue(
            bridge.is_point_two_to_three_leg(
                {"from_point": 2, "to_point": 3}
            )
        )
        self.assertFalse(
            bridge.taught_turn_requires_recorded_departure_view(
                {"from_point": 2, "to_point": 3}
            )
        )
        self.assertTrue(
            bridge.taught_turn_requires_recorded_departure_view(
                {"from_point": 1, "to_point": 2}
            )
        )
        self.assertEqual(bridge.taught_turn_direction_override(leg), "right")
        self.assertEqual(bridge.yaw_direction_for_angle(math.radians(102.0), "right"), 1.0)
        self.assertEqual(bridge.yaw_direction_for_angle(math.radians(102.0), None), 1.0)
        localizer = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("heading_leg_index in {1, 2, 3, 4}", localizer)
        self.assertIn(
            'int(route_context.get("leg_index") or 0) in {3, 4}',
            localizer,
        )
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn(
            '45.0 if turn_direction_override == "left" else 24.0',
            source,
        )
        self.assertIn('if turn_direction_override == "left":', source)
        self.assertIn("wrong_yaw_pulses = 0", source)

        successful_status_path = (
            Path(__file__).resolve().parents[1]
            / "viewer"
            / "public"
            / "live_dji_sessions"
            / "atlas_dji_live_20260811_115736_2b91ca"
            / "status.json"
        )
        successful_status = json.loads(
            successful_status_path.read_text(encoding="utf-8")
        )
        successful_progress = successful_status["last_control"]["progress"]
        self.assertEqual(successful_progress["recovery_phase"], "taught_turn_position_recovery")
        self.assertEqual(successful_progress["executed_steps"], 12)

    def test_taught_rotation_recovery_locks_yaw_then_requests_fresh_unlocked_position(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn('"phase": "taught_rotation_only_recovery"', source)
        self.assertIn('"phase": "taught_turn_wait_fresh_position"', source)
        self.assertIn("else 24.0", source)
        self.assertIn('or bool(gate_attempt.get("recent_hold_fallback"))', source)
        self.assertIn("required_rotation_heading_stable_frames = 3", source)
        self.assertIn("rotation_heading_stable_frames", source)
        self.assertIn(">= required_rotation_heading_stable_frames", source)
        self.assertIn(
            "taught_turn_requires_recorded_departure_view(taught_leg)",
            source,
        )
        self.assertIn("yaw_position_anchor = start_position", source)
        self.assertIn(
            "point_three_handoff\n"
            "                                or guarded_taught_rotation\n"
            "                                or rotation_position_untrusted",
            source,
        )
        self.assertIn("confirmed_rotation_angle = alignment_angle", source)
        self.assertIn("confirmed_rotation_gate = gate_attempt", source)
        self.assertIn("current_gate = confirmed_rotation_gate", source)
        self.assertIn("recorded_heading_acquired = False", source)
        self.assertIn("recorded_heading_acquired = True", source)
        self.assertIn("elif recorded_heading_acquired:", source)
        self.assertIn('"recorded_view_reacquisition_hold"', source)
        self.assertIn('"phase": "taught_turn_post_pulse_optical_return"', source)
        self.assertNotIn('"phase": "taught_turn_forward_reacquisition"', source)
        wait_start = source.index("if rotation_position_untrusted:")
        wait_block = source[
            wait_start:
            source.index("elif alignment_angle is not None", wait_start)
        ]
        self.assertIn("forward or lateral command is allowed", wait_block)
        self.assertIn('"phase": "taught_turn_wait_fresh_position"', wait_block)
        self.assertIn('"translation_locked": True', wait_block)
        self.assertIn('"route_visual_recovery_allowed": True', wait_block)
        self.assertIn('"rotation_release_requested": True', wait_block)
        self.assertIn('"body_forward_gain": 0.0', wait_block)
        self.assertIn('"body_lateral_gain": 0.0', wait_block)
        self.assertIn("wait_for_pose_recovery(", wait_block)
        self.assertIn('"taught_turn_recorded_heading_recovery"', wait_block)
        self.assertIn(
            "timeout=max_rotation_position_recovery_seconds",
            wait_block,
        )
        self.assertNotIn("gate_attempt.get(\"ok\")", wait_block)
        self.assertNotIn("release_rotation_lock = bool", wait_block)
        taught_block = source[source.index('"phase": "taught_rotation_only_recovery"'):source.index('"phase": "bounded_blind_yaw_recovery"')]
        self.assertIn("lr=0.0", taught_block)
        self.assertIn("bf=0.0", taught_block)
        self.assertIn("current_gate.get(\"recent_hold_fallback\")", source)

    def test_point_three_fails_closed_until_recorded_view_is_confirmed(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn(
            '"phase": "taught_turn_wait_recorded_departure_view"',
            source,
        )
        wait_start = source.index(
            "guarded_taught_rotation\n"
            "                        and not travel_started\n"
            "                        and confirmed_rotation_gate is None"
        )
        wait_block = source[
            wait_start:
            source.index("if (\n                        not gate_attempt.get", wait_start)
        ]
        self.assertIn('"translation_locked": True', wait_block)
        self.assertIn("neutral_hover(drone, 0.12)", wait_block)
        self.assertIn("continue", wait_block)

    def test_confirmed_point_three_departure_preempts_stale_blind_yaw(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        stale_yaw_start = source.index(
            "if (\n"
            "                        not gate_attempt.get(\"ok\")\n"
            "                        and confirmed_rotation_gate is None\n"
            "                        and not travel_started\n"
            "                        and abs(last_alignment_yaw_rc) <= 1e-6"
        )
        blind_yaw_start = source.index(
            "if (\n"
            "                        not gate_attempt.get(\"ok\")\n"
            "                        and confirmed_rotation_gate is None\n"
            "                        and not travel_started\n"
            "                        and abs(last_alignment_yaw_rc) > 1e-6",
            stale_yaw_start,
        )
        confirmed_gate_start = source.index(
            "if confirmed_rotation_gate is not None:",
            blind_yaw_start,
        )
        self.assertLess(stale_yaw_start, blind_yaw_start)
        self.assertLess(blind_yaw_start, confirmed_gate_start)
        self.assertIn(
            "current_gate = confirmed_rotation_gate",
            source[confirmed_gate_start : confirmed_gate_start + 800],
        )

    def test_point_three_turn_never_reuses_stale_heading_or_blind_yaws_near_target(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        wait_start = source.index(
            '"phase": "taught_rotation_wait_new_heading"'
        )
        yaw_command = source.index("taught_yaw_rc = (", wait_start)
        self.assertLess(wait_start, yaw_command)
        wait_block = source[wait_start - 900 : yaw_command]
        self.assertIn(
            "observation_instance_id\n                                == last_yaw_command_observation_id",
            wait_block,
        )
        self.assertIn("neutral_hover(drone, 0.12)", wait_block)
        self.assertIn("fine_point_three_alignment", source[yaw_command - 500 : yaw_command + 700])

        blind_start = source.index(
            '"phase": "bounded_blind_yaw_recovery"'
        )
        blind_condition = source[blind_start - 1600 : blind_start]
        self.assertIn("last_measured_alignment_error_rad", blind_condition)
        self.assertIn("<= math.radians(12.0)", blind_condition)
        self.assertIn("and not point_three_handoff", blind_condition)
        self.assertIn('"phase": "point3_yaw_feedback_timing_hold"', source)
        self.assertIn(
            '"recorded_point3_departure_bounded_recovery"', source
        )
        self.assertIn(
            "point_three_recorded_recovery_effort_limit_seconds(", source
        )
        recorded_recovery_limit = source.index(
            '"phase": "point3_recorded_recovery_limit_hold"',
            source.index("taught_yaw_rc = ("),
        )
        guarded_yaw_command = source.index(
            "sent = execute_rc_pulse(",
            source.index("taught_yaw_rc = ("),
        )
        self.assertLess(recorded_recovery_limit, guarded_yaw_command)

    def test_point_three_preserves_verified_endpoint_across_rotation(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        handoff_start = source.index(
            "point_three_handoff = is_guarded_point_three_to_four_turn("
        )
        handoff_end = source.index("if point_four_handoff:", handoff_start)
        handoff = source[handoff_start:handoff_end]
        self.assertIn("expected_leg_index=2", handoff)
        self.assertIn("point_three_handoff_limit = min(0.30, max_pose_step)", handoff)
        self.assertIn("verified_endpoint_turn_source_gate = gate", handoff)
        self.assertIn("verified_endpoint_turn_anchor = list(segment_start)", handoff)
        self.assertIn('"phase": "point3_handoff_verified"', handoff)

        departure_start = source.index(
            "if rotation_position_untrusted:", handoff_end
        )
        departure_end = source.index(
            "elif alignment_angle is not None", departure_start
        )
        departure = source[departure_start:departure_end]
        self.assertIn(
            "elif point_three_handoff or point_four_handoff:", departure
        )
        self.assertIn(
            "verified_optical_endpoint_turn_departure_gate(", departure
        )
        self.assertIn("heading_error_rad=alignment_angle", departure)
        self.assertIn(
            '{"verified_visual_endpoint", "metric_tsolve"}', departure
        )
        self.assertIn('"bounded_departure_commands": 1', departure)
        self.assertIn(
            '"require_observed_progress_after_command": True', departure
        )

    def test_point_four_handoff_accepts_verified_endpoint_before_turn(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        checkpoint_start = source.index(
            "point_four_handoff = is_point_four_to_one_leg(taught_leg)"
        )
        checkpoint_end = source.index(
            "if lap_metric_checkpoint_pending and not dynamic_lap_reentry:",
            checkpoint_start,
        )
        checkpoint = source[checkpoint_start:checkpoint_end]
        self.assertIn('"phase": "point4_endpoint_handoff"', checkpoint)
        self.assertIn('"point4_endpoint_handoff"', checkpoint)
        self.assertIn("require_endpoint_verified=True", checkpoint)
        self.assertIn("endpoint_leg_index=3", checkpoint)
        self.assertIn("require_translation_safe=False", checkpoint)
        self.assertIn("prior_verified_endpoint_arrival_record(", checkpoint)
        self.assertIn("prior_point_four_visual_ready", checkpoint)
        self.assertIn("timeout=8.0", checkpoint)
        self.assertIn("point_four_handoff_limit = min(0.30, max_pose_step)", checkpoint)
        self.assertIn('"phase": "point4_position_correction_required"', checkpoint)
        self.assertIn(
            'point_four_handoff_source = "verified_visual_endpoint"',
            checkpoint,
        )
        self.assertIn(
            "yaw_position_anchor = list(point_four_handoff_position)",
            checkpoint,
        )
        self.assertIn('"phase": "point4_handoff_verified"', checkpoint)
        self.assertNotIn("require_metric_pose=True,", checkpoint)

        departure_start = source.index(
            "endpoint_departure_gate = (",
            checkpoint_end,
        )
        departure_end = source.index(
            "elif alignment_angle is not None",
            departure_start,
        )
        departure = source[departure_start:departure_end]
        self.assertIn("verified_endpoint_turn_departure_gate(", departure)
        self.assertIn(
            '"phase": "verified_endpoint_turn_departure_ready"',
            departure,
        )
        self.assertIn('"bounded_departure_commands": 1', departure)
        self.assertIn(
            '"require_observed_progress_after_command": True',
            departure,
        )

        smooth_start = source.index("used_smooth_cruise = bool(")
        smooth_end = source.index(")\n                    smooth_cruise_issue", smooth_start)
        self.assertIn(
            'not current_gate.get(\n                            "verified_endpoint_turn_departure"',
            source[smooth_start:smooth_end],
        )

    def test_point_four_absolute_orb_heading_does_not_wait_for_optical_rebase(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        branch_start = source.index(
            "if recorded_heading_observation.get(\"ok\"):",
            source.index("if guarded_taught_rotation:"),
        )
        branch_end = source.index(
            "elif recorded_heading_acquired:",
            branch_start,
        )
        absolute_orb_branch = source[branch_start:branch_end]

        self.assertIn(
            'recorded_heading_observation.get(\n                                            "correction_deg"',
            absolute_orb_branch,
        )
        self.assertNotIn(
            "recorded_view_waiting_optical_rebase",
            absolute_orb_branch,
        )
        self.assertNotIn("alignment_angle = None", absolute_orb_branch)

    def test_latest_point_four_endpoint_has_authority_for_rotation_only_handoff(self):
        trace_path = (
            BRIDGE_PATH.parents[1]
            / "viewer"
            / "public"
            / "live_dji_sessions"
            / "atlas_dji_live_20260816_144116_df75b6"
            / "control_trace.jsonl"
        )
        if not trace_path.exists():
            self.skipTest("14:41 Point-4 endpoint regression trace is unavailable")
        endpoint_pose = None
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            pose = (
                (row.get("progress") or {}).get("pose_gate") or {}
            ).get("pose")
            if (
                isinstance(pose, dict)
                and pose.get("route_visual_endpoint_verified") is True
            ):
                # This archived trace predates the endpoint-scale field.  Its
                # frame was separately verified as the real Point-4 view; add
                # that historical fact while testing the rotation handoff.
                pose = {
                    **pose,
                    "route_visual_endpoint_view_geometry_verified": True,
                }
            if bridge.taught_endpoint_arrival_verified(
                pose,
                expected_leg_index=3,
            ):
                endpoint_pose = pose
        self.assertIsNotNone(endpoint_pose)
        self.assertEqual(endpoint_pose["route_visual_endpoint_hits"], 3)
        self.assertGreaterEqual(
            endpoint_pose["route_visual_endpoint_best_inliers"],
            endpoint_pose["route_visual_endpoint_minimum_inliers"],
        )
        self.assertFalse(
            bridge.pose_gate_has_fresh_metric_position(
                {"ok": True, "pose": endpoint_pose}
            )
        )

    def test_metric_checkpoint_releases_only_localizer_position_guard(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("metric_position_recovery = bool(", source)
        self.assertIn(
            'progress.get("metric_position_recovery_allowed") is True',
            source,
        )
        self.assertIn(
            'progress.get("require_metric_pose") is True',
            source,
        )
        self.assertIn(
            'and not metric_position_recovery',
            source,
        )
        self.assertIn(
            'and not post_translation_progress_recovery',
            source,
        )
        self.assertIn(
            '"metric_position_recovery": metric_position_recovery',
            source,
        )
        self.assertIn(
            'recovery_route_context.get("position_guard_locked") is True',
            source,
        )
        self.assertIn(
            '"controller_translation_locked": controller_translation_locked',
            source,
        )

    def test_point_four_turn_cannot_finish_from_optical_heading_alone(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "guarded_taught_rotation = (\n"
            "                    taught_turn_requires_recorded_departure_view(taught_leg)\n"
            "                )",
            source,
        )
        guard_start = source.index("if guarded_taught_rotation:")
        guard_end = source.index("heading_aligned = bool(", guard_start)
        guard = source[guard_start:guard_end]
        self.assertIn("if recorded_heading_observation.get(\"ok\"):", guard)
        self.assertIn("alignment_source = \"recorded_patrol_departure_view\"", guard)
        self.assertIn("alignment_tolerance = None", guard)
        self.assertIn("alignment_source = \"optical_flow_coarse_search\"", guard)

    def test_all_rotation_progress_publishes_fixed_position_for_viewer(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("active_route_translation_locked = rotation_only_command", source)
        self.assertIn('"translation_locked": active_route_translation_locked', source)
        self.assertIn('"position_anchor": active_route_position_anchor', source)
        self.assertIn('if rotation_only_command and yaw_position_anchor is None:', source)
        self.assertIn('"phase": "pre_yaw_translation_settle"', source)
        self.assertIn("stable_settle_samples < 2", source)
        self.assertIn(
            "if patrol_visual_yaw_anchor_ready(current_gate)",
            source,
        )
        self.assertIn("settle_wait_seconds = max(8.0, pose_recovery_seconds)", source)
        self.assertIn("settle_step <= 0.06", source)
        self.assertIn('"phase": "pre_yaw_translation_settled"', source)

    def test_completed_turn_opens_visual_recovery_before_forward(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        wait_start = source.index("if rotation_position_untrusted:")
        wait_block = source[
            wait_start:
            source.index("elif alignment_angle is not None", wait_start)
        ]
        self.assertIn('"phase": "taught_turn_wait_fresh_position"', wait_block)
        self.assertIn('"route_visual_recovery_allowed": True', wait_block)
        self.assertIn('"rotation_release_requested": True', wait_block)
        self.assertIn('"translation_locked": True', wait_block)
        self.assertIn("rotation_pose_locked = pose_gate_rotation_locked(gate_attempt)", source)
        self.assertIn("pose_gate_rotation_locked(current_gate)", source)
        self.assertIn('"rotation_release_requested": release_rotation_lock', source)
        release_start = source.index(
            '"phase": "translation_pose_gate"',
            source.index("release_rotation_lock = bool("),
        )
        release_end = source.index("neutral_hover(drone, 0.12)", release_start)
        release_block = source[release_start:release_end]
        self.assertIn(
            '"route_visual_recovery_allowed": (\n                                    release_rotation_lock',
            release_block,
        )
        self.assertIn('"physical_translation_active": False', release_block)
        self.assertIn("0.0 if release_rotation_lock else forward_gain", release_block)
        self.assertIn(
            "The localizer\n                    # can then release its rotation anchor",
            source,
        )

    def test_initial_rotation_lock_relocalizes_before_body_forward_verification(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("require_translation_safe: bool = False", source)
        self.assertIn("or not pose_gate_rotation_locked(gate)", source)
        self.assertIn('"phase": "initial_heading_release_rotation_lock"', source)
        self.assertIn('"rotation_release_requested": True', source)
        self.assertIn('"initial_heading_translation_unlock"', source)
        self.assertIn("require_translation_safe=True", source)
        release_at = source.index('"phase": "initial_heading_release_rotation_lock"')
        calibration_at = source.index("if calibrate_forward_heading(gate):", release_at)
        self.assertLess(release_at, calibration_at)

    def test_viewer_uses_rotation_position_anchor_without_freezing_heading(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("let liveRotationPositionAnchor = null;", source)
        self.assertIn("corrected.rawRotationRcenter = corrected.rcenter;", source)
        self.assertIn("corrected.rcenter = liveRotationPositionAnchor.slice(0, 3).map(Number);", source)
        self.assertIn("progress.translation_locked", source)
        self.assertIn("pose.rotation_position_locked", source)
        self.assertIn("corrected.rheading = opticalHeading;", source)
        self.assertNotIn(
            'pose.pose_source === "patrol_visual_route_recovery" ||',
            source,
        )
        self.assertIn("const latestPose = latestLivePoseForDisplay(room.poses);", source)

    def test_manual_teach_can_record_only_the_missing_three_to_one_half(self):
        viewer = APP_PATH.read_text(encoding="utf-8")
        server = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn("Teach Finish 3→4→1", viewer)
        self.assertIn('toggleManualPatrolRecording(patrol, "continuation_3_4_1")', viewer)
        self.assertIn('"continuation_3_4_1": {', server)
        self.assertIn('"route_points": [3, 4, 1]', server)

    def test_localizer_keeps_independent_optical_heading_when_map_pose_is_questionable(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("def optical_flow_yaw_delta(", source)
        self.assertIn('"rotation_heading_source": "optical_flow_yaw"', source)
        self.assertIn("def update_rotation_reference_from_accepted_pose", source)
        self.assertIn("or pose.get(\"rotation_position_locked\")", source)

    def test_manual_turns_freeze_position_inside_the_pose_stream(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("class RotationOnlyPositionStabilizer", source)
        self.assertIn('pose["rotation_position_locked"] = True', source)
        self.assertIn('pose["translation_allowed"] = False', source)
        self.assertIn('pose["rheading"] = optical_heading', source)
        self.assertIn('pose["rheading_source"] = "optical_flow_yaw"', source)
        self.assertGreaterEqual(
            source.count("rotation_position_stabilizer.apply("),
            4,
        )
        self.assertIn("if pose is None:", source)
        self.assertIn("max_step = max(0.22, min(0.35, 0.18 + 0.17 * dt))", source)

    def test_learned_patrol_geometry_is_locked_to_recovery_data(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        viewer = APP_PATH.read_text(encoding="utf-8")
        lock = json.loads(GEOMETRY_LOCK_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(
            (GEOMETRY_LOCK_PATH.parents[3] / "manifest.json").read_text(encoding="utf-8")
        )
        map_entry = next(item for item in manifest["maps"] if item["id"] == lock["map_id"])
        patrol = next(item for item in map_entry["patrols"] if item["id"] == lock["patrol_id"])
        self.assertEqual(
            [point["rxyz"] for point in patrol["points"]],
            lock["points"],
        )
        self.assertIn("load_taught_patrol_geometry_lock(entry, patrol_id)", source)
        self.assertIn("not patrol_geometry_matches(patrol, geometry_lock)", source)
        self.assertIn("Patrol geometry is locked to its taught recovery data.", source)
        self.assertIn('updatePatrolStatus(`Patrol was not saved. ${reason}`, "error")', viewer)
        self.assertIn("Patrol save failed: ${reason}", viewer)

    def test_stop_status_write_is_atomic_and_recovers_trailing_json_bytes(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        block = source[
            source.index("def mark_live_dji_status_stopped"):
            source.index("def load_config")
        ]
        self.assertIn("json.JSONDecoder().raw_decode", block)
        self.assertIn("atomic_write_json(path, payload)", block)
        self.assertIn('"status": "cancelled"', block)
        self.assertIn("cancel_running_control(payload[\"last_control\"])", block)
        self.assertIn("atomic_write_json(control_path, cancel_running_control(control))", block)
        self.assertNotIn("path.write_text", block)

    def test_global_recovery_database_is_removed_after_each_attempt(self):
        stream_source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("db.unlink(missing_ok=True)", stream_source)
        self.assertIn("shutil.rmtree(localized, ignore_errors=True)", stream_source)

    def test_global_recovery_rejects_a_map_alias_before_tsolve(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("def global_recovery_continuity_rejection(", source)
        self.assertIn(
            'return f"global_recovery_alias_{step:.3f}m_gt_{limit:.3f}m"',
            source,
        )
        registration = source[
            source.index("pool = registered_correspondence_pool("):
            source.index('pool["localization_method"] = attempt_name')
        ]
        self.assertIn(
            "continuity_reason = global_recovery_continuity_rejection(",
            registration,
        )
        self.assertIn("if continuity_reason is not None:", registration)
        self.assertIn("continue", registration)

    def test_sparse_recovery_accepts_guarded_fifteen_point_candidates(self):
        config_path = LOCALIZER_PATH.parents[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        reference_source = (
            LOCALIZER_PATH.parents[0] / "run_live_tsolve_existing_map_stream.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(config["min_query_correspondences"], 15)
        self.assertGreaterEqual(config["live_tracking_reference_image_cap"], 18)
        self.assertGreaterEqual(config["live_reference_image_cap"], 120)
        self.assertIn("def correspondence_spread_metrics(", source)
        self.assertIn("low_correspondence_spatial_concentration", source)
        self.assertIn("occupied >= 4", source)
        self.assertIn("span_x >= 0.20", source)
        self.assertIn("span_y >= 0.15", source)
        self.assertIn("self.heading_bins", reference_source)
        self.assertIn("used_heading_bins", reference_source)
        self.assertIn("respecting the configured total reference-image cap", reference_source)

    def test_soft_patrol_arrival_radius_is_point_three_eight(self):
        app_source = APP_PATH.read_text(encoding="utf-8")
        bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertGreaterEqual(
            app_source.count("arrival_deadband_map_units: 0.14"),
            2,
        )
        self.assertIn(
            'mission.get("arrival_deadband_map_units"),\n        0.14,',
            bridge_source,
        )

    def test_yaw_only_pulses_hold_navigation_position(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("yaw_position_anchor = None", source)
        self.assertIn("if yaw_position_anchor is not None", source)
        self.assertIn("after_position = yaw_position_anchor", source)
        self.assertIn("raw_after_position = pose_gate_position(after_gate)", source)

    def test_patrol_scan_yaw_is_disabled_until_localization_is_stable(self):
        app_source = APP_PATH.read_text(encoding="utf-8")
        bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("allow_patrol_scan_yaw: false", app_source)
        self.assertIn("patrol scan yaw disabled to preserve TSolve continuity", bridge_source)

    def test_proactive_pool_refresh_keeps_live_optical_flow_non_blocking(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("optical_flow_proactive_refresh_background", source)
        self.assertIn("proactive_map_refresh_background", source)
        self.assertIn("proactive_map_refresh_pending_keep_valid_optical_flow", source)
        self.assertIn("a proactive refresh must never pause", source)
        self.assertIn("proactive_fallback_pool = tracked", source)
        self.assertIn("pool = proactive_fallback_pool", source)
        self.assertIn("optical_flow_proactive_refresh_fallback", source)
        self.assertIn("proactive_rematch_failed_using_valid_optical_flow_pool", source)

    def test_proactive_pool_refresh_is_above_hard_tracking_minimum(self):
        config_path = LOCALIZER_PATH.parents[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn('"--proactive-relocalize-points", type=int, default=500', source)
        self.assertGreater(
            config["live_proactive_relocalize_points"],
            config["live_min_track_points"],
        )
        self.assertEqual(config["live_min_track_points"], 15)
        self.assertGreaterEqual(
            config["live_proactive_relocalize_points"],
            config["live_min_track_points"] + 10,
        )
        self.assertGreaterEqual(
            config["live_proactive_relocalize_points"],
            config["max_query_correspondences"] * 3,
        )
        self.assertLessEqual(
            config["live_proactive_relocalize_points"],
            config["live_tracking_pool_size"],
        )
        self.assertEqual(config["live_proactive_relocalize_cooldown_frames"], 15)

    def test_live_recovery_uses_bounded_frequent_sift(self):
        config_path = LOCALIZER_PATH.parents[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        server = SERVER_PATH.read_text(encoding="utf-8")
        self.assertEqual(config["live_sift_max_num_features"], 1024)
        self.assertEqual(config["live_pose_recovery_global_cooldown_frames"], 15)
        self.assertIn('"--SiftExtraction.max_num_features"', source)
        self.assertIn("self.sift_max_num_features", source)
        self.assertIn('"--sift-max-num-features"', server)
        self.assertIn('"--pose-recovery-global-cooldown-frames"', server)

    def test_pose_stasis_and_endpoint_hover_force_newest_frame_global_recovery(self):
        localizer = LOCALIZER_PATH.read_text(encoding="utf-8")
        bridge = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("post_translation_pose_stasis", localizer)
        self.assertIn("endpoint_pose_recovery", localizer)
        self.assertIn("POSE-RECOVERY NEWEST-FRAME GLOBAL SCHEDULED", localizer)
        self.assertIn('"endpoint_position_recovery_allowed"', bridge)
        self.assertIn("endpoint_metric_correction_hits >= 3", bridge)
        self.assertIn("bounded_endpoint_reverse_correction", bridge)
        self.assertIn("bounded_endpoint_forward_correction", bridge)

    def test_live_patrol_backward_tolerance_is_eight_centimeters(self):
        config_path = LOCALIZER_PATH.parents[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        localizer = LOCALIZER_PATH.read_text(encoding="utf-8")
        bridge = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertEqual(config["live_patrol_route_backward_tolerance"], 0.08)
        self.assertIn(
            'ap.add_argument("--patrol-route-backward-tolerance", type=float, default=0.08)',
            localizer,
        )
        self.assertIn("route_gate_backward_tolerance = 0.08", bridge)

    def test_live_patrol_uses_non_blocking_global_recovery(self):
        config_path = LOCALIZER_PATH.parents[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertFalse(config["live_blocking_global_recovery"])
        self.assertIn(
            "if args.blocking_global_recovery:\n            return False",
            source,
        )
        self.assertIn("def catch_up_background_recovery_pool(", source)
        self.assertIn("BACKGROUND RECOVERY CAUGHT UP:", source)
        self.assertIn("background_recovery_catchup_failed_", source)
        self.assertIn("Publishing the worker's old pose is never allowed", source)
        self.assertIn(
            "current_frame_idx=frame_idx,\n            current_gray=curr_gray,",
            source,
        )

    def test_live_patrol_enables_fast_direct_pnp_recovery(self):
        config_path = LOCALIZER_PATH.parents[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        server_source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertTrue(config["live_direct_pnp_recovery"])
        self.assertGreaterEqual(
            server_source.count('append("--direct-pnp-recovery")'),
            2,
        )

    def test_demo_live_output_uses_the_proven_thirty_centimeter_hard_cap(self):
        config_path = LOCALIZER_PATH.parents[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        server_source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertEqual(config["live_output_max_step"], 0.30)
        profile = server_source[
            server_source.index('if live_patrol_lock is not None and live_patrol_lock["reference_profile"]'):
            server_source.index("    return safe", server_source.index('if live_patrol_lock is not None and live_patrol_lock["reference_profile"]'))
        ]
        self.assertIn('"max_pose_step_hard_map_units": 0.30', profile)

    def test_drone_path_093832_marginal_objective_preserves_tracking_chain(self):
        config_path = LOCALIZER_PATH.parents[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        localizer_source = LOCALIZER_PATH.read_text(encoding="utf-8")
        server_source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertEqual(config["live_output_objective_threshold"], 30.0)
        self.assertEqual(config["live_global_recovery_after_failures"], 3)
        self.assertEqual(config["live_background_recovery_timeout_seconds"], 20.0)
        self.assertIn(
            '"faiss_ivf_current_frame_live_recovery"',
            localizer_source,
        )
        self.assertIn("31.127 at a 30.0 gate", localizer_source)
        self.assertIn("def output_rejection_requires_tracking_reset(", localizer_source)
        self.assertGreaterEqual(
            localizer_source.count("output_rejection_requires_tracking_reset("),
            3,
        )
        self.assertGreaterEqual(
            server_source.count('cfg.get("live_output_objective_threshold", 30.0)'),
            2,
        )
        self.assertGreaterEqual(
            server_source.count('cfg.get("live_background_recovery_timeout_seconds", 20.0)'),
            2,
        )
        self.assertIn("hard_cap: float = 0.55", localizer_source)
        self.assertIn('max_step: float = 0.85', localizer_source)

    def test_missing_tracking_anchor_does_not_block_live_rotation_heading(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        nonblocking_gate = source.index(
            "must_global\n"
            "            and interactive_recovery\n"
            "            and last_output_pose is not None"
        )
        synchronous_global = source.index("if must_global:\n", nonblocking_gate)
        synchronous_method = source.index(
            'method = f"global_colmap_',
            synchronous_global,
        )
        self.assertLess(nonblocking_gate, synchronous_global)
        self.assertLess(synchronous_global, synchronous_method)
        self.assertIn("must_global = False", source[nonblocking_gate:synchronous_global])
        self.assertIn(
            "holding_trusted_position_with_fresh_rotation_heading",
            source[nonblocking_gate:synchronous_global],
        )
        self.assertIn(
            "Forward/lateral movement remains\n"
            "            # locked until a new position passes",
            source[nonblocking_gate:synchronous_global],
        )

    def test_background_recovery_catchup_is_bounded_across_live_turn(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("def room_heading_separation_degrees(", source)
        self.assertIn('"rotation_heading": normalize_room_heading(rotation_heading)', source)
        self.assertIn("recovery_heading_shift > 15.0", source)
        self.assertIn("BACKGROUND RECOVERY DISCARDED AFTER TURN:", source)
        self.assertIn("max_sequential_catchup_frames = 12 if interactive_recovery", source)
        self.assertIn('"direct_catchup": True', source)

    def test_point_four_route_bank_gets_bounded_priority_before_full_map_recovery(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("route_visual_recovery_grace_frames = 30", source)
        self.assertIn("retained_route_hits > 0", source)
        recovery_due = source.index("        recovery_global_due = bool(")
        recovery_schedule = source.index(
            "        if recovery_global_due and pending_global is None:",
            recovery_due,
        )
        recovery_gate = source[recovery_due:recovery_schedule]
        self.assertIn(
            "and not route_visual_recovery_window_active",
            recovery_gate,
        )

    def test_newest_frame_faiss_recovery_has_one_owner_and_private_sqlite(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("self.faiss_current_frame_lock = threading.Lock()", source)
        self.assertIn("with self.faiss_current_frame_lock:", source)
        self.assertIn('call_id = f"{frame_idx:06d}_{uuid.uuid4().hex}"', source)
        self.assertIn('self.work_dir / "faiss_recovery_calls" / call_id', source)
        self.assertIn('database = call_dir / "query.db"', source)
        self.assertIn("shutil.rmtree(call_dir, ignore_errors=True)", source)
        self.assertNotIn(
            'database = self.work_dir / f"checkpoint_faiss_{frame_idx:06d}.db"',
            source,
        )

        scheduler_start = source.index("    def schedule_background_global_recovery(")
        scheduler_end = source.index("    def append_global_recovery_pose(", scheduler_start)
        scheduler = source[scheduler_start:scheduler_end]
        deadline_start = scheduler.index("elif pending_age > pending_limit:")
        deadline_end = scheduler.index("        center =", deadline_start)
        deadline = scheduler[deadline_start:deadline_end]
        self.assertIn('pending_global["deadline_exceeded"] = True', deadline)
        self.assertIn("return False", deadline)
        self.assertNotIn("pending_global = None", deadline)

        checkpoint_start = source.index(
            "        if (\n"
            "            absolute_metric_rebootstrap\n"
            "            and must_global"
        )
        checkpoint_end = source.index(
            "            checkpoint_stage:", checkpoint_start
        )
        self.assertIn(
            "and pending_global is None",
            source[checkpoint_start:checkpoint_end],
        )

    def test_yaw_recovery_cannot_overwrite_next_translation_phase(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        scheduler_start = source.index("    def schedule_background_global_recovery(")
        scheduler_end = source.index("    def append_global_recovery_pose(", scheduler_start)
        scheduler = source[scheduler_start:scheduler_end]
        self.assertIn('"route_key": recovery_route_key', scheduler)
        self.assertIn(
            '"controller_translation_locked": controller_translation_locked',
            scheduler,
        )
        self.assertIn(
            '"position_locked_recovery": position_locked_recovery',
            scheduler,
        )
        apply_start = source.index("    def apply_background_global_recovery(")
        apply_end = source.index("    write_partial_pose_stream(", apply_start)
        apply = source[apply_start:apply_end]
        self.assertIn("current_route_key != source_route_key", apply)
        self.assertIn(
            "current_translation_locked != source_translation_locked", apply
        )
        self.assertIn(
            "BACKGROUND RECOVERY DISCARDED AFTER CONTROLLER PHASE CHANGE:", apply
        )
        self.assertIn(
            '"global_colmap_background_recovery_phase_discard"', apply
        )

    def test_rejected_pose_cannot_advance_room_reference(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        accepted_gate = (
            'output_accepted = bool(result.get("success")) '
            "and output_rejection_reason is None"
        )
        self.assertGreaterEqual(source.count(accepted_gate), 2)
        self.assertGreaterEqual(source.count("if output_accepted:"), 2)
        self.assertGreaterEqual(
            source.count('reference_update_reason = "held_rejected_pose_not_trusted"'),
            2,
        )

    def test_translation_locked_post_yaw_candidate_reaches_stabilizer_without_moving_pose(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "tracking_reset_hard_motion_cap(max_step)\n"
            "            if controller_position_locked",
            source,
        )
        self.assertIn("and along_step <= route_motion_cap", source)
        self.assertIn(
            'observation["route_continuity_preserved_tracking_center"] = True',
            source,
        )
        self.assertIn("candidate_center = previous_center", source)

    def test_route_only_rejection_preserves_flow_chain_and_refreshes_map(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("def route_rejection_can_advance_flow_anchor(", source)
        self.assertIn('"route_backward_",', source)
        self.assertGreaterEqual(
            source.count("if route_rejection_can_advance_flow_anchor(output_rejection_reason):"),
            2,
        )
        self.assertGreaterEqual(
            source.count('reference_update_reason = "route_rejected_pose_flow_anchor_only"'),
            2,
        )
        self.assertGreaterEqual(
            source.count('reason=f"route_guard_rejection_{output_rejection_reason}"'),
            2,
        )
        self.assertGreaterEqual(
            source.count('"ROUTE REJECTION BACKGROUND REFRESH:"'),
            2,
        )

    def test_repeated_output_rejection_forces_clean_map_rematch(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("OUTPUT REJECTION RECOVERY ARMED:"), 2)
        self.assertGreaterEqual(source.count("current_pool = None"), 2)
        self.assertGreaterEqual(source.count("prev_gray = None"), 2)
        self.assertIn("must_global = current_pool is None or prev_gray is None", source)
        self.assertIn("def next_output_tracking_reset_streak(", source)
        self.assertGreaterEqual(
            source.count("consecutive_tracking_reset_rejections"),
            8,
        )

    def test_background_recovery_can_learn_anchor_after_accepting_pose(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        start = source.index("    def append_global_recovery_pose(")
        end = source.index("    def apply_background_global_recovery(", start)
        recovery = source[start:end]
        self.assertIn(
            "nonlocal last_online_recovery_anchor_frame, online_recovery_anchor_count",
            recovery,
        )
        self.assertIn("last_online_recovery_anchor_frame = frame_idx", recovery)
        self.assertIn("online_recovery_anchor_count += 1", recovery)

    def test_fixed_center_recovery_is_only_enabled_while_controller_locks_translation(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        start = source.index("    def schedule_background_global_recovery(")
        end = source.index("    def append_global_recovery_pose(", start)
        scheduler = source[start:end]
        self.assertIn(
            'recovery_route_context.get("controller_translation_locked") is True',
            scheduler,
        )
        self.assertIn('"position_locked_recovery": position_locked_recovery', scheduler)
        self.assertIn("position_locked=position_locked_recovery", scheduler)

    def test_bridge_shutdown_cancels_mission_before_socket_close(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        cancel_at = source.index("mission_cancel.set()", source.index("if args.max_frames > 0"))
        join_at = source.index("control_thread.join(timeout=2.5)", cancel_at)
        context_end = source.index("    except Exception as exc:", join_at)
        self.assertLess(cancel_at, join_at)
        self.assertLess(join_at, context_end)

    def test_open_dji_channel_order_is_yaw_vertical_lateral_forward(self):
        class FakeDrone:
            def __init__(self):
                self.calls = []

            def move(self, *args):
                self.calls.append(args)

        drone = FakeDrone()
        original_sleep = bridge.time.sleep
        bridge.time.sleep = lambda _seconds: None
        try:
            bridge.execute_rc_pulse(drone, yaw=0.01, du=0.02, lr=0.03, bf=0.04, seconds=0.05)
        finally:
            bridge.time.sleep = original_sleep
        self.assertEqual(drone.calls[0], (0.01, 0.02, 0.03, 0.04, True))
        self.assertEqual(drone.calls[-1], (0.0, 0.0, 0.0, 0.0, True))

    def test_broken_neutral_reconnects_but_never_retries_uncertain_motion(self):
        class FakeDrone:
            def __init__(self):
                self.calls = []

            def move(self, *args):
                self.calls.append(args)
                if args[:4] == (0.0, 0.0, 0.0, 0.0):
                    raise BrokenPipeError(32, "Broken pipe")
                return "success"

        drone = FakeDrone()
        reconnect_calls = []
        original_sleep = bridge.time.sleep
        original_reconnect = bridge.reconnect_control_and_confirm_neutral
        bridge.time.sleep = lambda _seconds: None
        bridge.reconnect_control_and_confirm_neutral = lambda _drone: (
            reconnect_calls.append(_drone)
            or {
                "neutral_confirmed": True,
                "control_link_recovered": True,
                "acknowledgement": "success",
            }
        )
        try:
            result = bridge.execute_rc_pulse(drone, bf=0.04, seconds=0.30)
        finally:
            bridge.time.sleep = original_sleep
            bridge.reconnect_control_and_confirm_neutral = original_reconnect

        self.assertTrue(result["neutral_confirmed"])
        self.assertTrue(result["control_link_recovered"])
        self.assertTrue(result["motion_outcome_uncertain"])
        self.assertTrue(result["requires_pose_recovery"])
        self.assertEqual(len(reconnect_calls), 1)
        self.assertEqual(drone.calls.count((0.0, 0.0, 0.0, 0.04, True)), 1)
        self.assertEqual(drone.calls[-1], (0.0, 0.0, 0.0, 0.0, True))

    def test_real_queue_path_never_retries_forward_after_neutral_broken_pipe(self):
        class FakeControlSocket:
            def __init__(self):
                self.payloads = []

            def sendall(self, payload):
                self.payloads.append(payload)
                if len(self.payloads) == 2:
                    raise BrokenPipeError(32, "Broken pipe")

        control_socket = FakeControlSocket()
        response_queue = bridge.queue.Queue()
        response_queue.put("success")
        drone = types.SimpleNamespace(
            _socket_control=control_socket,
            _background_control_messages=types.SimpleNamespace(_queue=response_queue),
        )
        original_sleep = bridge.time.sleep
        original_reconnect = bridge.reconnect_control_and_confirm_neutral
        bridge.time.sleep = lambda _seconds: None
        bridge.reconnect_control_and_confirm_neutral = lambda _drone: {
            "neutral_confirmed": True,
            "control_link_recovered": True,
            "acknowledgement": "success",
        }
        try:
            result = bridge.execute_rc_pulse(drone, bf=0.04, seconds=0.30)
        finally:
            bridge.time.sleep = original_sleep
            bridge.reconnect_control_and_confirm_neutral = original_reconnect

        self.assertTrue(result["neutral_confirmed"])
        self.assertTrue(result["control_link_recovered"])
        self.assertTrue(result["requires_pose_recovery"])
        self.assertEqual(
            control_socket.payloads.count(b"rc 0.0000 0.00 0.00 0.04\r\n"),
            1,
        )
        self.assertEqual(
            control_socket.payloads[-1],
            b"rc 0.0000 0.00 0.00 0.00\r\n",
        )

    def test_control_reconnect_confirms_neutral_before_installing_new_reader(self):
        events = []

        class FakeSocket:
            def settimeout(self, value):
                events.append(("timeout", value))

            def connect(self, address):
                events.append(("connect", address))

            def sendall(self, payload):
                events.append(("send", payload))

            def recv(self, _size):
                events.append(("recv", None))
                return b"success\r\n"

            def close(self):
                events.append(("close", None))

        class FakeBackground:
            def __init__(self, sock):
                self.sock = sock
                self.stopped = False
                events.append(("reader", sock))

            def stop(self, timeout=None):
                self.stopped = True
                events.append(("stop", timeout))

        old_socket = FakeSocket()
        old_background = FakeBackground(old_socket)
        drone = types.SimpleNamespace(
            host_address="192.0.2.8",
            PORT_CONTROL=9998,
            _socket_control=old_socket,
            _background_control_messages=old_background,
        )
        replacement = FakeSocket()
        original_socket_factory = bridge.socket.socket
        bridge.socket.socket = lambda *_args: replacement
        try:
            result = bridge.reconnect_control_and_confirm_neutral(drone, attempts=1)
        finally:
            bridge.socket.socket = original_socket_factory

        self.assertTrue(result["neutral_confirmed"])
        self.assertTrue(result["control_link_recovered"])
        self.assertTrue(old_background.stopped)
        self.assertIs(drone._socket_control, replacement)
        self.assertIs(drone._background_control_messages.sock, replacement)
        send_index = events.index(("send", b"rc 0.0000 0.00 0.00 0.00\r\n"))
        reader_index = events.index(("reader", replacement))
        self.assertLess(send_index, reader_index)

    def test_fresh_file_cannot_hide_stale_camera_observation(self):
        payload = {
            "updated_at": time.time(),
            "processed_count": 1,
            "poses": [{
                "instance_id": "instance_000001",
                "success": True,
                "held_pose": False,
                "rcenter": [0, 0, 0],
                "rheading": [1, 0, 0],
                "received_unix": time.time() - 5.0,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses_partial.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            gate = bridge.latest_tsolve_pose_gate(path, max_age_seconds=1.0)
        self.assertFalse(gate["ok"])
        self.assertIn("observation is stale", gate["reason"])

    def test_atomic_status_writes_do_not_share_temp_file(self):
        errors = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"

            def writer(index):
                try:
                    for sequence in range(30):
                        bridge.atomic_write_json(path, {"writer": index, "sequence": sequence})
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=writer, args=(index,)) for index in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(errors)
        self.assertIn("writer", payload)

    def test_live_session_records_replayable_control_trace_and_runtime_hashes(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn('session_control_trace_path = session_root / "control_trace.jsonl"', source)
        self.assertIn('"command_payload": command_payload', source)
        self.assertIn('{**control_progress, "event": "progress"}', source)
        self.assertIn('"runtime_fingerprints": current_runtime_fingerprints()', source)
        self.assertIn("startup_runtime_fingerprints", source)
        self.assertIn("runtime_fingerprint_changes(", source)
        self.assertIn("restart the live bridge before flight", source)
        self.assertIn('"pulse_trace": []', source)
        self.assertIn('"recorded_unix": time.time()', source)

    def test_runtime_fingerprint_is_content_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.txt"
            path.write_bytes(b"atlas-release")
            first = bridge.file_sha256(path)
            second = bridge.file_sha256(path)
            path.write_bytes(b"atlas-release-updated")
            changed = bridge.file_sha256(path)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_runtime_fingerprint_change_detection_is_fail_closed(self):
        expected = {"bridge": "aaa", "config": "bbb"}
        self.assertEqual(
            bridge.runtime_fingerprint_changes(expected, dict(expected)),
            [],
        )
        self.assertEqual(
            bridge.runtime_fingerprint_changes(
                expected,
                {"bridge": "aaa", "config": "changed"},
            ),
            ["config"],
        )

    def test_drone_stop_signal_remains_set_until_worker_cleanup(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        stop_handler = source[
            source.index('if url.path == "/api/drone/stop"'):
            source.index('if url.path == "/api/drone/flight-command"')
        ]
        self.assertIn("DRONE_STOP_EVENT.set()", stop_handler)
        self.assertNotIn("DRONE_STOP_EVENT.clear()", stop_handler)
        self.assertIn("DRONE_JOB_ACTIVE.is_set()", stop_handler)
        self.assertIn("release_drone_job()", source)
        self.assertIn("finally:\n        release_drone_job()", source)

    def test_cancelled_live_path_is_removed_from_pending_ui(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn(
            'const activeJobStates = new Set(["queued", "running", "stopping"]);',
            source,
        )
        self.assertIn(
            'const droneCancelledNow = ["cancelled", "failed"].includes(state.drone?.status)',
            source,
        )
        self.assertIn("if (droneCancelledNow)", source)

    def test_map_manifest_is_never_exposed_half_written(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        save_library = source[
            source.index("def save_library("):
            source.index("def default_enemy_library(")
        ]
        self.assertIn("atomic_write_json(MAP_MANIFEST, lib)", save_library)
        self.assertNotIn("MAP_MANIFEST.write_text", save_library)

    def test_pose_holds_use_the_latest_guarded_publication(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        safe_hold = "last_pose=latest_published_pose(partial_poses, last_output_pose)"
        # A route/rotation guard may publish a safer position without replacing
        # the solver's tracking anchor, and a late background worker can append
        # an older frame after the newest pose. Every fallback hold must copy
        # the freshest guarded publication so a later miss cannot jump back.
        self.assertEqual(4, source.count(safe_hold))
        self.assertNotIn("partial_poses[-1]", source)


if __name__ == "__main__":
    unittest.main()

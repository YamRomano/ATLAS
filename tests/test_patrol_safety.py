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


class PatrolSafetyTests(unittest.TestCase):
    def test_loop_keeps_exact_command_order(self):
        commands = [{"type": "gate"}, {"type": "cruise", "to": [1, 0, 0]}, {"type": "hover"}]
        observed = list(itertools.islice(bridge.mission_step_sequence(commands, True), 6))
        self.assertEqual([index for index, _step in observed], [0, 1, 2, 0, 1, 2])

    def test_non_loop_stops_after_one_pass(self):
        commands = [{"type": "gate"}, {"type": "cruise"}]
        self.assertEqual(list(bridge.mission_step_sequence(commands, False)), list(enumerate(commands)))

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
        self.assertIn("min(0.55, 0.18 + 0.85 * dt)", source)

    def test_observed_skipped_frame_motion_is_allowed(self):
        limit = bridge.bounded_pose_step_limit(100.0, 100.416)
        self.assertGreater(limit, 0.318)

    def test_large_jump_remains_rejected_after_skipped_frames(self):
        limit = bridge.bounded_pose_step_limit(100.0, 100.416)
        self.assertLess(limit, 0.932)
        self.assertLessEqual(bridge.bounded_pose_step_limit(100.0, 110.0), 0.55)

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

    def test_visual_heading_never_bypasses_physical_forward_verification(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("calibrated_heading_offset_rad: float | None = None", source)
        self.assertIn("operator_heading_seed_pending_physical_verification", source)

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
        self.assertIn("required_wrong_yaw_pulses = 3 if yaw_flip_count == 0 else 5", source)

    def test_navigation_yaw_is_faster_than_scan_yaw(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("max_yaw_rc: 0.050", source)
        self.assertIn("max_scan_yaw_rc: 0.025", source)
        bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("alignment_grace_seconds + max(4.0, planned_duration * 2.2)", bridge_source)

    def test_travel_budget_advances_only_with_commanded_motion(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("alignment_deadline = time.time()", source)
        self.assertIn("segment_safety_deadline = time.time() + max_cruise_seconds", source)
        self.assertIn("absolute_segment_deadline = time.time() + min(240.0, max_cruise_seconds * 2.0)", source)
        self.assertIn("alignment_progress_deadline = alignment_deadline", source)
        self.assertIn("now + 20.0", source)
        self.assertIn("forward_command_seconds = 0.0", source)
        self.assertIn("travel_yaw_command_seconds = 0.0", source)
        self.assertIn("forward_command_seconds += pulse_seconds", source)
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

    def test_adjusted_patrol_points_still_match_taught_leg_by_command_order(self):
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
        self.assertEqual(
            bridge.taught_leg_for_step(reference, adjusted_step),
            reference["legs"][2],
        )

    def test_point_three_to_four_uses_operator_requested_right_turn(self):
        leg = {"from_point": 3, "to_point": 4}
        self.assertTrue(bridge.is_guarded_point_three_to_four_turn(leg))
        self.assertFalse(
            bridge.is_guarded_point_three_to_four_turn(
                {"from_point": 2, "to_point": 3}
            )
        )
        self.assertEqual(bridge.taught_turn_direction_override(leg), "right")
        self.assertEqual(bridge.yaw_direction_for_angle(math.radians(102.0), "right"), 1.0)
        self.assertEqual(bridge.yaw_direction_for_angle(math.radians(102.0), None), 1.0)
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn(
            '45.0 if turn_direction_override == "left" else 24.0',
            source,
        )
        self.assertIn('if turn_direction_override == "left":', source)
        self.assertIn("wrong_yaw_pulses = 0", source)

    def test_taught_rotation_recovery_locks_translation_until_fresh_position(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn('"phase": "taught_rotation_only_recovery"', source)
        self.assertIn('"phase": "taught_turn_wait_fresh_position"', source)
        self.assertIn("else 24.0", source)
        self.assertIn('or bool(gate_attempt.get("recent_hold_fallback"))', source)
        self.assertIn("required_rotation_heading_stable_frames = 3", source)
        self.assertIn("rotation_heading_stable_frames", source)
        self.assertIn(">= required_rotation_heading_stable_frames", source)
        self.assertIn(
            "guarded_taught_rotation = is_guarded_point_three_to_four_turn(taught_leg)",
            source,
        )
        self.assertIn("yaw_position_anchor = start_position", source)
        self.assertIn(
            "(guarded_taught_rotation or rotation_position_untrusted)",
            source,
        )
        self.assertIn("confirmed_rotation_angle = visual_angle", source)
        self.assertIn("confirmed_rotation_gate = gate_attempt", source)
        self.assertIn("current_gate = confirmed_rotation_gate", source)
        self.assertIn('"phase": "taught_turn_post_pulse_optical_return"', source)
        self.assertIn(
            '"phase": "taught_turn_forward_reacquisition"',
            source,
        )
        self.assertIn("max_rotation_reacquisition_pulses = 3", source)
        reacquisition_block = source[
            source.index('"phase": "taught_turn_forward_reacquisition"'):
            source.index('"phase": "taught_rotation_only_recovery"')
        ]
        self.assertIn("yaw=0.0", reacquisition_block)
        self.assertIn("lr=0.0", reacquisition_block)
        taught_block = source[source.index('"phase": "taught_rotation_only_recovery"'):source.index('"phase": "bounded_blind_yaw_recovery"')]
        self.assertIn("lr=0.0", taught_block)
        self.assertIn("bf=0.0", taught_block)
        self.assertIn("current_gate.get(\"recent_hold_fallback\")", source)

    def test_localizer_keeps_independent_optical_heading_when_map_pose_is_questionable(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("def optical_flow_yaw_delta(", source)
        self.assertIn('"rotation_heading_source": "optical_flow_yaw"', source)
        self.assertIn("def update_rotation_reference_from_accepted_pose", source)
        self.assertIn("if pose is None:", source)
        self.assertIn("max_step = max(0.22, min(0.35, 0.18 + 0.17 * dt))", source)

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
        self.assertIn('"--proactive-relocalize-points", type=int, default=28', source)
        self.assertGreater(
            config["live_proactive_relocalize_points"],
            config["live_min_track_points"],
        )
        self.assertEqual(config["live_min_track_points"], 15)
        self.assertGreaterEqual(config["live_proactive_relocalize_cooldown_frames"], 30)

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

    def test_missing_tracking_anchor_does_not_block_live_rotation_heading(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        nonblocking_gate = source.index(
            "must_global\n"
            "            and args.follow_dir\n"
            "            and last_output_pose is not None"
        )
        synchronous_global = source.index(
            "if must_global:\n"
            '            method = f"global_colmap_',
            nonblocking_gate,
        )
        self.assertLess(nonblocking_gate, synchronous_global)
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
        self.assertIn("max_sequential_catchup_frames = 12 if args.follow_dir", source)
        self.assertIn('"direct_catchup": True', source)

    def test_rejected_pose_cannot_advance_tracking_reference(self):
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

    def test_repeated_output_rejection_forces_clean_map_rematch(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("OUTPUT REJECTION RECOVERY ARMED:"), 2)
        self.assertGreaterEqual(source.count("current_pool = None"), 2)
        self.assertGreaterEqual(source.count("prev_gray = None"), 2)
        self.assertIn("must_global = current_pool is None or prev_gray is None", source)

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
        self.assertEqual(drone.calls[0], (0.01, 0.02, 0.03, 0.04, False))
        self.assertEqual(drone.calls[-1], (0.0, 0.0, 0.0, 0.0, True))

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


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

LOCALIZER_PATH = SCRIPTS / "run_bounded_tsolve_video_stream.py"
SPEC = importlib.util.spec_from_file_location("run_bounded_tsolve_video_stream_alignment_test", LOCALIZER_PATH)
assert SPEC and SPEC.loader
localizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(localizer)


class LiveRoomAlignmentTests(unittest.TestCase):
    def test_replay_pacer_rebases_after_slow_bootstrap_instead_of_racing_backlog(self):
        pacer = localizer.ReplayPacer(enabled=True, scale=1.0, max_lag_seconds=0.25)
        with mock.patch.object(localizer.time, "perf_counter", side_effect=[0.0, 2.0, 2.0]), mock.patch.object(localizer.time, "sleep") as sleep:
            self.assertEqual(pacer.wait_until(0.0), 0.0)
            self.assertEqual(pacer.wait_until(1.0), 0.0)
            self.assertAlmostEqual(pacer.wall_start, 1.0)
            self.assertAlmostEqual(pacer.wait_until(1.1), 100.0)
            sleep.assert_called_once_with(mock.ANY)

    def test_tsolve_remains_primary_through_weak_leg_translation_and_rotation(self):
        last_pose = {"rcenter": [0.0, 0.0, 0.0]}
        verified = {"verified": True}

        self.assertIsNone(
            localizer.weak_patrol_leg_visual_primary_mode(
                route_context={"leg_index": 2},
                observation=verified,
                last_pose=last_pose,
            )
        )
        self.assertIsNone(
            localizer.weak_patrol_leg_visual_primary_mode(
                route_context={
                    "leg_index": 3,
                    "controller_translation_locked": False,
                    "translation_locked": False,
                },
                observation=verified,
                last_pose=last_pose,
            ),
        )
        self.assertIsNone(
            localizer.weak_patrol_leg_visual_primary_mode(
                route_context={"leg_index": 4, "controller_translation_locked": True},
                observation=None,
                last_pose=last_pose,
            )
        )
        self.assertIsNone(
            localizer.weak_patrol_leg_visual_primary_mode(
                route_context={
                    "leg_index": 3,
                    "controller_translation_locked": False,
                    "translation_locked": False,
                },
                observation={"verified": False},
                last_pose=last_pose,
            )
        )
        self.assertIsNone(
            localizer.weak_patrol_leg_visual_primary_mode(
                route_context={
                    "leg_index": 3,
                    "controller_translation_locked": False,
                    "translation_locked": True,
                },
                observation=verified,
                last_pose=last_pose,
            ),
        )

    def test_verified_departure_image_repairs_only_a_small_new_leg_floor(self):
        context = {
            "start": [-0.49, 0.0, 0.96],
            "end": [-3.07, 0.0, 1.11],
            "lap": 1,
            "leg_index": 3,
            "controller_translation_locked": False,
        }
        observation = {
            "verified": True,
            "translation_safe": True,
            "progress": 0.006,
            "inliers": 380,
            "minimum_inliers": 120,
            "acquisition_hits": 2,
        }
        gate = localizer.LivePatrolRouteGate(None, None)
        gate.last_key = gate._key(context)
        gate.last_progress = 0.061

        self.assertTrue(
            gate.reconcile_verified_departure_floor(observation, context)
        )
        self.assertAlmostEqual(gate.last_progress, 0.006)

        gate.last_progress = 0.03
        self.assertFalse(
            gate.reconcile_verified_departure_floor(
                {**observation, "matched_progress": 0.0}, context
            )
        )
        self.assertAlmostEqual(gate.last_progress, 0.03)

        # The same observation cannot rewind an established cruise track.
        gate.last_progress = 0.25
        self.assertFalse(
            gate.reconcile_verified_departure_floor(observation, context)
        )
        self.assertAlmostEqual(gate.last_progress, 0.25)

    def test_leg_four_second_acquisition_hit_can_release_departure_lock(self):
        context = {
            "start": [-3.07, 0.0, 1.11],
            "end": [-3.23, 0.0, -0.33],
            "lap": 1,
            "leg_index": 4,
            "controller_translation_locked": False,
        }
        observation = {
            "verified": True,
            "translation_safe": True,
            "progress": 0.052,
            "matched_progress": 0.052,
            "inliers": 1200,
            "minimum_inliers": 120,
            "acquisition_hits": 2,
        }
        gate = localizer.LivePatrolRouteGate(None, None)
        gate.last_key = gate._key(context)
        gate.last_progress = 0.0

        self.assertTrue(
            gate.reconcile_verified_departure_floor(observation, context)
        )
        self.assertAlmostEqual(gate.last_progress, 0.052)

        # The wider entry window is specific to the audited weak 4->1 leg.
        other_leg = {**context, "leg_index": 3}
        other_gate = localizer.LivePatrolRouteGate(None, None)
        other_gate.last_key = other_gate._key(other_leg)
        other_gate.last_progress = 0.0
        self.assertFalse(
            other_gate.reconcile_verified_departure_floor(observation, other_leg)
        )

    def test_departure_recovery_publishes_raw_match_not_false_metric_floor(self):
        start = [-0.49, 0.0, 0.96]
        end = [-3.07, 0.0, 1.11]
        context = {
            "start": start,
            "end": end,
            "lap": 1,
            "leg_index": 3,
            "controller_translation_locked": False,
        }
        gate = localizer.LivePatrolRouteGate(None, None)
        gate.last_key = gate._key(context)
        gate.last_progress = 0.0394
        gate.last_publish_time = 290.1

        class VisualRecovery:
            def __init__(self):
                self.progress = None

            def commit_published_progress(self, progress):
                self.progress = progress

        recovery = VisualRecovery()
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        pose = localizer.accepted_visual_route_recovery_pose(
            last_pose={"rcenter": [start[0] + 0.0394 * (end[0] - start[0]), 0.0,
                                   start[2] + 0.0394 * (end[2] - start[2])]},
            current_frame={
                "frame_index": 2902,
                "time_sec": 290.2,
                "received_unix": 1000.2,
                "image_name": "query/query_002902.jpg",
            },
            observation={
                "verified": True,
                "translation_safe": True,
                "center": [start[0] + 0.0394 * (end[0] - start[0]), 0.0,
                           start[2] + 0.0394 * (end[2] - start[2])],
                "heading": [-1.0, 0.0, 0.0],
                "progress": 0.0394,
                "matched_progress": 0.0,
                "inliers": 500,
                "ratio_matches": 550,
                "anchor_name": "query_002901.jpg",
                "source_frame": 2901,
                "acquisition_hits": 2,
                "minimum_inliers": 120,
                "map_id": "map",
                "patrol_id": "patrol",
                "baseline_replay_id": "baseline",
            },
            supervision={},
            route_context=context,
            rotation_heading=[-1.0, 0.0, 0.0],
            rotation_heading_tracks=300,
            rotation_position_stabilizer=stabilizer,
            route_gate=gate,
            visual_recovery=recovery,
            metric_route_observation={
                "progress": 0.0394,
                "unbiased_progress": 0.0394,
            },
        )

        self.assertTrue(pose["route_visual_departure_floor_reconciled"])
        self.assertAlmostEqual(pose["route_published_progress"], 0.0)
        self.assertAlmostEqual(gate.last_progress, 0.0)
        self.assertEqual(gate.departure_progress_bias_key, gate._key(context))
        self.assertAlmostEqual(gate.departure_progress_bias, -0.0394)
        self.assertAlmostEqual(recovery.progress, 0.0)

    def test_weak_leg_holds_anchor_until_visual_departure_calibrates_metric(self):
        start = [-0.49, 0.0, 0.96]
        end = [-3.07, 0.0, 1.11]
        context = {
            "start": start,
            "end": end,
            "anchor": start,
            "lap": 1,
            "leg_index": 3,
            "controller_translation_locked": False,
        }
        gate = localizer.LivePatrolRouteGate(None, None)
        key = gate._key(context)
        gate.last_key = key
        gate.last_progress = 0.0
        gate.last_publish_time = 10.0

        held = gate.constrain_published_pose(
            {"rcenter": [start[0] + 0.15 * (end[0] - start[0]), 0.0,
                         start[2] + 0.15 * (end[2] - start[2])], "time_sec": 10.1},
            {"key": key, "progress": 0.15, "cross_track": 0.0, "context": context},
        )
        self.assertTrue(held["route_departure_visual_lock"])
        self.assertAlmostEqual(held["route_published_progress"], 0.0)
        self.assertAlmostEqual(held["route_departure_unverified_metric_progress"], 0.15)

        verified = {
            "verified": True,
            "translation_safe": True,
            "progress": 0.0,
            "matched_progress": 0.0,
            "inliers": 500,
            "minimum_inliers": 120,
            "acquisition_hits": 2,
        }
        self.assertTrue(
            gate.reconcile_verified_departure_floor(
                verified,
                context,
                {"progress": 0.15, "unbiased_progress": 0.15},
            )
        )
        self.assertAlmostEqual(gate.departure_progress_bias, -0.15)

        released = gate.constrain_published_pose(
            {"rcenter": [start[0] + 0.01 * (end[0] - start[0]), 0.0,
                         start[2] + 0.01 * (end[2] - start[2])], "time_sec": 10.2},
            {"key": key, "progress": 0.01, "cross_track": 0.0, "context": context},
        )
        self.assertNotIn("route_departure_visual_lock", released)
        self.assertAlmostEqual(released["route_published_progress"], 0.01)

    def test_verified_anchor_filters_drifted_flow_and_replenishes_map_points(self):
        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
        tracked = {
            "K": K,
            "xy": np.array([[50.0, 40.0], [70.0, 40.0], [400.0, 400.0]]),
            "p3d": np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]]),
            "point3d_ids": np.array([10, 11, 12]),
        }
        recovered = {
            "K": K,
            "pose_prior_R": np.eye(3),
            "colmap_tvec_world_to_camera": [0.0, 0.0, 0.0],
            "xy": np.array([[50.0, 60.0], [30.0, 40.0]]),
            "p3d": np.array([[0.0, 1.0, 5.0], [-1.0, 0.0, 5.0]]),
            "point3d_ids": np.array([12, 13]),
        }

        merged = localizer.merge_verified_tracking_pool(tracked, recovered)

        self.assertEqual(set(merged["point3d_ids"].tolist()), {10, 11, 12, 13})
        self.assertEqual(merged["verified_lk_input_points"], 3)
        self.assertEqual(merged["verified_lk_inlier_points"], 2)
        self.assertTrue(merged["trusted_recovery"])

    def test_degraded_stable_tsolve_set_is_reselected_from_full_pool_consensus(self):
        xy = np.column_stack(
            [
                np.linspace(20.0, 1180.0, 100),
                np.resize(np.array([30.0, 210.0, 430.0, 640.0]), 100),
            ]
        )
        pool = {
            "xy": xy,
            "p3d": np.column_stack(
                [
                    np.linspace(-2.0, 2.0, 100),
                    np.resize(np.array([-1.0, 0.0, 1.0]), 100),
                    np.linspace(3.0, 7.0, 100),
                ]
            ),
            "point3d_ids": np.arange(100),
            "K": np.array([[800.0, 0.0, 600.0], [0.0, 800.0, 337.5], [0.0, 0.0, 1.0]]),
        }
        original_solver = getattr(localizer.cv2, "solvePnPRansac", None)
        original_flag = getattr(localizer.cv2, "SOLVEPNP_EPNP", None)
        original_error = getattr(localizer.cv2, "error", None)
        try:
            localizer.cv2.SOLVEPNP_EPNP = 1
            localizer.cv2.error = RuntimeError
            localizer.cv2.solvePnPRansac = lambda *args, **kwargs: (
                True,
                np.zeros((3, 1)),
                np.zeros((3, 1)),
                np.arange(40, 100, dtype=np.int32).reshape(-1, 1),
            )
            chosen, meta = localizer.stable_case_indices(
                pool,
                max_points=40,
                preferred_point3d_ids=np.arange(40),
            )
        finally:
            if original_solver is None:
                delattr(localizer.cv2, "solvePnPRansac")
            else:
                localizer.cv2.solvePnPRansac = original_solver
            if original_flag is None:
                delattr(localizer.cv2, "SOLVEPNP_EPNP")
            else:
                localizer.cv2.SOLVEPNP_EPNP = original_flag
            if original_error is None:
                delattr(localizer.cv2, "error")
            else:
                localizer.cv2.error = original_error

        self.assertTrue(meta["accepted"])
        self.assertTrue(meta["solve_set_reselected"])
        self.assertLess(meta["preferred_inlier_ratio"], 0.72)
        self.assertEqual(len(chosen), 40)
        self.assertTrue(np.all(np.asarray(chosen) >= 40))

    def test_healthy_stable_tsolve_set_is_retained(self):
        xy = np.column_stack([np.arange(80, dtype=float), np.arange(80, dtype=float)])
        pool = {
            "xy": xy,
            "p3d": np.column_stack([xy, np.ones(80)]),
            "point3d_ids": np.arange(80),
            "K": np.eye(3),
        }
        original_solver = getattr(localizer.cv2, "solvePnPRansac", None)
        original_flag = getattr(localizer.cv2, "SOLVEPNP_EPNP", None)
        original_error = getattr(localizer.cv2, "error", None)
        try:
            localizer.cv2.SOLVEPNP_EPNP = 1
            localizer.cv2.error = RuntimeError
            localizer.cv2.solvePnPRansac = lambda *args, **kwargs: (
                True,
                np.zeros((3, 1)),
                np.zeros((3, 1)),
                np.arange(80, dtype=np.int32).reshape(-1, 1),
            )
            chosen, meta = localizer.stable_case_indices(
                pool,
                max_points=40,
                preferred_point3d_ids=np.arange(40),
            )
        finally:
            if original_solver is None:
                delattr(localizer.cv2, "solvePnPRansac")
            else:
                localizer.cv2.solvePnPRansac = original_solver
            if original_flag is None:
                delattr(localizer.cv2, "SOLVEPNP_EPNP")
            else:
                localizer.cv2.SOLVEPNP_EPNP = original_flag
            if original_error is None:
                delattr(localizer.cv2, "error")
            else:
                localizer.cv2.error = original_error

        self.assertTrue(meta["accepted"])
        self.assertFalse(meta["solve_set_reselected"])
        self.assertTrue(meta["stable_solve_set"])
        self.assertTrue(np.array_equal(chosen, np.arange(40)))

    def test_new_patrol_leg_atomically_replaces_previous_turn_anchor(self):
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        point_two = [-0.65, -0.08, -0.49]
        point_three = [-0.50, -0.08, 0.88]
        stabilizer.observe(
            delta_degrees=1.0,
            tracks=400,
            last_published_center=point_two,
            commanded_yaw_only=True,
            commanded_position_anchor=point_two,
            route_key=(1, 2),
        )
        stabilizer.release_anchor = np.asarray(point_two, dtype=float)
        stabilizer.room_bias = np.asarray([0.4, 0.0, -0.3], dtype=float)

        stabilizer.observe(
            delta_degrees=1.0,
            tracks=400,
            last_published_center=point_three,
            commanded_yaw_only=True,
            commanded_position_anchor=point_three,
            route_key=(1, 3),
        )

        self.assertTrue(np.allclose(stabilizer.position_anchor, point_three))
        self.assertIsNone(stabilizer.release_anchor)
        self.assertTrue(np.allclose(stabilizer.room_bias, np.zeros(3)))
        self.assertEqual(stabilizer.route_transition_reset_count, 1)

    def test_fixed_center_recovery_finds_orientation_without_moving_position(self):
        rng = np.random.default_rng(20260810)
        yaw = np.deg2rad(37.0)
        pitch = np.deg2rad(-6.0)
        yaw_rotation = np.array(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ]
        )
        pitch_rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(pitch), -np.sin(pitch)],
                [0.0, np.sin(pitch), np.cos(pitch)],
            ]
        )
        expected_rotation = pitch_rotation @ yaw_rotation
        good_world = rng.normal(size=(42, 3))
        good_world[:, 2] += 4.0
        good_camera = good_world @ expected_rotation.T
        bad_world = rng.normal(size=(28, 3))
        bad_world[:, 2] += 4.0
        bad_camera = rng.normal(size=(28, 3))
        pixels = np.column_stack(
            [
                np.linspace(80.0, 1120.0, 70),
                np.resize(np.array([70.0, 220.0, 430.0, 620.0]), 70),
            ]
        )

        solution, diagnostic = localizer.fixed_center_orientation_consensus(
            world_rays=np.vstack([good_world, bad_world]),
            camera_rays=np.vstack([good_camera, bad_camera]),
            image_xy=pixels,
            width=1200,
            height=675,
            min_inliers=18,
            sample_count=8000,
        )

        self.assertIsNotNone(solution)
        self.assertGreaterEqual(diagnostic["inliers"], 40)
        self.assertLess(diagnostic["median_angle_degrees"], 0.01)
        self.assertTrue(diagnostic["correspondence_spread"]["ok"])
        self.assertTrue(np.allclose(solution["R"], expected_rotation, atol=1e-6))

    def test_visual_route_pose_is_explicitly_not_a_fake_map_solve(self):
        pose = localizer.visual_route_pose_from_last(
            last_pose={"rcenter": [-3.0, 0.0, 1.0], "center": [1.0, 2.0, 3.0]},
            current_frame={
                "frame_index": 4552,
                "time_sec": 10.0,
                "received_unix": 20.0,
                "image_name": "query/query_004552.jpg",
            },
            observation={
                "verified": True,
                "center": [-3.1, 0.0, 0.9],
                "heading": [0.0, 0.0, -1.0],
                "progress": 0.1,
                "inliers": 300,
                "ratio_matches": 340,
                "anchor_name": "query_004551.jpg",
                "source_frame": 4551,
                "acquisition_hits": 2,
                "minimum_inliers": 120,
                "map_id": "map",
                "patrol_id": "patrol",
                "baseline_replay_id": "baseline",
            },
        )
        self.assertTrue(pose["success"])
        self.assertEqual(pose["rcenter"], [-3.1, 0.0, 0.9])
        self.assertIsNone(pose["center"])
        self.assertIsNone(pose["R"])
        self.assertIsNone(pose["t"])
        self.assertEqual(pose["pose_source"], "patrol_visual_route_recovery")
        self.assertTrue(pose["route_visual_verified"])

    def test_verified_visual_route_pose_recovers_from_failed_metric_case(self):
        point_four = [-2.9376830260, -0.0802621970, 1.1032224272]
        point_one = [-3.2329557448, -0.0802621970, -0.3323657986]

        class RouteGate:
            def __init__(self):
                self.observation = None

            def commit(self, observation):
                self.observation = observation

        class VisualRecovery:
            def __init__(self):
                self.progress = None

            def commit_published_progress(self, progress):
                self.progress = progress

        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        stabilizer.position_anchor = np.asarray(point_four, dtype=float)
        stabilizer.release_anchor = np.asarray(point_four, dtype=float)
        stabilizer.room_bias = np.asarray([0.4, 0.0, -0.2], dtype=float)
        gate = RouteGate()
        recovery = VisualRecovery()
        observation = {
            "verified": True,
            "center": [-2.95, -0.0802621970, 1.10],
            "heading": [-0.20, 0.0, -0.98],
            "progress": 0.9474,
            "inliers": 190,
            "ratio_matches": 220,
            "anchor_name": "query_003198.jpg",
            "source_frame": 3198,
            "acquisition_hits": 2,
            "minimum_inliers": 120,
            "map_id": "map",
            "patrol_id": "patrol",
            "baseline_replay_id": "baseline",
            "translation_safe": True,
        }
        pose = localizer.accepted_visual_route_recovery_pose(
            last_pose={
                "rcenter": point_four,
                "held_pose": True,
                "success": False,
                "rotation_position_locked": True,
                "translation_allowed": False,
            },
            current_frame={
                "frame_index": 3198,
                "time_sec": 319.8,
                "received_unix": 1000.0,
                "image_name": "query/query_003198.jpg",
            },
            observation=observation,
            supervision={
                "route_visual_monitor_required": True,
                "route_visual_monitor_verified": True,
                "route_visual_monitor_inliers": 190,
            },
            route_context={
                "lap": 1,
                "leg_index": 4,
                "start": point_four,
                "end": point_one,
            },
            rotation_heading=[-0.18, 0.0, -0.98],
            rotation_heading_tracks=239,
            rotation_position_stabilizer=stabilizer,
            route_gate=gate,
            visual_recovery=recovery,
        )

        self.assertIsNotNone(pose)
        self.assertTrue(pose["success"])
        self.assertFalse(pose["held_pose"])
        self.assertFalse(pose["rotation_position_locked"])
        self.assertTrue(pose["translation_allowed"])
        self.assertEqual(pose["pose_source"], "patrol_visual_route_recovery")
        self.assertIsNone(stabilizer.position_anchor)
        self.assertIsNone(stabilizer.release_anchor)
        self.assertTrue(np.allclose(stabilizer.room_bias, np.zeros(3)))
        self.assertIsNotNone(gate.observation)
        self.assertAlmostEqual(recovery.progress, gate.observation["progress"])

    def test_pose_publisher_defensively_clamps_visual_progress_to_command_budget(self):
        start = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        end = [-3.0736291183109774, -0.0802621969998909, 1.1113942967930859]

        class RouteGate:
            def __init__(self):
                self.observation = None

            def commit(self, observation):
                self.observation = observation

        class VisualRecovery:
            def __init__(self):
                self.progress = None

            def commit_published_progress(self, progress):
                self.progress = progress

        gate = RouteGate()
        recovery = VisualRecovery()
        pose = localizer.accepted_visual_route_recovery_pose(
            last_pose={"rcenter": start},
            current_frame={
                "frame_index": 1923,
                "time_sec": 238.0,
                "received_unix": 1786874572.019,
                "image_name": "query/query_001923.jpg",
            },
            observation={
                "verified": True,
                "translation_safe": True,
                "center": [
                    start[index] + 0.6448975443 * (end[index] - start[index])
                    for index in range(3)
                ],
                "heading": [-1.0, 0.0, 0.06],
                "progress": 0.6448975443,
                "matched_progress": 0.6448975443,
                "inliers": 127,
                "ratio_matches": 150,
                "anchor_name": "query_001774.jpg",
                "source_frame": 1774,
                "acquisition_hits": 2,
                "minimum_inliers": 120,
                "map_id": "map",
                "patrol_id": "patrol",
                "baseline_replay_id": "baseline",
            },
            supervision={},
            route_context={
                "lap": 1,
                "leg_index": 3,
                "start": start,
                "end": end,
                "route_progress_command_ceiling": 0.0,
            },
            rotation_heading=[-1.0, 0.0, 0.06],
            rotation_heading_tracks=487,
            rotation_position_stabilizer=localizer.RotationOnlyPositionStabilizer(
                enabled=True
            ),
            route_gate=gate,
            visual_recovery=recovery,
        )

        self.assertIsNotNone(pose)
        self.assertTrue(np.allclose(pose["rcenter"], start))
        self.assertAlmostEqual(pose["route_visual_progress"], 0.0)
        self.assertTrue(pose["route_visual_command_progress_guarded"])
        self.assertAlmostEqual(pose["route_visual_unbounded_progress"], 0.6448975443)
        self.assertAlmostEqual(recovery.progress, 0.0)

    def test_verified_visual_progress_recovers_only_missing_or_rejected_metric_pose(self):
        start = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        end = [-3.0736291183109774, -0.0802621969998909, 1.1113942967930859]
        last_pose = {
            "rcenter": [-0.7600168455283054, -0.0802621969998909, 0.9723205098637462]
        }
        observation = {
            "verified": True,
            "translation_safe": True,
            "progress": 0.17447067696581248,
            "inliers": 121,
            "minimum_inliers": 120,
            "acquisition_hits": 2,
        }
        recovery_context = {
            "start": start,
            "end": end,
            "recovery_hover": True,
            "controller_translation_locked": True,
        }

        self.assertTrue(
            localizer.visual_recovery_supersedes_stalled_metric_pose(
                last_pose=last_pose,
                observation=observation,
                route_context=recovery_context,
                output_rejection_reason=None,
            )
        )

        self.assertTrue(
            localizer.visual_recovery_supersedes_stalled_metric_pose(
                last_pose=last_pose,
                observation={
                    **observation,
                    "translation_safe": False,
                    "weak_endpoint_recovery": True,
                    "progress": 0.96,
                    "inliers": 70,
                    "acquisition_hits": 0,
                },
                route_context=recovery_context,
                output_rejection_reason=None,
            )
        )
        normal_context = {
            **recovery_context,
            "recovery_hover": False,
            "controller_translation_locked": False,
        }
        self.assertFalse(
            localizer.visual_recovery_supersedes_stalled_metric_pose(
                last_pose=last_pose,
                observation=observation,
                route_context=normal_context,
                output_rejection_reason=None,
                metric_route_observation={"progress": 0.174},
            )
        )
        self.assertFalse(
            localizer.visual_recovery_supersedes_stalled_metric_pose(
                last_pose=last_pose,
                observation=observation,
                route_context=normal_context,
                output_rejection_reason=None,
                metric_route_observation={"progress": 0.10},
            )
        )
        self.assertFalse(
            localizer.visual_recovery_supersedes_stalled_metric_pose(
                last_pose=last_pose,
                observation=observation,
                route_context=normal_context,
                output_rejection_reason=None,
                metric_route_observation={"progress": 0.25},
            )
        )
        departure_pose = {
            "rcenter": [
                start[index] + 0.03 * (end[index] - start[index])
                for index in range(3)
            ]
        }
        departure_observation = {
            **observation,
            "progress": 0.03,
            "matched_progress": 0.0,
        }
        self.assertTrue(
            localizer.visual_recovery_supersedes_stalled_metric_pose(
                last_pose=departure_pose,
                observation=departure_observation,
                route_context=normal_context,
                output_rejection_reason=None,
                metric_route_observation={"progress": 0.03},
                departure_floor_repair_available=True,
            )
        )
        self.assertFalse(
            localizer.visual_recovery_supersedes_stalled_metric_pose(
                last_pose=departure_pose,
                observation=departure_observation,
                route_context=normal_context,
                output_rejection_reason=None,
                metric_route_observation={"progress": 0.03},
                departure_floor_repair_available=False,
            )
        )
        self.assertFalse(
            localizer.visual_recovery_supersedes_stalled_metric_pose(
                last_pose=last_pose,
                observation={**observation, "inliers": 119},
                route_context=recovery_context,
                output_rejection_reason=None,
            )
        )
        self.assertTrue(
            localizer.visual_recovery_supersedes_stalled_metric_pose(
                last_pose=last_pose,
                observation=observation,
                route_context={
                    **recovery_context,
                    "recovery_hover": False,
                    "controller_translation_locked": False,
                },
                output_rejection_reason=None,
            )
        )
        self.assertFalse(
            localizer.visual_recovery_supersedes_stalled_metric_pose(
                last_pose=last_pose,
                observation=observation,
                route_context={
                    **recovery_context,
                    "recovery_hover": False,
                    "controller_translation_locked": True,
                },
                output_rejection_reason="route_backward_-0.173_lt_0.060",
            )
        )
        self.assertTrue(
            localizer.visual_recovery_supersedes_stalled_metric_pose(
                last_pose=last_pose,
                observation={**observation, "progress": 0.10496179800000001},
                route_context=recovery_context,
                output_rejection_reason="route_backward_-0.173_lt_0.060",
            )
        )

    def test_verified_route_pose_uses_recorded_heading_not_drifting_optical_yaw(self):
        pose = localizer.visual_route_pose_from_last(
            last_pose={"rcenter": [0.0, 0.0, 0.0]},
            current_frame={"frame_index": 12},
            observation={
                "verified": True,
                "center": [0.0, 0.0, 0.0],
                "heading": [-0.9982, 0.0, 0.0600],
                "progress": 0.0,
                "inliers": 197,
                "ratio_matches": 261,
                "anchor_name": "query_002901.jpg",
                "source_frame": 2901,
                "acquisition_hits": 2,
                "minimum_inliers": 120,
                "map_id": "map",
                "patrol_id": "patrol",
                "baseline_replay_id": "baseline",
            },
            rotation_heading=[-0.8392, 0.0, 0.5437],
            rotation_heading_tracks=466,
        )

        self.assertTrue(np.allclose(pose["rheading"], [-0.9982, 0.0, 0.0600]))
        self.assertEqual(pose["rheading_source"], "recorded_patrol_leg_heading")
        self.assertTrue(
            np.allclose(
                pose["rotation_heading"],
                localizer.normalize_room_heading([-0.8392, 0.0, 0.5437]),
            )
        )
        self.assertEqual(pose["rotation_heading_tracks"], 466)

    def test_visual_route_pose_reconciles_large_position_change_without_translation(self):
        pose = localizer.visual_route_pose_from_last(
            last_pose={"rcenter": [0.0, 0.0, 0.0]},
            current_frame={"frame_index": 10, "image_name": "query/query_000010.jpg"},
            observation={
                "verified": True,
                "center": [0.60, 0.0, 0.0],
                "heading": [1.0, 0.0, 0.0],
                "progress": 0.3,
                "inliers": 220,
                "ratio_matches": 250,
                "anchor_name": "query_000010.jpg",
                "source_frame": 10,
                "acquisition_hits": 2,
                "minimum_inliers": 120,
                "map_id": "map",
                "patrol_id": "patrol",
                "baseline_replay_id": "baseline",
            },
        )
        self.assertAlmostEqual(pose["rcenter"][0], 0.18)
        self.assertFalse(pose["translation_allowed"])
        self.assertTrue(pose["route_visual_reconciling"])
        self.assertAlmostEqual(pose["route_visual_reconciliation_remaining_m"], 0.42)

    def test_visual_route_commits_published_center_not_farther_match_target(self):
        start = [-0.6480244338911889, -0.0802621969998909, -0.48774887233005093]
        end = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        pose = {
            "rcenter": [-0.6071068951128926, -0.0802621969998909, -0.11696897820442546],
            "route_visual_progress": 0.36675422655526185,
        }
        observation = localizer.published_visual_route_observation(
            pose,
            {
                "lap": 1,
                "leg_index": 2,
                "start": start,
                "end": end,
            },
        )
        expected, _cross_track = localizer.route_segment_projection_xz(
            pose["rcenter"],
            start,
            end,
        )
        self.assertAlmostEqual(expected, 0.2568, places=4)
        self.assertAlmostEqual(observation["progress"], expected)
        self.assertAlmostEqual(pose["route_visual_published_progress"], expected)
        self.assertGreater(pose["route_visual_progress_lag"], 0.10)

    def test_live_visual_route_reconciliation_is_time_bounded(self):
        pose = localizer.visual_route_pose_from_last(
            last_pose={
                "rcenter": [0.0, 0.0, 0.0],
                "received_unix": 20.0,
            },
            current_frame={
                "frame_index": 11,
                "received_unix": 20.1,
                "image_name": "query/query_000011.jpg",
            },
            observation={
                "verified": True,
                "center": [0.60, 0.0, 0.0],
                "heading": [1.0, 0.0, 0.0],
                "progress": 0.3,
                "inliers": 220,
                "ratio_matches": 250,
                "anchor_name": "query_000011.jpg",
                "source_frame": 11,
                "acquisition_hits": 2,
                "minimum_inliers": 120,
                "map_id": "map",
                "patrol_id": "patrol",
                "baseline_replay_id": "baseline",
            },
        )

        self.assertAlmostEqual(pose["rcenter"][0], 0.036, places=6)
        self.assertFalse(pose["translation_allowed"])

    def test_visual_route_pose_unlocks_only_after_bounded_reconciliation_finishes(self):
        observation = {
            "verified": True,
            "center": [-0.648, 0.0, -0.488],
            "heading": [0.0, 0.0, 1.0],
            "progress": 0.0,
            "inliers": 160,
            "ratio_matches": 190,
            "anchor_name": "point_2.jpg",
            "source_frame": 1902,
            "acquisition_hits": 2,
            "minimum_inliers": 120,
            "map_id": "map",
            "patrol_id": "patrol",
            "baseline_replay_id": "baseline",
        }
        pose = {"rcenter": [-1.020, 0.0, -0.471]}
        translation_states = []
        for frame_index in range(3):
            pose = localizer.visual_route_pose_from_last(
                last_pose=pose,
                current_frame={"frame_index": frame_index},
                observation=observation,
            )
            translation_states.append(pose["translation_allowed"])

        self.assertEqual(translation_states, [False, True, True])
        self.assertAlmostEqual(pose["rcenter"][0], -0.648)
        self.assertAlmostEqual(pose["rcenter"][2], -0.488)

    def test_reference_supervision_stops_after_point_four(self):
        for leg_index in (1, 2, 3):
            with self.subTest(leg_index=leg_index):
                metadata = localizer.visual_route_supervision_metadata(
                    context={
                        "leg_index": leg_index,
                        "start": [0.0, 0.0, 0.0],
                        "end": [2.0, 0.0, 0.0],
                        "translation_locked": False,
                    },
                    observation={"verified": True, "progress": 0.62, "inliers": 180},
                    diagnostic={"reason": ""},
                    progress_hint=0.37,
                    minimum_inliers=120,
                )
                self.assertTrue(metadata["route_visual_monitor_required"])
                self.assertTrue(metadata["route_visual_monitor_verified"])
                self.assertEqual(metadata["route_visual_monitor_leg_index"], leg_index)
                self.assertAlmostEqual(metadata["route_visual_monitor_disagreement_m"], 0.5)

        self.assertEqual(
            localizer.visual_route_supervision_metadata(
                context={
                    "leg_index": 4,
                    "start": [0.0, 0.0, 0.0],
                    "end": [2.0, 0.0, 0.0],
                    "translation_locked": False,
                },
                observation={"verified": True, "progress": 0.62, "inliers": 180},
                diagnostic={"reason": ""},
                progress_hint=0.37,
                minimum_inliers=120,
            ),
            {},
        )

    def test_recorded_departure_alignment_corrects_rendered_heading_only(self):
        metadata = localizer.visual_route_heading_metadata(
            context={"leg_index": 3, "controller_translation_locked": True},
            observation={
                "verified": True,
                "correction_deg": 25.8,
                "heading": [-0.9981982, 0.0, 0.0600028],
                "inliers": 197,
                "ratio_matches": 261,
                "anchor_name": "query_002901.jpg",
                "source_frame": 2901,
                "map_id": "map",
                "patrol_id": "patrol",
                "baseline_replay_id": "baseline",
            },
            diagnostic={"reason": ""},
            minimum_inliers=120,
        )
        pose = localizer.apply_visual_route_heading_alignment(
            {"rcenter": [1.0, 0.0, 2.0], "rheading": [0.0, 0.0, 1.0]},
            metadata,
        )

        self.assertEqual(pose["rcenter"], [1.0, 0.0, 2.0])
        self.assertEqual(pose["rheading_source"], "recorded_departure_image_alignment")
        self.assertTrue(metadata["route_visual_heading_verified"])
        self.assertAlmostEqual(metadata["route_visual_heading_correction_deg"], 25.8)

    def test_departure_heading_threshold_is_leg_specific(self):
        self.assertEqual(
            localizer.visual_route_heading_minimum_inliers(120, leg_index=1),
            75,
        )
        self.assertEqual(
            localizer.visual_route_heading_minimum_inliers(120, leg_index=2),
            120,
        )
        self.assertEqual(
            localizer.visual_route_heading_minimum_inliers(120, leg_index=3),
            120,
        )
        self.assertEqual(
            localizer.visual_route_heading_minimum_inliers(120, leg_index=4),
            30,
        )

    def test_recorded_departure_alignment_is_limited_to_audited_turns(self):
        point_three = localizer.visual_route_heading_metadata(
            context={"leg_index": 3, "controller_translation_locked": True},
            observation=None,
            diagnostic={"reason": "visual_heading_no_candidates"},
            minimum_inliers=120,
        )
        self.assertTrue(point_three["route_visual_heading_required"])
        self.assertFalse(point_three["route_visual_heading_verified"])

        for leg_index in (1, 2):
            with self.subTest(leg_index=leg_index):
                minimum_inliers = 75 if leg_index == 1 else 120
                metadata = localizer.visual_route_heading_metadata(
                    context={
                        "leg_index": leg_index,
                        "controller_translation_locked": True,
                    },
                    observation=None,
                    diagnostic={"reason": "visual_heading_no_candidates"},
                    minimum_inliers=minimum_inliers,
                )
                self.assertTrue(metadata["route_visual_heading_required"])
                self.assertEqual(
                    metadata["route_visual_heading_leg_index"],
                    leg_index,
                )
                self.assertFalse(metadata["route_visual_heading_verified"])
                self.assertEqual(
                    metadata["route_visual_heading_minimum_inliers"],
                    minimum_inliers,
                )

        self.assertEqual(
            localizer.visual_route_heading_metadata(
                context={"leg_index": 4, "controller_translation_locked": True},
                observation=None,
                diagnostic={"reason": "visual_heading_no_candidates"},
                minimum_inliers=120,
            ),
            {},
        )

        for leg_index in (0,):
            with self.subTest(leg_index=leg_index):
                self.assertEqual(
                    localizer.visual_route_heading_metadata(
                        context={
                            "leg_index": leg_index,
                            "controller_translation_locked": True,
                        },
                        observation=None,
                        diagnostic={"reason": "visual_heading_no_candidates"},
                        minimum_inliers=120,
                    ),
                    {},
                )

    def test_non_baseline_leg_does_not_require_visual_supervision(self):
        metadata = localizer.visual_route_supervision_metadata(
            context={
                "leg_index": 0,
                "start": [0.0, 0.0, 0.0],
                "end": [2.0, 0.0, 0.0],
                "translation_locked": False,
            },
            observation={"verified": True, "progress": 0.62, "inliers": 180},
            diagnostic={"reason": ""},
            progress_hint=0.37,
            minimum_inliers=120,
        )
        self.assertEqual(metadata, {})

    def test_live_atlas_141441_profile_uses_point_four_release_settings(self):
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        self.assertAlmostEqual(stabilizer.max_reanchor_step, 0.75)
        self.assertAlmostEqual(stabilizer.bias_decay, 0.92)

    def test_live_rotation_lock_requires_fresh_yaw_only_controller_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control_status.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "command": "mission",
                        "updated_at": time.time(),
                        "progress": {
                            "translation_locked": True,
                            "body_forward_gain": 0.0,
                            "body_lateral_gain": 0.0,
                            "position_anchor": [1.0, 0.0, 2.0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(localizer.live_rotation_commanded(path))
            commanded, anchor = localizer.live_rotation_command_state(path)
            self.assertTrue(commanded)
            self.assertEqual(anchor, [1.0, 0.0, 2.0])
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["progress"]["body_forward_gain"] = 1.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(localizer.live_rotation_commanded(path))

            payload["progress"].update(
                {
                    "body_forward_gain": 0.0,
                    "translation_locked": True,
                    "phase": "taught_turn_wait_fresh_position",
                    "route_visual_recovery_allowed": True,
                }
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(localizer.live_rotation_commanded(path))

    def test_controller_translation_releases_active_rotation_lock(self):
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        anchor = [1.0, 0.0, 2.0]
        self.assertTrue(
            stabilizer.observe(
                delta_degrees=0.0,
                tracks=0,
                last_published_center=anchor,
                commanded_yaw_only=True,
                commanded_position_anchor=anchor,
            )
        )
        self.assertFalse(
            stabilizer.observe(
                delta_degrees=0.5,
                tracks=300,
                last_published_center=anchor,
                commanded_yaw_only=False,
            )
        )
        self.assertFalse(stabilizer.active)
        self.assertTrue(np.allclose(stabilizer.release_anchor, anchor))

    def test_controller_anchor_freezes_first_post_command_pose(self):
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        controller_anchor = [-4.20, -0.08, -0.34]
        self.assertTrue(
            stabilizer.observe(
                delta_degrees=0.0,
                tracks=0,
                last_published_center=[-4.04, -0.08, -0.37],
                commanded_yaw_only=True,
                commanded_position_anchor=controller_anchor,
            )
        )
        pose = stabilizer.apply(
            {
                "success": True,
                "held_pose": False,
                "rcenter": [-3.98, -0.08, -0.39],
            }
        )
        self.assertEqual(pose["rcenter"], controller_anchor)
        self.assertEqual(pose["rotation_raw_rcenter"], [-3.98, -0.08, -0.39])
        self.assertTrue(pose["rotation_position_locked"])
        self.assertFalse(pose["translation_allowed"])

    def test_live_position_stabilizer_is_disabled_without_commanded_yaw_proof(self):
        stabilizer = localizer.RotationOnlyPositionStabilizer()
        anchor = [0.0, 0.0, 0.0]
        self.assertFalse(
            stabilizer.observe(delta_degrees=0.8, tracks=500, last_published_center=anchor)
        )
        self.assertFalse(
            stabilizer.observe(delta_degrees=0.9, tracks=500, last_published_center=anchor)
        )
        pose = stabilizer.apply(
            {"success": True, "held_pose": False, "rcenter": [0.6, 0.0, 0.0]}
        )
        self.assertEqual(pose["rcenter"], [0.6, 0.0, 0.0])
        self.assertNotIn("rotation_position_locked", pose)

    def test_rotation_only_stabilizer_freezes_and_reanchors_room_position(self):
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True, bias_decay=1.0)
        anchor = [1.0, 0.2, 2.0]
        self.assertFalse(
            stabilizer.observe(delta_degrees=0.4, tracks=200, last_published_center=anchor)
        )
        self.assertTrue(
            stabilizer.observe(delta_degrees=0.5, tracks=200, last_published_center=anchor)
        )
        turned = stabilizer.apply(
            {
                "success": True,
                "held_pose": False,
                "rcenter": [1.35, 0.1, 2.40],
                "rheading": [1.0, 0.0, 0.0],
                "rotation_heading": [0.0, 0.0, 1.0],
            }
        )
        self.assertEqual(turned["rcenter"], anchor)
        self.assertTrue(turned["rotation_position_locked"])
        self.assertFalse(turned["translation_allowed"])
        self.assertEqual(turned["rotation_raw_rcenter"], [1.35, 0.1, 2.4])
        self.assertEqual(turned["rheading"], [0.0, 0.0, 1.0])
        self.assertEqual(turned["rheading_raw"], [1.0, 0.0, 0.0])
        self.assertEqual(turned["rheading_source"], "optical_flow_yaw")

        for _ in range(4):
            stabilizer.observe(delta_degrees=0.01, tracks=200, last_published_center=anchor)
        released = stabilizer.apply(
            {"success": True, "held_pose": False, "rcenter": [1.38, 0.1, 2.44]}
        )
        self.assertEqual(released["rcenter"], anchor)
        self.assertTrue(released["rotation_reanchored_after_turn"])
        translated = stabilizer.apply(
            {"success": True, "held_pose": False, "rcenter": [1.43, 0.1, 2.49]}
        )
        self.assertTrue(np.allclose(translated["rcenter"], [1.05, 0.2, 2.05]))

    def test_point_four_held_pose_cannot_roll_back_after_yaw_release(self):
        # Live ATLAS 14:14:41 physically reached Point 4 and turned toward
        # Point 1. At yaw release, a held TSolve pose exposed a false center
        # about 1.5 m away. Keep Point 4 and the live optical heading instead.
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        point_four = [-2.9376830260, -0.0802621970, 1.1032224272]
        self.assertTrue(
            stabilizer.observe(
                delta_degrees=0.0,
                tracks=400,
                last_published_center=point_four,
                commanded_yaw_only=True,
                commanded_position_anchor=point_four,
            )
        )
        self.assertFalse(
            stabilizer.observe(
                delta_degrees=0.0,
                tracks=400,
                last_published_center=point_four,
                commanded_yaw_only=False,
            )
        )
        held = stabilizer.apply(
            {
                "success": False,
                "held_pose": True,
                "rcenter": [-1.5116238, -0.092399, 0.6392275],
                "rheading": [-0.998, 0.0, 0.061],
                "rotation_heading": [-0.316, 0.0, -0.949],
            }
        )
        self.assertTrue(np.allclose(held["rcenter"], point_four))
        self.assertTrue(held["rotation_position_locked"])
        self.assertFalse(held["translation_allowed"])
        self.assertTrue(held["rotation_reanchor_pending"])
        self.assertTrue(
            np.allclose(
                held["rheading"],
                localizer.normalize_room_heading([-0.316, 0.0, -0.949]),
            )
        )
        self.assertEqual(held["rheading_source"], "optical_flow_yaw")

    def test_real_translation_discards_an_older_turn_release_anchor(self):
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        old_point_two_anchor = [-0.7573072937, -0.1879364000, -0.5980327764]
        translated_model = [-0.6071068951, -0.0802621970, -0.1169689782]
        self.assertTrue(
            stabilizer.observe(
                delta_degrees=0.0,
                tracks=200,
                last_published_center=old_point_two_anchor,
                commanded_yaw_only=True,
                commanded_position_anchor=old_point_two_anchor,
            )
        )
        self.assertFalse(
            stabilizer.observe(
                delta_degrees=0.0,
                tracks=200,
                last_published_center=translated_model,
                commanded_yaw_only=False,
            )
        )
        self.assertIsNone(stabilizer.release_anchor)
        self.assertEqual(stabilizer.stale_release_discard_count, 1)
        held = stabilizer.apply(
            {
                "success": False,
                "held_pose": True,
                "rcenter": translated_model,
            }
        )
        self.assertEqual(held["rcenter"], translated_model)
        self.assertNotIn("rotation_reanchor_pending", held)

    def test_visual_translation_discards_release_anchor_already_left_by_prior_turn(self):
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        point_three = [-0.4972860149, -0.0802621970, 0.8781880023]
        point_four = [-2.9556796797, -0.0802621970, 1.1043042259]
        stabilizer.release_anchor = np.asarray(point_three, dtype=float)
        stabilizer.room_bias = np.asarray([0.4, 0.0, -0.2], dtype=float)
        self.assertFalse(
            stabilizer.observe(
                delta_degrees=0.01,
                tracks=500,
                last_published_center=point_four,
                commanded_yaw_only=False,
            )
        )
        self.assertIsNone(stabilizer.release_anchor)
        self.assertTrue(np.allclose(stabilizer.room_bias, np.zeros(3)))
        held = stabilizer.apply(
            {"success": False, "held_pose": True, "rcenter": point_four}
        )
        self.assertEqual(held["rcenter"], point_four)
        self.assertNotIn("rotation_reanchor_pending", held)

    def test_visual_route_pose_drops_inherited_metric_turn_anchor_metadata(self):
        pose = localizer.visual_route_pose_from_last(
            last_pose={
                "rcenter": [-0.49, 0.0, 0.88],
                "rotation_position_anchor": [-0.49, 0.0, 0.88],
                "rotation_position_source": "post_yaw_anchor_hold",
                "rotation_reanchor_pending": True,
                "route_raw_rcenter": [-0.65, 0.0, -0.49],
                "route_position_constrained": True,
            },
            current_frame={"frame_index": 2024},
            observation={
                "verified": True,
                "center": [-2.95, 0.0, 1.10],
                "heading": [-0.11, 0.0, -0.99],
                "progress": 0.95,
                "inliers": 253,
                "ratio_matches": 300,
                "anchor_name": "query_002024.jpg",
                "source_frame": 2024,
                "acquisition_hits": 2,
                "minimum_inliers": 120,
                "map_id": "map",
                "patrol_id": "patrol",
                "baseline_replay_id": "baseline",
            },
        )
        self.assertNotIn("rotation_position_anchor", pose)
        self.assertNotIn("rotation_reanchor_pending", pose)
        self.assertNotIn("route_raw_rcenter", pose)
        self.assertFalse(pose["rotation_position_locked"])

    def test_single_yaw_flow_spike_does_not_freeze_translation(self):
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        self.assertFalse(
            stabilizer.observe(
                delta_degrees=0.8,
                tracks=200,
                last_published_center=[0.0, 0.0, 0.0],
            )
        )
        self.assertFalse(
            stabilizer.observe(
                delta_degrees=0.01,
                tracks=200,
                last_published_center=[0.02, 0.0, 0.0],
            )
        )
        pose = stabilizer.apply(
            {"success": True, "held_pose": False, "rcenter": [0.04, 0.0, 0.0]}
        )
        self.assertEqual(pose["rcenter"], [0.04, 0.0, 0.0])
        self.assertNotIn("rotation_position_locked", pose)

    def test_large_point_three_turn_drift_stays_translation_locked_after_release(self):
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        anchor = [-0.49243012301642014, 0.11100994544145168, 0.9107450060397975]
        self.assertFalse(
            stabilizer.observe(delta_degrees=0.4, tracks=400, last_published_center=anchor)
        )
        self.assertTrue(
            stabilizer.observe(delta_degrees=0.5, tracks=400, last_published_center=anchor)
        )
        turned = stabilizer.apply(
            {
                "success": True,
                "held_pose": False,
                "rcenter": [-1.197261690960473, -0.11429148483458043, 0.43565012525803337],
            }
        )
        self.assertEqual(turned["rcenter"], anchor)

        for _ in range(4):
            stabilizer.observe(delta_degrees=0.01, tracks=400, last_published_center=anchor)
        raw_release = np.asarray([-1.2119898410684742, -0.10905360420707644, 0.4489930363115429])
        released = stabilizer.apply(
            {
                "success": True,
                "held_pose": False,
                "rcenter": raw_release.tolist(),
                "rheading": [-0.8340645636815801, 0.0, 0.5516668411375253],
                "rotation_heading": [-0.9925124632794704, 0.0, 0.12214340029210799],
            }
        )
        self.assertTrue(np.allclose(released["rcenter"], anchor))
        self.assertTrue(released["rotation_reanchor_rejected"])
        self.assertTrue(released["rotation_position_locked"])
        self.assertFalse(released["translation_allowed"])

        raw_forward = raw_release + np.asarray([-0.10, 0.0, 0.02])
        translated = stabilizer.apply(
            {"success": True, "held_pose": False, "rcenter": raw_forward.tolist()}
        )
        self.assertTrue(np.allclose(translated["rcenter"], anchor))
        self.assertTrue(translated["rotation_position_locked"])
        self.assertFalse(translated["translation_allowed"])

    def test_commanded_yaw_reanchors_live_120857_large_false_center_drift(self):
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        anchor = np.asarray(
            [-0.754698484909238, -0.0802621969998909, -0.48133657740202895]
        )
        self.assertTrue(
            stabilizer.observe(
                delta_degrees=0.40,
                tracks=500,
                last_published_center=anchor,
                commanded_yaw_only=True,
                commanded_position_anchor=anchor,
            )
        )
        false_raw = anchor + np.asarray([0.0, 0.0, 0.979])
        turned = stabilizer.apply(
            {"success": True, "held_pose": False, "rcenter": false_raw.tolist()}
        )
        self.assertTrue(np.allclose(turned["rcenter"], anchor))
        self.assertTrue(turned["rotation_position_locked"])
        self.assertTrue(turned["rotation_anchor_commanded"])

        self.assertFalse(
            stabilizer.observe(
                delta_degrees=0.01,
                tracks=500,
                last_published_center=anchor,
                commanded_yaw_only=False,
            )
        )
        released = stabilizer.apply(
            {"success": True, "held_pose": False, "rcenter": false_raw.tolist()}
        )
        self.assertTrue(np.allclose(released["rcenter"], anchor))
        self.assertTrue(released["rotation_reanchored_after_turn"])
        self.assertTrue(released["rotation_anchor_is_position_truth"])
        self.assertTrue(released["rotation_anchor_commanded"])
        self.assertNotIn("rotation_position_locked", released)

        raw_forward = false_raw + np.asarray([0.02, 0.0, 0.05])
        translated = stabilizer.apply(
            {"success": True, "held_pose": False, "rcenter": raw_forward.tolist()}
        )
        self.assertTrue(
            np.allclose(translated["rcenter"], anchor + np.asarray([0.02, 0.0, 0.05]))
        )
        self.assertTrue(translated["rotation_anchor_is_position_truth"])
        self.assertTrue(stabilizer.room_bias_commanded)

    def test_absolute_visual_position_clears_commanded_turn_bias(self):
        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        stabilizer.release_anchor = np.asarray([1.0, 0.0, 1.0])
        stabilizer.release_anchor_commanded = True
        stabilizer.room_bias = np.asarray([0.5, 0.0, -0.4])
        stabilizer.room_bias_commanded = True

        stabilizer.accept_absolute_position()

        self.assertIsNone(stabilizer.release_anchor)
        self.assertFalse(stabilizer.release_anchor_commanded)
        self.assertTrue(np.allclose(stabilizer.room_bias, np.zeros(3)))
        self.assertFalse(stabilizer.room_bias_commanded)

    def test_normal_track_keeps_the_ordinary_refresh_floor(self):
        self.assertEqual(
            localizer.required_tracking_points(
                {"trusted_recovery": False},
                normal_minimum=15,
                solver_minimum=8,
            ),
            15,
        )

    def test_consensus_recovery_track_uses_the_solver_safe_floor(self):
        self.assertEqual(
            localizer.required_tracking_points(
                {"trusted_recovery": True},
                normal_minimum=15,
                solver_minimum=8,
            ),
            8,
        )

    def test_live_pose_stream_json_is_compact(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "poses.json"
            localizer.atomic_write_json(path, {"poses": [{"success": True}]})
            self.assertEqual(path.read_text(encoding="utf-8"), '{"poses":[{"success":true}]}')

    def test_single_marginal_objective_rejection_keeps_tracking_pool(self):
        reason = localizer.output_objective_rejection({"objective": 31.127}, 30.0)
        self.assertEqual(reason, "objective_31.127_gt_30.000")
        self.assertFalse(
            localizer.output_rejection_requires_tracking_reset(reason, 1, 3)
        )
        self.assertFalse(
            localizer.output_rejection_requires_tracking_reset(reason, 2, 3)
        )
        self.assertFalse(
            localizer.output_rejection_requires_tracking_reset(reason, 3, 3)
        )

    def test_golden_path_marginal_objective_is_accepted_when_motion_is_continuous(self):
        result = {
            "success": True,
            "objective": 31.127,
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [-0.05, 0.0, 0.0],
        }
        reason, center, timestamp = localizer.output_continuity_rejection(
            case={"time_sec": 10.1},
            result=result,
            previous_center=np.asarray([0.0, 0.0, 0.0]),
            previous_time=10.0,
            objective_threshold=30.0,
        )
        self.assertIsNone(reason)
        self.assertTrue(np.allclose(center, [0.05, 0.0, 0.0]))
        self.assertEqual(timestamp, 10.1)

    def test_objective_only_rejection_never_erases_optical_anchor(self):
        self.assertFalse(
            localizer.output_rejection_requires_tracking_reset(
                "objective_61.000_gt_60.000",
                20,
                3,
            )
        )

    def test_sub_hard_cap_motion_catchup_keeps_optical_anchor(self):
        self.assertFalse(
            localizer.output_rejection_requires_tracking_reset(
                "motion_jump_0.526m_gt_0.355m",
                3,
                3,
                0.55,
            )
        )

    def test_over_hard_cap_motion_jump_resets_after_failure_limit(self):
        reason = "motion_jump_0.651m_gt_0.550m"
        self.assertFalse(
            localizer.output_rejection_requires_tracking_reset(reason, 2, 3, 0.55)
        )
        self.assertTrue(
            localizer.output_rejection_requires_tracking_reset(reason, 3, 3, 0.55)
        )

    def test_rejected_motion_keeps_consecutive_flow_anchor_until_reset_limit(self):
        reason = "motion_jump_0.350m_gt_0.300m"
        self.assertTrue(
            localizer.rejected_output_can_advance_flow_anchor(reason, 1, 3, 0.30)
        )
        self.assertTrue(
            localizer.rejected_output_can_advance_flow_anchor(reason, 2, 3, 0.30)
        )
        self.assertFalse(
            localizer.rejected_output_can_advance_flow_anchor(reason, 3, 3, 0.30)
        )

    def test_strict_publication_cap_does_not_become_destructive_tracking_cap(self):
        hard_cap = localizer.tracking_reset_hard_motion_cap(0.30)
        self.assertEqual(hard_cap, 0.55)
        for reason in (
            "motion_jump_0.418m_gt_0.300m",
            "motion_jump_0.445m_gt_0.300m",
            "motion_jump_0.457m_gt_0.300m",
        ):
            self.assertTrue(
                localizer.rejected_output_can_advance_flow_anchor(
                    reason, 3, 3, hard_cap
                )
            )
        self.assertFalse(
            localizer.rejected_output_can_advance_flow_anchor(
                "motion_jump_0.651m_gt_0.300m", 3, 3, hard_cap
            )
        )

    def test_rejected_output_flow_anchor_never_accepts_rejected_room_pose(self):
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertGreaterEqual(
            source.count('reference_update_reason = "output_rejected_pose_flow_anchor_only"'),
            2,
        )
        self.assertGreaterEqual(
            source.count('reference_update_reason = "held_rejected_pose_not_trusted"'),
            2,
        )

    def test_patrol_route_publish_is_monotonic_and_rate_limited(self):
        gate = localizer.LivePatrolRouteGate(None, None)
        context = {
            "start": [0.0, 0.0, 0.0],
            "end": [2.0, 0.0, 0.0],
            "anchor": [0.0, 0.0, 0.0],
        }
        key = (1, 1, 0.0, 0.0, 2.0, 0.0)
        first = gate.constrain_published_pose(
            {"rcenter": [0.50, 0.20, 0.30], "time_sec": 10.0},
            {"key": key, "progress": 0.25, "cross_track": 0.30, "context": context},
        )
        self.assertTrue(np.allclose(first["rcenter"], [0.0, 0.0, 0.0]))
        jumped = gate.constrain_published_pose(
            {"rcenter": [1.20, -0.30, -0.25], "time_sec": 10.1},
            {"key": key, "progress": 0.60, "cross_track": 0.25, "context": context},
        )
        self.assertLessEqual(float(jumped["rcenter"][0]), 0.051)
        corrected_back = gate.constrain_published_pose(
            {"rcenter": [0.20, 0.0, 0.0], "time_sec": 10.2},
            {"key": key, "progress": 0.10, "cross_track": 0.0, "context": context},
        )
        self.assertGreaterEqual(
            float(corrected_back["route_published_progress"]),
            float(jumped["route_published_progress"]),
        )

    def test_verified_hover_rewind_corrects_false_endpoint_clock_gradually(self):
        gate = localizer.LivePatrolRouteGate(None, None)
        context = {
            "start": [0.0, 0.0, 0.0],
            "end": [2.0, 0.0, 0.0],
            "anchor": [0.0, 0.0, 0.0],
            "lap": 1,
            "leg_index": 3,
            "recovery_hover": True,
        }
        key = localizer.LivePatrolRouteGate._key(context)
        gate.last_key = key
        gate.last_progress = 0.90
        gate.last_publish_time = 10.0
        pose = {
            "rcenter": [1.30, 0.0, 0.0],
            "time_sec": 10.1,
            "route_visual_verified_rewind": True,
            "route_visual_verified_rewind_hits": 5,
            "route_visual_verified_rewind_inliers": 120,
        }
        observation = {
            "key": key,
            "progress": 0.65,
            "cross_track": 0.0,
            "context": context,
            "verified_visual_rewind": True,
        }

        corrected = gate.constrain_published_pose(pose, observation)

        self.assertTrue(corrected["route_verified_visual_rewind_applied"])
        self.assertLess(corrected["route_published_progress"], 0.90)
        self.assertGreaterEqual(corrected["route_published_progress"], 0.875)
        self.assertEqual(gate.last_progress, corrected["route_published_progress"])

    def test_visual_route_observation_preserves_verified_rewind_authority(self):
        context = {
            "start": [0.0, 0.0, 0.0],
            "end": [2.0, 0.0, 0.0],
            "lap": 1,
            "leg_index": 3,
        }
        observation = localizer.published_visual_route_observation(
            {
                "rcenter": [1.3, 0.0, 0.0],
                "route_visual_progress": 0.65,
                "route_visual_verified_rewind": True,
            },
            context,
        )
        self.assertTrue(observation["verified_visual_rewind"])

    def test_patrol_route_never_publishes_beyond_a_waypoint(self):
        gate = localizer.LivePatrolRouteGate(None, None)
        context = {
            "start": [0.0, 0.0, 0.0],
            "end": [2.0, 0.0, 0.0],
            "anchor": [0.0, 0.0, 0.0],
        }
        key = (1, 1, 0.0, 0.0, 2.0, 0.0)
        gate.last_key = key
        gate.last_progress = 0.99
        gate.last_publish_time = 10.0
        published = gate.constrain_published_pose(
            {"rcenter": [2.10, 0.0, 0.0], "time_sec": 11.0},
            {"key": key, "progress": 1.05, "cross_track": 0.0, "context": context},
        )
        self.assertAlmostEqual(published["route_published_progress"], 1.0)
        self.assertTrue(np.allclose(published["rcenter"], context["end"]))

    def test_verified_endpoint_repairs_the_exact_1633_overrun_deadlock(self):
        start = [-3.2329557447702215, -0.0802621969998909, -0.33236579860361815]
        end = [-0.6480244338911889, -0.0802621969998909, -0.48774887233005093]
        context = {
            "start": start,
            "end": end,
            "anchor": start,
            "lap": 1,
            "leg_index": 1,
            "controller_translation_locked": True,
            "recovery_hover": True,
        }
        observation = {
            "verified": True,
            "translation_safe": True,
            "progress": 0.9527016699889285,
            "inliers": 153,
            "minimum_inliers": 120,
            "acquisition_hits": 2,
            "endpoint_verified": True,
        }
        gate = localizer.LivePatrolRouteGate(None, None)
        gate.last_key = gate._key(context)
        gate.last_progress = 1.0793975401231017

        self.assertTrue(
            localizer.visual_recovery_supersedes_stalled_metric_pose(
                last_pose={"rcenter": [-0.21717948187335478, -0.1386030527117187, -0.25355579610795353]},
                observation=observation,
                route_context=context,
                output_rejection_reason="route_backward_0.952_lt_1.034",
            )
        )
        self.assertTrue(gate.reconcile_verified_endpoint_floor(observation, context))
        self.assertAlmostEqual(gate.last_progress, 1.0)

    def test_running_patrol_status_survives_the_two_second_waypoint_hover(self):
        baseline = {
            "complete_loop": True,
            "enabled_for_live_route_gate": True,
            "map_id": "map_a",
            "patrol_id": "patrol_a",
            "legs": [
                {"from": [0.0, 0.0, 0.0], "to": [2.0, 0.0, 0.0]},
                {"from": [2.0, 0.0, 0.0], "to": [2.0, 0.0, 2.0]},
                {"from": [2.0, 0.0, 2.0], "to": [0.0, 0.0, 2.0]},
                {"from": [0.0, 0.0, 2.0], "to": [0.0, 0.0, 0.0]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            baseline_path = Path(td) / "baseline_a" / "reference_candidate.json"
            baseline_path.parent.mkdir()
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            status_path = Path(td) / "control_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "command": "mission",
                        "updated_at": time.time() - 2.2,
                        "progress": {
                            "map_id": "map_a",
                            "patrol_id": "patrol_a",
                            "baseline_replay_id": "baseline_a",
                            "lap": 1,
                            "segment_start": [0.0, 0.0, 0.0],
                            "target": [2.0, 0.0, 0.0],
                            "translation_locked": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            gate = localizer.LivePatrolRouteGate(
                baseline_path,
                status_path,
                max_status_age=5.0,
            )
            self.assertIsNotNone(gate.active_context())

    def test_metric_candidate_more_than_eight_percent_past_target_is_rejected(self):
        gate = localizer.LivePatrolRouteGate(None, None)
        context = {
            "start": [0.0, 0.0, 0.0],
            "end": [2.0, 0.0, 0.0],
            "anchor": [0.0, 0.0, 0.0],
            "lap": 1,
            "leg_index": 1,
            "position_guard_locked": False,
        }
        gate.active_context = lambda: context
        reason, observation = gate.rejection([2.32, 0.0, 0.0], [2.0, 0.0, 0.0])
        self.assertEqual(reason, "route_progress_outside_segment_1.160")
        self.assertAlmostEqual(observation["progress"], 1.16)

    def test_rejected_point2_jump_cannot_advance_route_floor_or_model(self):
        gate = localizer.LivePatrolRouteGate(None, None)
        start = [-0.6480244338911889, -0.0802621969998909, -0.48774887233005093]
        end = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        key = (1, 2, round(start[0], 4), round(start[2], 4), round(end[0], 4), round(end[2], 4))
        context = {"start": start, "end": end, "anchor": start}
        gate.last_key = key
        gate.last_progress = -0.0037985615350677885
        gate.last_publish_time = 129.762519
        trusted_center = [-0.648629645885969, -0.0802621969998909, -0.49323308389429066]
        held = {
            "instance_id": "hold_001077",
            "held_pose": True,
            "output_rejected": True,
            "hold_reason": "motion_jump_0.754m_gt_0.300m",
            "time_sec": 130.8,
            "rcenter": list(trusted_center),
            "route_published_progress": gate.last_progress,
        }

        published = gate.constrain_published_pose(
            held,
            {
                "key": key,
                "progress": 0.0634295802210962,
                "cross_track": 0.0,
                "context": context,
            },
        )

        self.assertEqual(published["rcenter"], trusted_center)
        self.assertAlmostEqual(gate.last_progress, -0.0037985615350677885)
        self.assertEqual(gate.last_publish_time, 129.762519)
        self.assertTrue(published["route_rejected_observation_ignored"])
        self.assertAlmostEqual(
            published["route_rejected_observation_progress"],
            0.0634295802210962,
        )

    def test_strong_route_tracking_can_absorb_lateral_solver_correction(self):
        context = {
            "start": [0.0, 0.0, 0.0],
            "end": [2.0, 0.0, 0.0],
            "anchor": [0.0, 0.0, 0.0],
        }

        class RouteGate:
            enabled = True
            last_key = ("leg",)
            last_progress = 0.30

            def active_context(self):
                return context

            def rejection(self, candidate, _previous):
                progress, cross_track = localizer.route_segment_projection_xz(
                    candidate, context["start"], context["end"]
                )
                return None, {
                    "key": self.last_key,
                    "progress": progress,
                    "cross_track": cross_track,
                    "context": context,
                }

        candidate = np.asarray([0.70, 0.40, 0.0])
        kwargs = {
            "result": {
                "success": True,
                "objective": 20.0,
                "R": np.eye(3).tolist(),
                "t": (-candidate).tolist(),
            },
            "previous_center": np.asarray([0.60, 0.0, 0.0]),
            "previous_time": 10.0,
            "previous_pose": {"rcenter": [0.60, 0.0, 0.0]},
            "route_gate": RouteGate(),
            "room_transform": lambda center: center,
            "output_center_bias": None,
            "max_step": 0.30,
            "max_speed": 0.85,
            "objective_threshold": 30.0,
        }
        reason, accepted, _timestamp, observation = (
            localizer.route_guarded_output_rejection(
                case={"time_sec": 10.1, "tracked_pool_points": 400},
                **kwargs,
            )
        )
        self.assertIsNone(reason)
        self.assertTrue(observation["route_continuity_override"])
        self.assertTrue(np.allclose(accepted, candidate))
        weak_reason, _accepted, _timestamp, _observation = (
            localizer.route_guarded_output_rejection(
                case={"time_sec": 10.1, "tracked_pool_points": 40},
                **kwargs,
            )
        )
        self.assertEqual(weak_reason, "motion_jump_0.412m_gt_0.300m")

        context["position_guard_locked"] = True
        locked_reason, accepted, _timestamp, observation = (
            localizer.route_guarded_output_rejection(
                case={"time_sec": 10.1, "tracked_pool_points": 40},
                **kwargs,
            )
        )
        self.assertIsNone(locked_reason)
        self.assertTrue(np.allclose(accepted, kwargs["previous_center"]))
        self.assertTrue(observation["route_continuity_override"])
        self.assertEqual(
            observation["route_continuity_override_source"],
            "controller_position_lock",
        )
        self.assertTrue(
            observation["route_continuity_preserved_tracking_center"]
        )
        source = LOCALIZER_PATH.read_text(encoding="utf-8")
        self.assertGreaterEqual(
            source.count("controller_locked_turn_flow_anchor_only"),
            2,
        )

        close_candidate = np.asarray([0.65, 0.0, 0.0])
        close_kwargs = {
            **kwargs,
            "result": {
                "success": True,
                "objective": 20.0,
                "R": np.eye(3).tolist(),
                "t": (-close_candidate).tolist(),
            },
        }
        close_reason, accepted, _timestamp, observation = (
            localizer.route_guarded_output_rejection(
                case={"time_sec": 10.1, "tracked_pool_points": 40},
                **close_kwargs,
            )
        )
        self.assertIsNone(close_reason)
        self.assertTrue(np.allclose(accepted, kwargs["previous_center"]))
        self.assertTrue(
            observation["route_continuity_preserved_tracking_center"]
        )

    def test_route_holds_do_not_preload_destructive_tracking_reset(self):
        streak = 0
        for _ in range(168):
            streak = localizer.next_output_tracking_reset_streak(
                "route_backward_0.855_lt_0.931",
                streak,
                0.55,
            )
        self.assertEqual(streak, 0)

        streak = localizer.next_output_tracking_reset_streak(
            "motion_jump_4.364m_gt_0.550m",
            streak,
            0.55,
        )
        self.assertEqual(streak, 1)
        self.assertFalse(
            localizer.output_rejection_requires_tracking_reset(
                "motion_jump_4.364m_gt_0.550m",
                streak,
                3,
                0.55,
            )
        )

    def test_live_point1_exit_waits_for_sub_hard_cap_catchup(self):
        previous_center = np.asarray([-1.0095929462, -0.1397335525, 1.3740492928])
        previous_time = 148.363547
        candidates = [
            (148.466664, [-0.5548, -0.1106, 1.1140]),
            (148.569300, [-0.5534, -0.1093, 1.1133]),
            (148.680452, [-0.5547, -0.1087, 1.1138]),
            (148.783881, [-0.5547, -0.1087, 1.1138]),
        ]
        for index, (timestamp, center) in enumerate(candidates):
            result = {
                "success": True,
                "objective": 27.0,
                "R": np.eye(3).tolist(),
                "t": (-np.asarray(center)).tolist(),
            }
            reason, accepted_center, accepted_time = localizer.output_continuity_rejection(
                case={"time_sec": timestamp},
                result=result,
                previous_center=previous_center,
                previous_time=previous_time,
                max_step=0.55,
                max_speed=0.85,
                objective_threshold=30.0,
            )
            if index < 3:
                self.assertIsNotNone(reason)
                self.assertTrue(str(reason).startswith("motion_jump_0.52"))
                self.assertFalse(
                    localizer.output_rejection_requires_tracking_reset(
                        reason,
                        index + 1,
                        3,
                        0.55,
                    )
                )
                self.assertTrue(np.allclose(accepted_center, previous_center))
                self.assertEqual(accepted_time, previous_time)
            else:
                self.assertIsNone(reason)
                self.assertTrue(np.allclose(accepted_center, center))
                self.assertEqual(accepted_time, timestamp)

    def test_live_164209_false_point1_to2_jump_is_rejected(self):
        previous_center = np.asarray([-2.3026461278, -0.1409000238, -0.4042403342])
        false_center = np.asarray([-2.1553105645, -0.3130697060, 0.0256585552])
        result = {
            "success": True,
            "objective": 24.735,
            "R": np.eye(3).tolist(),
            "t": (-false_center).tolist(),
        }
        reason, accepted_center, accepted_time = localizer.output_continuity_rejection(
            case={"time_sec": 103.02},
            result=result,
            previous_center=previous_center,
            previous_time=102.584,
            max_step=0.30,
            max_speed=0.85,
            objective_threshold=30.0,
        )
        self.assertEqual(reason, "motion_jump_0.486m_gt_0.300m")
        self.assertTrue(np.allclose(accepted_center, previous_center))
        self.assertEqual(accepted_time, 102.584)

    def test_live_094100_takeoff_keeps_horizontal_tracking_anchor(self):
        previous_center = np.zeros(3)
        takeoff_center = np.asarray([0.2624, 0.5226, 0.0])
        result = {
            "success": True,
            "objective": 13.898,
            "R": np.eye(3).tolist(),
            "t": (-takeoff_center).tolist(),
        }
        reason, accepted_center, accepted_time = localizer.output_continuity_rejection(
            case={"time_sec": 63.493460},
            result=result,
            previous_center=previous_center,
            previous_time=63.366642,
            max_step=0.30,
            max_speed=0.85,
            objective_threshold=30.0,
            allow_startup_vertical_motion=True,
            room_transform=lambda center: center,
        )
        self.assertIsNone(reason)
        self.assertTrue(np.allclose(accepted_center, takeoff_center))
        self.assertEqual(accepted_time, 63.493460)

    def test_startup_vertical_allowance_never_allows_horizontal_teleport(self):
        false_center = np.asarray([0.486, 0.10, 0.0])
        reason, accepted_center, accepted_time = localizer.output_continuity_rejection(
            case={"time_sec": 63.5},
            result={
                "success": True,
                "objective": 12.0,
                "R": np.eye(3).tolist(),
                "t": (-false_center).tolist(),
            },
            previous_center=np.zeros(3),
            previous_time=63.3,
            max_step=0.30,
            max_speed=0.85,
            allow_startup_vertical_motion=True,
            room_transform=lambda center: center,
        )
        self.assertEqual(reason, "motion_jump_0.486m_gt_0.300m")
        self.assertTrue(np.allclose(accepted_center, np.zeros(3)))
        self.assertEqual(accepted_time, 63.3)

    def test_startup_vertical_allowance_remains_bounded(self):
        false_center = np.asarray([0.05, 0.90, 0.0])
        reason, accepted_center, accepted_time = localizer.output_continuity_rejection(
            case={"time_sec": 63.5},
            result={
                "success": True,
                "objective": 12.0,
                "R": np.eye(3).tolist(),
                "t": (-false_center).tolist(),
            },
            previous_center=np.zeros(3),
            previous_time=63.3,
            max_step=0.30,
            max_speed=0.85,
            allow_startup_vertical_motion=True,
            room_transform=lambda center: center,
        )
        self.assertEqual(reason, "vertical_jump_0.900m_gt_0.750m")
        self.assertTrue(np.allclose(accepted_center, np.zeros(3)))
        self.assertEqual(accepted_time, 63.3)

    def test_consensus_recovery_cannot_expand_the_physical_jump_cap(self):
        result = {
            "success": True,
            "objective": 12.0,
            "R": np.eye(3).tolist(),
            "t": [-0.70, 0.0, 0.0],
        }
        reason, center, timestamp = localizer.output_continuity_rejection(
            case={
                "time_sec": 11.0,
                "trusted_recovery": True,
                "recovery_max_step": 0.85,
            },
            result=result,
            previous_center=np.zeros(3),
            previous_time=10.0,
            max_step=0.55,
            max_speed=0.85,
        )

        self.assertEqual(reason, "motion_jump_0.700m_gt_0.550m")
        self.assertTrue(np.allclose(center, np.zeros(3)))
        self.assertEqual(timestamp, 10.0)

    def test_verified_global_recovery_metadata_does_not_expand_output_cap(self):
        pool = localizer.mark_global_recovery_pool(
            {"accepted": True, "valid_2d3d": 877},
            last_center=np.zeros(3),
            recovery_max_step=0.85,
        )
        self.assertTrue(pool["trusted_recovery"])
        self.assertEqual(pool["recovery_max_step"], 0.85)

        reason, center, _timestamp = localizer.output_continuity_rejection(
            case={
                "time_sec": 11.0,
                "trusted_recovery": pool["trusted_recovery"],
                "recovery_max_step": pool["recovery_max_step"],
            },
            result={
                "success": True,
                "objective": 12.0,
                "R": np.eye(3).tolist(),
                "t": [-0.618, 0.0, 0.0],
            },
            previous_center=np.zeros(3),
            previous_time=10.0,
            max_step=0.55,
            max_speed=0.85,
        )
        self.assertEqual(reason, "motion_jump_0.618m_gt_0.550m")
        self.assertTrue(np.allclose(center, np.zeros(3)))

    def test_consensus_recovery_still_obeys_its_absolute_safety_cap(self):
        result = {
            "success": True,
            "objective": 12.0,
            "R": np.eye(3).tolist(),
            "t": [-0.90, 0.0, 0.0],
        }
        reason, center, timestamp = localizer.output_continuity_rejection(
            case={
                "time_sec": 11.0,
                "trusted_recovery": True,
                "recovery_max_step": 0.85,
            },
            result=result,
            previous_center=np.zeros(3),
            previous_time=10.0,
            max_step=0.55,
            max_speed=0.85,
        )

        self.assertEqual(reason, "motion_jump_0.900m_gt_0.550m")
        self.assertTrue(np.allclose(center, np.zeros(3)))
        self.assertEqual(timestamp, 10.0)

    def test_live_patrol_route_gate_rejects_backward_progress_and_anchors_turn_drift(self):
        baseline = {
            "complete_loop": True,
            "enabled_for_live_route_gate": True,
            "map_id": "map_a",
            "patrol_id": "patrol_a",
            "legs": [
                {"from": [0.0, 0.0, 0.0], "to": [2.0, 0.0, 0.0]},
                {"from": [2.0, 0.0, 0.0], "to": [2.0, 0.0, 2.0]},
                {"from": [2.0, 0.0, 2.0], "to": [0.0, 0.0, 2.0]},
                {"from": [0.0, 0.0, 2.0], "to": [0.0, 0.0, 0.0]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            baseline_path = Path(td) / "baseline_a" / "reference_candidate.json"
            baseline_path.parent.mkdir()
            status_path = Path(td) / "control_status.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            status = {
                "status": "running",
                "command": "mission",
                "updated_at": time.time(),
                "progress": {
                    "map_id": "map_a",
                    "patrol_id": "patrol_a",
                    "baseline_replay_id": "baseline_a",
                    "lap": 1,
                    "step_index": 4,
                    "segment_start": [0.0, 0.0, 0.0],
                    "target": [2.0, 0.0, 0.0],
                    "translation_locked": False,
                },
            }
            status_path.write_text(json.dumps(status), encoding="utf-8")
            gate = localizer.LivePatrolRouteGate(baseline_path, status_path, turn_max_drift=0.16)

            reason, observation = gate.rejection([1.0, 0.0, 0.05], [0.8, 0.0, 0.05])
            self.assertIsNone(reason)
            gate.commit(observation)
            reason, _observation = gate.rejection([0.6, 0.0, 0.05], [1.0, 0.0, 0.05])
            self.assertTrue(str(reason).startswith("route_backward_"))

            status["updated_at"] = time.time()
            status["progress"]["translation_locked"] = True
            status["progress"]["position_anchor"] = [1.0, 0.0, 0.0]
            status_path.write_text(json.dumps(status), encoding="utf-8")
            reason, observation = gate.rejection([1.30, 0.0, 0.0], [1.0, 0.0, 0.0])
            self.assertIsNone(reason)
            self.assertAlmostEqual(observation["progress"], 0.5)
            self.assertTrue(observation["raw_turn_drift_anchored"])

    def test_new_leg_bootstraps_from_waypoint_not_false_previous_yaw_center(self):
        """Regress Live ATLAS 13:51:24 at the Point-2-to-3 transition."""
        point_two = [-0.6480244338911889, -0.0802621969998909, -0.48774887233005093]
        point_three = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        real_anchor = [-0.7812130265840018, -0.0802621969998909, -0.4797427590415436]
        false_yaw_center = [-0.75272386146415, -0.007409724988571967, -0.03116405433004954]
        context = {
            "start": point_two,
            "end": point_three,
            "anchor": real_anchor,
            "lap": 1,
            "leg_index": 2,
            "position_guard_locked": False,
        }
        gate = localizer.LivePatrolRouteGate(None, None)
        gate.active_context = lambda: context
        # The preceding leg really reached Point 2 at 94.85%. Its key and
        # progress must not be reused as leg-2 coordinates.
        gate.last_key = (1, 1, -3.2330, -0.3324, -0.6480, -0.4877)
        gate.last_progress = 0.948474997330772

        false_progress, _ = localizer.route_segment_projection_xz(
            false_yaw_center, point_two, point_three
        )
        self.assertAlmostEqual(false_progress, 0.304535, places=5)
        reason, observation = gate.rejection(real_anchor, false_yaw_center)

        self.assertIsNone(reason)
        self.assertAlmostEqual(observation["progress"], -0.004579, places=5)
        gate.commit(observation)
        self.assertEqual(gate.last_key, localizer.LivePatrolRouteGate._key(context))
        self.assertAlmostEqual(gate.last_progress, -0.004579, places=5)

    def test_patrol_route_key_stays_stable_across_recovery_status_updates(self):
        cruise = {
            "lap": 1,
            "leg_index": 3,
            "step_index": 15,
            "phase": "cruise",
            "start": [-0.49, -0.08, 0.96],
            "end": [-3.07, -0.08, 1.11],
        }
        recovery = {
            **cruise,
            "step_index": None,
            "phase": "pose_recovery",
        }
        next_lap = {**cruise, "lap": 2}
        next_leg = {**cruise, "leg_index": 4}

        self.assertEqual(
            localizer.LivePatrolRouteGate._key(cruise),
            localizer.LivePatrolRouteGate._key(recovery),
        )
        self.assertNotEqual(
            localizer.LivePatrolRouteGate._key(cruise),
            localizer.LivePatrolRouteGate._key(next_lap),
        )
        self.assertNotEqual(
            localizer.LivePatrolRouteGate._key(cruise),
            localizer.LivePatrolRouteGate._key(next_leg),
        )

    def test_yaw_raw_center_cannot_poison_route_floor_during_neutral_recovery(self):
        """Regress Live ATLAS 16:12:17's permanent Point-1-to-2 hold."""
        baseline = {
            "complete_loop": True,
            "enabled_for_live_route_gate": True,
            "map_id": "map_a",
            "patrol_id": "patrol_a",
            "legs": [
                {"from": [0.0, 0.0, 0.0], "to": [1.0, 0.0, 0.0]},
                {"from": [1.0, 0.0, 0.0], "to": [1.0, 0.0, 1.0]},
                {"from": [1.0, 0.0, 1.0], "to": [0.0, 0.0, 1.0]},
                {"from": [0.0, 0.0, 1.0], "to": [0.0, 0.0, 0.0]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            baseline_path = Path(td) / "baseline_a" / "reference_candidate.json"
            baseline_path.parent.mkdir()
            status_path = Path(td) / "control_status.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            status = {
                "status": "running",
                "command": "mission",
                "updated_at": time.time(),
                "progress": {
                    "map_id": "map_a",
                    "patrol_id": "patrol_a",
                    "baseline_replay_id": "baseline_a",
                    "lap": 1,
                    "segment_start": [0.0, 0.0, 0.0],
                    "target": [1.0, 0.0, 0.0],
                    "translation_locked": False,
                },
            }
            status_path.write_text(json.dumps(status), encoding="utf-8")
            gate = localizer.LivePatrolRouteGate(baseline_path, status_path)

            reason, observation = gate.rejection([0.90, 0.0, 0.0], [0.88, 0.0, 0.0])
            self.assertIsNone(reason)
            gate.commit(observation)

            # A commanded yaw publishes and commits the fixed 0.95 anchor even
            # though the solver's audit-only raw center drifts beyond Point 2.
            status["updated_at"] = time.time()
            status["progress"].update(
                {
                    "translation_locked": True,
                    "position_anchor": [0.95, 0.0, 0.0],
                }
            )
            status_path.write_text(json.dumps(status), encoding="utf-8")
            reason, observation = gate.rejection([1.012, 0.0, 0.0], [0.90, 0.0, 0.0])
            self.assertIsNone(reason)
            gate.commit(observation)
            self.assertAlmostEqual(gate.last_progress, 0.95)

            # Neutral online recovery releases the position lock. The first
            # healthy solve is 3.1 cm behind the guarded anchor, well inside the
            # tolerance. The old code compared it with the raw 1.012 center and
            # rejected every subsequent frame as route_backward_*.
            status["updated_at"] = time.time()
            status["progress"].update(
                {
                    "phase": "pose_recovery",
                    "route_visual_recovery_allowed": True,
                }
            )
            status_path.write_text(json.dumps(status), encoding="utf-8")
            reason, observation = gate.rejection(
                [0.919, 0.0, 0.0],
                [1.012, 0.0, 0.0],
            )
            self.assertIsNone(reason)
            # Physical translation is still locked during the neutral hover,
            # so route validation stays at the controller anchor while the
            # visual matcher remains enabled.  The rotation stabilizer can now
            # reconcile the bounded raw drift and publish an unlocked pose.
            self.assertAlmostEqual(observation["progress"], 0.95)
            self.assertAlmostEqual(gate.last_progress, 0.95)

    def test_metric_checkpoint_can_measure_point_four_without_unlocking_controller(self):
        """A neutral Point-4 checkpoint may update pose, never flight motion."""
        baseline = {
            "complete_loop": True,
            "enabled_for_live_route_gate": True,
            "map_id": "map_a",
            "patrol_id": "patrol_a",
            "legs": [
                {"from": [0.0, 0.0, 0.0], "to": [1.0, 0.0, 0.0]},
                {"from": [1.0, 0.0, 0.0], "to": [1.0, 0.0, 1.0]},
                {"from": [1.0, 0.0, 1.0], "to": [0.0, 0.0, 1.0]},
                {"from": [0.0, 0.0, 1.0], "to": [0.0, 0.0, 0.0]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            baseline_path = Path(td) / "baseline_a" / "reference_candidate.json"
            baseline_path.parent.mkdir()
            status_path = Path(td) / "control_status.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            status = {
                "status": "running",
                "command": "mission",
                "updated_at": time.time(),
                "progress": {
                    "map_id": "map_a",
                    "patrol_id": "patrol_a",
                    "baseline_replay_id": "baseline_a",
                    "lap": 1,
                    "segment_start": [0.0, 0.0, 1.0],
                    "target": [0.0, 0.0, 0.0],
                    "translation_locked": True,
                    "position_anchor": [0.0, 0.0, 1.0],
                    "phase": "point4_metric_handoff",
                    "route_visual_recovery_allowed": True,
                },
            }
            status_path.write_text(json.dumps(status), encoding="utf-8")
            gate = localizer.LivePatrolRouteGate(baseline_path, status_path)

            ordinary_recovery = gate.active_context()
            self.assertTrue(ordinary_recovery["controller_translation_locked"])
            self.assertTrue(ordinary_recovery["position_guard_locked"])
            self.assertFalse(ordinary_recovery["metric_position_recovery"])

            status["updated_at"] = time.time()
            status["progress"].update(
                {
                    "phase": "pose_recovery",
                    "post_translation_progress_recovery": True,
                    "route_progress_command_ceiling": 0.42,
                    "route_progress_command_sequence": 5,
                }
            )
            status_path.write_text(json.dumps(status), encoding="utf-8")
            post_translation_recovery = gate.active_context()

            self.assertTrue(
                post_translation_recovery["post_translation_progress_recovery"]
            )
            self.assertFalse(post_translation_recovery["position_guard_locked"])

            status["updated_at"] = time.time()
            status["progress"].update(
                {
                    "metric_position_recovery_allowed": True,
                    "require_metric_pose": True,
                }
            )
            status_path.write_text(json.dumps(status), encoding="utf-8")
            metric_checkpoint = gate.active_context()

        self.assertTrue(metric_checkpoint["controller_translation_locked"])
        self.assertFalse(metric_checkpoint["position_guard_locked"])
        self.assertTrue(metric_checkpoint["metric_position_recovery"])
        self.assertTrue(metric_checkpoint["require_metric_pose"])

    def test_post_translation_recovery_accepts_only_issued_route_budget(self):
        segment_length = 2.589597223809809
        floor = 0.1933441553905859
        ceiling = 0.36243454803963226
        context = {
            "start": [0.0, 0.0, 0.0],
            "end": [segment_length, 0.0, 0.0],
            "anchor": [floor * segment_length, 0.0, 0.0],
            "position_guard_locked": False,
            "post_translation_progress_recovery": True,
            "route_progress_command_ceiling": ceiling,
        }

        class RouteGate:
            enabled = True
            last_key = ("leg",)
            last_progress = floor

            def active_context(self):
                return context

            def rejection(self, candidate, _previous):
                progress, cross_track = localizer.route_segment_projection_xz(
                    candidate, context["start"], context["end"]
                )
                return None, {
                    "key": self.last_key,
                    "progress": progress,
                    "cross_track": cross_track,
                    "context": context,
                }

        previous = np.asarray([floor * segment_length, 0.0, 0.0])

        def guarded(progress):
            candidate = np.asarray([progress * segment_length, 0.0, 0.0])
            return localizer.route_guarded_output_rejection(
                case={"time_sec": 10.1, "tracked_pool_points": 400},
                result={
                    "success": True,
                    "objective": 20.0,
                    "R": np.eye(3).tolist(),
                    "t": (-candidate).tolist(),
                },
                previous_center=previous,
                previous_time=10.0,
                previous_pose={"rcenter": previous.tolist()},
                route_gate=RouteGate(),
                room_transform=lambda center: center,
                output_center_bias=None,
                max_step=0.30,
                max_speed=0.85,
                objective_threshold=30.0,
            )

        reason, accepted, _timestamp, observation = guarded(0.3465)
        self.assertIsNone(reason)
        self.assertTrue(np.allclose(accepted[0], 0.3465 * segment_length))
        self.assertEqual(
            observation["route_continuity_override_source"],
            "command_bounded_post_translation_recovery",
        )

        reason, _accepted, _timestamp, _observation = guarded(0.39)
        self.assertIsNotNone(reason)

    def test_live_163028_recovery_drift_reaches_rotation_reanchor(self):
        """Regress the Point-2 deadlock from Live ATLAS 16:30:28."""
        baseline = {
            "complete_loop": True,
            "enabled_for_live_route_gate": True,
            "map_id": "map_a",
            "patrol_id": "patrol_a",
            "legs": [
                {"from": [0.0, 0.0, 0.0], "to": [1.0, 0.0, 0.0]},
                {"from": [1.0, 0.0, 0.0], "to": [1.0, 0.0, 1.0]},
                {"from": [1.0, 0.0, 1.0], "to": [0.0, 0.0, 1.0]},
                {"from": [0.0, 0.0, 1.0], "to": [0.0, 0.0, 0.0]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            baseline_path = Path(td) / "baseline_a" / "reference_candidate.json"
            baseline_path.parent.mkdir()
            status_path = Path(td) / "control_status.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            status_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "command": "mission",
                        "updated_at": time.time(),
                        "progress": {
                            "map_id": "map_a",
                            "patrol_id": "patrol_a",
                            "baseline_replay_id": "baseline_a",
                            "lap": 1,
                            "segment_start": [0.0, 0.0, 0.0],
                            "target": [1.0, 0.0, 0.0],
                            "translation_locked": True,
                            "position_anchor": [0.976, 0.0, 0.0],
                            "phase": "pose_recovery",
                            "route_visual_recovery_allowed": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            gate = localizer.LivePatrolRouteGate(baseline_path, status_path)
            gate.last_key = (1, 1, 0.0, 0.0, 1.0, 0.0)
            gate.last_progress = 0.976

            reason, observation = gate.rejection(
                [0.868, 0.0, 0.592],
                [0.976, 0.0, 0.0],
            )

        self.assertIsNone(reason)
        self.assertAlmostEqual(observation["progress"], 0.976)
        self.assertFalse(observation["context"]["translation_locked"])
        self.assertTrue(observation["context"]["position_guard_locked"])

        stabilizer = localizer.RotationOnlyPositionStabilizer(enabled=True)
        anchor = [0.976, 0.0, 0.0]
        self.assertTrue(
            stabilizer.observe(
                delta_degrees=0.0,
                tracks=0,
                last_published_center=anchor,
                commanded_yaw_only=True,
                commanded_position_anchor=anchor,
            )
        )
        self.assertFalse(
            stabilizer.observe(
                delta_degrees=0.0,
                tracks=200,
                last_published_center=anchor,
                commanded_yaw_only=False,
            )
        )
        recovered = stabilizer.apply(
            {
                "success": True,
                "held_pose": False,
                "rcenter": [0.868, 0.0, 0.592],
            }
        )
        self.assertTrue(np.allclose(recovered["rcenter"], anchor))
        self.assertTrue(recovered["rotation_reanchored_after_turn"])
        self.assertNotIn("rotation_position_locked", recovered)

    def test_production_turn_gate_anchors_normal_raw_localizer_drift(self):
        baseline = {
            "complete_loop": True,
            "enabled_for_live_route_gate": True,
            "map_id": "map_a",
            "patrol_id": "patrol_a",
            "legs": [
                {"from": [0.0, 0.0, 0.0], "to": [2.0, 0.0, 0.0]},
                {"from": [2.0, 0.0, 0.0], "to": [2.0, 0.0, 2.0]},
                {"from": [2.0, 0.0, 2.0], "to": [0.0, 0.0, 2.0]},
                {"from": [0.0, 0.0, 2.0], "to": [0.0, 0.0, 0.0]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            baseline_path = Path(td) / "baseline_a" / "reference_candidate.json"
            baseline_path.parent.mkdir()
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            status_path = Path(td) / "control_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "command": "mission",
                        "updated_at": time.time(),
                        "progress": {
                            "map_id": "map_a",
                            "patrol_id": "patrol_a",
                            "baseline_replay_id": "baseline_a",
                            "lap": 1,
                            "step_index": 4,
                            "segment_start": [0.0, 0.0, 0.0],
                            "target": [2.0, 0.0, 0.0],
                            "translation_locked": True,
                            "position_anchor": [1.0, 0.0, 0.0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            gate = localizer.LivePatrolRouteGate(baseline_path, status_path)
            reason, observation = gate.rejection([1.20, 0.0, 0.0], [1.0, 0.0, 0.0])
            self.assertIsNone(reason)
            self.assertAlmostEqual(observation["progress"], 0.50)
            reason, observation = gate.rejection([1.80, 0.0, 0.0], [1.0, 0.0, 0.0])
            self.assertIsNone(reason)
            self.assertAlmostEqual(observation["progress"], 0.50)
            self.assertTrue(observation["raw_turn_drift_anchored"])

    def test_live_120857_point_two_false_turn_drift_stays_at_command_anchor(self):
        """Regress the 0.979 m Point-2 yaw drift from Live ATLAS 12:08:57."""
        point_two = [-0.6480244338911889, -0.0802621969998909, -0.48774887233005093]
        point_three = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        anchor = [-0.754698484909238, -0.0802621969998909, -0.48133657740202895]
        baseline = {
            "complete_loop": True,
            "enabled_for_live_route_gate": True,
            "map_id": "map_a",
            "patrol_id": "patrol_a",
            "legs": [
                {"from": point_two, "to": point_three},
                {"from": point_three, "to": [-3.07, point_two[1], 1.11]},
                {"from": [-3.07, point_two[1], 1.11], "to": [-3.23, point_two[1], -0.33]},
                {"from": [-3.23, point_two[1], -0.33], "to": point_two},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            baseline_path = Path(td) / "baseline_a" / "reference_candidate.json"
            baseline_path.parent.mkdir()
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            status_path = Path(td) / "control_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "command": "mission",
                        "updated_at": time.time(),
                        "progress": {
                            "map_id": "map_a",
                            "patrol_id": "patrol_a",
                            "baseline_replay_id": "baseline_a",
                            "lap": 1,
                            "segment_start": point_two,
                            "target": point_three,
                            "translation_locked": True,
                            "position_anchor": anchor,
                            "phase": "pose_recovery",
                            "route_visual_recovery_allowed": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            gate = localizer.LivePatrolRouteGate(
                baseline_path,
                status_path,
                turn_max_drift=0.75,
            )
            raw_false_center = [anchor[0], anchor[1], anchor[2] + 0.979]
            reason, observation = gate.rejection(raw_false_center, point_two)

        self.assertIsNone(reason)
        self.assertAlmostEqual(observation["raw_turn_drift_m"], 0.979)
        self.assertTrue(observation["raw_turn_drift_anchored"])
        expected = localizer.route_segment_projection_xz(anchor, point_two, point_three)
        self.assertAlmostEqual(observation["progress"], expected[0])

    def test_live_patrol_route_gate_allows_visual_recovery_during_neutral_hover(self):
        baseline = {
            "complete_loop": True,
            "enabled_for_live_route_gate": True,
            "map_id": "map_a",
            "patrol_id": "patrol_a",
            "legs": [
                {"from": [0.0, 0.0, 0.0], "to": [2.0, 0.0, 0.0]},
                {"from": [2.0, 0.0, 0.0], "to": [2.0, 0.0, 2.0]},
                {"from": [2.0, 0.0, 2.0], "to": [0.0, 0.0, 2.0]},
                {"from": [0.0, 0.0, 2.0], "to": [0.0, 0.0, 0.0]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            baseline_path = Path(td) / "baseline_a" / "reference_candidate.json"
            baseline_path.parent.mkdir()
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            status_path = Path(td) / "control_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "command": "mission",
                        "updated_at": time.time(),
                        "progress": {
                            "map_id": "map_a",
                            "patrol_id": "patrol_a",
                            "baseline_replay_id": "baseline_a",
                            "lap": 1,
                            "step_index": 3,
                            "segment_start": [2.0, 0.0, 2.0],
                            "target": [0.0, 0.0, 2.0],
                            "translation_locked": True,
                            "position_anchor": [1.5, 0.0, 2.0],
                            "phase": "taught_turn_wait_fresh_position",
                            "route_visual_recovery_allowed": True,
                            "route_progress_command_ceiling": 0.0,
                            "route_progress_command_sequence": 0,
                            "route_progress_command_budget_m": 0.18,
                        },
                    }
                ),
                encoding="utf-8",
            )
            gate = localizer.LivePatrolRouteGate(baseline_path, status_path)
            context = gate.active_context()

        self.assertIsNotNone(context)
        self.assertTrue(context["controller_translation_locked"])
        self.assertTrue(context["recovery_hover"])
        self.assertFalse(context["translation_locked"])
        self.assertTrue(context["position_guard_locked"])
        self.assertEqual(context["route_progress_command_ceiling"], 0.0)
        self.assertEqual(context["route_progress_command_sequence"], 0)
        self.assertEqual(context["route_progress_command_budget_m"], 0.18)

    def test_visual_route_advances_only_during_explicit_physical_translation(self):
        baseline = {
            "complete_loop": True,
            "enabled_for_live_route_gate": True,
            "map_id": "map_a",
            "patrol_id": "patrol_a",
            "legs": [
                {"from": [0.0, 0.0, 0.0], "to": [1.0, 0.0, 0.0]},
                {"from": [1.0, 0.0, 0.0], "to": [1.0, 0.0, 1.0]},
                {"from": [1.0, 0.0, 1.0], "to": [0.0, 0.0, 1.0]},
                {"from": [0.0, 0.0, 1.0], "to": [0.0, 0.0, 0.0]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            baseline_path = Path(td) / "baseline_a" / "reference_candidate.json"
            baseline_path.parent.mkdir()
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            status_path = Path(td) / "control_status.json"
            status = {
                "status": "running",
                "command": "mission",
                "updated_at": time.time(),
                "progress": {
                    "map_id": "map_a",
                    "patrol_id": "patrol_a",
                    "baseline_replay_id": "baseline_a",
                    "lap": 1,
                    "segment_start": [1.0, 0.0, 1.0],
                    "target": [0.0, 0.0, 1.0],
                    # The metric rotation anchor is intentionally released,
                    # but the aircraft is still neutral.
                    "translation_locked": False,
                    "phase": "pre_yaw_translation_settle",
                    "physical_translation_active": False,
                    "body_forward_gain": 0.0,
                },
            }
            status_path.write_text(json.dumps(status), encoding="utf-8")
            gate = localizer.LivePatrolRouteGate(baseline_path, status_path)
            neutral = gate.active_context()

            status["updated_at"] = time.time()
            status["progress"].update(
                {
                    "phase": "translation_command_active",
                    "physical_translation_active": True,
                    "body_forward_gain": 0.7,
                }
            )
            status_path.write_text(json.dumps(status), encoding="utf-8")
            translating = gate.active_context()

        self.assertTrue(neutral["translation_locked"])
        self.assertFalse(neutral["position_guard_locked"])
        self.assertFalse(neutral["physical_translation_active"])
        self.assertFalse(translating["translation_locked"])
        self.assertTrue(translating["physical_translation_active"])

    def test_explicit_endpoint_reverse_can_lower_only_an_overrun_route_floor(self):
        context = {
            "start": [0.0, 0.0, 0.0],
            "end": [2.0, 0.0, 0.0],
            "anchor": [0.0, 0.0, 0.0],
            "lap": 1,
            "leg_index": 1,
            "position_guard_locked": False,
            "controller_translation_locked": False,
            "endpoint_overshoot_correction": True,
        }
        gate = localizer.LivePatrolRouteGate(None, None)
        gate.active_context = lambda: context
        gate.last_key = gate._key(context)
        gate.last_progress = 1.062

        reason, observation = gate.rejection(
            [2.0, 0.0, 0.0],
            [2.124, 0.0, 0.0],
        )
        self.assertIsNone(reason)
        self.assertTrue(observation["endpoint_overshoot_rollback"])
        gate.commit(observation)
        self.assertAlmostEqual(gate.last_progress, 1.0)

        gate.last_progress = 1.062
        context["endpoint_overshoot_correction"] = False
        reason, _observation = gate.rejection(
            [2.0, 0.0, 0.0],
            [2.124, 0.0, 0.0],
        )
        self.assertTrue(str(reason).startswith("route_backward_"))

    def test_case_metadata_keeps_colmap_anchor_when_pool_pose_is_reduced(self):
        with tempfile.TemporaryDirectory() as td:
            case_dir = Path(td)
            (case_dir / "input.json").write_text(
                json.dumps(
                    {
                        "colmap_qvec_world_to_camera": [1.0, 0.0, 0.0, 0.0],
                        "colmap_tvec_world_to_camera": [1.5, -2.0, 3.25],
                        "colmap_registered_points": 268,
                    }
                ),
                encoding="utf-8",
            )
            center = localizer.pose_reference_center_from_case({"case_dir": str(case_dir)})

        self.assertTrue(np.allclose(center, [-1.5, 2.0, -3.25]))

    def test_saved_alignment_overrides_scene_derived_frame(self):
        alignment = {
            "matrix": [
                [0.0, 0.0, 1.0, 4.0],
                [0.0, 1.0, 0.0, -2.0],
                [-1.0, 0.0, 0.0, 7.0],
            ]
        }
        transform = localizer.build_room_transform(None, -1.0, alignment)
        self.assertIsNotNone(transform)
        self.assertEqual(transform([1.0, 2.0, 3.0]), [7.0, 0.0, 6.0])
        self.assertEqual(transform.direction([1.0, 2.0, 3.0]), [3.0, 2.0, -1.0])

    def test_alignment_json_accepts_manifest_shaped_payload(self):
        alignment = {
            "matrix": [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
            ]
        }
        parsed = localizer.parse_room_alignment_json(json.dumps({"room_alignment": alignment}))
        self.assertEqual(parsed, alignment)

    def test_server_passes_fixed_alignment_to_live_and_uploaded_localizers(self):
        source = (SCRIPTS / "atlas_app_server.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count('"--room-alignment-json"'), 2)
        self.assertGreaterEqual(
            source.count('json.dumps(selected.get("room_alignment") or {})'),
            2,
        )

    def test_camera_path_output_bias_keeps_published_pose_consistent(self):
        class IdentityRoomTransform:
            def __call__(self, center):
                return list(center)

            def direction(self, direction):
                return list(direction)

        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory)
            (case_dir / "input.json").write_text(
                json.dumps({"time_sec": 1.25, "image_name": "frame_000001.jpg"}),
                encoding="utf-8",
            )
            pose = localizer.partial_pose_from_result(
                {"case_id": "instance_001", "case_dir": str(case_dir)},
                {
                    "success": True,
                    "R": np.eye(3).tolist(),
                    "t": [[-1.0, -2.0, -3.0]],
                },
                room_transform=IdentityRoomTransform(),
                output_center_bias=np.asarray([-0.10, 0.0, 0.20]),
            )

        self.assertTrue(pose["success"])
        self.assertTrue(np.allclose(pose["uncalibrated_center"], [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(pose["output_center_bias"], [-0.10, 0.0, 0.20]))
        self.assertTrue(np.allclose(pose["center"], [0.90, 2.0, 3.20]))
        self.assertTrue(np.allclose(pose["t"], [-0.90, -2.0, -3.20]))
        self.assertTrue(np.allclose(pose["rcenter"], pose["center"]))


if __name__ == "__main__":
    unittest.main()

import json
import math
import sys
import unittest
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from patrol_visual_route_recovery import (  # noqa: E402
    PatrolVisualRouteRecovery,
    conservative_candidate_progress,
    horizontal_distance,
    independent_endpoint_candidate_progress,
    sequence_candidate_progress,
    segment_progress,
    verified_recovery_rewind_candidate,
    weak_endpoint_candidate_progress,
)


class PatrolVisualRouteRecoveryTests(unittest.TestCase):
    def test_segment_progress_uses_room_xz_plane(self):
        self.assertAlmostEqual(
            segment_progress([1.0, 99.0, 0.0], [0.0, 0.0, 0.0], [2.0, -4.0, 0.0]),
            0.5,
        )

    def test_horizontal_distance_ignores_height(self):
        self.assertAlmostEqual(horizontal_distance([0, 10, 0], [3, -10, 4]), 5.0)

    def test_candidate_progress_is_conservative_and_monotonic(self):
        candidates = [
            {"progress": 0.32, "inliers": 900},
            {"progress": 0.20, "inliers": 880},
            {"progress": 0.07, "inliers": 100},
        ]
        self.assertAlmostEqual(
            conservative_candidate_progress(candidates, previous=0.12),
            0.20,
        )
        self.assertAlmostEqual(
            conservative_candidate_progress(candidates, previous=0.24),
            0.24,
        )

    def test_sequence_progress_advances_locally_and_ignores_far_alias(self):
        candidates = [
            {"progress": 0.04, "inliers": 150},
            {"progress": 0.08, "inliers": 158},
            {"progress": 0.11, "inliers": 160},
            {"progress": 0.89, "inliers": 180},
        ]
        selected = sequence_candidate_progress(candidates, previous=0.04)
        self.assertIsNotNone(selected)
        self.assertGreater(selected, 0.04)
        self.assertLessEqual(selected, 0.16)

    def test_recovery_rewind_prefers_same_depth_earlier_view(self):
        candidates = [
            {
                "progress": 0.95,
                "inliers": 125,
                "endpoint_view_geometry_verified": False,
            },
            {
                "progress": 0.66,
                "inliers": 120,
                "endpoint_view_geometry_verified": True,
            },
            {
                "progress": 0.54,
                "inliers": 110,
                "endpoint_view_geometry_verified": True,
            },
        ]
        selected = verified_recovery_rewind_candidate(
            candidates,
            previous=0.90,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["progress"], 0.66)
        self.assertIsNone(
            verified_recovery_rewind_candidate(
                [{**candidates[1], "endpoint_view_geometry_verified": False}],
                previous=0.90,
            )
        )

    def test_103815_point_four_alias_rewinds_to_real_midleg_view(self):
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_precision_20260813"
            / "visual_route_recovery_multirun.npz"
        )
        frame = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260817_103815_fa8c57"
            / "query_frames"
            / "query_002438.jpg"
        )
        if not bank.exists() or not frame.exists():
            self.skipTest("10:38 false Point-4 endpoint frames are unavailable")

        recovery = PatrolVisualRouteRecovery(bank)
        key = (1, 3)
        recovery.active_key = key
        recovery.last_progress = 0.90
        recovery.last_matched_progress = 0.90
        recovery.last_matched_source_frame = 2018
        recovery.last_sequence_index = 2437
        recovery.active_source_replay_id = (
            "dji_live_20260811_115736_2b91ca"
        )
        recovery.needs_acquisition = False
        gray = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
        observations = []
        for offset in range(12):
            observation, _diagnostic = recovery.recover(
                gray=gray,
                segment_start=[
                    -0.4886978074319452,
                    -0.0802621969998909,
                    0.9560112230666532,
                ],
                segment_end=[
                    -3.0736291183109774,
                    -0.0802621969998909,
                    1.1113942967930859,
                ],
                segment_key=key,
                progress_hint=0.90,
                progress_ceiling=1.0,
                recovery_hover=True,
                independent_progress=True,
                sequence_index=2438 + offset,
            )
            if observation is not None:
                observations.append(observation)

        rewinds = [
            item for item in observations if item.get("verified_rewind") is True
        ]
        self.assertGreaterEqual(len(rewinds), 2)
        self.assertGreaterEqual(rewinds[-1]["verified_rewind_hits"], 5)
        self.assertGreaterEqual(rewinds[-1]["verified_rewind_inliers"], 120)
        self.assertGreater(rewinds[-1]["progress"], 0.50)
        self.assertLess(rewinds[-1]["progress"], 0.75)
        self.assertLess(rewinds[-1]["progress"], 0.90)

    def test_latest_point_two_to_three_frames_track_forward_without_alias_jump(self):
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_20260809_154714_d26c33"
            / "visual_route_recovery.npz"
        )
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260811_130746_3810d2"
            / "query_frames"
        )
        required = [bank] + [
            frames / f"query_{index:06d}.jpg" for index in range(1427, 1445)
        ]
        if not all(path.exists() for path in required):
            self.skipTest("latest Point-2-to-3 regression frames are unavailable")

        recovery = PatrolVisualRouteRecovery(bank)
        recovery.active_key = (1, 2)
        recovery.last_progress = 0.042
        recovery.needs_acquisition = False
        published_progress = 0.042
        observations = 0
        for image_path in required[1:]:
            observation, _stage = recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=[-0.6480244339, -0.080262197, -0.4877488723],
                segment_end=[-0.4886978074, -0.080262197, 0.9560112231],
                segment_key=(1, 2),
                progress_hint=published_progress,
            )
            if observation is None:
                continue
            observations += 1
            published_progress = min(
                float(observation["progress"]),
                published_progress + 0.025,
            )
            recovery.commit_published_progress(published_progress)

        self.assertGreaterEqual(observations, 12)
        self.assertGreater(published_progress, 0.25)
        self.assertLess(published_progress, 0.65)

    def test_current_tsolve_progress_supersedes_stale_visual_progress(self):
        """Regress the false 0.085-vs-0.939 hold from Live ATLAS 13:23:59."""
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_20260809_154714_d26c33"
            / "visual_route_recovery.npz"
        )
        frame = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260811_132359_a58f04"
            / "query_frames"
            / "query_001409.jpg"
        )
        if not bank.exists() or not frame.exists():
            self.skipTest("saved stale-monitor regression assets are unavailable")

        recovery = PatrolVisualRouteRecovery(bank)
        recovery.active_key = (1, 1)
        recovery.last_progress = 0.016
        recovery.needs_acquisition = False
        observation, stage = recovery.recover(
            gray=cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE),
            segment_start=[-3.2329557448, -0.080262197, -0.3323657986],
            segment_end=[-0.6480244339, -0.080262197, -0.4877488723],
            segment_key=(1, 1),
            progress_hint=0.9393783720,
        )

        self.assertIsNotNone(observation, stage)
        self.assertAlmostEqual(observation["progress"], 0.9393783720)
        self.assertTrue(observation["endpoint_guarded"])
        self.assertFalse(observation["endpoint_verified"])

    def test_latest_point_three_hover_cannot_self_advance_without_command_budget(self):
        """Regress Live ATLAS 12:58:53's false 1.67 m hover translation."""
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_precision_20260813"
            / "visual_route_recovery_multirun.npz"
        )
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260816_125853_1b94db"
            / "query_frames"
        )
        indices = list(range(1877, 1924))
        required = [bank] + [
            frames / f"query_{frame_index:06d}.jpg"
            for frame_index in indices
        ]
        if not all(path.exists() for path in required):
            self.skipTest("latest Point-3 hover regression assets are unavailable")

        start = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        end = [-3.0736291183109774, -0.0802621969998909, 1.1113942967930859]
        recovery = PatrolVisualRouteRecovery(bank)
        accepted = []
        for frame_index, image_path in zip(indices, required[1:]):
            observation, _diagnostic = recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=start,
                segment_end=end,
                segment_key=("live_125853", 3, 4),
                progress_hint=0.0,
                progress_ceiling=0.0,
                recovery_hover=True,
                independent_progress=True,
                sequence_index=frame_index,
            )
            if observation is None:
                continue
            accepted.append(observation)
            self.assertAlmostEqual(float(observation["progress"]), 0.0)
            self.assertTrue(observation["command_progress_guarded"])
            self.assertGreater(float(observation["matched_progress"]), 0.0)
            recovery.commit_published_progress(observation["progress"])

        # The actual frames retain enough image evidence to reacquire heading,
        # but the model/matcher clock must remain exactly at Point 3 because no
        # forward or lateral command occurred in this interval.
        self.assertGreaterEqual(len(accepted), 8)
        self.assertAlmostEqual(float(recovery.last_matched_progress or 0.0), 0.0)
        self.assertAlmostEqual(float(recovery.last_progress or 0.0), 0.0)

    def test_one_forward_command_unlocks_only_its_bounded_route_distance(self):
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_precision_20260813"
            / "visual_route_recovery_multirun.npz"
        )
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260816_125853_1b94db"
            / "query_frames"
        )
        indices = list(range(1923, 1954))
        required = [bank] + [
            frames / f"query_{frame_index:06d}.jpg"
            for frame_index in indices
        ]
        if not all(path.exists() for path in required):
            self.skipTest("latest Point-3 post-pulse regression assets are unavailable")

        start = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        end = [-3.0736291183109774, -0.0802621969998909, 1.1113942967930859]
        leg_length = math.hypot(end[0] - start[0], end[2] - start[2])
        command_ceiling = 0.18 / leg_length
        recovery = PatrolVisualRouteRecovery(bank)
        published = 0.0
        accepted = []
        for frame_index, image_path in zip(indices, required[1:]):
            observation, _diagnostic = recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=start,
                segment_end=end,
                segment_key=("live_125853_post_pulse", 3, 4),
                progress_hint=published,
                progress_ceiling=command_ceiling,
                independent_progress=True,
                sequence_index=frame_index,
            )
            if observation is None:
                continue
            published = float(observation["progress"])
            recovery.commit_published_progress(published)
            accepted.append(published)

        self.assertTrue(accepted)
        self.assertGreater(max(accepted), 0.015 / leg_length)
        self.assertLessEqual(max(accepted), command_ceiling + 1e-9)

    def test_latest_point_two_recovery_uses_all_three_command_budgets(self):
        """Regress Live ATLAS 13:22:38's permanent Point-2 -> Point-3 hold."""
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_precision_20260813"
            / "visual_route_recovery_multirun.npz"
        )
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260816_132238_4c29b5"
            / "query_frames"
        )
        indices = list(range(1698, 1901))
        required = [bank] + [
            frames / f"query_{frame_index:06d}.jpg"
            for frame_index in indices
        ]
        if not all(path.exists() for path in required):
            self.skipTest("latest Point-2 recovery regression assets are unavailable")

        start = [-0.6480244338911889, -0.0802621969998909, -0.48774887233005093]
        end = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        leg_length = math.hypot(end[0] - start[0], end[2] - start[2])
        one_command = 0.18 / leg_length
        recovery = PatrolVisualRouteRecovery(bank)
        published = 0.0
        observations = []
        for frame_index, image_path in zip(indices, required[1:]):
            if frame_index <= 1750:
                command_ceiling = 0.0
            elif frame_index <= 1761:
                command_ceiling = one_command
            elif frame_index <= 1784:
                command_ceiling = 2.0 * one_command
            else:
                command_ceiling = 3.0 * one_command
            observation, _diagnostic = recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=start,
                segment_end=end,
                segment_key=("live_132238", 2, 3),
                progress_hint=published,
                progress_ceiling=command_ceiling,
                recovery_hover=True,
                independent_progress=False,
                sequence_index=frame_index,
            )
            if observation is None:
                continue
            published = float(observation["progress"])
            recovery.commit_published_progress(published)
            observations.append(observation)
            self.assertLessEqual(published, command_ceiling + 1e-9)

        temporal = [
            observation
            for observation in observations
            if observation.get("temporal_recovery") is True
        ]
        self.assertGreaterEqual(len(temporal), 20)
        self.assertTrue(
            all(
                int(observation["temporal_recovery_hits"]) >= 5
                and int(observation["inliers"]) >= 90
                for observation in temporal
            )
        )
        self.assertAlmostEqual(published, 3.0 * one_command)
        self.assertAlmostEqual(published * leg_length, 0.54)

    def test_verified_endpoint_stays_latched_until_the_leg_changes(self):
        recovery = object.__new__(PatrolVisualRouteRecovery)
        recovery.minimum_inliers = 120
        recovery.endpoint_candidate_progress = None
        recovery.endpoint_hits = 0
        recovery.endpoint_required_hits = 3
        recovery.endpoint_verified = False
        endpoint_candidates = [
            {"progress": 0.95, "inliers": 240, "anchor_name": "endpoint_a"},
            {"progress": 0.97, "inliers": 220, "anchor_name": "endpoint_b"},
            {"progress": 0.99, "inliers": 205, "anchor_name": "endpoint_c"},
            {"progress": 0.55, "inliers": 150, "anchor_name": "leg_alias"},
        ]

        for _ in range(3):
            metadata = recovery._update_endpoint_verification(endpoint_candidates)

        self.assertTrue(metadata["endpoint_verified"])
        self.assertEqual(metadata["endpoint_hits"], 3)

        # One blurred/aliased forward-pulse frame cannot undo a completed
        # whole-leg proof and send route progress back behind the aircraft.
        metadata = recovery._update_endpoint_verification([])
        self.assertTrue(metadata["endpoint_verified"])
        self.assertEqual(metadata["endpoint_hits"], 3)

    def test_weak_endpoint_requires_multiple_endpoint_supporters(self):
        candidates = [
            {"progress": 0.99, "inliers": 82},
            {"progress": 0.98, "inliers": 77},
            {"progress": 0.96, "inliers": 73},
            {"progress": 0.61, "inliers": 70},
        ]
        self.assertAlmostEqual(
            weak_endpoint_candidate_progress(candidates),
            0.96,
        )

    def test_independent_endpoint_requires_global_endpoint_winner(self):
        endpoint = [
            {"progress": 0.96, "inliers": 208},
            {"progress": 0.98, "inliers": 206},
            {"progress": 1.00, "inliers": 184},
            {"progress": 0.45, "inliers": 176},
        ]
        false_arrival = [
            {"progress": 0.75, "inliers": 177},
            {"progress": 0.45, "inliers": 176},
            {"progress": 0.97, "inliers": 174},
            {"progress": 0.99, "inliers": 170},
        ]
        self.assertAlmostEqual(
            independent_endpoint_candidate_progress(endpoint), 0.96
        )
        self.assertIsNone(
            independent_endpoint_candidate_progress(false_arrival)
        )
        low_light_endpoint = [
            {"progress": 0.989, "inliers": 99},
            {"progress": 0.968, "inliers": 90},
            {"progress": 0.935, "inliers": 88},
            {"progress": 0.317, "inliers": 93},
        ]
        self.assertAlmostEqual(
            independent_endpoint_candidate_progress(
                low_light_endpoint,
                minimum_inliers=75,
            ),
            0.935,
        )

    def test_endpoint_rejects_strong_same_wall_match_from_wrong_depth(self):
        candidates = [
            {
                "progress": 1.00,
                "inliers": 160,
                "endpoint_view_geometry_verified": False,
                "endpoint_view_scale_min": 0.66,
                "endpoint_view_scale_max": 0.70,
            },
            {
                "progress": 0.99,
                "inliers": 151,
                "endpoint_view_geometry_verified": False,
                "endpoint_view_scale_min": 0.65,
                "endpoint_view_scale_max": 0.69,
            },
            {"progress": 0.72, "inliers": 130},
        ]

        self.assertIsNone(independent_endpoint_candidate_progress(candidates))

    def test_latest_point_four_false_arrival_stays_before_endpoint(self):
        """Regress the 09:49 cabinet alias accepted while 1/4 leg remained."""
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_precision_20260813"
            / "visual_route_recovery_multirun.npz"
        )
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260817_094913_5604e4"
            / "query_frames"
        )
        indices = list(range(2175, 2187))
        required = [bank] + [
            frames / f"query_{frame_index:06d}.jpg"
            for frame_index in indices
        ]
        if not all(path.exists() for path in required):
            self.skipTest("latest Point-4 false-arrival assets are unavailable")

        start = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        end = [-3.0736291183109774, -0.0802621969998909, 1.1113942967930859]
        recovery = PatrolVisualRouteRecovery(bank)
        recovery.active_key = (1, 3)
        recovery.last_progress = 0.903545324519272
        # Recreate the exact pre-handoff tracker state: the route matcher had
        # already walked to the final anchors even though the published model
        # was still near 90% and the physical aircraft was farther behind.
        recovery.last_matched_progress = 0.9997619680666205
        recovery.last_matched_source_frame = 2014
        recovery.last_sequence_index = indices[0] - 1
        recovery.active_source_replay_id = "dji_live_20260811_115736_2b91ca"
        recovery.needs_acquisition = False
        observations = []
        for frame_index, image_path in zip(indices, required[1:]):
            observation, _diagnostic = recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=start,
                segment_end=end,
                segment_key=(1, 3),
                progress_hint=0.903545324519272,
                progress_ceiling=1.0,
                independent_progress=True,
                sequence_index=frame_index,
            )
            if observation is None:
                continue
            observations.append(observation)
            recovery.commit_published_progress(observation["progress"])

        self.assertTrue(observations)
        self.assertFalse(any(item["endpoint_verified"] for item in observations))
        self.assertTrue(observations[-1]["endpoint_guarded"])
        self.assertLessEqual(
            float(observations[-1]["progress"]),
            0.903545324519272 + 1.0e-9,
        )

    def test_landed_live_point_two_endpoint_passes_low_light_consensus(self):
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_20260809_154714_d26c33"
            / "visual_route_recovery.npz"
        )
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260811_143856_f3a6bd"
            / "query_frames"
        )
        required = [bank] + [
            frames / f"query_{index:06d}.jpg" for index in range(1269, 1277)
        ]
        if not all(path.exists() for path in required):
            self.skipTest("landed live Point-2 endpoint frames are unavailable")

        recovery = PatrolVisualRouteRecovery(bank)
        recovery.active_key = (1, 1)
        recovery.last_progress = 0.80
        recovery.needs_acquisition = False
        observation = None
        for image_path in required[1:]:
            observation, stage = recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=[-3.2329557448, -0.080262197, -0.3323657986],
                segment_end=[-0.6480244339, -0.080262197, -0.4877488723],
                segment_key=(1, 1),
                progress_hint=0.80,
                recovery_hover=True,
            )
            if observation is not None:
                recovery.commit_published_progress(observation["progress"])

        self.assertIsNotNone(observation, stage)
        self.assertTrue(observation["endpoint_verified"])
        self.assertEqual(observation["endpoint_minimum_inliers"], 75)
        self.assertLess(observation["endpoint_best_inliers"], 120)
        self.assertGreater(observation["progress"], 0.90)

    def test_1633_point_two_frames_recover_the_verified_endpoint_after_metric_overrun(self):
        """Regress the live 1.079 floor that held 330 correct frames."""
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_20260809_154714_d26c33"
            / "visual_route_recovery.npz"
        )
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260811_163315_1fd889"
            / "query_frames"
        )
        required = [bank] + [
            frames / f"query_{index:06d}.jpg" for index in range(1596, 1604)
        ]
        if not all(path.exists() for path in required):
            self.skipTest("16:33 Point-2 endpoint regression frames are unavailable")

        recovery = PatrolVisualRouteRecovery(bank)
        recovery.active_key = (1, 1)
        recovery.last_progress = 1.0
        recovery.needs_acquisition = False
        observation = None
        for image_path in required[1:]:
            candidate, stage = recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=[-3.2329557448, -0.080262197, -0.3323657986],
                segment_end=[-0.6480244339, -0.080262197, -0.4877488723],
                segment_key=(1, 1),
                progress_hint=1.0793975401231017,
                recovery_hover=True,
            )
            if candidate is not None:
                observation = candidate
                recovery.commit_published_progress(candidate["progress"])

        self.assertIsNotNone(observation, stage)
        self.assertTrue(observation["endpoint_verified"])
        self.assertGreaterEqual(observation["inliers"], 120)
        self.assertGreaterEqual(observation["progress"], 0.95)

    def test_saved_success_endpoints_pass_and_false_arrivals_stay_guarded(self):
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_20260809_154714_d26c33"
            / "visual_route_recovery.npz"
        )
        sessions = {
            "success": (
                "atlas_dji_live_20260811_115736_2b91ca",
                3,
                range(2030, 2033),
            ),
            "false": (
                "atlas_dji_live_20260811_140849_2f60f0",
                3,
                range(1662, 1665),
            ),
        }
        required = [bank]
        for session, _leg, indices in sessions.values():
            required.extend(
                public
                / "live_dji_sessions"
                / session
                / "query_frames"
                / f"query_{index:06d}.jpg"
                for index in indices
            )
        if not all(path.exists() for path in required):
            self.skipTest("saved endpoint regression assets are unavailable")

        segment_start = [-0.4886978074, -0.080262197, 0.9560112231]
        segment_end = [-3.0736291183, -0.080262197, 1.1113942968]
        outcomes = {}
        for label, (session, leg, indices) in sessions.items():
            recovery = PatrolVisualRouteRecovery(bank)
            recovery.active_key = (1, leg)
            recovery.last_progress = 0.80
            recovery.needs_acquisition = False
            observation = None
            for index in indices:
                observation, stage = recovery.recover(
                    gray=cv2.imread(
                        str(
                            public
                            / "live_dji_sessions"
                            / session
                            / "query_frames"
                            / f"query_{index:06d}.jpg"
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    ),
                    segment_start=segment_start,
                    segment_end=segment_end,
                    segment_key=(1, leg),
                    progress_hint=0.80,
                )
                self.assertIsNotNone(observation, stage)
                recovery.commit_published_progress(observation["progress"])
            outcomes[label] = observation

        self.assertTrue(outcomes["success"]["endpoint_verified"])
        self.assertGreater(outcomes["success"]["progress"], 0.90)
        self.assertFalse(outcomes["false"]["endpoint_verified"])
        self.assertTrue(outcomes["false"]["endpoint_guarded"])
        self.assertAlmostEqual(outcomes["false"]["progress"], 0.84)

    def test_endpoint_consensus_resets_between_laps(self):
        recovery = object.__new__(PatrolVisualRouteRecovery)
        recovery.active_key = (1, 3)
        recovery.last_progress = 0.95
        recovery.pending_progress = 0.95
        recovery.pending_hits = 2
        recovery.needs_acquisition = False
        recovery.weak_endpoint_progress = None
        recovery.weak_endpoint_hits = 0
        recovery.endpoint_candidate_progress = 0.96
        recovery.endpoint_hits = 3
        recovery.endpoint_verified = True

        recovery._reset_for_key((2, 3))

        self.assertFalse(recovery.endpoint_verified)
        self.assertEqual(recovery.endpoint_hits, 0)
        self.assertIsNone(recovery.endpoint_candidate_progress)
        self.assertIsNone(recovery.last_progress)
        self.assertIsNone(
            weak_endpoint_candidate_progress(
                [{"progress": 0.99, "inliers": 82}, {"progress": 0.61, "inliers": 80}]
            )
        )

    def test_matcher_progress_changes_only_when_published_position_is_committed(self):
        recovery = object.__new__(PatrolVisualRouteRecovery)
        recovery.last_progress = 0.257
        recovery.commit_published_progress(0.277)
        self.assertAlmostEqual(recovery.last_progress, 0.277)
        recovery.commit_published_progress(0.266)
        self.assertAlmostEqual(recovery.last_progress, 0.266)

    def test_visual_loss_requires_fresh_acquisition_without_rewinding_progress(self):
        recovery = object.__new__(PatrolVisualRouteRecovery)
        recovery.last_progress = 0.62
        recovery.pending_progress = 0.66
        recovery.pending_hits = 1
        recovery.needs_acquisition = False

        recovery._mark_unverified()

        self.assertAlmostEqual(recovery.last_progress, 0.62)
        self.assertIsNone(recovery.pending_progress)
        self.assertEqual(recovery.pending_hits, 0)
        self.assertTrue(recovery.needs_acquisition)

    def test_point_three_failed_heading_is_rejected_and_taught_view_is_accepted(self):
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_20260809_154714_d26c33"
            / "visual_route_recovery.npz"
        )
        failed_frame = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260810_133246_d7a064"
            / "query_frames"
            / "query_001732.jpg"
        )
        taught_frame = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260809_154714_d26c33"
            / "query_frames"
            / "query_002901.jpg"
        )
        if not all(path.exists() for path in (bank, failed_frame, taught_frame)):
            self.skipTest("saved Point-3 heading regression assets are unavailable")

        recovery = PatrolVisualRouteRecovery(bank)
        segment_start = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        segment_end = [-3.0736291183109774, -0.0802621969998909, 1.1113942967930859]
        failed, _failed_stage = recovery.departure_heading_alignment(
            gray=cv2.imread(str(failed_frame), cv2.IMREAD_GRAYSCALE),
            segment_start=segment_start,
            segment_end=segment_end,
            focal_px=851.6865528775178,
        )
        taught, _taught_stage = recovery.departure_heading_alignment(
            gray=cv2.imread(str(taught_frame), cv2.IMREAD_GRAYSCALE),
            segment_start=segment_start,
            segment_end=segment_end,
            focal_px=851.6865528775178,
        )

        self.assertIsNotNone(failed)
        self.assertGreater(failed["inliers"], 120)
        self.assertGreater(failed["correction_deg"], 20.0)
        self.assertIsNotNone(taught)
        self.assertGreater(taught["inliers"], 120)
        self.assertLess(abs(taught["correction_deg"]), 1.0)

    def test_live_1109_point_two_absolute_heading_detects_alignment_and_overshoot(self):
        """Regress the physical 2->3 over-rotation hidden by optical yaw."""
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_precision_20260813"
            / "visual_route_recovery.npz"
        )
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260816_110954_b9d9d1"
            / "query_frames"
        )
        aligned_frame = frames / "query_001480.jpg"
        overshot_frame = frames / "query_001540.jpg"
        if not all(path.exists() for path in (bank, aligned_frame, overshot_frame)):
            self.skipTest("saved Point-2 heading regression assets are unavailable")

        recovery = PatrolVisualRouteRecovery(bank)
        segment_start = [-0.6480244338911889, -0.0802621969998909, -0.48774887233005093]
        segment_end = [-0.4886978074319452, -0.0802621969998909, 0.9560112230666532]
        aligned, _aligned_stage = recovery.departure_heading_alignment(
            gray=cv2.imread(str(aligned_frame), cv2.IMREAD_GRAYSCALE),
            segment_start=segment_start,
            segment_end=segment_end,
            focal_px=851.6865528775178,
            minimum_inliers=120,
        )
        overshot, _overshot_stage = recovery.departure_heading_alignment(
            gray=cv2.imread(str(overshot_frame), cv2.IMREAD_GRAYSCALE),
            segment_start=segment_start,
            segment_end=segment_end,
            focal_px=851.6865528775178,
            minimum_inliers=120,
        )

        self.assertIsNotNone(aligned)
        self.assertGreater(aligned["inliers"], 200)
        self.assertLess(abs(aligned["correction_deg"]), 1.0)
        self.assertIsNotNone(overshot)
        self.assertGreater(overshot["inliers"], 140)
        self.assertLess(overshot["correction_deg"], -15.0)

    def test_point_four_correct_point_one_view_reacquires_and_old_wrong_view_does_not(self):
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_20260809_154714_d26c33"
            / "visual_route_recovery.npz"
        )
        latest_frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260810_141441_b89541"
            / "query_frames"
        )
        baseline_frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260809_154714_d26c33"
            / "query_frames"
        )
        required = [
            bank,
            latest_frames / "query_005252.jpg",
            latest_frames / "query_005253.jpg",
            baseline_frames / "query_004551.jpg",
            baseline_frames / "query_004552.jpg",
        ]
        if not all(path.exists() for path in required):
            self.skipTest("saved Point-4-to-1 regression assets are unavailable")

        segment_start = [-3.0736291183109774, -0.0802621969998909, 1.1113942967930859]
        segment_end = [-3.2329557447702215, -0.0802621969998909, -0.33236579860361815]
        recovery = PatrolVisualRouteRecovery(bank)
        first, first_stage = recovery.recover(
            gray=cv2.imread(str(required[1]), cv2.IMREAD_GRAYSCALE),
            segment_start=segment_start,
            segment_end=segment_end,
            segment_key=("point4_to_point1",),
            translation_locked=False,
            progress_hint=0.0,
        )
        second, _second_stage = recovery.recover(
            gray=cv2.imread(str(required[2]), cv2.IMREAD_GRAYSCALE),
            segment_start=segment_start,
            segment_end=segment_end,
            segment_key=("point4_to_point1",),
            translation_locked=False,
            progress_hint=0.0,
        )

        self.assertIsNone(first)
        self.assertEqual(first_stage["reason"], "visual_route_acquiring")
        self.assertIsNotNone(second)
        self.assertGreaterEqual(second["inliers"], 120)
        self.assertLess(second["progress"], 0.01)
        self.assertLessEqual(second["source_frame"], 4270)

        wrong_recovery = PatrolVisualRouteRecovery(bank)
        wrong = None
        for image_path in required[3:]:
            wrong, _wrong_stage = wrong_recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=segment_start,
                segment_end=segment_end,
                segment_key=("point4_to_point1_wrong_view",),
                translation_locked=False,
                progress_hint=0.0,
            )
        self.assertIsNone(wrong)

    def test_live_0903_point_four_turn_has_three_safe_aligned_departure_frames(self):
        """Regress the 120-inlier gate that forced a correct 4->1 turn to abort."""
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_precision_20260813"
            / "visual_route_recovery.npz"
        )
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260817_090303_f65cee"
            / "query_frames"
        )
        wrong_frame = frames / "query_002700.jpg"
        aligned_frames = [
            frames / f"query_{index:06d}.jpg"
            for index in (2753, 2754, 2755)
        ]
        if not all(path.exists() for path in [bank, wrong_frame, *aligned_frames]):
            self.skipTest("09:03 Point-4 turn regression assets are unavailable")

        start = [-3.0736291183109774, -0.0802621969998909, 1.1113942967930859]
        end = [-3.2329557447702215, -0.0802621969998909, -0.33236579860361815]
        recovery = PatrolVisualRouteRecovery(bank)
        wrong, wrong_stage = recovery.departure_heading_alignment(
            gray=cv2.imread(str(wrong_frame), cv2.IMREAD_GRAYSCALE),
            segment_start=start,
            segment_end=end,
            focal_px=851.6865528775178,
            minimum_inliers=30,
        )
        aligned = [
            recovery.departure_heading_alignment(
                gray=cv2.imread(str(path), cv2.IMREAD_GRAYSCALE),
                segment_start=start,
                segment_end=end,
                focal_px=851.6865528775178,
                minimum_inliers=30,
            )[0]
            for path in aligned_frames
        ]

        self.assertIsNone(wrong)
        self.assertEqual(wrong_stage["reason"], "visual_heading_inliers_below_threshold")
        self.assertTrue(all(observation is not None for observation in aligned))
        self.assertTrue(
            all(observation["inliers"] >= 30 for observation in aligned)
        )
        self.assertTrue(
            all(abs(observation["correction_deg"]) <= 4.0 for observation in aligned)
        )

    def test_full_point_four_to_one_frames_do_not_walk_ahead_then_freeze(self):
        """Regress the 4326-anchor alias seen at live frame 4286."""
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        replay = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_20260809_154714_d26c33"
        )
        bank = replay / "visual_route_recovery.npz"
        frame_dir = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260809_154714_d26c33"
            / "query_frames"
        )
        required = [bank, replay / "poses.json"] + [
            frame_dir / f"query_{index:06d}.jpg"
            for index in range(4261, 4401)
        ]
        if not all(path.exists() for path in required):
            self.skipTest("full Point-4-to-1 regression assets are unavailable")

        baseline = json.loads((replay / "poses.json").read_text(encoding="utf-8"))
        truth = {
            int(pose["source_frame"]): float(pose["route_progress"])
            for pose in baseline["poses"]
            if pose.get("source_frame") is not None
        }
        start = [-3.0736291183, -0.080262197, 1.1113942968]
        end = [-3.2329557448, -0.080262197, -0.3323657986]
        leg_length = math.hypot(end[0] - start[0], end[2] - start[2])
        recovery = PatrolVisualRouteRecovery(bank)
        published = 0.0
        rows = []
        for index in range(4261, 4401):
            observation, _stage = recovery.recover(
                gray=cv2.imread(
                    str(frame_dir / f"query_{index:06d}.jpg"),
                    cv2.IMREAD_GRAYSCALE,
                ),
                segment_start=start,
                segment_end=end,
                segment_key=(1, 4),
                progress_hint=published,
                independent_progress=True,
                sequence_index=index,
            )
            if observation is not None:
                # Mirror the localizer's 5 cm per-frame route publication cap.
                published = min(
                    float(observation["progress"]),
                    published + 0.05 / leg_length,
                )
                recovery.commit_published_progress(published)
            rows.append((index, published, truth[index]))

        errors = [
            abs(published_progress - truth_progress) * leg_length
            for _index, published_progress, truth_progress in rows
        ]
        self.assertLess(max(errors), 0.08)
        self.assertAlmostEqual(rows[25][1], rows[25][2], places=3)
        self.assertAlmostEqual(rows[-1][1], 1.0, places=6)

        longest_freeze = 0
        freeze_start = 0
        for offset in range(1, len(rows) + 1):
            if (
                offset < len(rows)
                and abs(rows[offset][1] - rows[offset - 1][1]) < 1e-9
            ):
                continue
            physical_motion = (
                rows[offset - 1][2] - rows[freeze_start][2]
            ) * leg_length
            if physical_motion > 0.05:
                longest_freeze = max(longest_freeze, offset - freeze_start)
            freeze_start = offset
        self.assertLessEqual(longest_freeze, 6)

    def test_landed_point_three_frames_recover_endpoint_without_translation_authority(self):
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_20260809_154714_d26c33"
            / "visual_route_recovery.npz"
        )
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260811_103720_2a620c"
            / "query_frames"
        )
        required = [bank] + [frames / f"query_{index:06d}.jpg" for index in range(1795, 1860)]
        if not all(path.exists() for path in required):
            self.skipTest("landed Point-3 regression frames are unavailable")
        recovery = PatrolVisualRouteRecovery(bank)
        observation = None
        last_observation = None
        committed_progress = 0.257
        weak_observations = 0
        for image_path in required[1:]:
            observation, _stage = recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=[-0.6480244339, -0.080262197, -0.4877488723],
                segment_end=[-0.4886978074, -0.080262197, 0.9560112231],
                segment_key=(1, 2),
                progress_hint=0.257,
                recovery_hover=True,
            )
            if observation is None:
                continue
            last_observation = observation
            weak_observations += 1
            self.assertTrue(observation["weak_endpoint_recovery"])
            self.assertFalse(observation["translation_safe"])
            self.assertLess(observation["inliers"], observation["minimum_inliers"])
            committed_progress = min(
                float(observation["progress"]),
                committed_progress + 0.025,
            )
            recovery.commit_published_progress(committed_progress)
        self.assertGreaterEqual(weak_observations, 20)
        self.assertIsNotNone(last_observation)
        self.assertAlmostEqual(
            committed_progress,
            last_observation["endpoint_safe_prearrival_progress"],
        )
        self.assertLessEqual(committed_progress, 0.90)
        self.assertFalse(last_observation["endpoint_verified"])
        self.assertTrue(last_observation["endpoint_guarded"])

    def test_multirun_bank_tracks_the_short_independent_leg_without_freezing(self):
        """One safe metric pulse must fit the normalized short-leg window."""
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        replay = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_precision_20260813"
        )
        bank = replay / "visual_route_recovery_multirun.npz"
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260811_115736_2b91ca"
            / "query_frames"
        )
        indices = [*range(1121, 1180, 2), 1180]
        required = [bank] + [
            frames / f"query_{frame_index:06d}.jpg"
            for frame_index in indices
        ]
        if not all(path.exists() for path in required):
            self.skipTest("multi-run short-leg regression assets are unavailable")

        recovery = PatrolVisualRouteRecovery(bank)
        published = 0.0
        accepted = []
        rejections = []
        for frame_index, image_path in zip(indices, required[1:]):
            observation, diagnostic = recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=[-0.6480244339, 0.0100528347, -0.4877488723],
                segment_end=[-0.4886978074, 0.0100528347, 0.9560112231],
                segment_key=("independent_115736", 2, 3),
                progress_hint=published,
                independent_progress=True,
                sequence_index=frame_index,
            )
            if observation is None:
                rejections.append(str(diagnostic.get("reason") or "unknown"))
                continue
            self.assertGreaterEqual(float(observation["progress"]), published)
            self.assertEqual(
                observation["source_replay_id"],
                "dji_live_20260811_115736_2b91ca",
            )
            published = float(observation["progress"])
            recovery.commit_published_progress(published)
            accepted.append(published)

        self.assertEqual(rejections, ["visual_route_acquiring"])
        self.assertEqual(len(accepted), len(indices) - 1)
        self.assertGreaterEqual(published, 0.95)
        self.assertEqual(recovery.active_source_replay_id, "dji_live_20260811_115736_2b91ca")

if __name__ == "__main__":
    unittest.main()

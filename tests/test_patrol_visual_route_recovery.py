import json
import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from patrol_visual_route_recovery import (  # noqa: E402
    PatrolVisualRouteRecovery,
    command_bounded_recovery_candidate,
    conservative_candidate_progress,
    geometric_candidate_rank,
    horizontal_distance,
    independent_endpoint_candidate_progress,
    sequence_candidate_progress,
    segment_progress,
    verified_recovery_rewind_candidate,
    weak_endpoint_candidate_progress,
)


class PatrolVisualRouteRecoveryTests(unittest.TestCase):
    def test_query_orb_features_are_shared_for_same_live_frame(self):
        class CountingDetector:
            def __init__(self):
                self.calls = 0

            def detectAndCompute(self, image, mask):
                self.calls += 1
                return [object()], np.asarray([[self.calls]], dtype=np.uint8)

        recovery = object.__new__(PatrolVisualRouteRecovery)
        recovery.detector = CountingDetector()
        recovery._query_feature_image = None
        recovery._query_feature_keypoints = None
        recovery._query_feature_descriptors = None
        image = np.zeros((12, 12), dtype=np.uint8)

        first = recovery._extract_query_features(image)
        second = recovery._extract_query_features(image)
        third = recovery._extract_query_features(image.copy())

        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertEqual(recovery.detector.calls, 2)
        self.assertNotEqual(int(first[1][0, 0]), int(third[1][0, 0]))

    def test_command_bounded_recovery_can_see_past_one_step_window(self):
        candidates = [
            {"progress": 0.080452, "inliers": 42},
            {"progress": 0.382282, "inliers": 91},
            {"progress": 0.62, "inliers": 180},
        ]
        selected = command_bounded_recovery_candidate(
            candidates,
            previous=0.080452,
            command_progress_ceiling=0.371766,
            minimum_inliers=50,
        )
        self.assertIsNotNone(selected)
        self.assertAlmostEqual(float(selected["progress"]), 0.382282)

    def test_temporal_recovery_uses_fresh_rolling_consensus(self):
        recovery = object.__new__(PatrolVisualRouteRecovery)
        recovery.recovery_acquisition_hits = 5
        recovery.temporal_recovery_progress = None
        recovery.temporal_recovery_source_replay_id = None
        recovery.temporal_recovery_hits = 0
        recovery.temporal_recovery_samples = []
        recovery.temporal_recovery_command_ceiling = None

        result = None
        for progress in (0.0015, 0.0804, 0.0498, 0.0917, 0.1021):
            result = recovery._collect_temporal_recovery_progress(
                proposed=progress,
                source_replay_id="recorded_4_to_1",
                command_progress_ceiling=0.123922,
            )
        self.assertIsNotNone(result)
        self.assertGreater(float(result), 0.04)
        self.assertNotAlmostEqual(float(result), 0.0015)

    def test_last_live_point_four_to_one_frames_advance_frozen_model(self):
        """Regress the 23-Aug run that stayed at 8% after real movement."""
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
        bank = replay / "visual_route_recovery_manual_tail_point1_20260820.npz"
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260823_115821_8e4a29"
            / "query_frames"
        )
        indices = list(range(3825, 3835))
        required = [bank, replay / "reference_candidate.json"] + [
            frames / f"query_{index:06d}.jpg" for index in indices
        ]
        if not all(path.exists() for path in required):
            self.skipTest("latest live Point-4 -> Point-1 frames are unavailable")

        leg = json.loads(required[1].read_text(encoding="utf-8"))["legs"][3]
        recovery = PatrolVisualRouteRecovery(bank)
        segment_key = ("last-live-point4-to-point1",)
        frozen_progress = 0.08045224996831972
        command_ceiling = 0.37176646744829467
        recovery.active_key = segment_key
        recovery.last_matched_progress = frozen_progress
        recovery.last_progress = frozen_progress
        recovery.last_matched_source_frame = 2911
        recovery.last_sequence_index = indices[0] - 1
        recovery.active_source_replay_id = "dji_live_20260819_164415_524a50"
        recovery.needs_acquisition = False
        published = frozen_progress
        observations = []
        for index, image_path in zip(indices, required[2:]):
            observation, _diagnostic = recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=leg["from"],
                segment_end=leg["to"],
                segment_key=segment_key,
                progress_hint=published,
                progress_ceiling=command_ceiling,
                recovery_hover=True,
                recovery_minimum_inliers=50,
                independent_progress=True,
                allow_endpoint_only_recovery=True,
                sequence_index=index,
            )
            if observation is None:
                continue
            published = float(observation["progress"])
            recovery.commit_published_progress(published)
            observations.append(observation)

        self.assertTrue(observations)
        self.assertGreaterEqual(published, 0.22)
        self.assertLessEqual(published, command_ceiling + 1.0e-9)
        self.assertTrue(all(item["temporal_recovery"] for item in observations))

    def test_restored_point1_extension_is_active_and_valid(self):
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
        bank = replay / "visual_route_recovery_manual_tail_point1_20260820.npz"
        audit_path = (
            replay / "visual_route_recovery_manual_tail_point1_20260820_audit.json"
        )
        manual_frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260819_164415_524a50"
            / "query_frames"
        )
        point1_frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260820_121156_b6ef3c"
            / "query_frames"
        )
        point4_before_path = manual_frames / "query_002740.jpg"
        point4_aligned_path = manual_frames / "query_002886.jpg"
        point1_heading_path = manual_frames / "query_003037.jpg"
        point1_endpoint_paths = [
            point1_frames / f"query_{index:06d}.jpg"
            for index in (1652, 1657, 1662)
        ]
        required = [
            bank,
            audit_path,
            point4_before_path,
            point4_aligned_path,
            point1_heading_path,
            *point1_endpoint_paths,
        ]
        if not all(path.exists() for path in required):
            self.skipTest("latest Point-1 endpoint audit assets are unavailable")

        reference = json.loads(
            (replay / "reference_candidate.json").read_text(encoding="utf-8")
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["preservation"]["base_prefix_preserved"])
        self.assertFalse(audit["preservation"]["route_anchors_replaced"])
        self.assertEqual(audit["preservation"]["added_endpoint_anchor_count"], 11)
        self.assertFalse(audit["negative_endpoint_verified"])
        self.assertEqual(
            reference["visual_route_recovery_bank"],
            "visual_route_recovery_manual_tail_point1_20260820.npz",
        )

        with np.load(bank, allow_pickle=False) as contents:
            starts = np.asarray(contents["anchor_from"], dtype=float)
            ends = np.asarray(contents["anchor_to"], dtype=float)
            sources = [
                str(value)
                for value in contents["anchor_source_replay_ids"].tolist()
            ]
            priorities = np.asarray(
                contents["anchor_heading_priority"], dtype=int
            )
            source_frames = np.asarray(contents["source_frames"], dtype=int)
        leg4 = reference["legs"][3]
        leg4_mask = np.asarray(
            [
                horizontal_distance(start, leg4["from"]) <= 0.08
                and horizontal_distance(end, leg4["to"]) <= 0.08
                for start, end in zip(starts, ends)
            ]
        )
        self.assertEqual(
            {source for source, selected in zip(sources, leg4_mask) if selected},
            {
                "dji_live_20260819_164415_524a50",
                "dji_live_20260820_121156_b6ef3c",
            },
        )
        priority_frames = sorted(
            int(frame)
            for frame, source, priority in zip(
                source_frames, sources, priorities
            )
            if source == "dji_live_20260819_164415_524a50"
            and priority == 100
        )
        self.assertEqual(
            priority_frames,
            [2878, 2880, 2882, 2884, 2886, 2888, 2890, 3035, 3036, 3037, 3038, 3039],
        )

        recovery = PatrolVisualRouteRecovery(bank)
        point4_before, _ = recovery.departure_heading_alignment(
            gray=cv2.imread(str(point4_before_path), cv2.IMREAD_GRAYSCALE),
            segment_start=leg4["from"],
            segment_end=leg4["to"],
            focal_px=882.4866783165957,
            minimum_inliers=50,
        )
        point4_aligned, _ = recovery.departure_heading_alignment(
            gray=cv2.imread(str(point4_aligned_path), cv2.IMREAD_GRAYSCALE),
            segment_start=leg4["from"],
            segment_end=leg4["to"],
            focal_px=882.4866783165957,
            minimum_inliers=50,
        )
        point1_leg = reference["legs"][0]
        point1_aligned, _ = recovery.departure_heading_alignment(
            gray=cv2.imread(str(point1_heading_path), cv2.IMREAD_GRAYSCALE),
            segment_start=point1_leg["from"],
            segment_end=point1_leg["to"],
            focal_px=882.4866783165957,
            minimum_inliers=72,
        )
        self.assertIsNotNone(point4_before)
        self.assertLess(point4_before["correction_deg"], -15.0)
        self.assertIsNotNone(point4_aligned)
        self.assertLess(abs(point4_aligned["correction_deg"]), 1.0)
        self.assertIsNotNone(point1_aligned)
        self.assertLess(abs(point1_aligned["correction_deg"]), 1.0)

        endpoint_recovery = PatrolVisualRouteRecovery(bank)
        endpoint_observation = None
        for sequence_index, image_path in enumerate(point1_endpoint_paths, start=1):
            endpoint_observation, diagnostic = endpoint_recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=leg4["from"],
                segment_end=leg4["to"],
                segment_key=("stale-live-point1", 1, 4),
                translation_locked=False,
                progress_hint=0.080452,
                progress_ceiling=0.619611,
                recovery_hover=True,
                recovery_minimum_inliers=50,
                independent_progress=True,
                allow_endpoint_only_recovery=True,
                sequence_index=sequence_index,
            )
            if sequence_index < 3:
                self.assertIsNone(endpoint_observation)
                self.assertEqual(
                    diagnostic.get("reason"),
                    "visual_route_endpoint_only_acquiring",
                )
        self.assertIsNotNone(endpoint_observation)
        self.assertTrue(endpoint_observation["endpoint_verified"])
        self.assertTrue(endpoint_observation["command_progress_guarded"])
        self.assertAlmostEqual(endpoint_observation["progress"], 0.080452)
        self.assertGreaterEqual(endpoint_observation["endpoint_best_inliers"], 90)

    def test_complete_121156_single_run_remains_an_audited_candidate(self):
        public = ROOT / "viewer" / "public"
        replay = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_precision_20260813"
        )
        bank = replay / "visual_route_recovery_single_run_121156_20260820.npz"
        audit_path = (
            replay
            / "visual_route_recovery_single_run_121156_20260820_audit.json"
        )
        reference_path = replay / "reference_candidate.json"
        if not all(path.exists() for path in (bank, audit_path, reference_path)):
            self.skipTest("single-run patrol-bank assets are unavailable")

        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertNotEqual(reference["visual_route_recovery_bank"], bank.name)
        self.assertNotEqual(reference["visual_route_recovery_audit"], audit_path.name)
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["source_is_single"])
        self.assertTrue(all(lap["passed"] for lap in audit["laps"]))
        self.assertTrue(
            all(
                leg["passed"]
                for lap in audit["laps"]
                for leg in lap["legs"]
            )
        )

        with np.load(bank, allow_pickle=False) as contents:
            sources = {
                str(value)
                for value in contents["anchor_source_replay_ids"].tolist()
            }
            advertised_sources = [
                str(value) for value in contents["source_replay_ids"].tolist()
            ]
            progress = np.asarray(contents["anchor_progress"], dtype=float)
        self.assertEqual(sources, {"dji_live_20260820_121156_b6ef3c"})
        self.assertEqual(advertised_sources, ["dji_live_20260820_121156_b6ef3c"])
        self.assertTrue(np.all((0.0 <= progress) & (progress <= 1.0)))

    def test_completed_095043_tail_tracks_point_four_to_one_and_next_heading(self):
        """Keep the newly completed live/manual tail as a real-frame regression."""
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
        bank = replay / "visual_route_recovery_manual_tail.npz"
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260820_095043_85a3b7"
            / "query_frames"
        )
        indices = list(range(2910, 3046, 5))
        required = [bank, replay / "reference_candidate.json"] + [
            frames / f"query_{index:06d}.jpg"
            for index in [*indices, 3090, 3100]
        ]
        if not all(path.exists() for path in required):
            self.skipTest("09:50 completed patrol-tail frames are unavailable")

        legs = json.loads(required[1].read_text(encoding="utf-8"))["legs"]
        recovery = PatrolVisualRouteRecovery(bank)
        published = 0.0
        accepted = []
        for frame_index in indices:
            observation, diagnostic = recovery.recover(
                gray=cv2.imread(
                    str(frames / f"query_{frame_index:06d}.jpg"),
                    cv2.IMREAD_GRAYSCALE,
                ),
                segment_start=legs[3]["from"],
                segment_end=legs[3]["to"],
                segment_key=("completed-095043-tail", 4),
                translation_locked=False,
                progress_hint=published,
                independent_progress=True,
                sequence_index=frame_index,
            )
            if observation is None:
                self.assertEqual(diagnostic.get("reason"), "visual_route_acquiring")
                continue
            progress = float(observation["progress"])
            self.assertGreaterEqual(progress + 1.0e-9, published)
            published = progress
            recovery.commit_published_progress(progress)
            accepted.append(observation)

        self.assertGreaterEqual(len(accepted), len(indices) - 2)
        self.assertGreaterEqual(published, 0.99)
        self.assertTrue(accepted[-1].get("endpoint_verified"))
        self.assertGreaterEqual(min(int(item["inliers"]) for item in accepted), 120)

        aligned, _ = recovery.departure_heading_alignment(
            gray=cv2.imread(str(frames / "query_003090.jpg"), cv2.IMREAD_GRAYSCALE),
            segment_start=legs[0]["from"],
            segment_end=legs[0]["to"],
            focal_px=882.4866783165957,
            minimum_inliers=72,
        )
        overshot, _ = recovery.departure_heading_alignment(
            gray=cv2.imread(str(frames / "query_003100.jpg"), cv2.IMREAD_GRAYSCALE),
            segment_start=legs[0]["from"],
            segment_end=legs[0]["to"],
            focal_px=882.4866783165957,
            minimum_inliers=72,
        )
        self.assertIsNotNone(aligned)
        self.assertLess(abs(float(aligned["correction_deg"])), 1.0)
        self.assertIsNotNone(overshot)
        self.assertGreater(abs(float(overshot["correction_deg"])), 4.0)

    def test_point_two_recovery_releases_heading_only_source_lock(self):
        """A Point-1 yaw clip must not deadlock Point-2 ORB verification."""
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
        bank = replay / "visual_route_recovery_manual_tail.npz"
        endpoint_frame = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260811_115736_2b91ca"
            / "query_frames"
            / "query_000920.jpg"
        )
        reference_path = replay / "reference_candidate.json"
        if not all(path.exists() for path in (bank, endpoint_frame, reference_path)):
            self.skipTest("Point-2 source-lock regression assets are unavailable")

        leg = json.loads(reference_path.read_text(encoding="utf-8"))["legs"][0]
        key = (1, 1, "point1_to_point2")
        recovery = PatrolVisualRouteRecovery(bank)
        recovery.active_key = key
        recovery.active_source_replay_id = "dji_live_20260819_164415_524a50"
        recovery.needs_acquisition = False
        recovery.last_progress = 0.9591210210207715
        recovery.last_matched_progress = 0.0
        recovery.last_matched_source_frame = 3039
        recovery.last_sequence_index = 2233
        gray = cv2.imread(str(endpoint_frame), cv2.IMREAD_GRAYSCALE)

        observations = []
        diagnostics = []
        for sequence_index in range(2234, 2238):
            observation, diagnostic = recovery.recover(
                gray=gray,
                segment_start=leg["from"],
                segment_end=leg["to"],
                segment_key=key,
                progress_hint=0.9591210210207715,
                progress_ceiling=0.9970323371525861,
                recovery_hover=True,
                sequence_index=sequence_index,
            )
            diagnostics.append(diagnostic)
            if observation is not None:
                observations.append(observation)

        self.assertEqual(
            diagnostics[0].get("released_source_replay_id"),
            "dji_live_20260819_164415_524a50",
        )
        self.assertNotIn(
            "visual_route_no_anchor_in_progress_window",
            [diagnostic.get("reason") for diagnostic in diagnostics],
        )
        self.assertTrue(observations)
        self.assertTrue(observations[-1]["endpoint_verified"])
        self.assertGreaterEqual(observations[-1]["endpoint_hits"], 3)
        self.assertGreaterEqual(observations[-1]["inliers"], 120)
        self.assertEqual(
            recovery.active_source_replay_id,
            "dji_live_20260811_115736_2b91ca",
        )

    def test_geometric_rank_never_trades_inliers_for_secondary_quality(self):
        stronger = {
            "inliers": 121,
            "inlier_ratio": 0.51,
            "source_coverage": 0.10,
            "query_coverage": 0.10,
            "median_reprojection_error_px": 2.0,
        }
        cleaner_but_weaker = {
            "inliers": 120,
            "inlier_ratio": 0.95,
            "source_coverage": 0.90,
            "query_coverage": 0.90,
            "median_reprojection_error_px": 0.1,
        }
        self.assertGreater(
            geometric_candidate_rank(stronger),
            geometric_candidate_rank(cleaner_but_weaker),
        )

    def test_geometric_rank_resolves_equal_inlier_alias_by_geometry(self):
        concentrated_alias = {
            "inliers": 120,
            "inlier_ratio": 0.60,
            "source_coverage": 0.06,
            "query_coverage": 0.05,
            "median_reprojection_error_px": 1.8,
        }
        supported_view = {
            "inliers": 120,
            "inlier_ratio": 0.60,
            "source_coverage": 0.30,
            "query_coverage": 0.25,
            "median_reprojection_error_px": 0.9,
        }
        self.assertGreater(
            geometric_candidate_rank(supported_view),
            geometric_candidate_rank(concentrated_alias),
        )

    def test_hierarchical_route_match_equals_exact_recover_decision(self):
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
            / "atlas_dji_live_20260809_154714_d26c33"
            / "query_frames"
            / "query_002939.jpg"
        )
        if not bank.exists() or not frame.exists():
            self.skipTest("saved hierarchical ORB audit assets are unavailable")

        gray = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
        segment_start = [
            -0.4886978074319452,
            0.010052834697950513,
            0.9560112230666532,
        ]
        segment_end = [
            -3.0736291183109774,
            0.010052834697950513,
            1.1113942967930859,
        ]
        exact = PatrolVisualRouteRecovery(bank, matching_profile="exact")
        hierarchical = PatrolVisualRouteRecovery(
            bank, matching_profile="hierarchical"
        )
        exact_observation = hierarchical_observation = None
        for sequence_index in range(2):
            exact_observation, _ = exact.recover(
                gray=gray,
                segment_start=segment_start,
                segment_end=segment_end,
                segment_key=("hierarchy_audit",),
                progress_hint=0.06,
                progress_ceiling=0.18,
                sequence_index=sequence_index,
            )
            hierarchical_observation, _ = hierarchical.recover(
                gray=gray,
                segment_start=segment_start,
                segment_end=segment_end,
                segment_key=("hierarchy_audit",),
                progress_hint=0.06,
                progress_ceiling=0.18,
                sequence_index=sequence_index,
            )

        self.assertIsNotNone(exact_observation)
        self.assertIsNotNone(hierarchical_observation)
        for field in (
            "progress",
            "inliers",
            "ratio_matches",
            "anchor_name",
            "source_frame",
            "source_replay_id",
            "center",
            "heading",
        ):
            self.assertEqual(exact_observation[field], hierarchical_observation[field])
        diagnostic = hierarchical.last_match_diagnostic
        self.assertTrue(diagnostic["hierarchy_attempted"])
        self.assertTrue(diagnostic["hierarchy_winner_proven"])
        self.assertFalse(diagnostic["hierarchy_fallback"])
        self.assertLess(
            diagnostic["hierarchy_selected_anchor_count"],
            diagnostic["anchor_count"],
        )

    def test_invalid_matching_profile_is_rejected(self):
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_precision_20260813"
            / "visual_route_recovery_multirun.npz"
        )
        if not bank.exists():
            self.skipTest("saved hierarchical ORB audit bank is unavailable")
        with self.assertRaises(ValueError):
            PatrolVisualRouteRecovery(bank, matching_profile="approximate")

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

    def test_122039_point_four_rewind_switches_source_and_stays_midleg(self):
        """The 57% correction must not fall back to the old Point-4 window."""
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
            / "atlas_dji_live_20260817_122039_40e9c0"
            / "query_frames"
            / "query_001748.jpg"
        )
        if not bank.exists() or not frame.exists():
            self.skipTest("12:20 false Point-4 endpoint frame is unavailable")

        recovery = PatrolVisualRouteRecovery(bank)
        key = (1, 3)
        recovery.active_key = key
        recovery.last_progress = 0.90
        recovery.last_matched_progress = 0.90
        recovery.last_matched_source_frame = 2018
        recovery.last_sequence_index = 1747
        recovery.active_source_replay_id = (
            "dji_live_20260811_115736_2b91ca"
        )
        recovery.needs_acquisition = False
        gray = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
        observations = []
        diagnostics = []
        for offset in range(7):
            observation, diagnostic = recovery.recover(
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
                sequence_index=1748 + offset,
            )
            diagnostics.append(diagnostic)
            if observation is not None:
                observations.append(observation)

        rewinds = [
            item for item in observations if item.get("verified_rewind") is True
        ]
        self.assertTrue(rewinds)
        self.assertAlmostEqual(rewinds[-1]["matched_progress"], 0.5722543352601156)
        self.assertEqual(
            recovery.active_source_replay_id,
            "patrol_baseline_precision_20260813",
        )
        self.assertGreaterEqual(recovery.last_matched_source_frame, 3500)
        self.assertNotEqual(
            diagnostics[-1].get("reason"),
            "visual_route_progress_consensus_missing",
        )
        self.assertLess(observations[-1]["matched_progress"], 0.75)
        self.assertNotIn("query_00201", observations[-1]["anchor_name"])

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
            focal_px=882.4866783165957,
        )
        taught, _taught_stage = recovery.departure_heading_alignment(
            gray=cv2.imread(str(taught_frame), cv2.IMREAD_GRAYSCALE),
            segment_start=segment_start,
            segment_end=segment_end,
            focal_px=882.4866783165957,
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
            focal_px=882.4866783165957,
            minimum_inliers=120,
        )
        overshot, _overshot_stage = recovery.departure_heading_alignment(
            gray=cv2.imread(str(overshot_frame), cv2.IMREAD_GRAYSCALE),
            segment_start=segment_start,
            segment_end=segment_end,
            focal_px=882.4866783165957,
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
            focal_px=882.4866783165957,
            minimum_inliers=30,
        )
        aligned = [
            recovery.departure_heading_alignment(
                gray=cv2.imread(str(path), cv2.IMREAD_GRAYSCALE),
                segment_start=start,
                segment_end=end,
                focal_px=882.4866783165957,
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

    def test_live_152645_point_four_recovery_uses_five_frame_50_inlier_gate(self):
        """The saved Point-4 view is valid but weaker than the generic gate."""
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
            / "atlas_dji_live_20260819_152645_3344db"
            / "query_frames"
        )
        indices = [3322, 3323, 3324, 3325, 3326]
        required = [bank, replay / "reference_candidate.json"] + [
            frames / f"query_{frame_index:06d}.jpg"
            for frame_index in indices
        ]
        if not all(path.exists() for path in required):
            self.skipTest("15:26 Point-4 recovery assets are unavailable")

        leg = json.loads(required[1].read_text(encoding="utf-8"))["legs"][3]
        strict_recovery = PatrolVisualRouteRecovery(bank)
        strict_observation = None
        for frame_index, image_path in zip(indices, required[2:]):
            strict_observation, _stage = strict_recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=leg["from"],
                segment_end=leg["to"],
                segment_key=("strict_point4_to_point1",),
                progress_hint=0.0,
                progress_ceiling=0.123922,
                recovery_hover=True,
                independent_progress=True,
                sequence_index=frame_index,
            )
        self.assertIsNone(strict_observation)

        audited_recovery = PatrolVisualRouteRecovery(bank)
        observations = []
        for frame_index, image_path in zip(indices, required[2:]):
            observation, _stage = audited_recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=leg["from"],
                segment_end=leg["to"],
                segment_key=("audited_point4_to_point1",),
                progress_hint=0.0,
                progress_ceiling=0.123922,
                recovery_hover=True,
                recovery_minimum_inliers=50,
                independent_progress=True,
                sequence_index=frame_index,
            )
            observations.append(observation)

        self.assertEqual(observations[:4], [None, None, None, None])
        self.assertIsNotNone(observations[4])
        self.assertTrue(observations[4]["temporal_recovery"])
        self.assertGreaterEqual(observations[4]["inliers"], 50)
        self.assertGreaterEqual(observations[4]["acquisition_hits"], 5)
        self.assertLessEqual(observations[4]["progress"], 0.123922)

    def test_live_123007_mid_leg_endpoint_alias_does_not_starve_local_recovery(self):
        """Regress the Point-1 alias that froze 4->1 after its fourth pulse."""
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("OpenCV is stubbed by another combined-suite module")
        public = ROOT / "viewer" / "public"
        bank = (
            public
            / "maps"
            / "map_copy_20260730_114851_cfefdc"
            / "replays"
            / "patrol_baseline_precision_20260813"
            / "visual_route_recovery_manual_tail_point1_20260820.npz"
        )
        frames = (
            public
            / "live_dji_sessions"
            / "atlas_dji_live_20260823_123007_6ab36a"
            / "query_frames"
        )
        indices = list(range(2266, 2271))
        required = [bank] + [
            frames / f"query_{frame_index:06d}.jpg"
            for frame_index in indices
        ]
        if not all(path.exists() for path in required):
            self.skipTest("12:30:07 Point-4-to-1 freeze assets are unavailable")

        recovery = PatrolVisualRouteRecovery(bank)
        key = ("live_123007_point4_to_point1", 4)
        recovery._reset_for_key(key)
        recovery.last_progress = 0.37176646744829467
        recovery.last_matched_progress = 0.38228182370214714
        recovery.last_matched_source_frame = 2929
        recovery.last_sequence_index = 2265
        recovery.active_source_replay_id = "dji_live_20260819_164415_524a50"
        recovery.needs_acquisition = False
        observations = []
        stages = []
        for frame_index, image_path in zip(indices, required[1:]):
            observation, stage = recovery.recover(
                gray=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
                segment_start=[
                    -3.0736291183109774,
                    -0.0802621969998909,
                    1.1113942967930859,
                ],
                segment_end=[
                    -3.2329557447702215,
                    -0.0802621969998909,
                    -0.33236579860361815,
                ],
                segment_key=key,
                translation_locked=False,
                progress_hint=0.37176646744829467,
                progress_ceiling=0.4956886232643929,
                recovery_hover=True,
                recovery_minimum_inliers=50,
                independent_progress=True,
                allow_endpoint_only_recovery=True,
                sequence_index=frame_index,
            )
            observations.append(observation)
            stages.append(stage)

        self.assertTrue(
            all(
                stage.get("reason")
                != "visual_route_endpoint_only_acquiring"
                for stage in stages
            )
        )
        self.assertEqual(observations[:4], [None, None, None, None])
        self.assertIsNotNone(observations[4])
        self.assertTrue(observations[4]["temporal_recovery"])
        self.assertGreaterEqual(observations[4]["inliers"], 50)
        self.assertLessEqual(observations[4]["progress"], 0.4956886232643929)

if __name__ == "__main__":
    unittest.main()

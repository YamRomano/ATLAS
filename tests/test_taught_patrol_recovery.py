import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from colmap_io import qvec_to_rotmat
from taught_patrol_recovery import (
    consensus_candidate_cluster,
    rotmat_to_qvec,
    select_anchor_match_window,
    unique_ann_point_matches,
)


class TaughtPatrolRecoveryTests(unittest.TestCase):
    def test_recovery_qvec_round_trips_world_to_camera_rotation_and_center(self):
        rotation = np.asarray(
            [
                [-0.6854202942077334, 0.04728293560639424, -0.7266108616643616],
                [-0.07379626251518355, 0.9882405115382702, 0.13392088333553065],
                [0.7243984621235571, 0.1454132571501659, -0.6738707982380638],
            ]
        )
        translation = np.asarray([1.2290249268274316, 0.06836623570921496, 0.2059937377902017])

        restored_rotation = qvec_to_rotmat(np.asarray(rotmat_to_qvec(rotation)))

        self.assertTrue(np.allclose(restored_rotation, rotation, atol=1e-9))
        self.assertTrue(
            np.allclose(
                -(restored_rotation.T @ translation),
                -(rotation.T @ translation),
                atol=1e-9,
            )
        )

    def test_consensus_ignores_a_strong_repeated_room_alias(self):
        candidates = [
            {"center": np.asarray([0.01, 0.00, 0.00]), "inliers": 21, "center_step": 0.65},
            {"center": np.asarray([0.04, -0.02, 0.01]), "inliers": 20, "center_step": 0.66},
            {"center": np.asarray([-0.02, 0.03, -0.01]), "inliers": 19, "center_step": 0.64},
            {"center": np.asarray([0.75, 0.00, 0.00]), "inliers": 24, "center_step": 0.20},
        ]

        cluster = consensus_candidate_cluster(candidates, radius=0.18, minimum=3)

        self.assertEqual(len(cluster), 3)
        self.assertTrue(all(float(np.linalg.norm(item["center"])) < 0.1 for item in cluster))

    def test_single_lookalike_never_recovers_pose(self):
        cluster = consensus_candidate_cluster(
            [{"center": np.asarray([0.2, 0.0, 0.0]), "inliers": 50, "center_step": 0.2}],
            radius=0.18,
            minimum=3,
        )

        self.assertEqual(cluster, [])

    def test_ann_ratio_uses_a_different_3d_point_as_competitor(self):
        matches = unique_ann_point_matches(
            distances=np.asarray([[1.0, 1.1, 4.0, 9.0]], dtype=np.float32),
            neighbor_ids=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            ann_rows=np.asarray([0, 1, 2, 3], dtype=np.int64),
            point3d_ids=np.asarray([10, 10, 20, 30], dtype=np.int64),
            anchor_ids=np.asarray([0, 1, 2, 3], dtype=np.int32),
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["point3d_id"], 10)
        self.assertEqual(matches[0]["anchor_ids"], [0, 1])
        self.assertEqual(matches[0]["second_point_distance"], 4.0)

    def test_anchor_window_never_crosses_recorded_replays(self):
        matches = [
            {
                "query_index": index,
                "point3d_id": 100 + index,
                "source_row": index,
                "distance": 1.0 + index * 0.01,
                "anchor_ids": [index % 2],
            }
            for index in range(9)
        ]
        # Anchor 2 is numerically adjacent but belongs to another replay and
        # must not become support for the selected correspondence window.
        matches.append(
            {
                "query_index": 9,
                "point3d_id": 109,
                "source_row": 9,
                "distance": 0.1,
                "anchor_ids": [2],
            }
        )

        selected, anchors = select_anchor_match_window(
            matches,
            anchor_names=["lap_a/f0", "lap_a/f1", "lap_b/f0"],
            radius=12,
            minimum_points=8,
            minimum_anchors=2,
        )

        self.assertEqual(len(selected), 9)
        self.assertEqual(anchors, [0, 1])

    def test_taught_recovery_does_not_estimate_pose_with_opencv(self):
        source = (
            ROOT / "scripts" / "taught_patrol_recovery.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("solvePnPRansac", source)
        self.assertIn('"solver": "tsolve"', source)
        self.assertIn('"tsolve_only_correspondences": True', source)

if __name__ == "__main__":
    unittest.main()

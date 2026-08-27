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

if __name__ == "__main__":
    unittest.main()

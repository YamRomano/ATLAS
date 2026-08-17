import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_live_patrol_runs import analyze_run, ordered_waypoint_hits, patrol_waypoints  # noqa: E402


class LivePatrolRunAuditTests(unittest.TestCase):
    def setUp(self):
        self.reference = {
            "legs": [
                {"from": [0, 0, 0], "to": [2, 0, 0]},
                {"from": [2, 0, 0], "to": [2, 0, 2]},
                {"from": [2, 0, 2], "to": [0, 0, 2]},
                {"from": [0, 0, 2], "to": [0, 0, 0]},
            ]
        }
        self.waypoints = patrol_waypoints(self.reference)

    def test_ordered_hits_do_not_skip_a_missing_waypoint(self):
        poses = [
            {"rcenter": [0, 0, 0]},
            {"rcenter": [2, 0, 2]},
            {"rcenter": [0, 0, 2]},
        ]
        hits = ordered_waypoint_hits(poses, self.waypoints, radius=0.1)
        self.assertEqual([item["waypoint"] for item in hits], [1])

    def test_completed_loop_requires_return_to_point_one(self):
        poses = [
            {"rcenter": point, "held_pose": index == 2, "pose_source": "metric"}
            for index, point in enumerate(self.waypoints)
        ]
        result = analyze_run({"replay_id": "test", "poses": poses}, self.waypoints, radius=0.1)
        self.assertTrue(result["reached_point_4"])
        self.assertTrue(result["closed_first_loop"])
        self.assertEqual(result["reached_waypoints"], [1, 2, 3, 4, 1])
        self.assertAlmostEqual(result["held_ratio"], 0.2)

    def test_published_held_pose_is_counted_but_not_hidden(self):
        poses = [
            {"rcenter": [0, 0, 0]},
            {
                "rcenter": [2, 0, 0],
                "held_pose": True,
                "pose_source": "patrol_visual_route_recovery",
                "hold_reason": "global timeout",
            },
        ]
        result = analyze_run({"poses": poses}, self.waypoints, radius=0.1)
        self.assertEqual(result["reached_waypoints"], [1, 2])
        self.assertEqual(result["held_count"], 1)
        self.assertEqual(result["visual_route_count"], 1)
        self.assertEqual(result["timeout_frame_count"], 1)


if __name__ == "__main__":
    unittest.main()

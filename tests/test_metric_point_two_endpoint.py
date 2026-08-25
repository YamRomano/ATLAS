import importlib.util
import sys
import types
import unittest
from pathlib import Path


sys.modules.setdefault("cv2", types.ModuleType("cv2"))
BRIDGE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "atlas_dji_live_bridge.py"
)
SPEC = importlib.util.spec_from_file_location("atlas_dji_live_bridge", BRIDGE_PATH)
assert SPEC and SPEC.loader
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def metric_gate(instance_id: str, position: list[float]) -> dict:
    return {
        "ok": True,
        "pose": {
            "instance_id": instance_id,
            "pose_source": "tsolve_global",
            "rcenter": list(position),
            "center": list(position),
            "t": [0.1, 0.2, 0.3],
            "R": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
    }


class MetricPointTwoEndpointTests(unittest.TestCase):
    def setUp(self):
        self.start = [-1.60, -0.08, -0.49]
        self.target = [-0.65, -0.08, -0.49]

    def candidate(self, gate: dict, leg: int = 1) -> bool:
        return bridge.tight_metric_point_two_endpoint_arrival_candidate(
            gate,
            target=self.target,
            segment_start=self.start,
            expected_leg_index=leg,
        )

    def test_accepts_only_fresh_metric_pose_inside_three_centimetres_on_leg_one(self):
        self.assertTrue(self.candidate(metric_gate("pose-1", [-0.63, -0.08, -0.49])))
        self.assertFalse(self.candidate(metric_gate("pose-2", [-0.61, -0.08, -0.49])))
        self.assertFalse(self.candidate(metric_gate("pose-3", [-0.63, -0.08, -0.49]), leg=2))

        route_only = metric_gate("pose-4", [-0.63, -0.08, -0.49])
        route_only["pose"]["pose_source"] = "patrol_visual_route_recovery"
        self.assertFalse(self.candidate(route_only))

    def test_consensus_requires_three_distinct_stable_metric_frames(self):
        state = {}
        first = metric_gate("pose-1", [-0.630, -0.08, -0.490])
        second = metric_gate("pose-2", [-0.631, -0.08, -0.489])
        third = metric_gate("pose-3", [-0.629, -0.08, -0.491])

        self.assertFalse(
            bridge.update_stable_metric_endpoint_consensus(
                state, first, candidate=self.candidate(first)
            )
        )
        self.assertFalse(
            bridge.update_stable_metric_endpoint_consensus(
                state, first, candidate=self.candidate(first)
            )
        )
        self.assertFalse(
            bridge.update_stable_metric_endpoint_consensus(
                state, second, candidate=self.candidate(second)
            )
        )
        self.assertTrue(
            bridge.update_stable_metric_endpoint_consensus(
                state, third, candidate=self.candidate(third)
            )
        )
        self.assertEqual(state["hits"], 3)

    def test_consensus_restarts_when_metric_position_jumps(self):
        state = {}
        left = metric_gate("pose-1", [-0.675, -0.08, -0.49])
        right = metric_gate("pose-2", [-0.625, -0.08, -0.49])
        self.assertTrue(self.candidate(left))
        self.assertTrue(self.candidate(right))

        bridge.update_stable_metric_endpoint_consensus(
            state, left, candidate=True
        )
        bridge.update_stable_metric_endpoint_consensus(
            state, right, candidate=True
        )
        self.assertEqual(state["hits"], 1)


if __name__ == "__main__":
    unittest.main()

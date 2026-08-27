from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_route_constrained_patrol_baseline.py"
SPEC = importlib.util.spec_from_file_location("route_baseline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rotation_phase_locks_position_and_changes_heading() -> None:
    points = [
        [-3.0, 4.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 6.0, 2.0],
        [-3.0, 7.0, 2.0],
    ]
    boundaries = {
        "point1": 0,
        "point2_arrival": 2,
        "point2_departure": 4,
        "point3_arrival": 6,
        "point3_departure": 8,
        "point4_arrival": 10,
        "point4_departure": 12,
        "point1_return": 14,
    }
    rows = {
        index: {
            "image_name": f"query_{index:06d}.jpg",
            "time_sec": float(index),
            "received_unix": 1000.0 + index,
        }
        for index in range(15)
    }
    poses = MODULE.build_poses(frame_rows=rows, points=points, boundaries=boundaries, cruise_y=1.25)
    turn = [pose for pose in poses if pose["route_phase"] == "turn_at_3"]
    assert len(turn) == 2
    assert all(pose["rcenter"] == [0.0, 1.25, 2.0] for pose in turn)
    assert all(pose["rotation_position_locked"] for pose in turn)
    assert turn[0]["rheading"] != turn[-1]["rheading"]


def test_all_translation_phases_are_monotonic() -> None:
    points = [
        [-3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [-3.0, 0.0, 2.0],
    ]
    boundaries = {
        "point1": 0,
        "point2_arrival": 3,
        "point2_departure": 5,
        "point3_arrival": 8,
        "point3_departure": 10,
        "point4_arrival": 13,
        "point4_departure": 15,
        "point1_return": 18,
    }
    rows = {
        index: {
            "image_name": f"query_{index:06d}.jpg",
            "time_sec": float(index),
            "received_unix": 1000.0 + index,
        }
        for index in range(19)
    }
    poses = MODULE.build_poses(frame_rows=rows, points=points, boundaries=boundaries, cruise_y=0.0)
    for phase in ("leg_1_2", "leg_2_3", "leg_3_4", "leg_4_1"):
        progress = [pose["route_progress"] for pose in poses if pose["route_phase"] == phase]
        assert progress == sorted(progress)
        assert progress[0] >= 0.0
        assert progress[-1] == 1.0


def test_return_leg_uses_denser_visual_samples() -> None:
    points = [
        [-3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [-3.0, 0.0, 2.0],
    ]
    boundaries = {
        "point1": 0,
        "point2_arrival": 20,
        "point2_departure": 22,
        "point3_arrival": 42,
        "point3_departure": 44,
        "point4_arrival": 64,
        "point4_departure": 66,
        "point1_return": 86,
    }
    rows = {
        index: {
            "image_name": f"query_{index:06d}.jpg",
            "time_sec": float(index),
            "received_unix": 1000.0 + index,
        }
        for index in range(87)
    }
    poses = MODULE.build_poses(
        frame_rows=rows,
        points=points,
        boundaries=boundaries,
        cruise_y=0.0,
    )
    reference = MODULE.build_reference(
        map_id="map_a",
        patrol_id="patrol_a",
        source_replay_id="replay_a",
        points=points,
        poses=poses,
        boundaries=boundaries,
        visual_recovery_bank="visual_route_recovery.npz",
        visual_recovery_audit="visual_route_recovery_audit.json",
    )

    assert len(reference["legs"][0]["samples"]) == 3
    assert len(reference["legs"][3]["samples"]) == 5
    assert reference["enabled_for_visual_route_recovery"] is True
    assert reference["visual_route_recovery_bank"] == "visual_route_recovery.npz"
    assert reference["visual_route_recovery_audit"] == "visual_route_recovery_audit.json"

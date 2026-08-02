import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAP_ID = "map_copy_20260730_114851_cfefdc"
YAW_DEG = -7.662
BACKUP = (
    ROOT
    / "viewer/public/maps"
    / MAP_ID
    / "alignment_backup_20260730_202557_before_7p662deg.json"
)


class CopyMapAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "viewer/public/maps/manifest.json").read_text())
        cls.map_entry = next(entry for entry in cls.manifest["maps"] if entry["id"] == MAP_ID)
        cls.backup = json.loads(BACKUP.read_text())

    def test_viewer_uses_map_specific_structure_yaw(self):
        source = (ROOT / "viewer/app.js").read_text()
        self.assertIn("room_alignment?.structure_yaw_deg", source)
        self.assertIn("configuredStructureYawDeg * Math.PI / 180", source)
        self.assertAlmostEqual(
            self.map_entry["room_alignment"]["structure_yaw_deg"],
            YAW_DEG,
            places=9,
        )
        configured = [
            entry["id"]
            for entry in self.manifest["maps"]
            if "structure_yaw_deg" in (entry.get("room_alignment") or {})
        ]
        self.assertEqual(configured, [MAP_ID])

    def test_safety_walls_are_a_rigid_rotation_of_backup(self):
        before = self.backup["safety_barriers"]
        after = self.map_entry["safety_barriers"]
        self.assertEqual([wall["id"] for wall in before], [wall["id"] for wall in after])

        pivot_x = sum(wall["a"][0] for wall in before) / len(before)
        pivot_z = sum(wall["a"][2] for wall in before) / len(before)
        angle = math.radians(YAW_DEG)
        cosine = math.cos(angle)
        sine = math.sin(angle)

        def rotate(point):
            dx = point[0] - pivot_x
            dz = point[2] - pivot_z
            return [
                pivot_x + cosine * dx - sine * dz,
                point[1],
                pivot_z + sine * dx + cosine * dz,
            ]

        for old_wall, new_wall in zip(before, after):
            for old_point, new_point in (
                (old_wall["a"], new_wall["a"]),
                (old_wall["b"], new_wall["b"]),
                *zip(old_wall["corners"], new_wall["corners"]),
            ):
                for expected, actual in zip(rotate(old_point), new_point):
                    self.assertAlmostEqual(actual, expected, places=9)

        for index, wall in enumerate(after):
            next_wall = after[(index + 1) % len(after)]
            for endpoint, next_start in zip(wall["b"], next_wall["a"]):
                self.assertAlmostEqual(endpoint, next_start, places=12)


if __name__ == "__main__":
    unittest.main()

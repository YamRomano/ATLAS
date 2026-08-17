import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAP_ID = "map_copy_20260730_114851_cfefdc"
PATROL_ID = "patrol_ms4br5xr_4xclts"
YAW_DEG = -7.662


class CopyMapAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "viewer/public/maps/manifest.json").read_text())
        cls.map_entry = next(entry for entry in cls.manifest["maps"] if entry["id"] == MAP_ID)

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

    def test_current_safety_walls_enclose_patrol_with_clearance(self):
        walls = self.map_entry["safety_barriers"]
        self.assertEqual(len(walls), 4)
        self.assertEqual([wall["label"] for wall in walls], ["Wall 1", "Wall 2", "Wall 3", "Wall 4"])
        self.assertEqual(len({wall["id"] for wall in walls}), len(walls))

        for index, wall in enumerate(walls):
            next_wall = walls[(index + 1) % len(walls)]
            for endpoint, next_start in zip(wall["b"], next_wall["a"]):
                self.assertAlmostEqual(endpoint, next_start, places=12)
            self.assertGreaterEqual(wall["clearance_m"], 0.45)
            self.assertAlmostEqual(wall["height_m"], 3.5, places=12)
            self.assertEqual(wall["corners"][0], wall["a"])
            self.assertEqual(wall["corners"][1], wall["b"])
            self.assertEqual(wall["corners"][2][::2], wall["b"][::2])
            self.assertEqual(wall["corners"][3][::2], wall["a"][::2])
            self.assertAlmostEqual(wall["corners"][2][1] - wall["b"][1], wall["height_m"], places=12)
            self.assertAlmostEqual(wall["corners"][3][1] - wall["a"][1], wall["height_m"], places=12)

        polygon = [(wall["a"][0], wall["a"][2]) for wall in walls]

        def cross(origin, end, point):
            return ((end[0] - origin[0]) * (point[1] - origin[1])) - (
                (end[1] - origin[1]) * (point[0] - origin[0])
            )

        corner_turns = [
            cross(polygon[index], polygon[(index + 1) % 4], polygon[(index + 2) % 4])
            for index in range(4)
        ]
        self.assertTrue(all(value < 0.0 for value in corner_turns))

        patrol = next(item for item in self.map_entry["patrols"] if item["id"] == PATROL_ID)
        for point in patrol["points"]:
            room_point = (point["rxyz"][0], point["rxyz"][2])
            edge_sides = [
                cross(polygon[index], polygon[(index + 1) % 4], room_point)
                for index in range(4)
            ]
            self.assertTrue(all(value < 0.0 for value in edge_sides))

            # For a convex clockwise polygon, this is the perpendicular distance
            # from the patrol point to each wall line. The 0.75 m bound includes
            # the saved 0.45 m wall clearance plus a 0.30 m operational margin.
            distances = []
            for index, side in enumerate(edge_sides):
                start = polygon[index]
                end = polygon[(index + 1) % 4]
                edge_length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
                distances.append(abs(side) / edge_length)
            self.assertGreaterEqual(min(distances), 0.75)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


sys.modules.setdefault("cv2", types.ModuleType("cv2"))
ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "scripts" / "atlas_app_server.py"

SPEC = importlib.util.spec_from_file_location("atlas_app_server_patrol_import_test", SERVER_PATH)
assert SPEC and SPEC.loader
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


class PatrolImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.original_maps_dir = server.MAPS_DIR
        self.original_manifest = server.MAP_MANIFEST
        server.MAPS_DIR = Path(self.temporary.name)
        server.MAP_MANIFEST = server.MAPS_DIR / "manifest.json"
        self.patrol = {
            "id": "patrol_lab_1",
            "title": "Patrol 1",
            "points": [
                {"rxyz": [0.0, 1.0, 0.0], "rgb": [20, 30, 40]},
                {"rxyz": [1.0, 1.0, 0.0], "rgb": [40, 50, 60]},
                {"rxyz": [1.0, 1.0, 1.0], "rgb": None},
            ],
            "speed": 0.12,
            "altitude_m": 1.0,
            "dwell_s": 2.0,
            "scan_mode": "yaw-sweep",
            "patrol_mode": "circle",
            "loop": True,
            "created_at": "2026-07-29 12:00:00",
            "updated_at": "2026-07-29 12:00:00",
        }
        self.library = {
            "selected_map_id": "enhanced_copy",
            "hidden_builtin_ids": ["default_demo"],
            "maps": [
                {
                    "id": "source_map",
                    "title": "Video Map 20:07:46",
                    "patrols": [self.patrol],
                },
                {
                    "id": "enhanced_copy",
                    "title": "Video Map 20:07:46 Copy",
                    "source_map_id": "source_map",
                    "localization_map_id": "enhanced_copy",
                    "patrols": [],
                },
                {
                    "id": "unrelated_map",
                    "title": "Other Room",
                    "patrols": [],
                },
                {
                    "id": "independently_aligned_map",
                    "title": "Armour Map",
                    "coordinate_frame_id": "source_map",
                    "patrols": [],
                },
            ],
        }
        server.MAP_MANIFEST.write_text(json.dumps(self.library, indent=2), encoding="utf-8")

    def tearDown(self):
        server.MAPS_DIR = self.original_maps_dir
        server.MAP_MANIFEST = self.original_manifest
        self.temporary.cleanup()

    def test_import_copies_patrol_without_changing_source(self):
        source_before = json.loads(json.dumps(self.patrol))

        target, imported = server.import_map_patrol(
            "enhanced_copy",
            "source_map",
            "patrol_lab_1",
        )

        self.assertEqual(imported["id"], "patrol_lab_1")
        self.assertEqual(imported["title"], "Patrol 1")
        self.assertEqual(imported["points"], source_before["points"])
        self.assertEqual(len(target["patrols"]), 1)

        saved = server.load_library()
        saved_by_id = {entry["id"]: entry for entry in saved["maps"]}
        self.assertEqual(saved_by_id["source_map"]["patrols"][0], source_before)
        self.assertEqual(saved_by_id["enhanced_copy"]["patrols"][0]["id"], "patrol_lab_1")

    def test_duplicate_import_is_blocked(self):
        server.import_map_patrol("enhanced_copy", "source_map", "patrol_lab_1")

        with self.assertRaisesRegex(RuntimeError, "already present"):
            server.import_map_patrol("enhanced_copy", "source_map", "patrol_lab_1")

    def test_unrelated_coordinate_frames_are_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "different map coordinate frame"):
            server.import_map_patrol("unrelated_map", "source_map", "patrol_lab_1")

    def test_independent_map_can_share_coordinates_without_inheriting_localization(self):
        target, imported = server.import_map_patrol(
            "independently_aligned_map",
            "source_map",
            "patrol_lab_1",
        )

        self.assertEqual(imported["id"], "patrol_lab_1")
        self.assertEqual(target["coordinate_frame_id"], "source_map")
        self.assertNotIn("source_map_id", target)
        self.assertNotIn("localization_map_id", target)

    def test_active_recovery_banks_include_full_loop_but_exclude_backups(self):
        asset_dir = Path(self.temporary.name) / "map_assets"
        patrol_dir = asset_dir / "taught_patrols" / "patrol_lab_1"
        patrol_dir.mkdir(parents=True)
        current = patrol_dir / "recovery_bank.npz"
        full_loop = patrol_dir / "recovery_bank_full_loop_154714.npz"
        backup = patrol_dir / "recovery_bank_before_20260809.npz"
        for path in (current, full_loop, backup):
            path.touch()

        banks = server.active_taught_recovery_banks(asset_dir)

        self.assertEqual(banks, sorted([current.resolve(), full_loop.resolve()]))


if __name__ == "__main__":
    unittest.main()

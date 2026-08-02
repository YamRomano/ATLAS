import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

LOCALIZER_PATH = SCRIPTS / "run_bounded_tsolve_video_stream.py"
SPEC = importlib.util.spec_from_file_location("run_bounded_tsolve_video_stream_alignment_test", LOCALIZER_PATH)
assert SPEC and SPEC.loader
localizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(localizer)


class LiveRoomAlignmentTests(unittest.TestCase):
    def test_saved_alignment_overrides_scene_derived_frame(self):
        alignment = {
            "matrix": [
                [0.0, 0.0, 1.0, 4.0],
                [0.0, 1.0, 0.0, -2.0],
                [-1.0, 0.0, 0.0, 7.0],
            ]
        }
        transform = localizer.build_room_transform(None, -1.0, alignment)
        self.assertIsNotNone(transform)
        self.assertEqual(transform([1.0, 2.0, 3.0]), [7.0, 0.0, 6.0])
        self.assertEqual(transform.direction([1.0, 2.0, 3.0]), [3.0, 2.0, -1.0])

    def test_alignment_json_accepts_manifest_shaped_payload(self):
        alignment = {
            "matrix": [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
            ]
        }
        parsed = localizer.parse_room_alignment_json(json.dumps({"room_alignment": alignment}))
        self.assertEqual(parsed, alignment)

    def test_server_passes_fixed_alignment_to_live_and_uploaded_localizers(self):
        source = (SCRIPTS / "atlas_app_server.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count('"--room-alignment-json"'), 2)
        self.assertGreaterEqual(
            source.count('json.dumps(selected.get("room_alignment") or {})'),
            2,
        )


if __name__ == "__main__":
    unittest.main()

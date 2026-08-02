import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_room_mesh_colab_input.py"
NOTEBOOK = ROOT / "colab" / "two_video_room_mesh_vggt.ipynb"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_room_mesh_colab_input_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = load_module()


class RoomMeshColabTests(unittest.TestCase):
    def test_hash_distance_and_coverage_selection(self):
        self.assertEqual(PREPARE.hash_distance(0b1010, 0b1111), 2)
        records = [
            PREPARE.FrameRecord(index, f"v{index}.mov", float(time), f"v{index}_{time}.jpg", 100 + time, "0" * 16)
            for index in (1, 2)
            for time in range(20)
        ]
        selected = PREPARE.coverage_select(records, 10)
        self.assertEqual(len(selected), 10)
        self.assertEqual({record.video_index for record in selected}, {1, 2})

    def test_perceptual_hash_is_deterministic(self):
        if not hasattr(PREPARE.cv2, "resize"):
            self.skipTest("another test installed a minimal cv2 stub")
        image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
        self.assertEqual(PREPARE.perceptual_hash(image), PREPARE.perceptual_hash(image.copy()))

    def test_notebook_has_pilot_gate_and_dense_export(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("scene_pilot", source)
        self.assertIn("RUN_LARGER = False", source)
        self.assertIn("RUN_DENSE = False", source)
        self.assertIn("make_reduced_pilot", source)
        self.assertIn("scene_pilot_24", source)
        self.assertIn("VGGT/COLMAP failed with exit code", source)
        self.assertIn("pilot_query_points = 1024 if gpu_gb < 20 else 2048", source)
        self.assertIn("pycolmap.patch_match_stereo", source)
        self.assertIn("pycolmap.poisson_meshing", source)


if __name__ == "__main__":
    unittest.main()

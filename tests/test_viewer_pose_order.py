import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


sys.modules.setdefault("numpy", types.ModuleType("numpy"))
colmap_io = types.ModuleType("colmap_io")
colmap_io.camera_center = None
colmap_io.read_images_text = None
colmap_io.read_points3d_text = None
colmap_io.qvec_to_rotmat = None
sys.modules.setdefault("colmap_io", colmap_io)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_viewer_data.py"
SPEC = importlib.util.spec_from_file_location("build_viewer_data_pose_order_test", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ViewerPoseOrderTests(unittest.TestCase):
    def test_instance_ids_are_sorted_numerically(self):
        ids = ["instance_999", "instance_1001", "instance_1000", "instance_099"]
        self.assertEqual(
            sorted(ids, key=module.instance_sort_key),
            ["instance_099", "instance_999", "instance_1000", "instance_1001"],
        )

    def test_live_output_rejections_are_loaded_for_final_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            (runtime / "live_stream_summary.json").write_text(
                json.dumps(
                    {
                        "output_rejected": [
                            {"case_id": "instance_12", "reason": "objective"},
                            {"reason": "missing id"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                module.load_output_rejected_case_ids(runtime),
                {"instance_12"},
            )


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "scripts" / "atlas_app_server.py"
CONVERTER_PATH = ROOT / "scripts" / "convert_camera_path_lab_mesh.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


SERVER = load_module("atlas_app_server_camera_path_lab_test", SERVER_PATH)
CONVERTER = load_module("convert_camera_path_lab_mesh_test", CONVERTER_PATH)


class CameraPathLabTests(unittest.TestCase):
    def test_standalone_page_has_isolated_upload_and_live_coordinates(self):
        html = (ROOT / "viewer" / "camera-path-lab.html").read_text(encoding="utf-8")
        script = (ROOT / "viewer" / "camera-path-lab.js").read_text(encoding="utf-8")
        for element_id in (
            "lab-canvas",
            "video-input",
            "start-button",
            "camera-label",
            "camera-coordinates",
            "accepted-count",
            "processed-count",
        ):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn('const REFERENCE_MAP_ID = "map_copy_20260730_114851_cfefdc"', script)
        self.assertIn('fetch("/api/camera-path-lab/upload"', script)
        self.assertIn("pose.rcenter", script)
        self.assertIn("roomMatrix", script)

    def test_server_keeps_side_project_out_of_map_manifest(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn('def drone_video_job(', source)
        self.assertIn('publish_to_map: bool = True', source)
        self.assertIn('if publish_to_map:\n            add_replay_to_map', source)
        self.assertIn('ROOT / "results" / "camera_path_lab_runs"', source)
        self.assertIn('url.path == "/api/camera-path-lab/status"', source)
        self.assertIn('"/api/camera-path-lab/upload"', source)

    def test_camera_lab_snapshot_is_a_copy(self):
        SERVER.set_camera_path_lab_stream({"pose_count": 3})
        snapshot = SERVER.camera_path_lab_snapshot()
        snapshot["stream"]["pose_count"] = 99
        self.assertEqual(SERVER.camera_path_lab_snapshot()["stream"]["pose_count"], 3)

    def test_mesh_conversion_applies_room_transform_and_face_cap(self):
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64
        )
        faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        matrix = np.asarray(
            [[0, 0, 1, 4], [0, 1, 0, -2], [-1, 0, 0, 7]], dtype=np.float64
        )
        positions, normals, indices, colors = CONVERTER.prepare_mesh(
            vertices, faces, None, matrix, max_faces=1
        )
        self.assertEqual(len(indices), 3)
        self.assertIsNone(colors)
        self.assertTrue(np.isfinite(normals).all())
        self.assertTrue(any(np.allclose(position, [4, -2, 7]) for position in positions))


if __name__ == "__main__":
    unittest.main()

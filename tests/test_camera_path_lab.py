import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SERVER_PATH = ROOT / "scripts" / "atlas_app_server.py"
CONVERTER_PATH = ROOT / "scripts" / "convert_camera_path_lab_mesh.py"
CAMERA_CONVERTER_PATH = ROOT / "scripts" / "convert_analog_camera_asset.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


SERVER = load_module("atlas_app_server_camera_path_lab_test", SERVER_PATH)
CONVERTER = load_module("convert_camera_path_lab_mesh_test", CONVERTER_PATH)
CAMERA_CONVERTER = load_module("convert_analog_camera_asset_test", CAMERA_CONVERTER_PATH)


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
        self.assertIn('fallbackCanvas.className = "fallback-canvas"', script)
        self.assertIn("function drawFallbackScene()", script)
        self.assertIn("function drawFallbackCamera(context)", script)
        self.assertIn("function applyDisplayedHeading(heading)", script)
        self.assertIn("displayedHeading.angleTo(targetHeading)", script)
        self.assertIn("YAW ${yaw.toFixed(1)}°", script)
        self.assertIn('const CAMERA_MODEL_URL = "./public/camera_path_lab/analog_camera.glb"', script)
        self.assertIn("function loadAnalogCameraModel()", script)
        self.assertIn("node.material = new THREE.MeshBasicMaterial({", script)
        self.assertIn("THREE.SRGBColorSpace", script)
        self.assertIn("opacity: 0.5", script)
        self.assertIn("new THREE.PointsMaterial({", script)
        self.assertIn("new THREE.Points(geometry, material)", script)
        self.assertIn("rig.scale.setScalar(1.3)", script)
        self.assertIn("colored surface points · GPU display", script)
        self.assertIn('camera-path-lab.js?v=20260802-dark-ombre', html)
        self.assertIn('window.location.protocol === "file:"', html)
        self.assertIn('window.location.replace("http://127.0.0.1:8767/camera-path-lab.html")', html)
        stylesheet = (ROOT / "viewer" / "camera-path-lab.css").read_text(encoding="utf-8")
        self.assertIn("color-scheme: dark", stylesheet)
        self.assertIn("linear-gradient(135deg, #061725", stylesheet)
        self.assertIn("CAMERA PATH", html)
        self.assertNotIn("atlas", (html + script + stylesheet).lower())
        self.assertNotIn("3D preview unavailable here", script)

    def test_analog_camera_converter_preserves_material_color_and_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = Path(directory) / "camera.obj"
            obj.write_text(
                "\n".join(
                    (
                        "v 0 0 0",
                        "v 2 0 0",
                        "v 0 1 0",
                        "vn 0 0 1",
                        "usemtl Lente",
                        "f 1//1 2//1 3//1",
                    )
                ),
                encoding="utf-8",
            )
            positions, normals, indices, colors = CAMERA_CONVERTER.read_obj(obj)
            scaled = CAMERA_CONVERTER.center_and_scale(positions, 0.42)
        self.assertEqual(len(indices), 3)
        self.assertTrue(np.all(colors == np.asarray(CAMERA_CONVERTER.MATERIAL_COLORS["Lente"])))
        self.assertTrue(np.allclose(normals, [[0, 0, 1]] * 3))
        self.assertAlmostEqual(float(np.ptp(scaled[:, 0])), 0.42)

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

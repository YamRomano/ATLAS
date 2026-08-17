import importlib.util
import json
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
    def test_camera_path_recovery_is_non_blocking(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        source = (ROOT / "scripts" / "run_bounded_tsolve_video_stream.py").read_text(
            encoding="utf-8"
        )
        self.assertFalse(config["simulated_live_blocking_global_recovery"])
        self.assertFalse(config["camera_path_lab_blocking_global_recovery"])
        self.assertEqual(config["camera_path_lab_min_track_points"], 6)
        self.assertEqual(config["camera_path_lab_global_recovery_max_step"], 1.6)
        self.assertEqual(config["camera_path_lab_output_max_step"], 1.6)
        self.assertEqual(config["camera_path_lab_output_max_speed"], 2.2)
        self.assertIn("--follow-all-frames", source)
        self.assertIn("args.wait_for_background_recovery", source)
        self.assertIn("interactive_recovery = bool(args.follow_dir or pacer.enabled)", source)
        self.assertIn("and interactive_recovery", source)
        self.assertIn(
            "max_sequential_catchup_frames = 12 if interactive_recovery",
            source,
        )
        self.assertIn('stage["registration_profile"] = "robust_initial_anchor"', source)
        self.assertIn('stage["registration_profile"] = "fast_fixed_map_recovery"', source)
        self.assertIn("--calibrate-output-to-first-global-anchor", source)
        self.assertIn("OUTPUT ANCHOR CALIBRATED", source)
        self.assertIn("if last_center is None:", source)
        self.assertIn('"--Mapper.fix_existing_images"', source)

    def test_camera_path_lab_never_reuses_precomputed_poses(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("find_cached_camera_path_replay", source)
        self.assertNotIn('"cached_replay": True', source)
        self.assertNotIn("Exact video match", source)
        self.assertIn('"fresh_localization": not publish_to_map', source)
        self.assertIn("Starting fresh frame-by-frame localization", source)

    def test_upload_does_not_flash_previous_cancelled_status(self):
        source = (ROOT / "viewer" / "camera-path-lab.js").read_text(encoding="utf-8")
        self.assertIn("let uploadInFlight = false", source)
        self.assertIn('uploadInFlight ? "queued"', source)
        self.assertIn('uploadInFlight ? "Uploading video…"', source)
        self.assertIn("finally {\n    uploadInFlight = false;", source)

    def test_standalone_page_has_isolated_upload_and_live_coordinates(self):
        html = (ROOT / "viewer" / "camera-path-lab.html").read_text(encoding="utf-8")
        script = (ROOT / "viewer" / "camera-path-lab.js").read_text(encoding="utf-8")
        for element_id in (
            "lab-canvas",
            "video-input",
            "start-button",
            "replay-button",
            "camera-label",
            "camera-coordinates",
            "coordinate-kicker",
            "coordinate-link",
            "accepted-count",
            "processed-count",
            "adjust-preview-path",
            "place-preview-start",
            "preview-motion-scale",
            "lock-preview-path",
            "playback-speed-control",
            "preview-timing-segments",
            "preview-timing-segment-list",
            "toggle-preview-timing",
            "preview-transform-controls",
            "preview-rotation",
            "preview-scale-x",
            "preview-scale-y",
            "video-panel-size",
            "video-panel-size-value",
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
        self.assertIn("function displayablePose(pose)", script)
        self.assertIn("pose.rotation_heading", script)
        self.assertIn("RECOVERING FRAME ${currentInputFrameIndex}", script)
        self.assertIn("latestDisplayHeld", script)
        self.assertIn("displayedHeading.angleTo(targetHeading)", script)
        self.assertIn("YAW ${yaw.toFixed(1)}°", script)
        self.assertIn('const CAMERA_MODEL_URL = "./public/camera_path_lab/analog_camera.glb"', script)
        self.assertIn("function loadAnalogCameraModel()", script)
        self.assertIn("node.material = new THREE.MeshStandardMaterial({", script)
        self.assertIn("THREE.SRGBColorSpace", script)
        self.assertIn("opacity: 0.5", script)
        self.assertIn("new THREE.PointsMaterial({", script)
        self.assertIn("new THREE.Points(geometry, material)", script)
        self.assertIn("rig.scale.setScalar(2.1)", script)
        self.assertIn("Initializing the first camera pose… video is held until localization is ready.", script)
        self.assertIn("function syncVideoToLocalizedFrame(time)", script)
        self.assertIn("function startReplay()", script)
        self.assertIn("function updateReplayFrame()", script)
        self.assertIn("function setPlaybackRate(", script)
        self.assertIn("function poseIndexForTime(", script)
        self.assertIn("sourceVideo.playbackRate = playbackRate", script)
        self.assertIn("DEFAULT_PREVIEW_TIMING_SEGMENTS", script)
        self.assertIn("function previewTimingOffsetForFrame(", script)
        self.assertIn("function previewTimingForPlaybackTime(", script)
        self.assertIn("PREVIEW_TIMING_BLEND_FRAMES", script)
        self.assertIn("PREVIEW_TIMING_MIN_SECTION_FRAMES", script)
        self.assertIn("function splitPreviewTimingSegment(", script)
        self.assertIn("function removePreviewTimingSegment(", script)
        self.assertIn("function changePreviewTimingBoundary(", script)
        self.assertIn('sourceVideo.currentTime = 0', script)
        self.assertIn("segmentIndex > 0 ? previewTimingSegments[segmentIndex - 1].offset_sec : 0", script)
        self.assertIn("timing_offset_sec", script)
        self.assertIn("timing_segments", script)
        self.assertIn("previewRotationDeg", script)
        self.assertIn("previewScaleX", script)
        self.assertIn("previewScaleY", script)
        self.assertIn("function updatePreviewFloorTransform()", script)
        self.assertIn("function setVideoPanelWidth(", script)
        self.assertIn("function updateVideoAspectRatio()", script)
        self.assertIn("new ResizeObserver(() => resize())", script)
        self.assertIn('stage.style.setProperty("--video-panel-width"', script)
        self.assertIn('videoCard.style.setProperty("--phone-video-aspect"', script)
        self.assertIn('videoPanelSizeInput.addEventListener("input"', script)
        self.assertIn("movement_scale_x", script)
        self.assertIn("movement_scale_y", script)
        self.assertIn("rotation_deg", script)
        self.assertIn("function maybeStartLivePlayback()", script)
        self.assertIn("|| previewAdjustMode || previewPlaceStartMode", script)
        self.assertIn("function updateLivePlaybackFrame()", script)
        self.assertIn("function stopLivePlayback(", script)
        self.assertIn("function beginSyntheticPlaybackClock(", script)
        self.assertIn("async function resumeLiveMediaPlayback()", script)
        self.assertIn("function livePresentationTime(latestTime)", script)
        self.assertIn("return Math.min(mediaTime, latestTime)", script)
        self.assertIn("later video frames were not played without synchronized camera poses", script)
        self.assertIn("livePlaybackSyntheticClock", script)
        self.assertIn("sourceVideo.currentTime = syntheticTime", script)
        self.assertIn("const completedPoseUrl = stream.asset_base", script)
        self.assertIn("(stream.complete || stream.failed) ? completedPoseUrl", script)
        self.assertIn("livePlaybackUserPaused", script)
        self.assertNotIn("cachedReplayActive", script)
        self.assertIn("Playing the freshly localized camera path", script)
        self.assertIn("Camera-path playback paused.", script)
        self.assertIn("LIVE_START_BUFFER_SECONDS", script)
        self.assertIn("LIVE_RESUME_BUFFER_SECONDS", script)
        self.assertIn("Localization is catching up… playback will resume automatically.", script)
        self.assertIn("function updateCoordinateKicker(", script)
        self.assertIn("function placePreviewStart(", script)
        self.assertIn("function initializePreviewCalibration(", script)
        self.assertIn("function setPreviewAdjustMode(", script)
        self.assertIn("function beginPreviewPathDrag(", script)
        self.assertIn("async function savePreviewCalibration(", script)
        self.assertIn('fetch("/api/camera-path-lab/preview-calibration"', script)
        self.assertIn("preview_movement_scale", script)
        self.assertIn("previewMotionScaleInput.addEventListener", script)
        self.assertIn("roomCeilingY + 0.68", script)
        self.assertIn('coordinateLinkPath.setAttribute("d"', script)
        self.assertIn("for (const marker of [coordinateCameraRing, coordinateCameraDot])", script)
        self.assertIn("LOCALIZING FRAME ${currentInputFrameIndex}", script)
        self.assertIn("const cameraOverlayScene = new THREE.Scene()", script)
        self.assertIn("renderer.clearDepth()", script)
        self.assertIn("renderer.autoClear = false", script)
        self.assertIn("renderer.autoClear = true", script)
        self.assertIn("cameraOverlayScene.add(rig)", script)
        self.assertIn("const pathGlowGroup = new THREE.Group()", script)
        self.assertIn("new THREE.TubeGeometry(curve, segments, 0.022", script)
        self.assertIn("new THREE.TubeGeometry(curve, segments, 0.067", script)
        self.assertIn("blending: THREE.AdditiveBlending", script)
        self.assertIn("const viewField = new THREE.Mesh", script)
        self.assertIn("const centerRay = new THREE.Mesh", script)
        self.assertIn("function loadStoredStreamVideo(mediaUrl", script)
        self.assertIn("loadStoredStreamVideo(stream.media_url", script)
        self.assertIn('sourceVideo.addEventListener("loadedmetadata"', script)
        self.assertIn("window.setTimeout(pollStatus, 300)", script)
        self.assertNotIn("active && selectedFile && sourceVideo.paused", script)
        self.assertIn("colored surface points · GPU display", script)
        self.assertIn('const MESH_GLB_URL = "./public/camera_path_lab/room_scan_textured.glb"', script)
        self.assertIn("const ROOM_MESH_OPACITY = 0.82", script)
        self.assertIn("if (material.map) material.map.colorSpace = THREE.SRGBColorSpace", script)
        self.assertIn("Textured room scan · GPU display", script)
        self.assertIn('camera-path-lab.js?v=20260804-flexible-segment-sync-v2', html)
        self.assertIn("const POSITION_ONLY_CAMERA_MARKER = true", script)
        self.assertIn("rig.userData.positionOnlyMarker", script)
        self.assertIn("model.visible = !POSITION_ONLY_CAMERA_MARKER", script)
        self.assertIn('id="toggle-ceiling"', html)
        self.assertIn("renderer.localClippingEnabled = true", script)
        self.assertIn("function updateCeilingVisibility()", script)
        self.assertIn("ceilingCutY = bounds.min.y + size.y * 0.64", script)
        self.assertIn("const ROOM_SCAN_SOURCE_FLOOR_Y = -1.362", script)
        self.assertIn("const ROOM_SCAN_SOURCE_AXIS_DEG = 174.616", script)
        self.assertIn("const ROOM_SCAN_SOURCE_CENTER_XZ", script)
        self.assertIn("const ROOM_SCAN_SOURCE_LONG_M", script)
        self.assertIn("const ROOM_SCAN_REFINEMENT_YAW_DEG", script)
        self.assertIn("const ROOM_SCAN_REFINEMENT_XZ", script)
        self.assertIn("const FLOOR_GRID_CLEARANCE_M = 0.025", script)
        self.assertIn("function alignRoomScanForDisplay(roomScan)", script)
        self.assertIn("roomLongAxisDeg - ROOM_SCAN_SOURCE_AXIS_DEG", script)
        self.assertIn("roomLongAxisLength / ROOM_SCAN_SOURCE_LONG_M", script)
        self.assertIn("makeScale(horizontalScale, 1, horizontalScale)", script)
        self.assertIn("refinement.multiply(correction)", script)
        self.assertIn("roomFootprintCenter.x", script)
        self.assertIn("floorGrid.position.y = roomFloorY - FLOOR_GRID_CLEARANCE_M", script)
        self.assertIn("alignRoomScanForDisplay(gltf.scene)", script)
        self.assertIn('<video id="source-video" autoplay muted playsinline', html)
        self.assertIn('window.location.protocol === "file:"', html)
        self.assertIn('window.location.replace("http://127.0.0.1:8767/camera-path-lab.html")', html)
        stylesheet = (ROOT / "viewer" / "camera-path-lab.css").read_text(encoding="utf-8")
        self.assertIn("color-scheme: dark", stylesheet)
        self.assertIn("linear-gradient(135deg, #061725", stylesheet)
        self.assertIn(".playback-speed-control", stylesheet)
        self.assertIn(".preview-transform-controls", stylesheet)
        self.assertIn(".preview-timing-segments", stylesheet)
        self.assertIn(".preview-timing-segments.collapsed", stylesheet)
        self.assertIn(".timing-segment-row", stylesheet)
        self.assertIn("aspect-ratio: var(--phone-video-aspect, 9 / 16)", stylesheet)
        self.assertIn("right: calc(var(--video-panel-width) + var(--video-panel-gap))", stylesheet)
        self.assertIn(".video-size-control", stylesheet)
        self.assertIn("CAMERA PATH", html)
        self.assertNotIn("atlas", (html + script + stylesheet).lower())
        self.assertNotIn("3D preview unavailable here", script)

    def test_textured_room_scan_is_aligned_and_visual_only(self):
        asset = ROOT / "viewer" / "public" / "camera_path_lab" / "room_scan_textured.glb"
        metadata_path = asset.with_suffix(".json")
        builder = (ROOT / "scripts" / "build_camera_path_scan_mesh.py").read_text(encoding="utf-8")
        self.assertIn('"room_scan_textured.glb"', builder)
        if not asset.is_file() or not metadata_path.is_file():
            self.skipTest("Local scan-derived display asset has not been generated.")
        self.assertGreater(asset.stat().st_size, 1_000_000)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["format"], "camera-path-textured-room-scan-v1")
        self.assertEqual(metadata["reference_map_id"], "map_copy_20260730_114851_cfefdc")
        self.assertTrue(metadata["visual_only"])
        self.assertGreater(metadata["vertices"], 1_000_000)
        self.assertGreater(metadata["faces"], 1_000_000)
        matrix = np.asarray(metadata["scan_to_room_matrix"], dtype=np.float64)
        self.assertEqual(matrix.shape, (4, 4))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertLess(metadata["alignment"]["median_source_to_reference_m"], 0.5)

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
        self.assertIn('cfg.get("camera_path_lab_blocking_global_recovery", False)', source)
        self.assertIn('cfg.get("camera_path_lab_min_track_points", 6)', source)
        self.assertIn('"--follow-all-frames"', source)
        self.assertIn('"--wait-for-background-recovery"', source)
        self.assertIn('"--direct-pnp-recovery"', source)
        self.assertIn('cfg.get("camera_path_lab_reference_image_cap", 36)', source)
        self.assertIn('cfg.get("camera_path_lab_global_recovery_max_step", 1.6)', source)
        self.assertIn('cfg.get("camera_path_lab_output_max_step", 1.6)', source)
        self.assertIn('cfg.get("camera_path_lab_output_max_speed", 2.2)', source)
        self.assertIn("def extract_live_frames()", source)
        self.assertIn("target=monitor_partial_pose_file", source)
        self.assertIn("update_camera_path_lab_stream,", source)
        self.assertIn(
            "Camera Path localization writes accepted and held observations",
            source,
        )
        self.assertIn('if blocking_recovery:', source)
        self.assertIn('failure_pose_path = failure_asset_dir / "poses.json"', source)
        self.assertIn('final_pose_url=public_rel(failure_pose_path)', source)
        self.assertIn('url.path == "/api/camera-path-lab/status"', source)
        self.assertIn('"/api/camera-path-lab/upload"', source)
        self.assertIn('url.path == "/api/camera-path-lab/preview-calibration"', source)
        self.assertIn("def save_camera_path_preview_calibration(", source)

    def test_camera_lab_snapshot_is_a_copy(self):
        SERVER.set_camera_path_lab_stream({"pose_count": 3})
        snapshot = SERVER.camera_path_lab_snapshot()
        snapshot["stream"]["pose_count"] = 99
        self.assertEqual(SERVER.camera_path_lab_snapshot()["stream"]["pose_count"], 3)

    def test_preview_path_calibration_is_scoped_saved_and_restored(self):
        original_path = SERVER.CAMERA_PATH_PREVIEW_CALIBRATIONS
        with SERVER.CAMERA_PATH_LAB_LOCK:
            original_state = json.loads(json.dumps(SERVER.CAMERA_PATH_LAB_STATE))
        try:
            with tempfile.TemporaryDirectory() as directory:
                SERVER.CAMERA_PATH_PREVIEW_CALIBRATIONS = Path(directory) / "preview_calibrations.json"
                with SERVER.CAMERA_PATH_LAB_LOCK:
                    SERVER.CAMERA_PATH_LAB_STATE.update(
                        {
                            "status": "running",
                            "message": "Preview active",
                            "updated_at": 1.0,
                            "stream": {
                                "replay_id": "preview_test",
                                "validation_preview": True,
                            },
                        }
                    )
                calibration = SERVER.save_camera_path_preview_calibration(
                    {
                        "replay_id": "preview_test",
                        "target_start": [-4.2, 0.1, -0.7],
                        "movement_scale": 0.35,
                        "movement_scale_x": 0.4,
                        "movement_scale_y": 0.3,
                        "rotation_deg": 17.0,
                        "timing_offset_sec": -4.2,
                        "timing_segments": [
                            {"start_frame": 0, "end_frame": 333, "offset_sec": -4.2},
                            {"start_frame": 334, "end_frame": 599, "offset_sec": 2.0},
                            {"start_frame": 600, "end_frame": 749, "offset_sec": -1.5},
                            {"start_frame": 750, "end_frame": None, "offset_sec": 1.0},
                        ],
                    }
                )
                self.assertTrue(calibration["locked"])
                self.assertEqual(calibration["target_start"], [-4.2, 0.1, -0.7])
                self.assertEqual(calibration["timing_offset_sec"], -4.2)
                self.assertEqual(
                    [segment["offset_sec"] for segment in calibration["timing_segments"]],
                    [-4.2, 2.0, -1.5, 1.0],
                )
                self.assertEqual(calibration["movement_scale_x"], 0.4)
                self.assertEqual(calibration["movement_scale_y"], 0.3)
                self.assertEqual(calibration["rotation_deg"], 17.0)
                snapshot = SERVER.camera_path_lab_snapshot()
                self.assertEqual(snapshot["stream"]["preview_calibration"]["movement_scale"], 0.35)
                self.assertEqual(snapshot["stream"]["preview_calibration"]["timing_offset_sec"], -4.2)
                self.assertEqual(
                    snapshot["stream"]["preview_calibration"]["timing_segments"][2]["end_frame"],
                    749,
                )
                self.assertEqual(snapshot["stream"]["preview_calibration"]["rotation_deg"], 17.0)
                expanded = SERVER.save_camera_path_preview_calibration(
                    {
                        "replay_id": "preview_test",
                        "target_start": [-4.2, 0.1, -0.7],
                        "movement_scale": 0.35,
                        "timing_segments": [
                            {"start_frame": 0, "end_frame": 166, "offset_sec": -4.2},
                            {"start_frame": 167, "end_frame": 333, "offset_sec": -3.0},
                            {"start_frame": 334, "end_frame": 599, "offset_sec": 2.0},
                            {"start_frame": 600, "end_frame": 749, "offset_sec": -1.5},
                            {"start_frame": 750, "end_frame": None, "offset_sec": 1.0},
                        ],
                    }
                )
                self.assertEqual(len(expanded["timing_segments"]), 5)
                self.assertEqual(expanded["timing_segments"][1]["start_frame"], 167)
                with self.assertRaisesRegex(ValueError, "contiguous"):
                    SERVER.save_camera_path_preview_calibration(
                        {
                            "replay_id": "preview_test",
                            "target_start": [-4.2, 0.1, -0.7],
                            "movement_scale": 0.35,
                            "timing_segments": [
                                {"start_frame": 0, "end_frame": 200, "offset_sec": 0.0},
                                {"start_frame": 202, "end_frame": None, "offset_sec": 1.0},
                            ],
                        }
                    )
        finally:
            SERVER.CAMERA_PATH_PREVIEW_CALIBRATIONS = original_path
            with SERVER.CAMERA_PATH_LAB_LOCK:
                SERVER.CAMERA_PATH_LAB_STATE.clear()
                SERVER.CAMERA_PATH_LAB_STATE.update(original_state)

    def test_video_byte_ranges_support_seek_and_replay(self):
        self.assertEqual(SERVER.parse_http_byte_range("bytes=100-199", 1000), (100, 199))
        self.assertEqual(SERVER.parse_http_byte_range("bytes=900-", 1000), (900, 999))
        self.assertEqual(SERVER.parse_http_byte_range("bytes=-100", 1000), (900, 999))
        with self.assertRaises(ValueError):
            SERVER.parse_http_byte_range("bytes=1000-", 1000)

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

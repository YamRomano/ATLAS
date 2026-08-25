import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MapMeshOverlayTests(unittest.TestCase):
    def test_good_copy_has_read_only_visual_mesh_registry(self):
        registry = json.loads((ROOT / "viewer/public/map_visual_layers.json").read_text())
        layer = registry["layers"][0]
        self.assertEqual(layer["root_map_id"], "map_copy_20260730_114851_cfefdc")
        self.assertTrue(layer["inherit_to_copies"])
        self.assertEqual(layer["alignment"]["mode"], "fixed_room_footprint")
        self.assertIn("target_footprint", layer["alignment"])
        self.assertEqual(layer["alignment"]["target_footprint"]["source"], "colmap_safety_wall_footprint")
        self.assertAlmostEqual(layer["alignment"]["target_footprint"]["axis_deg"], 172.381)
        self.assertAlmostEqual(layer["alignment"]["target_footprint"]["short_m"], 6.787087)
        self.assertAlmostEqual(layer["alignment"]["source_short_m"], 7.289)
        audit = layer["alignment"]["registration_audit"]
        self.assertEqual(audit["method"], "two_axis_room_footprint_plus_user_corner_projection")
        self.assertEqual(layer["alignment"]["visual_anchor_offset_xz"], [2.74, -0.10])
        self.assertEqual(audit["anchor_screen_delta_px"], [55, 85])
        self.assertEqual(audit["anchor_room_delta_m"], [1.32, -1.54])
        self.assertEqual(audit["visual_anchor_offset_xz"], [2.74, -0.10])
        self.assertTrue(audit["automatic_camera_facing_wall_cutaway"])
        self.assertTrue(audit["visual_only"])
        self.assertTrue((ROOT / "viewer" / layer["url"]).is_file())

    def test_armour_map_reuses_mesh_without_inheriting_reference_localization(self):
        registry = json.loads((ROOT / "viewer/public/map_visual_layers.json").read_text())
        armour_layer = next(
            layer
            for layer in registry["layers"]
            if layer["root_map_id"] == "video_map_20260823_191301_f1572f"
        )
        baseline_layer = next(
            layer
            for layer in registry["layers"]
            if layer["root_map_id"] == "map_copy_20260730_114851_cfefdc"
        )
        self.assertEqual(armour_layer["url"], baseline_layer["url"])
        self.assertTrue(armour_layer["alignment"]["registration_audit"]["shared_asset"])

        manifest = json.loads((ROOT / "viewer/public/maps/manifest.json").read_text())
        armour_map = next(
            entry
            for entry in manifest["maps"]
            if entry["id"] == "video_map_20260823_191301_f1572f"
        )
        self.assertIsNone(armour_map["source_map_id"])
        self.assertIsNone(armour_map["localization_map_id"])
        self.assertEqual(
            armour_map["coordinate_frame_id"],
            "map_copy_20260730_114851_cfefdc",
        )

    def test_main_viewer_uses_physical_room_registration_plus_visual_anchor(self):
        registry = json.loads((ROOT / "viewer/public/map_visual_layers.json").read_text())
        alignment = registry["layers"][0]["alignment"]
        target = alignment["target_footprint"]
        self.assertAlmostEqual(target["long_m"], 12.001491)
        self.assertAlmostEqual(target["short_m"], 6.787087)
        self.assertNotIn("operator_anchor_offset_xz", alignment)
        self.assertEqual(alignment["visual_anchor_offset_xz"], [2.74, -0.10])

    def test_registered_footprint_matches_saved_map_wall_corners(self):
        registry = json.loads((ROOT / "viewer/public/map_visual_layers.json").read_text())
        layer = registry["layers"][0]
        target = layer["alignment"]["target_footprint"]
        map_export = json.loads(
            (ROOT / "viewer/public/camera_path_lab/good_copy_mesh.json").read_text()
        )
        self.assertEqual(map_export["source_map_id"], layer["root_map_id"])
        walls = map_export["safety_barriers"]
        corners = {
            (round(float(point[0]), 9), round(float(point[2]), 9))
            for wall in walls
            for point in (wall["a"], wall["b"])
        }
        self.assertEqual(len(corners), 4)
        center = (
            sum(point[0] for point in corners) / 4,
            sum(point[1] for point in corners) / 4,
        )
        lengths = sorted(
            math.dist((wall["a"][0], wall["a"][2]), (wall["b"][0], wall["b"][2]))
            for wall in walls
        )
        self.assertAlmostEqual(target["center_xz"][0], center[0], places=6)
        self.assertAlmostEqual(target["center_xz"][1], center[1], places=6)
        self.assertAlmostEqual(target["short_m"], sum(lengths[:2]) / 2, places=5)
        self.assertAlmostEqual(target["long_m"], sum(lengths[2:]) / 2, places=5)

    def test_viewer_has_lazy_mesh_toggle_and_separate_canvas(self):
        html = (ROOT / "viewer/index.html").read_text()
        self.assertIn('id="map-mesh"', html)
        self.assertIn('id="toggle-mesh"', html)
        self.assertIn('id="toggle-mesh-ceiling"', html)
        self.assertIn('id="align-map-mesh"', html)
        self.assertIn('id="lock-map-mesh"', html)
        self.assertIn("20260806-one-time-placement-v8", html)
        self.assertIn('id="restore-mesh-wall"', html)
        self.assertIn('src="map_mesh_overlay.js?v=20260806-one-time-placement-v8"', html)

        styles = (ROOT / "viewer/style.css").read_text()
        self.assertIn('body[data-map-mesh-placement="editing"] .map-control-dock', styles)
        self.assertIn('body[data-map-mesh-placement="editing"] #lock-map-mesh', styles)

        script = (ROOT / "viewer/map_mesh_overlay.js").read_text()
        self.assertIn("inherit_to_copies", script)
        self.assertIn("source_map_id", script)
        self.assertIn("room_footprint", script)
        self.assertIn("fixed_room_footprint", script)
        self.assertIn("display-only", script)
        self.assertIn("renderer.localClippingEnabled = true", script)
        self.assertIn("material.clippingPlanes", script)
        self.assertIn("isCeilingHidden", script)
        self.assertIn("calibratedFootprint", script)
        self.assertIn('addEventListener("dblclick"', script)
        self.assertIn("wallCutawayPlaneRoom", script)
        self.assertIn("isWallCutawayActive", script)
        self.assertIn("automaticWallCutawayPlaneRoom", script)
        self.assertIn("syncAutomaticWallCutaway", script)
        self.assertIn("isAutomaticWallCutawayActive", script)
        self.assertIn("dataset.mapMeshAutomaticWall", script)
        self.assertIn("atlas.map-mesh-placement.v1", script)
        self.assertIn("roomFloorPointFromClient", script)
        self.assertIn("lockPlacement", script)
        self.assertIn("dataset.mapMeshPlacement", script)
        self.assertIn("getEffectiveVisualAnchorOffsetXZ", script)
        self.assertIn("isPlacementLocked", script)
        self.assertIn("localStorage.setItem", script)
        self.assertIn('document.getElementById("reset")?.click()', script)
        self.assertIn("mainCanvas.getBoundingClientRect()", script)
        self.assertIn("canvas.offsetParent?.getBoundingClientRect?.()", script)
        self.assertIn('canvas.style.inset = "auto"', script)
        self.assertIn("getRegistrationVersion", script)
        self.assertIn("dataset.mapMeshRegistration", script)
        self.assertIn("source_short_m", script)
        self.assertIn("shortScale", script)
        self.assertIn("visual_anchor_offset_xz", script)
        self.assertIn("getVisualAnchorOffsetXZ", script)
        self.assertIn("layer?.version", script)
        self.assertNotIn('method: "POST"', script)
        self.assertNotIn("/api/maps/", script)

    def test_overlay_reads_current_map_without_mutating_it(self):
        app = (ROOT / "viewer/app.js").read_text()
        self.assertIn("getCurrentMapEntry: () => currentMapEntry", app)
        self.assertIn("getMapLibrary: () => mapLibraryData", app)
        self.assertIn("isMapInteractionBusy: () => Boolean", app)

    def test_live_view_keeps_held_pose_and_orders_async_results_by_frame(self):
        app = (ROOT / "viewer/app.js").read_text()
        self.assertIn("chronologicalPoseObservations(roomPoses)", app)
        self.assertIn("poseObservationFrameIndex", app)
        self.assertIn("pose.success && !pose.held_pose", app)

    def test_orbit_pivot_uses_oriented_footprint_midpoint(self):
        app = (ROOT / "viewer/app.js").read_text()
        html = (ROOT / "viewer/index.html").read_text()
        self.assertIn(
            "const center = toRoom(0.5 * (u0 + u1), 0.5 * (v0 + v1), 0.5 * (y0 + y1));",
            app,
        )
        self.assertIn("return { bottom, top, yaw, center };", app)
        self.assertNotIn(
            "return { bottom, top, yaw, center: [cx, 0.5 * (y0 + y1), cz] };",
            app,
        )
        self.assertIn('src="app.js?v=211"', html)


if __name__ == "__main__":
    unittest.main()

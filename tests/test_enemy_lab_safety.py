import importlib.util
import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path


sys.modules.setdefault("cv2", types.ModuleType("cv2"))
ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "scripts" / "atlas_app_server.py"
APP_PATH = ROOT / "viewer" / "app.js"
INDEX_PATH = ROOT / "viewer" / "index.html"
TRAIN_PATH = ROOT / "scripts" / "train_enemy_yolo.py"
FIXED_ENHANCE_PATH = ROOT / "scripts" / "enhance_colmap_fixed_reference.py"
BRIDGE_PATH = ROOT / "scripts" / "atlas_dji_live_bridge.py"

SPEC = importlib.util.spec_from_file_location("atlas_app_server_enemy_test", SERVER_PATH)
assert SPEC and SPEC.loader
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)

BRIDGE_SPEC = importlib.util.spec_from_file_location("atlas_dji_enemy_pursuit_test", BRIDGE_PATH)
assert BRIDGE_SPEC and BRIDGE_SPEC.loader
bridge = importlib.util.module_from_spec(BRIDGE_SPEC)
BRIDGE_SPEC.loader.exec_module(bridge)


class EnemyLabSafetyTests(unittest.TestCase):
    def test_patrol_one_and_two_resolve_to_independent_live_profiles(self):
        library = server.load_library()
        locks = server.resolved_live_patrol_locks(library)
        by_patrol = {lock["patrol_id"]: lock for lock in locks}

        patrol_one_id = "patrol_ms4br5xr_4xclts"
        patrol_two_id = "patrol_mszqwnot_awshxl"
        self.assertEqual(server.resolved_live_patrol_lock(library)["patrol_id"], patrol_one_id)
        self.assertEqual(
            by_patrol[patrol_one_id]["baseline_replay_id"],
            "patrol_baseline_precision_20260813",
        )
        self.assertEqual(
            by_patrol[patrol_two_id]["baseline_replay_id"],
            "patrol2_hybrid_baseline_20260819",
        )
        self.assertNotEqual(
            by_patrol[patrol_one_id]["baseline_reference_path"],
            by_patrol[patrol_two_id]["baseline_reference_path"],
        )

    def test_patrol_one_geometry_is_preserved_and_patrol_two_matches_its_frame_bank(self):
        library = server.load_library()
        map_entry = next(
            item for item in library["maps"]
            if item["id"] == "map_copy_20260730_114851_cfefdc"
        )
        patrols = {item["id"]: item for item in map_entry["patrols"]}
        patrol_one = patrols["patrol_ms4br5xr_4xclts"]
        patrol_two = patrols["patrol_mszqwnot_awshxl"]
        self.assertEqual(
            [point["rxyz"] for point in patrol_one["points"]],
            [
                [-3.2329557447702215, -0.17877615459930907, -0.33236579860361815],
                [-0.6480244338911889, -0.48003389609676617, -0.48774887233005093],
                [-0.4886978074319452, 1.7599605504441844, 0.9560112230666532],
                [-3.0736291183109774, -0.20678855225328296, 1.1113942967930859],
            ],
        )
        patrol_two_reference = json.loads(
            (
                ROOT
                / "viewer/public/maps/map_copy_20260730_114851_cfefdc/replays"
                / "patrol2_hybrid_baseline_20260819/reference_candidate.json"
            ).read_text(encoding="utf-8")
        )
        frame_bank_points = [leg["from"] for leg in patrol_two_reference["legs"]]
        self.assertEqual(
            [[point["rxyz"][0], point["rxyz"][2]] for point in patrol_two["points"]],
            [[point[0], point[2]] for point in frame_bank_points],
        )
        self.assertEqual(patrol_two["points"][1], patrol_one["points"][1])
        self.assertEqual(patrol_two["points"][2], patrol_one["points"][2])

    def test_taught_recovery_banks_never_cross_patrol_directories(self):
        library = server.load_library()
        map_entry = next(
            item for item in library["maps"]
            if item["id"] == "map_copy_20260730_114851_cfefdc"
        )
        base = server.map_asset_dir(map_entry)
        patrol_one_banks = server.active_taught_recovery_banks(
            base, "patrol_ms4br5xr_4xclts"
        )
        patrol_two_banks = server.active_taught_recovery_banks(
            base, "patrol_mszqwnot_awshxl"
        )
        self.assertTrue(patrol_one_banks)
        self.assertTrue(all(path.parent.name == "patrol_ms4br5xr_4xclts" for path in patrol_one_banks))
        self.assertTrue(all(path.parent.name == "patrol_mszqwnot_awshxl" for path in patrol_two_banks))

    def test_live_ui_selects_and_binds_one_patrol_before_localization(self):
        source = APP_PATH.read_text(encoding="utf-8")
        html = INDEX_PATH.read_text(encoding="utf-8")
        self.assertIn('id="live-atlas-patrol"', html)
        self.assertIn("configuredLivePatrolProfiles", source)
        self.assertIn("patrol_id: patrolId || null", source)
        self.assertIn("This localization session is isolated to another patrol", source)

    def test_live_check_mode_blocks_every_flight_command(self):
        status = {
            "status": "streaming",
            "updated_at": time.time(),
            "control_enabled": False,
            "view_only": True,
        }
        for command in ("takeoff", "land", "hover", "enable", "disable", "mission"):
            ready, reason = server.dji_live_bridge_readiness(status, command)
            self.assertFalse(ready)
            self.assertIn("disabled", reason)

    def test_training_entry_point_exists_and_compiles(self):
        self.assertTrue(TRAIN_PATH.exists())
        compile(TRAIN_PATH.read_text(encoding="utf-8"), str(TRAIN_PATH), "exec")

    def test_live_response_requires_fresh_multi_frame_confirmation(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("ENEMY_CONFIRM_HITS = 3", source)
        self.assertIn("ENEMY_CONFIRM_WINDOW = 5", source)
        self.assertIn("enemyDetectionIsFresh(payload)", source)

    def test_enemy_detection_is_operator_opt_in_and_cannot_interrupt_when_off(self):
        source = APP_PATH.read_text(encoding="utf-8")
        html = INDEX_PATH.read_text(encoding="utf-8")
        self.assertIn('id="enemy-detection-enabled"', html)
        self.assertNotIn('id="enemy-detection-enabled" type="checkbox" checked', html)
        self.assertIn('localStorage.getItem(ENEMY_DETECTION_ENABLED_STORAGE_KEY) === "true"', source)
        poll_start = source.index("async function pollEnemyLiveDetections")
        poll_fetch = source.index("fetch(`public/live_dji/enemy_detections.json", poll_start)
        self.assertLess(source.index("if (!enemyDetectionEnabled())", poll_start), poll_fetch)
        pause_start = source.index("async function pauseForEnemyDetection")
        self.assertIn("if (!enemyDetectionEnabled()) return;", source[pause_start:pause_start + 180])
        self.assertIn("Enemy detection is off. Detector results cannot interrupt this patrol.", source)
        self.assertIn('postJson("/api/enemy-drone/live-detection", { enabled: Boolean(enabled) })', source)

    def test_runtime_enemy_detection_gate_fails_closed_and_clears_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "enemy_detection_control.json"
            detections = root / "enemy_detections.json"
            self.assertFalse(bridge.enemy_detection_control_enabled(control))
            control.write_text(json.dumps({"enabled": True}), encoding="utf-8")
            self.assertTrue(bridge.enemy_detection_control_enabled(control))
            payload = server.set_live_enemy_detection_enabled(False, control, detections)
            self.assertFalse(payload["enabled"])
            self.assertFalse(bridge.enemy_detection_control_enabled(control))
            cleared = json.loads(detections.read_text(encoding="utf-8"))
            self.assertEqual(cleared["status"], "disabled")
            self.assertEqual(cleared["detections"], [])

    def test_bridge_checks_runtime_gate_before_yolo_inference(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        loop = source.index("enemy_detection_is_enabled = enemy_detection_control_enabled")
        inference = source.index("detections = detect_enemy_drones", loop)
        self.assertLess(loop, inference)
        self.assertIn("enemy_detector is not None and enemy_detection_is_enabled", source[loop:inference])

    def test_lock_on_plan_has_no_forward_cruise(self):
        source = APP_PATH.read_text(encoding="utf-8")
        start = source.index("function buildEnemyLockOnPlan")
        end = source.index("async function confirmEnemyLockOn", start)
        self.assertNotIn('type: "cruise"', source[start:end])

    def test_dataset_uses_independent_split_directories(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        start = source.index("def prepare_enemy_yolo_dataset")
        end = source.index("def selected_enemy_model_path", start)
        implementation = source[start:end]
        self.assertIn('"val: images/val', implementation)
        self.assertIn('"test: images/test', implementation)
        self.assertNotIn('"val: images/train', implementation)
        self.assertIn('"negative_frame_count"', implementation)

    def test_enemy_manifest_save_is_atomic(self):
        original_dir = server.ENEMY_DIR
        original_manifest = server.ENEMY_MANIFEST
        try:
            with tempfile.TemporaryDirectory() as directory:
                server.ENEMY_DIR = Path(directory)
                server.ENEMY_MANIFEST = Path(directory) / "manifest.json"
                server.save_enemy_library(server.default_enemy_library())
                loaded = server.load_enemy_library()
                self.assertEqual(loaded["model_status"], "not_trained")
                self.assertFalse(list(Path(directory).glob("*.tmp")))
        finally:
            server.ENEMY_DIR = original_dir
            server.ENEMY_MANIFEST = original_manifest

    def test_range_calibration_requires_and_validates_varied_measured_samples(self):
        original_dir = server.ENEMY_DIR
        original_manifest = server.ENEMY_MANIFEST
        try:
            with tempfile.TemporaryDirectory() as directory:
                server.ENEMY_DIR = Path(directory)
                server.ENEMY_MANIFEST = Path(directory) / "manifest.json"
                profile = server.normalize_enemy_profile(
                    {"id": "enemy_neo", "name": "NEO", "class_name": "neo", "videos": []}
                )
                lib = server.default_enemy_library()
                lib["enemies"] = [profile]
                server.save_enemy_library(lib)
                scale = 0.32
                for distance in (0.60, 0.60, 0.80, 0.80, 1.00, 1.00, 1.35, 1.35):
                    area = (scale / distance) ** 2
                    side = area ** 0.5
                    server.capture_enemy_range_sample(
                        "enemy_neo",
                        distance,
                        {
                            "status": "detected",
                            "frame": f"frame_{distance}",
                            "updated_at": 1.0,
                            "detections": [
                                {
                                    "class_name": "neo",
                                    "confidence": 0.95,
                                    "box": {"width": side, "height": side},
                                }
                            ],
                        },
                    )
                result = server.fit_enemy_range_calibration("enemy_neo", 0.50)
                self.assertTrue(result["validation"]["accepted"])
                calibration = result["enemy"]["range_calibration"]
                self.assertEqual(calibration["status"], "validated")
                self.assertAlmostEqual(calibration["model"]["scale"], scale, places=6)
                self.assertGreaterEqual(calibration["model"]["conservative_margin_m"], 0.12)
        finally:
            server.ENEMY_DIR = original_dir
            server.ENEMY_MANIFEST = original_manifest

    def test_moving_target_prediction_and_calibrated_range_are_bounded(self):
        history = [
            {"updated_at": 1.0, "center_x": 0.40, "center_y": 0.50},
            {"updated_at": 1.5, "center_x": 0.50, "center_y": 0.50},
            {"updated_at": 2.0, "center_x": 0.60, "center_y": 0.50},
        ]
        predicted_x, predicted_y = bridge.predict_enemy_image_center(history, 0.30)
        self.assertGreater(predicted_x, 0.60)
        self.assertAlmostEqual(predicted_y, 0.50, places=6)
        estimate, margin = bridge.estimate_enemy_clearance(
            {"area": 0.16},
            {"type": "inverse_sqrt_area", "scale": 0.40, "conservative_margin_m": 0.12},
        )
        self.assertAlmostEqual(estimate, 1.0, places=6)
        self.assertAlmostEqual(margin, 0.12, places=6)
        width_estimate, _ = bridge.estimate_enemy_clearance(
            {"width": 0.20},
            {"type": "inverse_width", "scale": 0.30, "conservative_margin_m": 0.12},
        )
        self.assertAlmostEqual(width_estimate, 1.5, places=6)
        normalized = server.normalize_enemy_range_calibration(
            {
                "status": "validated",
                "model": {"type": "inverse_width", "scale": 0.30, "conservative_margin_m": 0.12},
            }
        )
        self.assertEqual(normalized["model"]["type"], "inverse_width")

    def test_approaching_target_predicts_an_earlier_stop(self):
        history = [
            {"updated_at": 1.0, "clearance_m": 1.20},
            {"updated_at": 1.2, "clearance_m": 1.00},
            {"updated_at": 1.4, "clearance_m": 0.80},
        ]
        smoothed, predicted = bridge.predict_enemy_clearance(history, 0.35)
        self.assertAlmostEqual(smoothed, 1.0, places=6)
        self.assertLess(predicted, 0.80)

    def test_tracked_detection_rejects_large_identity_jump(self):
        payload = {
            "detections": [
                {
                    "class_name": "neo",
                    "confidence": 0.99,
                    "box": {"x1": 0.80, "y1": 0.80, "width": 0.10, "height": 0.10},
                }
            ]
        }
        selected = bridge.select_tracked_enemy_detection(payload, "neo", (0.20, 0.20), 0.35)
        self.assertIsNone(selected)

    def test_guarded_pursuit_follows_fresh_frames_and_stops_conservatively(self):
        class FakeDrone:
            def __init__(self):
                self.moves = []
                self.enabled = False

            def enableControl(self, _wait):
                self.enabled = True
                return None

            def disableControl(self, _wait):
                self.enabled = False
                return None

            def move(self, yaw, vertical, lateral, forward, _wait):
                self.moves.append((yaw, vertical, lateral, forward))
                return None

        original_gate = bridge.latest_tsolve_pose_gate
        original_hover = bridge.neutral_hover
        original_pulse = bridge.execute_rc_pulse
        fake = FakeDrone()
        sent_pulses = []
        try:
            bridge.latest_tsolve_pose_gate = lambda *_args, **_kwargs: {
                "ok": True,
                "pose": {"rcenter": [0.0, 0.0, 0.0], "received_unix": time.time()},
            }
            bridge.neutral_hover = lambda drone, _duration: drone.move(0.0, 0.0, 0.0, 0.0, False)

            def fake_pulse(drone, *, yaw=0.0, du=0.0, lr=0.0, bf=0.0, seconds=0.1):
                sent = {"yaw": yaw, "vertical": du, "lateral": lr, "forward": bf, "seconds": seconds}
                sent_pulses.append(sent)
                drone.move(yaw, du, lr, bf, False)
                return sent

            bridge.execute_rc_pulse = fake_pulse
            with tempfile.TemporaryDirectory() as directory:
                detection_path = Path(directory) / "enemy.json"

                def write_frame(index, clearance):
                    width = 0.30 / clearance
                    payload = {
                        "status": "detected",
                        "frame": f"frame_{index}",
                        "updated_at": time.time(),
                        "detections": [
                            {
                                "class_name": "neo1",
                                "confidence": 0.95,
                                "box": {
                                    "x1": 0.5 - width / 2,
                                    "y1": 0.40,
                                    "width": width,
                                    "height": 0.20,
                                },
                            }
                        ],
                    }
                    bridge.atomic_write_json(detection_path, payload)

                write_frame(1, 1.50)

                def update_detections():
                    for index, clearance in enumerate((1.50, 1.50, 1.30, 1.05, 0.82, 0.60), start=2):
                        time.sleep(0.06)
                        write_frame(index, clearance)

                updater = threading.Thread(target=update_detections, daemon=True)
                updater.start()
                result = bridge.execute_guarded_enemy_pursuit(
                    fake,
                    {
                        "guided_enabled": True,
                        "operator_confirmed": True,
                        "client_safety_version": 3,
                        "target_class_name": "neo1",
                        "stop_clearance_m": 0.50,
                        "max_pursuit_seconds": 5.0,
                        "pursuit_yaw_sign": 1,
                        "range_model": {
                            "type": "inverse_width",
                            "scale": 0.30,
                            "conservative_margin_m": 0.12,
                            "trained_min_clearance_m": 0.50,
                            "trained_max_clearance_m": 2.00,
                        },
                        "safety_barriers": [
                            {"label": "Wall 1", "a": [-3, -1, -3], "b": [3, -1, -3], "corners": [[-3, -1, -3], [3, -1, -3], [3, 3, -3], [-3, 3, -3]], "clearance_m": 0.45},
                            {"label": "Wall 2", "a": [3, -1, -3], "b": [3, -1, 3], "corners": [[3, -1, -3], [3, -1, 3], [3, 3, 3], [3, 3, -3]], "clearance_m": 0.45},
                            {"label": "Wall 3", "a": [3, -1, 3], "b": [-3, -1, 3], "corners": [[3, -1, 3], [-3, -1, 3], [-3, 3, 3], [3, 3, 3]], "clearance_m": 0.45},
                            {"label": "Wall 4", "a": [-3, -1, 3], "b": [-3, -1, -3], "corners": [[-3, -1, 3], [-3, -1, -3], [-3, 3, -3], [-3, 3, 3]], "clearance_m": 0.45},
                        ],
                    },
                    pose_stream_path=Path(directory) / "poses.json",
                    enemy_detection_path=detection_path,
                )
                updater.join(timeout=1.0)
            self.assertTrue(result["ok"], result)
            self.assertTrue(result["reached"])
            self.assertGreaterEqual(result["rc_summary"]["pulse_counts"]["forward"], 1)
            self.assertEqual(result["rc_summary"]["pulse_counts"]["vertical"], 0)
            self.assertTrue(all(pulse["lateral"] == 0.0 for pulse in sent_pulses))
            self.assertTrue(all(pulse["forward"] >= 0.0 for pulse in sent_pulses))
            self.assertFalse(fake.enabled)
        finally:
            bridge.latest_tsolve_pose_gate = original_gate
            bridge.neutral_hover = original_hover
            bridge.execute_rc_pulse = original_pulse

    def test_pursuit_geofence_blocks_wall_and_outside_room(self):
        walls = [
            {"label": "South", "a": [-2, -1, -2], "b": [2, -1, -2], "corners": [[-2, -1, -2], [2, -1, -2], [2, 3, -2], [-2, 3, -2]], "clearance_m": 0.45},
            {"label": "East", "a": [2, -1, -2], "b": [2, -1, 2], "corners": [[2, -1, -2], [2, -1, 2], [2, 3, 2], [2, 3, -2]], "clearance_m": 0.45},
            {"label": "North", "a": [2, -1, 2], "b": [-2, -1, 2], "corners": [[2, -1, 2], [-2, -1, 2], [-2, 3, 2], [2, 3, 2]], "clearance_m": 0.45},
            {"label": "West", "a": [-2, -1, 2], "b": [-2, -1, -2], "corners": [[-2, -1, 2], [-2, -1, -2], [-2, 3, -2], [-2, 3, 2]], "clearance_m": 0.45},
        ]
        self.assertIsNone(bridge.pursuit_geofence_issue([0.0, 1.0, 0.0], walls, []))
        self.assertIn("East", bridge.pursuit_geofence_issue([1.4, 1.0, 0.0], walls, []))
        self.assertIn("outside", bridge.pursuit_geofence_issue([3.0, 1.0, 0.0], walls, []))

    def test_geofence_adds_at_least_thirty_centimeters_to_saved_clearance(self):
        walls = [
            {"label": "South", "a": [-2, -1, -2], "b": [2, -1, -2], "corners": [[-2, -1, -2], [2, -1, -2], [2, 3, -2], [-2, 3, -2]], "clearance_m": 0.45},
            {"label": "East", "a": [2, -1, -2], "b": [2, -1, 2], "corners": [[2, -1, -2], [2, -1, 2], [2, 3, 2], [2, 3, -2]], "clearance_m": 0.45},
            {"label": "North", "a": [2, -1, 2], "b": [-2, -1, 2], "corners": [[2, -1, 2], [-2, -1, 2], [-2, 3, 2], [2, 3, 2]], "clearance_m": 0.45},
            {"label": "West", "a": [-2, -1, 2], "b": [-2, -1, -2], "corners": [[-2, -1, 2], [-2, -1, -2], [-2, 3, -2], [-2, 3, 2]], "clearance_m": 0.45},
        ]
        self.assertIsNone(bridge.pursuit_geofence_issue([1.24, 1.0, 0.0], walls, []))
        issue = bridge.pursuit_geofence_issue([1.26, 1.0, 0.0], walls, [])
        self.assertIn("East", issue)
        self.assertIn("0.75 m", issue)

    def test_server_replaces_patrol_barriers_with_authoritative_saved_map_geometry(self):
        original_load_library = server.load_library
        try:
            server.load_library = lambda: {
                "maps": [{
                    "id": "map_safe",
                    "patrols": [{"id": "patrol_safe", "title": "Safe patrol"}],
                    "safety_barriers": [{"id": "wall_1"}, {"id": "wall_2"}, {"id": "wall_3"}, {"id": "wall_4"}],
                    "safety_obstacles": [{"id": "desk"}],
                }]
            }
            mission = server.validated_guarded_patrol_mission({
                "client_safety_version": 3,
                "patrol": True,
                "map_id": "map_safe",
                "patrol_id": "patrol_safe",
                "safety_barriers": [{"id": "forged"}],
                "safety_motion_buffer_m": 0.05,
            })
            self.assertEqual([item["id"] for item in mission["safety_barriers"]], ["wall_1", "wall_2", "wall_3", "wall_4"])
            self.assertEqual(mission["safety_obstacles"][0]["id"], "desk")
            self.assertAlmostEqual(mission["safety_motion_buffer_m"], 0.30)
        finally:
            server.load_library = original_load_library

    def test_live_patrol_lock_rejects_a_different_map_or_patrol(self):
        original_load_library = server.load_library
        try:
            server.load_library = lambda: {
                "live_patrol_lock": {
                    "enabled": True,
                    "map_id": "map_pinned",
                    "patrol_id": "patrol_1",
                },
                "maps": [
                    {
                        "id": "map_pinned",
                        "title": "Video Map 20:07:46 Copy",
                        "patrols": [{"id": "patrol_1", "title": "Patrol 1"}],
                        "safety_barriers": [{"id": "w1"}, {"id": "w2"}, {"id": "w3"}],
                    },
                    {
                        "id": "map_other",
                        "patrols": [{"id": "patrol_other"}],
                        "safety_barriers": [{"id": "w1"}, {"id": "w2"}, {"id": "w3"}],
                    },
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "pinned to Video Map 20:07:46 Copy / Patrol 1"):
                server.validated_guarded_patrol_mission(
                    {
                        "client_safety_version": 3,
                        "patrol": True,
                        "map_id": "map_other",
                        "patrol_id": "patrol_other",
                    }
                )
        finally:
            server.load_library = original_load_library

    def test_suspended_live_patrol_is_rejected_before_mission_queueing(self):
        original_load_library = server.load_library
        try:
            server.load_library = lambda: {
                "live_patrol_lock": {
                    "enabled": True,
                    "flight_enabled": False,
                    "map_id": "map_pinned",
                    "patrol_id": "patrol_1",
                    "suspension_reason": "raw and published poses diverged",
                },
                "maps": [{
                    "id": "map_pinned",
                    "title": "Video Map 20:07:46 Copy",
                    "patrols": [{"id": "patrol_1", "title": "Patrol 1"}],
                    "safety_barriers": [{"id": "w1"}, {"id": "w2"}, {"id": "w3"}],
                }],
            }
            with self.assertRaisesRegex(RuntimeError, "Live Patrol is suspended"):
                server.validated_guarded_patrol_mission(
                    {
                        "client_safety_version": 3,
                        "patrol": True,
                        "map_id": "map_pinned",
                        "patrol_id": "patrol_1",
                    }
                )
        finally:
            server.load_library = original_load_library

    def test_server_accepts_one_connected_entry_then_pins_exactly_two_laps(self):
        original_load_library = server.load_library
        original_resolved_lock = server.resolved_live_patrol_lock
        points = [
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ]
        try:
            server.load_library = lambda: {
                "maps": [{
                    "id": "map_pinned",
                    "title": "Video Map 20:07:46 Copy",
                    "patrols": [{
                        "id": "patrol_1",
                        "title": "Patrol 1",
                        "points": [{"rxyz": point} for point in points],
                    }],
                    "safety_barriers": [{"id": "w1"}, {"id": "w2"}, {"id": "w3"}],
                }],
            }
            server.resolved_live_patrol_lock = lambda _library, **_selection: {
                "map_id": "map_pinned",
                "map_title": "Video Map 20:07:46 Copy",
                "patrol_id": "patrol_1",
                "patrol_title": "Patrol 1",
                "flight_enabled": True,
                "suspension_reason": "",
                "reference_profile": "off",
                "baseline_replay_id": "baseline_full",
                "patrol_laps": 2,
            }
            connected_route = [[-1.0, 1.0, -0.4], points[0], *points[1:], points[0]]
            mission = server.validated_guarded_patrol_mission({
                "client_safety_version": 3,
                "patrol": True,
                "patrol_stage": "combined",
                "map_id": "map_pinned",
                "patrol_id": "patrol_1",
                "route": connected_route,
                "loop": True,
            })
            self.assertEqual(mission["patrol_stage"], "combined")
            self.assertEqual(mission["patrol_laps"], 2)
            self.assertTrue(mission["continuous_relocalization"])

            with self.assertRaisesRegex(RuntimeError, "Connected patrol must enter Point 1"):
                server.validated_guarded_patrol_mission({
                    "client_safety_version": 3,
                    "patrol": True,
                    "patrol_stage": "combined",
                    "map_id": "map_pinned",
                    "patrol_id": "patrol_1",
                    "route": [[-1.0, 1.0, -0.4], points[0], points[2], points[1], points[3], points[0]],
                    "loop": True,
                })
        finally:
            server.load_library = original_load_library
            server.resolved_live_patrol_lock = original_resolved_lock

    def test_successful_pursuit_resumes_interrupted_patrol_from_nearest_point(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('options.entryStrategy === "nearest"', source)
        self.assertIn('validatePatrolPreview(true, { entryStrategy: "nearest" })', source)
        self.assertIn("resumeInterruptedPatrolAfterPursuit", source)
        self.assertIn("enemyTargetSuppressedUntilClear", source)
        self.assertIn("FLIGHT_SAFETY_PULSE_BUFFER_M = 0.30", source)
        bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("enforce_patrol_geofence", bridge_source)
        self.assertIn('safety_motion_buffer = clamp_float(mission.get("safety_motion_buffer_m"), 0.30, 0.30, 1.0)', bridge_source)

    def test_enemy_pursuit_is_range_gated_and_uses_live_detection_stream(self):
        server_source = SERVER_PATH.read_text(encoding="utf-8")
        bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
        app_source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("validated_enemy_pursuit_mission", server_source)
        self.assertIn('calibration.get("status") != "validated"', server_source)
        self.assertIn("execute_guarded_enemy_pursuit", bridge_source)
        self.assertIn("enemy_detection_path.read_text", bridge_source)
        self.assertIn("confirmation_count >= confirmation_hits", bridge_source)
        self.assertIn("Start Guarded Pursuit", (ROOT / "viewer" / "index.html").read_text(encoding="utf-8"))
        self.assertIn("vertical_tracking_enabled: false", app_source)
        self.assertIn("pursuit_yaw_sign", app_source)
        self.assertIn('"enemy_live_detect_fps": 5.0', (ROOT / "config.json").read_text(encoding="utf-8"))

    def test_fixed_reference_enhancement_locks_existing_cameras(self):
        source = FIXED_ENHANCE_PATH.read_text(encoding="utf-8")
        self.assertIn('"mapper"', source)
        self.assertIn('"--Mapper.fix_existing_images"', source)
        self.assertIn("choose_largest_sparse_model", source)
        self.assertIn('relative_name = f"enhancement/{source.name}"', source)
        self.assertIn('"temporal_coverage"', source)
        self.assertIn('"max_reference_pose_delta"', source)
        self.assertIn('"coordinate_frame_preserved": True', source)

    def test_map_copy_uses_fixed_reference_pipeline(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn("enhance_fixed_reference_map_job(selected, videos)", source)
        self.assertIn('selected.get("kind") == "map_copy"', source)


if __name__ == "__main__":
    unittest.main()

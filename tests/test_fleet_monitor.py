import importlib.util
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "scripts" / "atlas_app_server.py"
SPEC = importlib.util.spec_from_file_location("atlas_app_server_fleet_test", SERVER_PATH)
SERVER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(SERVER)


class FleetMonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.public = Path(self.tmp.name) / "public"
        self.viewer = Path(self.tmp.name)
        self.fleet_dir = self.public / "fleet"
        self.patches = [
            patch.object(SERVER, "PUBLIC", self.public),
            patch.object(SERVER, "VIEWER", self.viewer),
            patch.object(SERVER, "FLEET_DIR", self.fleet_dir),
            patch.object(SERVER, "FLEET_MANIFEST", self.fleet_dir / "manifest.json"),
        ]
        for item in self.patches:
            item.start()
        SERVER.FLEET_SESSIONS.clear()

    def tearDown(self):
        SERVER.FLEET_SESSIONS.clear()
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def test_each_drone_requires_a_unique_android_endpoint(self):
        first = SERVER.upsert_fleet_drone({"name": "Scout One", "phone_ip": "192.168.50.235"})
        second = SERVER.upsert_fleet_drone({"name": "Scout Two", "phone_ip": "192.168.50.236"})
        self.assertNotEqual(first["id"], second["id"])
        with self.assertRaisesRegex(RuntimeError, "already assigned"):
            SERVER.upsert_fleet_drone({"name": "Duplicate Link", "phone_ip": "192.168.50.235"})

    def test_fleet_snapshot_reports_independent_active_sessions(self):
        SERVER.upsert_fleet_drone({"name": "Scout One", "phone_ip": "192.168.50.235"})
        SERVER.upsert_fleet_drone({"name": "Scout Two", "phone_ip": "192.168.50.236"})
        for index, drone_id in enumerate(("scout_one", "scout_two")):
            SERVER.FLEET_SESSIONS[drone_id] = {
                "drone_id": drone_id,
                "status": "running",
                "airborne": index == 0,
                "patrol_running": index == 0,
                "events": [],
                "stop_event": threading.Event(),
                "thread": None,
            }
        snapshot = SERVER.fleet_snapshot()
        self.assertEqual(snapshot["summary"]["registered"], 2)
        self.assertEqual(snapshot["summary"]["active"], 2)
        self.assertEqual(snapshot["summary"]["airborne"], 1)
        self.assertEqual(snapshot["hardware"]["architecture"], "one_android_endpoint_per_drone")

    def test_saved_patrol_is_compiled_into_guarded_fleet_mission(self):
        asset = self.viewer / "public" / "maps" / "lab"
        asset.mkdir(parents=True)
        (asset / "scene.json").write_text(
            json.dumps({"room": {"floorY": 0.0, "bounds": {"min": [-2, 0, -2], "max": [2, 3, 2]}}}),
            encoding="utf-8",
        )
        barriers = [
            {"id": "w1", "points": [[-2, 0, -2], [2, 0, -2]]},
            {"id": "w2", "points": [[2, 0, -2], [2, 0, 2]]},
            {"id": "w3", "points": [[2, 0, 2], [-2, 0, 2]]},
            {"id": "w4", "points": [[-2, 0, 2], [-2, 0, -2]]},
        ]
        map_entry = {
            "id": "lab",
            "title": "Lab",
            "asset_base": "public/maps/lab",
            "safety_barriers": barriers,
            "safety_obstacles": [],
            "patrols": [
                {
                    "id": "patrol_1",
                    "title": "Patrol 1",
                    "points": [
                        {"rxyz": [-1, 1, -1]},
                        {"rxyz": [1, 1, -1]},
                        {"rxyz": [1, 1, 1]},
                    ],
                    "patrol_mode": "circle",
                    "speed": 0.10,
                    "altitude_m": 1.0,
                    "dwell_s": 1.0,
                }
            ],
        }
        SERVER.FLEET_SESSIONS["scout"] = {
            "drone_id": "scout",
            "map_id": "lab",
            "patrol_id": "patrol_1",
            "partial_pose_url": None,
            "events": [],
            "stop_event": threading.Event(),
            "thread": None,
        }
        with patch.object(SERVER, "load_library", return_value={"maps": [map_entry]}):
            mission = SERVER.build_fleet_patrol_mission("scout")
        self.assertTrue(mission["guided_enabled"])
        self.assertTrue(mission["patrol"])
        self.assertEqual(mission["safety_motion_buffer_m"], 0.30)
        self.assertEqual(mission["route"][0], mission["route"][-1])
        self.assertTrue(any(step["type"] == "cruise" for step in mission["commands"]))

    def test_takeoff_waits_for_bridge_acknowledgement(self):
        drone_id = "scout_one"
        SERVER.FLEET_SESSIONS[drone_id] = {
            "drone_id": drone_id,
            "status": "running",
            "phone_ip": "192.168.50.235",
            "localization_ready": True,
            "airborne": False,
            "events": [],
        }
        with patch.object(SERVER, "fleet_bridge_status", return_value={"status": "streaming"}), patch.object(
            SERVER, "dji_live_bridge_readiness", return_value=(True, "ready")
        ):
            result = SERVER.control_fleet_drone(
                {"drone_id": drone_id, "action": "takeoff", "height_m": 1.0}
            )
        session = SERVER.FLEET_SESSIONS[drone_id]
        self.assertTrue(result["queued"])
        self.assertFalse(session["airborne"])
        self.assertTrue(session["takeoff_pending"])
        self.assertTrue(session["control_pending"])

    def test_matching_bridge_acknowledgement_marks_drone_airborne(self):
        drone = SERVER.upsert_fleet_drone({"name": "Scout One", "phone_ip": "192.168.50.235"})
        SERVER.FLEET_SESSIONS[drone["id"]] = {
            "drone_id": drone["id"],
            "status": "running",
            "airborne": False,
            "takeoff_pending": True,
            "control_pending": True,
            "last_command_id": "takeoff-1",
            "events": [],
        }
        public_root = SERVER.fleet_session_public_root(drone["id"])
        public_root.mkdir(parents=True)
        (public_root / "control_status.json").write_text(
            json.dumps({"id": "takeoff-1", "command": "takeoff", "ok": True}),
            encoding="utf-8",
        )
        snapshot = SERVER.fleet_snapshot()
        session = snapshot["drones"][0]["session"]
        self.assertTrue(session["airborne"])
        self.assertFalse(session["takeoff_pending"])
        self.assertFalse(session["control_pending"])
        self.assertEqual(session["last_control_status"], "ok")

    def test_fleet_live_replay_is_scoped_to_requested_drone_and_supports_deltas(self):
        pose_path = self.viewer / "public" / "fleet" / "drones" / "scout_one" / "poses_partial.json"
        pose_path.parent.mkdir(parents=True)
        pose_path.write_text(
            json.dumps(
                {
                    "processed_count": 3,
                    "complete": False,
                    "poses": [
                        {"instance_id": "one", "rcenter": [0, 1, 0]},
                        {"instance_id": "two", "rcenter": [1, 1, 0]},
                        {"instance_id": "three", "rcenter": [2, 1, 0]},
                    ],
                }
            ),
            encoding="utf-8",
        )
        SERVER.FLEET_SESSIONS["scout_one"] = {
            "drone_id": "scout_one",
            "status": "running",
            "map_id": "lab",
            "partial_pose_url": "public/fleet/drones/scout_one/poses_partial.json",
            "events": [],
        }
        SERVER.FLEET_SESSIONS["scout_two"] = {
            "drone_id": "scout_two",
            "status": "running",
            "map_id": "other_lab",
            "partial_pose_url": None,
            "events": [],
        }

        payload, status = SERVER.fleet_live_replay_payload("scout_one", requested_after=1)
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["delta_start"], 1)
        self.assertEqual([pose["instance_id"] for pose in payload["poses"]], ["two", "three"])
        self.assertEqual(payload["stream"]["drone_id"], "scout_one")
        self.assertEqual(payload["stream"]["map_id"], "lab")

        waiting, waiting_status = SERVER.fleet_live_replay_payload("scout_two", requested_after=0)
        self.assertEqual(waiting_status, 200)
        self.assertEqual(waiting["poses"], [])
        self.assertEqual(waiting["stream"]["drone_id"], "scout_two")

    def test_monitor_ui_contains_dispatch_and_emergency_controls(self):
        html = (ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "viewer" / "app.js").read_text(encoding="utf-8")
        for element_id in (
            "fleet-monitor-button",
            "fleet-dispatch",
            "fleet-stop-all",
            "fleet-overview-grid",
            "fleet-overview-count",
            "fleet-start-patrol",
            "fleet-hover",
            "fleet-smart-log",
        ):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn('postJson("/api/fleet/dispatch"', script)
        self.assertIn('postJson("/api/fleet/stop"', script)
        self.assertIn("function renderFleetOverview()", script)
        self.assertIn("fleetControl(action, droneId", script)
        self.assertIn('data-field="map"', script)
        self.assertIn("fleet-embed=1&fleet-drone=", script)
        self.assertIn("function refreshFleetEmbed()", script)
        self.assertIn("/api/fleet/live-replay?drone_id=", script)


if __name__ == "__main__":
    unittest.main()

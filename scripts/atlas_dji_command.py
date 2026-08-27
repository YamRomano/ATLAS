#!/usr/bin/env python3
"""Send one explicit DJI MSDK control command for ATLAS.

This helper is intentionally small and single-shot.  It is used when the live
video bridge is not already connected to the Android MSDK app.  If the live
bridge is connected, atlas_app_server writes the command to the bridge command
file so the existing OpenDJI socket sends it instead.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTATION_OPENDJI_ROOT = Path("/Users/yamromano/Desktop/DJI-MSDK-to-PC-main")
VENDORED_OPENDJI_ROOT = ROOT / "vendor" / "opendji"
DEFAULT_OPENDJI_ROOT = (
    WORKSTATION_OPENDJI_ROOT
    if (WORKSTATION_OPENDJI_ROOT / "OpenDJI.py").is_file()
    else VENDORED_OPENDJI_ROOT
)
TAKEOFF_VERTICAL_SPEED = 0.03
TAKEOFF_STEP_SECONDS = 0.50
TAKEOFF_MAX_ASSIST_SECONDS = 16.0
PRE_LAND_STABILIZE_SECONDS = 1.50


def load_opendji(opendji_root: Path):
    opendji_path = opendji_root / "OpenDJI.py"
    if not opendji_path.exists():
        raise FileNotFoundError(
            f"OpenDJI.py was not found in {opendji_root}. "
            "Pass --opendji-root /path/to/DJI-MSDK-to-PC-main."
        )
    sys.path.insert(0, str(opendji_root))
    try:
        from OpenDJI import OpenDJI  # type: ignore

        return OpenDJI
    except TypeError as exc:
        if "unsupported operand type(s) for |" not in str(exc):
            raise

    # The public OpenDJI.py uses Python 3.10 union annotations.  ATLAS may run
    # under Python 3.9, so load it with postponed annotation evaluation.
    sys.modules.pop("OpenDJI", None)
    module = types.ModuleType("OpenDJI")
    module.__file__ = str(opendji_path)
    module.__package__ = ""
    source = opendji_path.read_text(encoding="utf-8")
    code = compile("from __future__ import annotations\n" + source, str(opendji_path), "exec")
    exec(code, module.__dict__)
    sys.modules["OpenDJI"] = module
    return module.OpenDJI


def read_altitude(drone: Any, OpenDJI: Any) -> str | None:
    try:
        return drone.getValue(OpenDJI.MODULE_FLIGHTCONTROLLER, "AircraftLocation3D")
    except Exception:
        return None


def parse_altitude_m(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        value = raw.get("altitude")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    text = str(raw).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict) and "altitude" in value:
            return float(value["altitude"])
    except Exception:
        pass
    match = re.search(r'"altitude"\s*:\s*([-+]?\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1))
    return None


def climb_to_requested_height(
    drone: Any,
    OpenDJI: Any,
    altitude_before: Any,
    target_height_m: float | None,
) -> dict[str, Any]:
    if target_height_m is None:
        return {"enabled": False, "reason": "no requested height"}
    target_height_m = max(0.1, min(2.0, float(target_height_m)))
    ground_alt = parse_altitude_m(altitude_before)
    if ground_alt is None:
        return {
            "enabled": False,
            "reason": "altitude telemetry unavailable before takeoff",
        }

    target_abs = ground_alt + target_height_m
    # Give DJI's built-in takeoff action a moment to settle before adding any
    # explicit upward stick input.
    time.sleep(3.0)
    current_raw = read_altitude(drone, OpenDJI)
    current_alt = parse_altitude_m(current_raw)
    if current_alt is None:
        return {
            "enabled": False,
            "reason": "altitude telemetry unavailable after takeoff",
            "target_height_m": target_height_m,
            "ground_altitude_m": ground_alt,
            "target_altitude_m": target_abs,
        }
    if current_alt >= target_abs - 0.15:
        return {
            "enabled": True,
            "reached": True,
            "reason": "takeoff altitude already reached requested guard height",
            "target_height_m": target_height_m,
            "ground_altitude_m": ground_alt,
            "target_altitude_m": target_abs,
            "final_altitude_m": current_alt,
            "move_steps": 0,
        }

    steps = 0
    control_enabled = False
    last_alt = current_alt
    started = time.time()
    try:
        drone.enableControl(True)
        control_enabled = True
        while time.time() - started < TAKEOFF_MAX_ASSIST_SECONDS:
            current_raw = read_altitude(drone, OpenDJI)
            current_alt = parse_altitude_m(current_raw)
            if current_alt is not None:
                last_alt = current_alt
                if current_alt >= target_abs - 0.15:
                    break
            # Conservative vertical-only climb. No yaw, no lateral, no forward.
            drone.move(0.0, TAKEOFF_VERTICAL_SPEED, 0.0, 0.0, False)
            steps += 1
            time.sleep(TAKEOFF_STEP_SECONDS)
    finally:
        try:
            drone.move(0.0, 0.0, 0.0, 0.0, True)
        except Exception:
            pass
        if control_enabled:
            try:
                drone.disableControl(True)
            except Exception:
                pass

    return {
        "enabled": True,
        "reached": last_alt >= target_abs - 0.15,
        "target_height_m": target_height_m,
        "ground_altitude_m": ground_alt,
        "target_altitude_m": target_abs,
        "final_altitude_m": last_alt,
        "move_steps": steps,
        "elapsed_seconds": time.time() - started,
    }


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    OpenDJI = load_opendji(args.opendji_root.resolve())
    started = time.time()
    with OpenDJI(args.phone_ip) as drone:
        altitude_before = read_altitude(drone, OpenDJI)
        if args.command == "takeoff":
            result = drone.takeoff(True)
            height_guard = climb_to_requested_height(drone, OpenDJI, altitude_before, args.height_m)
        elif args.command == "land":
            time.sleep(PRE_LAND_STABILIZE_SECONDS)
            result = drone.land(True)
            height_guard = {
                "enabled": False,
                "reason": "native DJI landing command; OpenDJI does not expose landing-speed control",
                "pre_land_stabilize_seconds": PRE_LAND_STABILIZE_SECONDS,
            }
        elif args.command == "enable":
            result = drone.enableControl(True)
            height_guard = {"enabled": False, "reason": "not a takeoff command"}
        elif args.command == "disable":
            result = drone.disableControl(True)
            height_guard = {"enabled": False, "reason": "not a takeoff command"}
        elif args.command == "hover":
            result = drone.move(0, 0, 0, 0, True)
            height_guard = {"enabled": False, "reason": "not a takeoff command"}
        else:
            raise ValueError(f"Unsupported command: {args.command}")
        altitude_after = read_altitude(drone, OpenDJI)
    note = ""
    if args.command == "takeoff" and args.height_m is not None:
        note = (
            "Takeoff command sent. The requested height is enforced with a "
            "conservative telemetry-based upward-only guard when altitude "
            "telemetry is available."
        )
    return {
        "ok": True,
        "command": args.command,
        "phone_ip": args.phone_ip,
        "height_m": args.height_m,
        "result": result,
        "altitude_before": altitude_before,
        "altitude_after": altitude_after,
        "height_guard": height_guard,
        "note": note,
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Send one ATLAS DJI MSDK control command.")
    ap.add_argument("--phone-ip", required=True)
    ap.add_argument("--opendji-root", type=Path, default=DEFAULT_OPENDJI_ROOT)
    ap.add_argument("--command", required=True, choices=["takeoff", "land", "enable", "disable", "hover"])
    ap.add_argument("--height-m", type=float, default=None)
    args = ap.parse_args()
    payload = run_command(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

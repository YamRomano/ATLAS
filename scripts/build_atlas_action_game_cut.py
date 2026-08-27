#!/usr/bin/env python3
"""Build the clean ATLAS action-opening cut requested on 2026-08-13.

Editorial rules for this cut:
- App screenshots are never used as full-frame backgrounds.
- Every recorded segment is unique; no shot or interval is repeated.
- The response sequence reads from the ATLAS camera toward NEO, using real
  NEO approach frames accelerated in time.
- Technical copy is kept to verified project facts and essential story beats.
"""

from __future__ import annotations

import math
import json
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

from build_atlas_spy_demo import (
    AMBER,
    CYAN,
    CYAN_SOFT,
    GREEN,
    MUTED,
    RED,
    TYPE,
    WHITE,
    VideoSource,
    alpha_rect,
    clamp,
    contain,
    cover,
    paste,
    pulse,
    smooth,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "spy_demo"
WORK = OUT / "work" / "action_game_cut"
BRANDING = ROOT / "viewer" / "public" / "branding"
MUSIC = Path("/Users/yamromano/Downloads/41 Minutes of Spy Music - Instrumental Spy Themes.mp3")
SCREEN_RECORDING = Path(
    "/var/folders/yj/0rnc6hws72n_zrcq_z42nzgr0000gn/T/TemporaryItems/"
    "NSIRD_screencaptureui_iNPvJU/Screen Recording 2026-08-13 at 19.18.57.mov"
)
SILENT = WORK / "atlas_action_game_cut_silent.mp4"
SFX = WORK / "atlas_action_game_sfx.wav"
FINAL = OUT / "ATLAS_ACTION_GAME_OPENING_2MIN.mp4"

WIDTH, HEIGHT, FPS, DURATION = 1920, 1080, 24, 120.0


def line_text(frame: np.ndarray, text: str, xy: tuple[int, int], size: int, color: tuple[int, int, int], anchor: str = "la") -> None:
    TYPE.draw(frame, xy, text, size, color, "tech", anchor=anchor, stroke=1, stroke_color=(0, 0, 0))


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


class ActionFilm:
    def __init__(self) -> None:
        if not SCREEN_RECORDING.is_file():
            raise FileNotFoundError(f"Supplied screen recording is unavailable: {SCREEN_RECORDING}")
        self.w, self.h, self.fps = WIDTH, HEIGHT, FPS
        self.mesh_motion = VideoSource(SCREEN_RECORDING)
        self.path_a = VideoSource(ROOT / "runtime/manual_path_videos/Live_ATLAS_11-07-56_20260729.mp4")
        self.path_b = VideoSource(ROOT / "data/app_uploads/drone_upload_1699b1a3.MP4")
        self.path_c = VideoSource(ROOT / "data/app_uploads/drone_upload_5afa5460.MP4")
        enemy = ROOT / "viewer/public/enemy_drones/enemy_20260802_104805_b94cef/videos"
        self.neo_a = VideoSource(enemy / "enemy_calib_211d7230.MOV")
        self.neo_b = VideoSource(enemy / "enemy_calib_3f046629.MOV")
        manifest = json.loads((ROOT / "viewer/public/enemy_drones/manifest.json").read_text(encoding="utf-8"))
        frames = manifest["enemies"][0]["frames"]
        self.neo_boxes = {
            "neo_a": [item for item in frames if item.get("source_filename") == "neo1.MOV" and item.get("box")],
            "neo_b": [item for item in frames if item.get("source_filename") == "neo2.MOV" and item.get("box")],
        }
        self.poster = cv2.imread(str(BRANDING / "atlas-space-poster.png"), cv2.IMREAD_COLOR)
        self.symbol = cv2.imread(str(BRANDING / "atlas-space-symbol.png"), cv2.IMREAD_COLOR)
        if self.poster is None or self.symbol is None:
            raise FileNotFoundError("ATLAS branding assets are missing.")

        yy, xx = np.mgrid[0 : self.h, 0 : self.w]
        dx = (xx - self.w / 2) / (self.w / 2)
        dy = (yy - self.h / 2) / (self.h / 2)
        self.vignette = np.clip(1.04 - 0.24 * (dx * dx + dy * dy), 0.68, 1.0)[..., None]

    def base(self) -> np.ndarray:
        frame = np.full((self.h, self.w, 3), (13, 8, 3), np.uint8)
        for x in range(0, self.w, 120):
            cv2.line(frame, (x, 0), (x, self.h), (20, 20, 17), 1)
        for y in range(0, self.h, 120):
            cv2.line(frame, (0, y), (self.w, y), (20, 20, 17), 1)
        return frame

    def mesh_frame(self, seconds: float, *, dark: float = 0.0, zoom: float = 1.0, pan_x: float = 0.0, pan_y: float = 0.0) -> np.ndarray:
        """Crop only the moving 3D room viewport from the supplied recording."""
        source = self.mesh_motion.frame(seconds)
        height, width = source.shape[:2]
        # The recording's left viewport contains the mesh. Removing the top,
        # left and right chrome makes the room itself the cinematic image.
        x0, x1 = round(width * 0.055), round(width * 0.638)
        y0, y1 = round(height * 0.145), round(height * 0.985)
        mesh = source[y0:y1, x0:x1]
        frame = cover(mesh, self.w, self.h, zoom=zoom, pan_x=pan_x, pan_y=pan_y)
        if dark > 0:
            alpha_rect(frame, (0, 0, self.w, self.h), (4, 7, 10), dark)
        return frame

    def grade_camera(self, image: np.ndarray, *, alpha: float = 0.91, beta: int = -6) -> np.ndarray:
        frame = cover(image, self.w, self.h)
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    def top_line(self, frame: np.ndarray, chapter: str, t: float, *, alert: bool = False) -> None:
        accent = RED if alert else CYAN
        alpha_rect(frame, (0, 0, self.w, 56), (3, 6, 9), 0.91)
        cv2.line(frame, (0, 56), (self.w, 56), accent, 1, cv2.LINE_AA)
        line_text(frame, "ATLAS // MISSION 01", (44, 28), 16, CYAN, "lm")
        line_text(frame, chapter.upper(), (self.w - 44, 28), 16, WHITE, "rm")
        progress = round((self.w - 88) * clamp(t / DURATION))
        cv2.line(frame, (44, self.h - 28), (self.w - 44, self.h - 28), (42, 55, 58), 2, cv2.LINE_AA)
        cv2.line(frame, (44, self.h - 28), (44 + progress, self.h - 28), accent, 3, cv2.LINE_AA)

    def heading(self, frame: np.ndarray, eyebrow: str, title: str, subtitle: str = "", *, alert: bool = False, x: int = 82, y: int = 130) -> None:
        accent = RED if alert else CYAN
        cv2.line(frame, (x, y), (x + 190, y), accent, 4, cv2.LINE_AA)
        line_text(frame, eyebrow.upper(), (x, y + 25), 21, accent)
        TYPE.draw(frame, (x, y + 66), title.upper(), 64, WHITE, "condensed", stroke=2, stroke_color=(0, 0, 0), spacing=5)
        if subtitle:
            TYPE.draw(frame, (x, y + 222), subtitle, 24, MUTED, "regular", stroke=1, stroke_color=(0, 0, 0))

    def stat(self, frame: np.ndarray, x: int, y: int, label: str, value: str, *, alert: bool = False, width: int = 250) -> None:
        accent = RED if alert else CYAN
        alpha_rect(frame, (x, y, x + width, y + 92), (3, 8, 12), 0.85)
        cv2.rectangle(frame, (x, y), (x + width, y + 92), accent, 1, cv2.LINE_AA)
        line_text(frame, label.upper(), (x + 15, y + 23), 14, accent)
        TYPE.draw(frame, (x + 15, y + 72), value, 27, WHITE, "condensed", anchor="ls")

    def scan(self, frame: np.ndarray, local: float, color: tuple[int, int, int] = CYAN) -> None:
        y = round(80 + ((local * 0.22) % 1.0) * (self.h - 150))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y - 14), (self.w, y + 14), color, -1)
        cv2.addWeighted(overlay, 0.035, frame, 0.965, 0, frame)
        cv2.line(frame, (0, y), (self.w, y), color, 1, cv2.LINE_AA)

    def tracking_points(self, frame: np.ndarray, local: float, count: int = 32) -> None:
        rng = np.random.default_rng(4096)
        for index in range(count):
            if (index + round(local * 12)) % 4:
                continue
            x = round((0.08 + rng.random() * 0.84) * self.w)
            y = round((0.12 + rng.random() * 0.74) * self.h)
            cv2.circle(frame, (x, y), 4, CYAN, 1, cv2.LINE_AA)
            cv2.line(frame, (x - 7, y), (x + 7, y), CYAN_SOFT, 1, cv2.LINE_AA)

    def route(self, frame: np.ndarray, progress: float, lap: int) -> None:
        x0, y0, x1, y1 = 1395, 104, 1850, 420
        alpha_rect(frame, (x0, y0, x1, y1), (3, 8, 11), 0.82)
        cv2.rectangle(frame, (x0, y0), (x1, y1), CYAN_SOFT, 1, cv2.LINE_AA)
        pts = np.array([[x0 + 42, y0 + 45], [x1 - 45, y0 + 45], [x1 - 45, y1 - 56], [x0 + 42, y1 - 56], [x0 + 42, y0 + 45]], np.int32)
        cv2.polylines(frame, [pts], False, (42, 56, 60), 3, cv2.LINE_AA)
        lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        distance = clamp(progress) * float(np.sum(lengths))
        drawn = [pts[0]]
        for index, length in enumerate(lengths):
            if distance >= length:
                drawn.append(pts[index + 1])
                distance -= length
            else:
                partial = pts[index] + (pts[index + 1] - pts[index]) * distance / max(length, 1e-6)
                drawn.append(partial.astype(np.int32))
                break
        cv2.polylines(frame, [np.array(drawn)], False, CYAN, 6, cv2.LINE_AA)
        for index, point in enumerate(pts[:-1], start=1):
            cv2.circle(frame, tuple(point), 12, (5, 10, 12), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(point), 12, CYAN, 2, cv2.LINE_AA)
            line_text(frame, str(index), tuple(point), 13, WHITE, "mm")
        line_text(frame, f"LAP {lap} / 2", (x0 + 18, y1 - 17), 17, CYAN, "ls")

    def scene_open(self, local: float) -> np.ndarray:
        frame = self.mesh_frame(0.5 + local * 0.85, dark=0.44, zoom=1.015 + 0.025 * smooth(local / 8))
        alpha_rect(frame, (0, 0, 815, self.h), (2, 6, 10), 0.74)
        if local > 0.25:
            self.heading(frame, "INDOOR AUTONOMY", "THE ROOM IS DARK.\nTHE MISSION IS NOT.", "No GPS. No beacons. No blind spots.", x=74, y=180)
        if local > 4.3:
            line_text(frame, "ATLAS", (76, 720), 31, WHITE)
            line_text(frame, "AUTONOMOUS TRACKING AND LOCALIZATION", (76, 764), 17, CYAN)
        self.scan(frame, local)
        return frame

    def scene_map(self, local: float) -> np.ndarray:
        frame = self.mesh_frame(11.0 + local * 0.9, dark=0.18, zoom=1.03, pan_x=0.08)
        alpha_rect(frame, (42, 74, 850, 420), (3, 7, 10), 0.90)
        self.heading(frame, "MISSION PREPARATION", "BUILD THE WORLD\nBEFORE FLIGHT.", "COLMAP + SIFT create the localization map.", x=70, y=108)
        self.stat(frame, 70, 790, "COLMAP POINTS", "440,249")
        self.stat(frame, 340, 790, "MAP CAMERAS", "3,235")
        self.stat(frame, 610, 790, "FEATURES", "SIFT")
        self.scan(frame, local)
        return frame

    def scene_localize(self, local: float) -> np.ndarray:
        frame = self.grade_camera(self.path_a.frame(20 + local * 1.35))
        alpha_rect(frame, (52, 680, 910, 935), (3, 8, 11), 0.82)
        line_text(frame, "LIVE SELF-LOCALIZATION", (82, 726), 22, CYAN)
        TYPE.draw(frame, (82, 770), "SEE THE ROOM. KNOW THE POSITION.", 46, WHITE, "condensed", stroke=1, stroke_color=(0, 0, 0))
        line_text(frame, "TSolve pose + optical-flow continuity", (82, 852), 18, MUTED)
        self.stat(frame, 1370, 745, "POSE", "CONFIRMED", width=420)
        self.tracking_points(frame, local)
        self.scan(frame, local)
        return frame

    def panel(self, frame: np.ndarray, source: VideoSource, seconds: float, box: tuple[int, int, int, int], label: str) -> None:
        x0, y0, x1, y1 = box
        image = cover(source.frame(seconds), x1 - x0, y1 - y0)
        paste(frame, image, x0, y0)
        cv2.rectangle(frame, (x0, y0), (x1, y1), CYAN_SOFT, 2, cv2.LINE_AA)
        alpha_rect(frame, (x0, y1 - 58, x1, y1), (2, 6, 9), 0.86)
        line_text(frame, label, (x0 + 16, y1 - 29), 16, WHITE, "lm")
        cv2.circle(frame, (x1 - 24, y1 - 29), 5, GREEN, -1, cv2.LINE_AA)

    def scene_fleet(self, local: float) -> np.ndarray:
        frame = self.base()
        line_text(frame, "DISTRIBUTED AWARENESS", (62, 105), 21, CYAN)
        TYPE.draw(frame, (62, 150), "ONE SYSTEM. THREE LIVE VIEWS.", 52, WHITE, "condensed")
        top, bottom, gap = 270, 880, 24
        width = (self.w - 124 - gap * 2) // 3
        self.panel(frame, self.path_a, 76 + local * 1.05, (62, top, 62 + width, bottom), "ATLAS-01 // LAB LOOP")
        self.panel(frame, self.path_b, 210 + local * 1.15, (62 + width + gap, top, 62 + 2 * width + gap, bottom), "ATLAS-02 // NORTH SECTOR")
        self.panel(frame, self.path_c, 10 + local * 0.92, (62 + 2 * (width + gap), top, self.w - 62, bottom), "ATLAS-03 // EAST SECTOR")
        self.scan(frame, local)
        return frame

    def scene_patrol_one(self, local: float) -> np.ndarray:
        frame = self.grade_camera(self.path_a.frame(108 + local * 1.45))
        self.route(frame, local / 12.0, 1)
        alpha_rect(frame, (62, 730, 780, 900), (2, 7, 10), 0.80)
        line_text(frame, "AUTONOMOUS PATROL", (88, 770), 21, CYAN)
        TYPE.draw(frame, (88, 812), "LAP ONE // LIVE", 43, WHITE, "condensed")
        line_text(frame, "LOCALIZE  •  VERIFY  •  ADVANCE", (88, 869), 17, MUTED)
        self.tracking_points(frame, local, 24)
        return frame

    def scene_patrol_two(self, local: float) -> np.ndarray:
        frame = self.grade_camera(self.path_b.frame(482 + local * 1.55))
        self.route(frame, local / 12.0, 2)
        alpha_rect(frame, (62, 730, 780, 900), (2, 7, 10), 0.80)
        line_text(frame, "ROUTE REACQUIRED", (88, 770), 21, GREEN)
        TYPE.draw(frame, (88, 812), "LAP TWO // CONTINUOUS", 43, WHITE, "condensed")
        line_text(frame, "A NEW PASS. THE SAME GLOBAL MAP.", (88, 869), 17, MUTED)
        self.tracking_points(frame, local + 10, 24)
        return frame

    def neo_feed(self, source: VideoSource, source_key: str, seconds: float, *, label: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Fill the screen with real NEO footage and keep the target in view."""
        rows = self.neo_boxes[source_key]
        item = min(rows, key=lambda row: abs(float(row["time_sec"]) - seconds))
        box = item["box"]
        image = source.frame(seconds)
        src_h, src_w = image.shape[:2]
        scale = self.w / src_w
        dst_h = max(self.h, round(src_h * scale))
        resized = cv2.resize(image, (self.w, dst_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        target_y = float(box["y_center"]) * dst_h
        crop_y = round(target_y - self.h * 0.50)
        crop_y = max(0, min(dst_h - self.h, crop_y))
        frame = resized[crop_y : crop_y + self.h].copy()
        cx = round(float(box["x_center"]) * self.w)
        cy = round(target_y - crop_y)
        bw = max(90, round(float(box["width"]) * self.w))
        bh = max(70, round(float(box["height"]) * dst_h))
        detected = (
            max(8, cx - bw // 2),
            max(66, cy - bh // 2),
            min(self.w - 8, cx + bw // 2),
            min(self.h - 38, cy + bh // 2),
        )
        alpha_rect(frame, (0, 0, self.w, 56), (3, 6, 9), 0.40)
        line_text(frame, label, (self.w // 2, 28), 16, WHITE, "mm")
        return frame, detected

    def target_box(self, frame: np.ndarray, detected: tuple[int, int, int, int], *, confidence: float = 94.7) -> None:
        # The box comes from the project's annotated NEO frames, transformed
        # into the smart full-screen crop used for this shot.
        x0, y0, x1, y1 = detected
        cv2.rectangle(frame, (x0, y0), (x1, y1), RED, 3, cv2.LINE_AA)
        for x, y, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(frame, (x, y), (x + sx * 28, y), WHITE, 2, cv2.LINE_AA)
            cv2.line(frame, (x, y), (x, y + sy * 28), WHITE, 2, cv2.LINE_AA)
        alpha_rect(frame, (x0, y0 - 42, x0 + 250, y0), (10, 10, 14), 0.86)
        line_text(frame, f"NEO1  //  {confidence:.1f}%", (x0 + 12, y0 - 20), 16, RED, "lm")

    def scene_detect(self, local: float) -> np.ndarray:
        raw_time = 1.0 + local * 1.45
        frame, detected = self.neo_feed(self.neo_a, "neo_a", raw_time, label="RECORDED NEO FEED // A")
        alpha_rect(frame, (48, 96, 460, 510), (4, 5, 9), 0.92)
        self.heading(frame, "PROXIMITY EVENT", "UNKNOWN\nAIRCRAFT.", "Target signature acquired.", alert=True, x=76, y=130)
        self.stat(frame, 1490, 700, "CLASS", "NEO1", alert=True, width=340)
        self.stat(frame, 1490, 812, "STATE", "LOCKED", alert=True, width=340)
        self.target_box(frame, detected, confidence=91.4 + 3.3 * smooth(local / 9))
        self.scan(frame, local, RED)
        return frame

    def scene_confirm(self, local: float) -> np.ndarray:
        raw_time = 14.5 + local * 1.45
        frame, detected = self.neo_feed(self.neo_a, "neo_a", raw_time, label="RECORDED NEO FEED // B")
        alpha_rect(frame, (45, 705, 470, 930), (4, 5, 9), 0.90)
        line_text(frame, "VISUAL MATCH", (72, 748), 20, RED)
        TYPE.draw(frame, (72, 792), "NEO1 CONFIRMED.", 42, WHITE, "condensed")
        line_text(frame, "RESPONSE VECTOR AUTHORIZED", (72, 864), 16, MUTED)
        self.target_box(frame, detected, confidence=94.7)
        self.stat(frame, 1490, 760, "ATLAS", "TRACKING", alert=True, width=340)
        self.scan(frame, local, RED)
        return frame

    def scene_intercept(self, local: float) -> np.ndarray:
        # The raw NEO recording advances 2x. As NEO grows in the camera, the
        # point of view reads as ATLAS closing the range toward the target.
        raw_time = 0.5 + local * 2.0
        frame, detected = self.neo_feed(self.neo_b, "neo_b", raw_time, label="ATLAS CAMERA // INTERCEPT VECTOR")
        p = smooth(local / 12.0)
        self.target_box(frame, detected, confidence=94.7)
        alpha_rect(frame, (40, 88, 455, 285), (4, 6, 9), 0.91)
        line_text(frame, "ATLAS → NEO1", (68, 128), 24, RED)
        TYPE.draw(frame, (68, 166), "CLOSING", 54, WHITE, "condensed")
        line_text(frame, "REAL APPROACH FRAMES // 2.0×", (68, 242), 16, CYAN)
        distance = 8.4 * (1.0 - p) + 0.42
        self.stat(frame, 1490, 685, "RANGE", f"{distance:0.2f} m", alert=True, width=340)
        self.stat(frame, 1490, 797, "VECTOR", "ATLAS → NEO", alert=True, width=340)
        # Speed lines point inward toward the real center image.
        rng = np.random.default_rng(802)
        for index in range(24):
            y = round(90 + rng.random() * 890)
            length = round(50 + 110 * pulse(local + index * 0.03, 0.7))
            cv2.line(frame, (0, y), (length, y), CYAN_SOFT, 1, cv2.LINE_AA)
            cv2.line(frame, (self.w, y), (self.w - length, y), CYAN_SOFT, 1, cv2.LINE_AA)
        self.scan(frame, local, RED)
        return frame

    def scene_impact(self, local: float) -> np.ndarray:
        frame = self.base()
        center = (self.w // 2, self.h // 2)
        p = smooth(local / 4.0)
        glow = np.full_like(frame, AMBER)
        cv2.addWeighted(glow, 0.10 + 0.35 * pulse(local, 0.8), frame, 0.90 - 0.35 * pulse(local, 0.8), 0, frame)
        for index, radius in enumerate((round(90 + 710 * p), round(40 + 470 * p), round(20 + 250 * p))):
            cv2.circle(frame, center, radius, WHITE if index == 2 else AMBER, max(2, 10 - index * 2), cv2.LINE_AA)
        TYPE.draw(frame, center, "CONTACT", 104, WHITE, "tech", anchor="mm", stroke=3, stroke_color=(20, 25, 32))
        line_text(frame, "CINEMATIC RESPONSE SEQUENCE", (self.w // 2, 690), 19, RED, "mm")
        return frame

    def scene_relocalize(self, local: float) -> np.ndarray:
        frame = self.mesh_frame(64 + local * 0.85, dark=0.22, zoom=1.04, pan_x=-0.05)
        alpha_rect(frame, (58, 120, 770, 450), (3, 8, 11), 0.90)
        self.heading(frame, "GLOBAL RECOVERY", "RELOCALIZE.\nRETURN TO MISSION.", "The map re-anchors the live pose.", x=86, y=155)
        status = "SEARCHING" if local < 3.3 else "POSE REACQUIRED"
        color = AMBER if local < 3.3 else GREEN
        self.stat(frame, 86, 765, "RECOVERY", status, width=470)
        cv2.circle(frame, (1730, 790), 7, color, -1, cv2.LINE_AA)
        line_text(frame, "TSOLVE CONFIRMED" if local >= 3.3 else "GLOBAL MATCH", (1708, 790), 19, color, "rm")
        self.scan(frame, local)
        return frame

    def scene_resume(self, local: float) -> np.ndarray:
        frame = self.grade_camera(self.path_c.frame(46 + local * 1.4))
        self.route(frame, min(1.0, 0.72 + local / 20.0), 2)
        alpha_rect(frame, (58, 710, 820, 900), (3, 8, 10), 0.82)
        line_text(frame, "POSE VERIFIED", (86, 752), 21, GREEN)
        TYPE.draw(frame, (86, 798), "PATROL CONTINUES.", 50, WHITE, "condensed")
        line_text(frame, "COMMAND FLOW RESTORED", (86, 866), 17, CYAN)
        self.tracking_points(frame, local + 30, 22)
        return frame

    def scene_outro(self, local: float) -> np.ndarray:
        frame = cover(self.poster, self.w, self.h, zoom=1.0 + 0.015 * smooth(local / 4.0))
        alpha_rect(frame, (0, 0, self.w, self.h), (2, 5, 8), 0.34)
        symbol = contain(self.symbol, 360, 220, (3, 6, 9))
        paste(frame, symbol, self.w // 2 - 180, 125)
        TYPE.draw(frame, (self.w // 2, 430), "ATLAS", 126, WHITE, "tech", anchor="mm")
        cv2.line(frame, (630, 535), (1290, 535), CYAN, 2, cv2.LINE_AA)
        TYPE.draw(frame, (self.w // 2, 635), "DETECT. LOCALIZE. RESPOND.", 55, WHITE, "condensed", anchor="mm")
        line_text(frame, "AUTONOMY INSIDE THE ROOM", (self.w // 2, 755), 23, CYAN, "mm")
        return frame

    def render(self, t: float) -> np.ndarray:
        if t < 8:
            frame, chapter, alert = self.scene_open(t), "MISSION BOOT", False
        elif t < 18:
            frame, chapter, alert = self.scene_map(t - 8), "MAP ONLINE", False
        elif t < 30:
            frame, chapter, alert = self.scene_localize(t - 18), "LIVE LOCALIZATION", False
        elif t < 42:
            frame, chapter, alert = self.scene_fleet(t - 30), "DISTRIBUTED PATROL", False
        elif t < 54:
            frame, chapter, alert = self.scene_patrol_one(t - 42), "PATROL // LAP 1", False
        elif t < 66:
            frame, chapter, alert = self.scene_patrol_two(t - 54), "PATROL // LAP 2", False
        elif t < 75:
            frame, chapter, alert = self.scene_detect(t - 66), "THREAT DETECTED", True
        elif t < 84:
            frame, chapter, alert = self.scene_confirm(t - 75), "TARGET CONFIRMED", True
        elif t < 96:
            frame, chapter, alert = self.scene_intercept(t - 84), "ATLAS INTERCEPT", True
        elif t < 100:
            frame, chapter, alert = self.scene_impact(t - 96), "CONTACT", True
        elif t < 110:
            frame, chapter, alert = self.scene_relocalize(t - 100), "GLOBAL RECOVERY", False
        elif t < 116:
            frame, chapter, alert = self.scene_resume(t - 110), "PATROL RESTORED", False
        else:
            frame, chapter, alert = self.scene_outro(t - 116), "MISSION CONTINUES", False

        self.top_line(frame, chapter, t, alert=alert)
        frame = np.clip(frame.astype(np.float32) * self.vignette, 0, 255).astype(np.uint8)
        for y in range(0, self.h, 7):
            frame[y : y + 1] = (frame[y : y + 1].astype(np.float32) * 0.965).astype(np.uint8)
        # Brief cinematic dips, never dead holds.
        fade = min(1.0, t / 0.28, (DURATION - t) / 0.72)
        for boundary in (8, 18, 30, 42, 54, 66, 75, 84, 96, 100, 110, 116):
            distance = abs(t - boundary)
            if distance < 0.08:
                fade = min(fade, 0.84 + distance * 2.0)
        return (frame.astype(np.float32) * clamp(fade)).astype(np.uint8)


def make_sfx(path: Path) -> None:
    rate = 48_000
    count = round(DURATION * rate)
    stereo = np.zeros((count, 2), np.float64)
    rng = np.random.default_rng(813)

    def add_whoosh(start: float, duration: float = 0.55, volume: float = 0.13) -> None:
        i0 = round(start * rate)
        length = min(round(duration * rate), count - i0)
        x = np.arange(length) / rate
        envelope = np.sin(np.pi * np.clip(x / duration, 0, 1)) ** 1.8
        noise = rng.standard_normal(length)
        shaped = (noise - np.roll(noise, 17)) * envelope * volume
        stereo[i0 : i0 + length, 0] += shaped
        stereo[i0 : i0 + length, 1] += np.roll(shaped, 240) * 0.9

    for boundary in (8, 18, 30, 42, 54, 66, 75, 84, 96, 100, 110, 116):
        add_whoosh(max(0, boundary - 0.22))

    i0 = round(96.05 * rate)
    length = min(round(2.4 * rate), count - i0)
    x = np.arange(length) / rate
    boom = np.sin(math.tau * (58 * x - 12 * x * x)) * np.exp(-x * 1.9)
    crack = rng.standard_normal(length) * np.exp(-x * 8.5)
    hit = 0.58 * boom + 0.12 * crack
    stereo[i0 : i0 + length, 0] += hit
    stereo[i0 : i0 + length, 1] += np.roll(hit, 380) * 0.91

    pcm = np.clip(stereo * 32767, -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


def render_silent() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    film = ActionFilm()
    writer = cv2.VideoWriter(str(SILENT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {SILENT}")
    try:
        for index in range(round(DURATION * FPS)):
            if index % (FPS * 5) == 0:
                print(f"rendered {index / FPS:5.1f}/{DURATION:.1f}s", flush=True)
            writer.write(film.render(index / FPS))
    finally:
        writer.release()


def mux() -> None:
    if not MUSIC.is_file():
        raise FileNotFoundError(MUSIC)
    make_sfx(SFX)
    run(
        [
            "/opt/homebrew/bin/ffmpeg",
            "-y",
            "-i",
            str(SILENT),
            "-i",
            str(MUSIC),
            "-i",
            str(SFX),
            "-filter_complex",
            "[1:a]atrim=start=0:end=120,asetpts=PTS-STARTPTS,loudnorm=I=-16:LRA=10:TP=-1.5,afade=t=out:st=118:d=2[music];"
            "[2:a]volume=0.9[sfx];[music][sfx]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.94[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-t",
            "120",
            str(FINAL),
        ]
    )


def main() -> None:
    render_silent()
    mux()
    print(FINAL)


if __name__ == "__main__":
    main()

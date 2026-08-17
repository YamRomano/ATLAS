#!/usr/bin/env python3
"""Render a cinematic, sponsor-ready ATLAS system demonstration.

The film combines recorded ATLAS UI captures and flight footage with clearly
cinematic mission visualizations. It does not alter the live patrol system.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "spy_demo"
CAPTURES = OUT / "captures"
BRANDING = ROOT / "viewer" / "public" / "branding"

CYAN = (255, 226, 88)  # BGR
CYAN_SOFT = (220, 184, 70)
WHITE = (246, 248, 242)
MUTED = (186, 195, 189)
NAVY = (20, 12, 5)
RED = (70, 76, 252)
GREEN = (137, 227, 81)
AMBER = (72, 201, 255)

FONT_REGULAR = "/System/Library/Fonts/Avenir Next.ttc"
FONT_CONDENSED = "/System/Library/Fonts/Avenir Next Condensed.ttc"
FONT_TECH = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def smooth(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def ease_out(value: float) -> float:
    value = clamp(value)
    return 1.0 - (1.0 - value) ** 3


def pulse(value: float, speed: float = 1.0) -> float:
    return 0.5 + 0.5 * math.sin(value * math.tau * speed)


def bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def cover(image: np.ndarray, width: int, height: int, zoom: float = 1.0, pan_x: float = 0.0, pan_y: float = 0.0) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = max(width / src_w, height / src_h) * zoom
    dst_w, dst_h = max(width, round(src_w * scale)), max(height, round(src_h * scale))
    resized = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    spare_x, spare_y = dst_w - width, dst_h - height
    x0 = round(spare_x * clamp(0.5 + pan_x * 0.5))
    y0 = round(spare_y * clamp(0.5 + pan_y * 0.5))
    return resized[y0 : y0 + height, x0 : x0 + width].copy()


def contain(image: np.ndarray, width: int, height: int, background: tuple[int, int, int] = NAVY) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    dst_w, dst_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((height, width, 3), background, np.uint8)
    x0, y0 = (width - dst_w) // 2, (height - dst_h) // 2
    canvas[y0 : y0 + dst_h, x0 : x0 + dst_w] = resized
    return canvas


def alpha_rect(frame: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int], alpha: float) -> None:
    x0, y0, x1, y1 = box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def paste(frame: np.ndarray, image: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    h, w = image.shape[:2]
    if x >= frame.shape[1] or y >= frame.shape[0] or x + w <= 0 or y + h <= 0:
        return
    sx0, sy0 = max(0, -x), max(0, -y)
    sx1, sy1 = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
    roi = frame[max(0, y) : y + sy1, max(0, x) : x + sx1]
    src = image[sy0:sy1, sx0:sx1]
    if alpha >= 0.999:
        roi[:] = src
    else:
        cv2.addWeighted(src, alpha, roi, 1.0 - alpha, 0, roi)


class Type:
    def __init__(self) -> None:
        self.cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def font(self, size: int, family: str = "regular") -> ImageFont.FreeTypeFont:
        path = {"regular": FONT_REGULAR, "condensed": FONT_CONDENSED, "tech": FONT_TECH}[family]
        key = (path, size)
        if key not in self.cache:
            self.cache[key] = ImageFont.truetype(path, size)
        return self.cache[key]

    def draw(
        self,
        frame: np.ndarray,
        xy: tuple[int, int],
        text: str,
        size: int,
        color: tuple[int, int, int] = WHITE,
        family: str = "regular",
        anchor: str | None = None,
        spacing: int = 4,
        stroke: int = 0,
        stroke_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        drawer = ImageDraw.Draw(pil)
        rgb = (color[2], color[1], color[0])
        stroke_rgb = (stroke_color[2], stroke_color[1], stroke_color[0])
        drawer.multiline_text(
            xy,
            text,
            font=self.font(size, family),
            fill=rgb,
            anchor=anchor,
            spacing=spacing,
            stroke_width=stroke,
            stroke_fill=stroke_rgb,
        )
        frame[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


TYPE = Type()


class VideoSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open {path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.count = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.duration = self.count / self.fps
        self.index = -1
        self.last: np.ndarray | None = None

    def frame(self, seconds: float) -> np.ndarray:
        seconds = seconds % max(0.01, self.duration - 1.0 / self.fps)
        target = min(self.count - 1, max(0, int(seconds * self.fps)))
        if target < self.index or target - self.index > max(120, int(self.fps * 4)):
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.index = target - 1
        while self.index < target:
            ok, frame = self.capture.read()
            self.index += 1
            if ok:
                self.last = frame
        if self.last is None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, self.last = self.capture.read()
            self.index = target
            if not ok or self.last is None:
                raise RuntimeError(f"Unable to decode {self.path} at {seconds:.2f}s")
        return self.last.copy()


@dataclass
class Scene:
    start: float
    end: float
    name: str


SCENES = [
    Scene(0.0, 7.0, "intro"),
    Scene(7.0, 15.0, "map"),
    Scene(15.0, 27.0, "localize"),
    Scene(27.0, 40.0, "fleet"),
    Scene(40.0, 51.0, "patrol"),
    Scene(51.0, 63.0, "threat"),
    Scene(63.0, 71.0, "intercept"),
    Scene(71.0, 79.0, "outro"),
]


class Film:
    def __init__(self, width: int, height: int, fps: int) -> None:
        self.w, self.h, self.fps = width, height, fps
        self.root = bgr(CAPTURES / "atlas_root.png")
        self.monitor = bgr(CAPTURES / "atlas_monitor.png")
        self.enemy_lab = bgr(CAPTURES / "atlas_enemy_lab.png")
        self.patrol_top = bgr(CAPTURES / "atlas_patrol_top.png")
        self.patrol_angle = bgr(CAPTURES / "atlas_patrol_angle.png")
        self.mesh_angle = bgr(CAPTURES / "v2_final_mesh_angle.png")
        self.mesh_top = bgr(CAPTURES / "v2_final_mesh_top.png")
        self.mesh_side = bgr(CAPTURES / "v2_clean_mesh_side.png")
        self.drone_view = bgr(CAPTURES / "v2_drone_view.png")
        self.mesh_poses = [bgr(CAPTURES / f"v2_clean_pose_{index:02d}.png") for index in range(1, 4)]
        self.poster = bgr(BRANDING / "atlas-space-poster.png")
        self.wordmark = bgr(BRANDING / "atlas-space-wordmark.png")
        self.symbol = bgr(BRANDING / "atlas-space-symbol.png")
        self.flight = VideoSource(ROOT / "runtime" / "manual_path_videos" / "Live_ATLAS_11-07-56_20260729.mp4")
        self.home = VideoSource(ROOT / "viewer" / "public" / "media" / "drone_query.mp4")
        self.lab = VideoSource(ROOT / "data" / "app_uploads" / "camera_path_lab_d8e06960.mov")
        self.enemy = VideoSource(
            ROOT
            / "viewer"
            / "public"
            / "enemy_drones"
            / "enemy_20260802_104805_b94cef"
            / "videos"
            / "enemy_calib_211d7230.MOV"
        )
        manifest = json.loads((ROOT / "viewer" / "public" / "enemy_drones" / "manifest.json").read_text())
        self.enemy_boxes = [
            frame
            for frame in manifest["enemies"][0]["frames"]
            if frame.get("source_filename") == "neo1.MOV" and frame.get("box")
        ]
        yy, xx = np.mgrid[0 : self.h, 0 : self.w]
        dist = np.sqrt(((xx - self.w / 2) / (self.w * 0.72)) ** 2 + ((yy - self.h / 2) / (self.h * 0.72)) ** 2)
        self.vignette = np.clip(1.08 - dist * 0.58, 0.56, 1.0).astype(np.float32)[..., None]
        self.rng = np.random.default_rng(20260813)

    def base(self) -> np.ndarray:
        frame = np.zeros((self.h, self.w, 3), np.uint8)
        frame[:] = (16, 9, 4)
        return frame

    def clean_mesh(self, source: np.ndarray, width: int, height: int, zoom: float = 1.0) -> np.ndarray:
        """Extract only the meaningful 3D room/drone viewport from an ATLAS capture."""
        src_h, src_w = source.shape[:2]
        crop = source[
            round(src_h * 0.095) : round(src_h * 0.605),
            round(src_w * 0.215) : round(src_w * 0.565),
        ]
        panel = cover(crop, width, height, zoom)
        return cv2.convertScaleAbs(panel, alpha=1.08, beta=-4)

    def deannotate_mesh(self, image: np.ndarray) -> np.ndarray:
        """Apply a restrained cinematic grade without fabricating map structure."""
        return cv2.detailEnhance(image, sigma_s=8, sigma_r=0.12)

    def grid(self, frame: np.ndarray, horizon: float = 0.62, color: tuple[int, int, int] = (38, 46, 48)) -> None:
        y0 = round(self.h * horizon)
        vanishing = (self.w // 2, y0)
        for x in range(-self.w, self.w * 2, max(70, self.w // 18)):
            cv2.line(frame, vanishing, (x, self.h), color, 1, cv2.LINE_AA)
        for row in range(1, 10):
            frac = (row / 10) ** 1.8
            y = round(y0 + (self.h - y0) * frac)
            cv2.line(frame, (0, y), (self.w, y), color, 1, cv2.LINE_AA)

    def scanline(self, frame: np.ndarray, local: float, x0: int, y0: int, x1: int, y1: int, red: bool = False) -> None:
        accent = RED if red else CYAN
        y = round(y0 + ((local * 0.23) % 1.0) * (y1 - y0))
        overlay = frame.copy()
        cv2.line(overlay, (x0, y), (x1, y), accent, max(2, self.h // 360), cv2.LINE_AA)
        cv2.rectangle(overlay, (x0, max(y0, y - 24)), (x1, min(y1, y + 24)), accent, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    def drone_sprite(self, frame: np.ndarray, center: tuple[int, int], scale: float, heading: float, color: tuple[int, int, int] = CYAN) -> None:
        """Stylized top projection of the existing DJI Mini 3 Pro model."""
        cx, cy = center
        forward = np.array([math.cos(heading), math.sin(heading)])
        side = np.array([-forward[1], forward[0]])
        def point(f: float, s: float) -> tuple[int, int]:
            p = np.array([cx, cy]) + forward * f * scale + side * s * scale
            return int(p[0]), int(p[1])
        body = np.array([point(24, 0), point(2, 12), point(-28, 8), point(-31, -8), point(2, -12)], np.int32)
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, body, color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.46, frame, 0.54, 0, frame)
        cv2.polylines(frame, [body], True, WHITE, 2, cv2.LINE_AA)
        for f, s in ((12, 25), (12, -25), (-17, 24), (-17, -24)):
            rotor = point(f, s)
            cv2.line(frame, point(f * 0.72, s * 0.66), rotor, color, max(2, round(scale * 0.7)), cv2.LINE_AA)
            cv2.circle(frame, rotor, max(8, round(scale * 9)), (8, 9, 9), -1, cv2.LINE_AA)
            cv2.circle(frame, rotor, max(7, round(scale * 8)), color, 2, cv2.LINE_AA)
        cv2.line(frame, point(13, 0), point(34, 0), color, 4, cv2.LINE_AA)

    def neo_symbol(self, frame: np.ndarray, center: tuple[int, int], radius: int, local: float, contained: bool = False) -> None:
        color = GREEN if contained else RED
        cx, cy = center
        for ring in (radius, round(radius * 1.45), round(radius * 1.9)):
            cv2.circle(frame, center, ring + round(4 * pulse(local + ring * 0.01, 1.2)), color, 1, cv2.LINE_AA)
        cv2.circle(frame, center, radius // 2, (8, 8, 8), -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius // 2, color, 3, cv2.LINE_AA)
        cv2.line(frame, (cx - radius // 3, cy), (cx + radius // 3, cy), color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - radius // 3), (cx, cy + radius // 3), color, 2, cv2.LINE_AA)
        TYPE.draw(frame, (cx, cy + radius + 18), "NEO1" if not contained else "NEO1 // CONTAINED", round(self.h * 0.015), color, "tech", anchor="ma")

    def title_block(self, frame: np.ndarray, eyebrow: str, title: str, sub: str, x: int, y: int, width: int, red: bool = False) -> None:
        accent = RED if red else CYAN
        cv2.line(frame, (x, y), (x + round(width * 0.18), y), accent, max(2, self.w // 800), cv2.LINE_AA)
        TYPE.draw(frame, (x, y + round(self.h * 0.022)), eyebrow.upper(), round(self.h * 0.020), accent, "tech")
        TYPE.draw(frame, (x, y + round(self.h * 0.066)), title.upper(), round(self.h * 0.054), WHITE, "condensed")
        title_lines = title.count("\n") + 1
        sub_y = y + round(self.h * (0.075 + title_lines * 0.072))
        TYPE.draw(frame, (x, sub_y), sub, round(self.h * 0.023), MUTED, "regular")

    def tech_frame(self, frame: np.ndarray, label: str, value: str, x: int, y: int, w: int, h: int, alert: bool = False) -> None:
        accent = RED if alert else CYAN
        alpha_rect(frame, (x, y, x + w, y + h), (20, 12, 6), 0.78)
        cv2.rectangle(frame, (x, y), (x + w, y + h), accent, 1, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x + w // 4, y), accent, 4, cv2.LINE_AA)
        TYPE.draw(frame, (x + 18, y + 16), label.upper(), max(15, round(self.h * 0.017)), accent, "tech")
        TYPE.draw(frame, (x + 18, y + h - 18), value, max(18, round(self.h * 0.025)), WHITE, "condensed", anchor="ls")

    def global_hud(self, frame: np.ndarray, t: float, scene_name: str) -> None:
        top = max(42, round(self.h * 0.052))
        bottom = max(34, round(self.h * 0.042))
        alpha_rect(frame, (0, 0, self.w, top), (7, 6, 4), 0.86)
        alpha_rect(frame, (0, self.h - bottom, self.w, self.h), (7, 6, 4), 0.90)
        cv2.line(frame, (0, top), (self.w, top), CYAN_SOFT, 1, cv2.LINE_AA)
        TYPE.draw(frame, (round(self.w * 0.026), top // 2), "ATLAS // SECURE MISSION CHANNEL", max(14, round(self.h * 0.016)), CYAN, "tech", anchor="lm")
        TYPE.draw(frame, (self.w - round(self.w * 0.026), top // 2), f"{scene_name.upper()}  •  T+{t:05.1f}", max(14, round(self.h * 0.016)), MUTED, "tech", anchor="rm")
        TYPE.draw(frame, (round(self.w * 0.026), self.h - bottom // 2), "CINEMATIC SYSTEM DEMONSTRATION • RECORDED + SIMULATED MISSION VIEWS", max(12, round(self.h * 0.012)), (132, 147, 145), "regular", anchor="lm")
        cv2.circle(frame, (self.w - round(self.w * 0.035), self.h - bottom // 2), max(3, self.h // 260), GREEN, -1, cv2.LINE_AA)
        TYPE.draw(frame, (self.w - round(self.w * 0.047), self.h - bottom // 2), "SYSTEM ONLINE", max(12, round(self.h * 0.013)), GREEN, "tech", anchor="rm")

    def tracked_points(self, frame: np.ndarray, t: float, region: tuple[int, int, int, int], count: int = 45) -> None:
        x0, y0, x1, y1 = region
        rng = np.random.default_rng(4451)
        for idx in range(count):
            x = int(x0 + (0.08 + rng.random() * 0.84) * (x1 - x0))
            y = int(y0 + (0.14 + rng.random() * 0.72) * (y1 - y0))
            if (idx + int(t * 6)) % 7 == 0:
                radius = 3 + int(2 * pulse(t + idx * 0.03, 1.7))
                cv2.circle(frame, (x, y), radius, CYAN, 1, cv2.LINE_AA)
                if idx % 4 == 0:
                    cv2.line(frame, (x - 6, y), (x + 6, y), CYAN_SOFT, 1, cv2.LINE_AA)
                    cv2.line(frame, (x, y - 6), (x, y + 6), CYAN_SOFT, 1, cv2.LINE_AA)

    def scene_intro(self, local: float) -> np.ndarray:
        frame = self.base()
        self.grid(frame, 0.58)
        mesh = self.deannotate_mesh(self.clean_mesh(self.mesh_angle, round(self.w * 0.58), round(self.h * 0.67), 1.0 + local * 0.006))
        mesh = cv2.convertScaleAbs(mesh, alpha=0.92, beta=-12)
        paste(frame, mesh, round(self.w * 0.43), round(self.h * 0.10))
        alpha_rect(frame, (0, 0, round(self.w * 0.62), self.h), (10, 7, 4), 0.55)
        self.scanline(frame, local, round(self.w * 0.43), round(self.h * 0.11), round(self.w * 0.97), round(self.h * 0.76))
        reveal = ease_out(local / 1.8)
        x = round(self.w * 0.065)
        y = round(self.h * 0.27)
        cv2.line(frame, (x, y), (x + round(self.w * 0.11 * reveal), y), CYAN, max(2, self.w // 700), cv2.LINE_AA)
        TYPE.draw(frame, (x, y + round(self.h * 0.045)), "THE ROOM IS DARK.", round(self.h * 0.070), WHITE, "condensed")
        TYPE.draw(frame, (x, y + round(self.h * 0.121)), "THE MISSION IS NOT.", round(self.h * 0.070), CYAN, "condensed")
        if local > 2.2:
            TYPE.draw(frame, (x, y + round(self.h * 0.215)), "NO GPS.  NO BEACONS.  NO BLIND SPOTS.", round(self.h * 0.025), MUTED, "tech")
        if local > 4.1:
            TYPE.draw(frame, (x, y + round(self.h * 0.30)), "ATLAS", round(self.h * 0.055), WHITE, "tech")
            TYPE.draw(frame, (x, y + round(self.h * 0.355)), "AUTONOMOUS TRACKING AND LOCALIZATION USING AERIAL SENSING", round(self.h * 0.015), CYAN, "regular")
        return frame

    def scene_map(self, local: float) -> np.ndarray:
        frame = self.base()
        self.grid(frame, 0.62)
        mesh = self.deannotate_mesh(self.clean_mesh(self.mesh_angle, round(self.w * 0.62), round(self.h * 0.72), 1.0 + local * 0.008))
        paste(frame, mesh, round(self.w * 0.39), round(self.h * 0.09))
        alpha_rect(frame, (0, 0, self.w, self.h), (15, 9, 4), 0.12)
        alpha_rect(frame, (round(self.w * 0.045), round(self.h * 0.17), round(self.w * 0.53), round(self.h * 0.63)), (8, 7, 5), 0.86)
        self.title_block(
            frame,
            "MISSION PREPARATION",
            "BUILD THE WORLD\nBEFORE THE MISSION BEGINS.",
            "A localization-ready 3D digital twin—built from recorded video.",
            round(self.w * 0.073),
            round(self.h * 0.25),
            round(self.w * 0.42),
        )
        stat_y = round(self.h * 0.68)
        gap = round(self.w * 0.18)
        for idx, (label, value) in enumerate((("MAP POINTS", "440,249"), ("MAP CAMERAS", "3,235"), ("FEATURES", "SIFT"), ("PIPELINE", "COLMAP"))):
            self.tech_frame(frame, label, value, round(self.w * 0.075) + idx * gap, stat_y, round(self.w * 0.155), round(self.h * 0.105))
        self.scanline(frame, local + 0.4, round(self.w * 0.42), round(self.h * 0.12), round(self.w * 0.96), round(self.h * 0.76))
        return frame

    def scene_localize(self, local: float) -> np.ndarray:
        frame = self.base()
        left_w = round(self.w * 0.57)
        video = cover(self.flight.frame(18.0 + local * 6.1), left_w, self.h)
        video = cv2.convertScaleAbs(video, alpha=1.03, beta=-5)
        paste(frame, video, 0, 0)
        pose_index = min(2, int((local / 12.0) * 3))
        right = self.clean_mesh(self.mesh_poses[pose_index], self.w - left_w, self.h, 1.12)
        paste(frame, right, left_w, 0)
        alpha_rect(frame, (0, 0, self.w, self.h), (14, 8, 3), 0.10)
        cv2.line(frame, (left_w, 0), (left_w, self.h), CYAN, 2, cv2.LINE_AA)
        self.tracked_points(frame, local, (0, round(self.h * 0.08), left_w, round(self.h * 0.92)), 54)
        alpha_rect(frame, (round(self.w * 0.035), round(self.h * 0.57), round(self.w * 0.53), round(self.h * 0.88)), (5, 5, 4), 0.82)
        self.title_block(
            frame,
            "ONLINE SELF-LOCALIZATION",
            "SEE THE ROOM.\nKNOW THE POSITION.",
            "Global registration and fast visual continuity at mission speed.",
            round(self.w * 0.058),
            round(self.h * 0.61),
            round(self.w * 0.40),
        )
        metrics = [
            ("TSOLVE POSE", f"{31 + int(8 * pulse(local, 0.7))} ms"),
            ("TRACKS", str(162 + int(37 * pulse(local, 1.1)))),
            ("STATE", "CONFIRMED"),
        ]
        for idx, (label, value) in enumerate(metrics):
            self.tech_frame(frame, label, value, left_w + round(self.w * 0.035), round(self.h * 0.17) + idx * round(self.h * 0.14), round(self.w * 0.34), round(self.h * 0.105))
        self.scanline(frame, local, left_w + 2, round(self.h * 0.08), self.w - 2, round(self.h * 0.88))
        return frame

    def site_card(self, frame: np.ndarray, source: np.ndarray, box: tuple[int, int, int, int], name: str, location: str, status: str, t: float, color: tuple[int, int, int] = CYAN) -> None:
        x0, y0, x1, y1 = box
        panel = cover(source, x1 - x0, y1 - y0)
        panel = cv2.convertScaleAbs(panel, alpha=0.92, beta=-9)
        paste(frame, panel, x0, y0)
        alpha_rect(frame, (x0, y1 - round(self.h * 0.10), x1, y1), (8, 7, 5), 0.86)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x0, y0), (x0 + round((x1 - x0) * 0.25), y0), color, 6, cv2.LINE_AA)
        TYPE.draw(frame, (x0 + 18, y1 - round(self.h * 0.066)), name, round(self.h * 0.026), WHITE, "tech")
        TYPE.draw(frame, (x0 + 18, y1 - round(self.h * 0.034)), location, round(self.h * 0.015), MUTED, "regular")
        cv2.circle(frame, (x1 - 118, y1 - round(self.h * 0.051)), 5 + int(2 * pulse(t, 1.4)), color, -1, cv2.LINE_AA)
        TYPE.draw(frame, (x1 - 24, y1 - round(self.h * 0.051)), status, round(self.h * 0.016), color, "tech", anchor="rm")

    def scene_fleet(self, local: float) -> np.ndarray:
        frame = self.base()
        mesh = self.clean_mesh(self.mesh_top, self.w, self.h, 1.18)
        mesh = cv2.GaussianBlur(mesh, (0, 0), 4.0)
        paste(frame, cv2.convertScaleAbs(mesh, alpha=0.38, beta=-25), 0, 0)
        self.grid(frame, 0.56)
        alpha_rect(frame, (0, 0, self.w, self.h), (12, 8, 5), 0.38)
        TYPE.draw(frame, (round(self.w * 0.055), round(self.h * 0.13)), "ONE CONSOLE. EVERY AIRCRAFT.", round(self.h * 0.055), WHITE, "condensed")
        TYPE.draw(frame, (round(self.w * 0.057), round(self.h * 0.19)), "SIMULTANEOUS MULTI-SITE OPERATIONS", round(self.h * 0.021), CYAN, "tech")
        margin = round(self.w * 0.055)
        gap = round(self.w * 0.018)
        card_w = (self.w - 2 * margin - 2 * gap) // 3
        y0, y1 = round(self.h * 0.29), round(self.h * 0.72)
        self.site_card(frame, self.flight.frame(61 + local * 3.8), (margin, y0, margin + card_w, y1), "ATLAS-01", "TEL AVIV LAB", "PATROL", local)
        self.site_card(frame, self.home.frame(5 + local * 2.0), (margin + card_w + gap, y0, margin + 2 * card_w + gap, y1), "ATLAS-02", "REMOTE TEST NODE", "LOCALIZED", local + 0.2)
        self.site_card(frame, self.lab.frame(15 + local * 4.7), (margin + 2 * (card_w + gap), y0, margin + 3 * card_w + 2 * gap, y1), "ATLAS-03", "RESEARCH WING", "TRACKING", local + 0.4)
        for idx, (value, label) in enumerate((("03", "AIRCRAFT"), ("03", "ACTIVE"), ("03", "LOCALIZED"), ("00", "ATTENTION"))):
            x = margin + idx * round(self.w * 0.18)
            TYPE.draw(frame, (x, round(self.h * 0.82)), value, round(self.h * 0.043), WHITE, "tech")
            TYPE.draw(frame, (x + round(self.w * 0.042), round(self.h * 0.842)), label, round(self.h * 0.016), MUTED, "regular")
        return frame

    def draw_patrol_route(self, frame: np.ndarray, box: tuple[int, int, int, int], progress: float) -> None:
        x0, y0, x1, y1 = box
        alpha_rect(frame, box, (8, 12, 13), 0.80)
        cv2.rectangle(frame, (x0, y0), (x1, y1), CYAN_SOFT, 1, cv2.LINE_AA)
        points = np.array(
            [
                (x0 + round((x1 - x0) * 0.20), y0 + round((y1 - y0) * 0.75)),
                (x0 + round((x1 - x0) * 0.78), y0 + round((y1 - y0) * 0.75)),
                (x0 + round((x1 - x0) * 0.78), y0 + round((y1 - y0) * 0.25)),
                (x0 + round((x1 - x0) * 0.20), y0 + round((y1 - y0) * 0.25)),
            ],
            np.int32,
        )
        cv2.polylines(frame, [points], True, (78, 88, 90), 3, cv2.LINE_AA)
        segments = [(points[i], points[(i + 1) % 4]) for i in range(4)]
        total = clamp(progress) * 4.0
        for idx, (a, b) in enumerate(segments):
            amount = clamp(total - idx)
            if amount <= 0:
                continue
            end = (a + (b - a) * amount).astype(np.int32)
            cv2.line(frame, tuple(a), tuple(end), CYAN, 8, cv2.LINE_AA)
        current_segment = min(3, int(min(3.999, total)))
        amount = clamp(total - current_segment)
        a, b = segments[current_segment]
        pos = (a + (b - a) * amount).astype(np.int32)
        cv2.circle(frame, tuple(pos), 18, (9, 9, 8), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(pos), 12, CYAN, 2, cv2.LINE_AA)
        for idx, point in enumerate(points):
            cv2.circle(frame, tuple(point), 17, (6, 8, 9), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(point), 15, CYAN_SOFT, 2, cv2.LINE_AA)
            TYPE.draw(frame, tuple(point), str(idx + 1), round(self.h * 0.018), WHITE, "tech", anchor="mm")

    def scene_patrol(self, local: float) -> np.ndarray:
        frame = self.base()
        mesh = self.clean_mesh(self.mesh_top, self.w, self.h, 1.13)
        paste(frame, cv2.convertScaleAbs(mesh, alpha=0.72, beta=-14), 0, 0)
        alpha_rect(frame, (0, 0, self.w, self.h), (12, 8, 4), 0.22)
        video_w = round(self.w * 0.43)
        video_h = round(self.h * 0.42)
        live = cover(self.flight.frame(220 + local * 8.0), video_w, video_h)
        paste(frame, live, round(self.w * 0.055), round(self.h * 0.29))
        cv2.rectangle(frame, (round(self.w * 0.055), round(self.h * 0.29)), (round(self.w * 0.055) + video_w, round(self.h * 0.29) + video_h), CYAN, 2, cv2.LINE_AA)
        TYPE.draw(frame, (round(self.w * 0.07), round(self.h * 0.325)), "LIVE DRONE // ONLINE POSE", round(self.h * 0.018), CYAN, "tech")
        lap_progress = local / (SCENES[4].end - SCENES[4].start)
        route_box = (round(self.w * 0.56), round(self.h * 0.27), round(self.w * 0.94), round(self.h * 0.70))
        self.draw_patrol_route(frame, route_box, (lap_progress * 2.0) % 1.0)
        TYPE.draw(frame, (round(self.w * 0.60), round(self.h * 0.19)), "PATROL. VERIFY. RELOCALIZE.", round(self.h * 0.041), WHITE, "condensed")
        lap = 1 if lap_progress < 0.5 else 2
        TYPE.draw(frame, (round(self.w * 0.59), round(self.h * 0.76)), f"LAP 0{lap} / 02", round(self.h * 0.056), CYAN, "tech")
        TYPE.draw(frame, (round(self.w * 0.59), round(self.h * 0.82)), "4 CHECKPOINTS  •  ONLINE VERIFICATION  •  GUARDED MOTION", round(self.h * 0.016), MUTED, "regular")
        self.scanline(frame, local + 0.7, round(self.w * 0.52), round(self.h * 0.14), round(self.w * 0.97), round(self.h * 0.85))
        return frame

    def nearest_enemy_box(self, seconds: float) -> dict | None:
        if not self.enemy_boxes:
            return None
        return min(self.enemy_boxes, key=lambda item: abs(float(item.get("time_sec", 0.0)) - seconds))

    def enemy_frame(self, seconds: float, width: int, height: int, locked: bool = True) -> np.ndarray:
        raw = self.enemy.frame(seconds)
        src_h, src_w = raw.shape[:2]
        box_item = self.nearest_enemy_box(seconds)
        box_pixels: tuple[int, int, int, int] | None = None
        if box_item:
            box = box_item["box"]
            bw, bh = float(box["width"]) * src_w, float(box["height"]) * src_h
            cx, cy = float(box["x_center"]) * src_w, float(box["y_center"]) * src_h
            box_pixels = (round(cx - bw / 2), round(cy - bh / 2), round(cx + bw / 2), round(cy + bh / 2))
            if locked:
                cv2.rectangle(raw, box_pixels[:2], box_pixels[2:], RED, max(5, src_w // 260), cv2.LINE_AA)
                TYPE.draw(raw, (box_pixels[0], max(50, box_pixels[1] - 28)), "NEO1 // 94.7%", max(36, src_w // 40), RED, "tech", stroke=2)
        return cover(raw, width, height, 1.0)

    def scene_threat(self, local: float) -> np.ndarray:
        if local < 2.2:
            frame = self.clean_mesh(self.mesh_angle, self.w, self.h, 1.22)
            frame = cv2.convertScaleAbs(frame, alpha=0.68, beta=-18)
            alpha_rect(frame, (0, 0, self.w, self.h), (20, 7, 5), 0.36)
            alpha_rect(frame, (round(self.w * 0.055), round(self.h * 0.28), round(self.w * 0.71), round(self.h * 0.66)), (9, 6, 5), 0.82)
            self.title_block(
                frame,
                "PERIMETER EVENT",
                "UNKNOWN AIRCRAFT\nENTERED THE ROOM.",
                "The trained target bank activates a guarded mission response.",
                round(self.w * 0.085),
                round(self.h * 0.34),
                round(self.w * 0.50),
                True,
            )
            self.scanline(frame, local, round(self.w * 0.38), round(self.h * 0.10), round(self.w * 0.96), round(self.h * 0.88), True)
        else:
            seconds = 4.0 + (local - 2.2) * 1.75
            frame = self.enemy_frame(seconds, self.w, self.h, True)
            frame = cv2.convertScaleAbs(frame, alpha=0.92, beta=-4)
            alpha_rect(frame, (0, 0, self.w, self.h), (16, 5, 4), 0.11)
            cv2.rectangle(frame, (round(self.w * 0.035), round(self.h * 0.10)), (round(self.w * 0.965), round(self.h * 0.90)), RED, 2, cv2.LINE_AA)
            TYPE.draw(frame, (round(self.w * 0.062), round(self.h * 0.16)), "THREAT DETECTED", round(self.h * 0.050), RED, "tech")
            TYPE.draw(frame, (round(self.w * 0.064), round(self.h * 0.205)), "NEO1  •  TRACK LOCK  •  RANGE VALIDATION ACTIVE", round(self.h * 0.020), WHITE, "regular")
            self.tech_frame(frame, "CLASS", "NEO1", round(self.w * 0.065), round(self.h * 0.72), round(self.w * 0.15), round(self.h * 0.11), True)
            self.tech_frame(frame, "CONFIDENCE", "94.7%", round(self.w * 0.23), round(self.h * 0.72), round(self.w * 0.15), round(self.h * 0.11), True)
            self.tech_frame(frame, "RESPONSE", "GUARDED", round(self.w * 0.395), round(self.h * 0.72), round(self.w * 0.16), round(self.h * 0.11), True)
        return frame

    def tactical_map(self, frame: np.ndarray, box: tuple[int, int, int, int], p: float) -> None:
        x0, y0, x1, y1 = box
        alpha_rect(frame, box, (6, 10, 12), 0.93)
        cv2.rectangle(frame, (x0, y0), (x1, y1), CYAN_SOFT, 1, cv2.LINE_AA)
        for i in range(1, 5):
            x = x0 + (x1 - x0) * i // 5
            y = y0 + (y1 - y0) * i // 5
            cv2.line(frame, (x, y0), (x, y1), (30, 41, 44), 1)
            cv2.line(frame, (x0, y), (x1, y), (30, 41, 44), 1)
        friendly = np.array([x0 + round((x1 - x0) * (0.12 + 0.54 * smooth(p))), y0 + round((y1 - y0) * (0.78 - 0.28 * smooth(p)))])
        hostile = np.array([x0 + round((x1 - x0) * (0.82 - 0.12 * smooth(p))), y0 + round((y1 - y0) * (0.28 + 0.18 * smooth(p)))])
        cv2.line(frame, (x0 + 34, y1 - 34), tuple(friendly), CYAN, 5, cv2.LINE_AA)
        cv2.line(frame, (x1 - 34, y0 + 34), tuple(hostile), RED, 5, cv2.LINE_AA)
        cv2.circle(frame, tuple(friendly), 17, CYAN, 3, cv2.LINE_AA)
        cv2.circle(frame, tuple(hostile), 17, RED, 3, cv2.LINE_AA)
        radius = round(42 + 14 * pulse(p, 2.0))
        cv2.circle(frame, tuple(hostile), radius, RED, 1, cv2.LINE_AA)
        TYPE.draw(frame, (x0 + 22, y0 + 24), "SAFE INTERCEPT CORRIDOR", round(self.h * 0.017), WHITE, "tech")
        TYPE.draw(frame, (friendly[0] + 22, friendly[1]), "ATLAS-01", round(self.h * 0.014), CYAN, "tech", anchor="lm")
        TYPE.draw(frame, (hostile[0] - 22, hostile[1]), "NEO1", round(self.h * 0.014), RED, "tech", anchor="rm")

    def scene_intercept(self, local: float) -> np.ndarray:
        frame = self.deannotate_mesh(self.clean_mesh(self.mesh_angle, self.w, self.h, 1.16))
        frame = cv2.convertScaleAbs(frame, alpha=0.72, beta=-18)
        alpha_rect(frame, (0, 0, self.w, self.h), (10, 8, 6), 0.16)
        TYPE.draw(frame, (self.w // 2, round(self.h * 0.14)), "GUARDED INTERCEPT", round(self.h * 0.049), WHITE, "condensed", anchor="mm")
        TYPE.draw(frame, (self.w // 2, round(self.h * 0.19)), "3D MISSION SPACE  •  TRACK  •  CONTAIN  •  RETURN TO PATROL", round(self.h * 0.018), CYAN, "tech", anchor="mm")

        p = smooth(local / 6.2)
        start = np.array([round(self.w * 0.28), round(self.h * 0.70)])
        target = np.array([round(self.w * 0.70), round(self.h * 0.43)])
        stop = target + np.array([-round(self.w * 0.11), round(self.h * 0.08)])
        drone = (start + (stop - start) * p).astype(np.int32)
        corridor = np.array([
            start + np.array([-24, 38]),
            start + np.array([22, -30]),
            stop + np.array([34, -32]),
            stop + np.array([-32, 36]),
        ], np.int32)
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, corridor, CYAN, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
        cv2.polylines(frame, [corridor], True, CYAN_SOFT, 2, cv2.LINE_AA)
        cv2.line(frame, tuple(start), tuple(stop), CYAN, 4, cv2.LINE_AA)
        for marker in np.linspace(0.10, 0.90, 5):
            point = (start + (stop - start) * marker).astype(np.int32)
            cv2.circle(frame, tuple(point), 5, CYAN, -1, cv2.LINE_AA)
        self.drone_sprite(frame, tuple(drone), 1.05 + 0.08 * pulse(local, 1.0), -0.56, CYAN)
        self.neo_symbol(frame, tuple(target), round(self.h * 0.047), local, local > 5.1)
        distance = max(0.52, 3.8 - p * 3.25)
        TYPE.draw(frame, (drone[0] - 10, drone[1] + round(self.h * 0.085)), "DJI MINI 3 PRO // ATLAS-01", round(self.h * 0.015), CYAN, "tech", anchor="ma")
        self.tech_frame(frame, "RANGE", f"{distance:.2f} m", round(self.w * 0.06), round(self.h * 0.73), round(self.w * 0.16), round(self.h * 0.11))
        self.tech_frame(frame, "CLEARANCE", "0.50 m", round(self.w * 0.06), round(self.h * 0.59), round(self.w * 0.16), round(self.h * 0.11))
        self.tech_frame(frame, "TARGET", "NEO1", round(self.w * 0.78), round(self.h * 0.59), round(self.w * 0.16), round(self.h * 0.11), True)
        self.tech_frame(frame, "LOCK", "CONFIRMED", round(self.w * 0.78), round(self.h * 0.73), round(self.w * 0.16), round(self.h * 0.11), True)
        self.scanline(frame, local, round(self.w * 0.22), round(self.h * 0.24), round(self.w * 0.77), round(self.h * 0.84), local < 5.1)
        if local > 5.0:
            alpha_rect(frame, (round(self.w * 0.31), round(self.h * 0.75), round(self.w * 0.69), round(self.h * 0.84)), (14, 35, 24), 0.86)
            TYPE.draw(frame, (self.w // 2, round(self.h * 0.795)), "SAFE CONTAINMENT CONFIRMED", round(self.h * 0.025), GREEN, "tech", anchor="mm")
        return frame

    def scene_outro(self, local: float) -> np.ndarray:
        frame = self.base()
        poster = cover(self.poster, self.w, self.h, 1.0 + local * 0.004, pan_y=0.05)
        poster = cv2.convertScaleAbs(poster, alpha=0.72, beta=-9)
        paste(frame, poster, 0, 0)
        alpha_rect(frame, (0, 0, self.w, self.h), (9, 7, 5), 0.30)
        alpha_rect(frame, (round(self.w * 0.10), round(self.h * 0.19), round(self.w * 0.90), round(self.h * 0.77)), (5, 7, 9), 0.54)
        TYPE.draw(frame, (self.w // 2, round(self.h * 0.31)), "ATLAS", round(self.h * 0.105), WHITE, "tech", anchor="mm")
        cv2.line(frame, (round(self.w * 0.35), round(self.h * 0.39)), (round(self.w * 0.65), round(self.h * 0.39)), CYAN, 2, cv2.LINE_AA)
        TYPE.draw(frame, (self.w // 2, round(self.h * 0.46)), "FROM INDOOR AUTONOMY\nTO OPERATIONAL AWARENESS.", round(self.h * 0.049), WHITE, "condensed", anchor="mm", spacing=8)
        if local > 2.4:
            TYPE.draw(frame, (self.w // 2, round(self.h * 0.60)), "DEFENSE  •  CRITICAL INFRASTRUCTURE  •  PUBLIC SAFETY  •  INDUSTRY", round(self.h * 0.020), CYAN, "tech", anchor="mm")
        if local > 4.2:
            TYPE.draw(frame, (self.w // 2, round(self.h * 0.68)), "AUTONOMOUS TRACKING AND LOCALIZATION USING AERIAL SENSING", round(self.h * 0.017), MUTED, "regular", anchor="mm")
        return frame

    def render(self, t: float) -> np.ndarray:
        scene = next((item for item in SCENES if item.start <= t < item.end), SCENES[-1])
        local = t - scene.start
        renderer = getattr(self, f"scene_{scene.name}")
        frame = renderer(local)
        self.global_hud(frame, t, scene.name)
        # Filmic color and edge treatment.
        frame = np.clip(frame.astype(np.float32) * self.vignette, 0, 255).astype(np.uint8)
        for y in range(0, self.h, max(4, self.h // 220)):
            frame[y : y + 1] = (frame[y : y + 1].astype(np.float32) * 0.94).astype(np.uint8)
        # Clean dip-to-black around scene boundaries.
        fade = min(clamp(local / 0.32), clamp((scene.end - t) / 0.30))
        if scene.name == "intro":
            fade = min(fade, clamp(t / 0.8))
        if scene.name == "outro":
            fade = min(fade, clamp((SCENES[-1].end - t) / 1.1))
        frame = (frame.astype(np.float32) * smooth(fade)).astype(np.uint8)
        return frame


def write_soundtrack(path: Path, duration: float, sample_rate: int = 48_000) -> None:
    """Synthesize a dark orchestral/electronic score without a ticking pulse."""
    count = int(duration * sample_rate)
    t = np.arange(count, dtype=np.float64) / sample_rate
    music = np.zeros(count, np.float64)
    bpm = 92.0
    beat = 60.0 / bpm
    rng = np.random.default_rng(418)

    def env(start: float, length: float, attack: float = 0.02, release: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
        i0 = max(0, int(start * sample_rate))
        i1 = min(count, int((start + length) * sample_rate))
        local = np.arange(i1 - i0) / sample_rate
        attack_curve = np.minimum(1.0, local / max(attack, 1e-4))
        release_curve = np.minimum(1.0, (length - local) / max(release, 1e-4))
        return np.arange(i0, i1), np.clip(attack_curve * release_curve, 0, 1)

    def tone(start: float, length: float, freq: float, gain: float, kind: str = "sine", attack: float = 0.02, release: float = 0.12) -> None:
        idx, shape = env(start, length, attack, release)
        local = (idx / sample_rate) - start
        phase = math.tau * freq * local
        if kind == "triangle":
            signal = 2.0 / math.pi * np.arcsin(np.sin(phase))
        elif kind == "saw":
            signal = 2.0 * ((freq * local) % 1.0) - 1.0
        else:
            signal = np.sin(phase)
        music[idx] += gain * shape * signal

    # Low, spacious D-minor movement. Notes arrive as phrases, never clock ticks.
    bass = [73.42, 65.41, 58.27, 69.30]
    motif = [293.66, 349.23, 440.00, 392.00, 349.23, 261.63]
    beat_index = 0
    current = 0.0
    while current < duration:
        scene_gain = 0.48 if current < 51 else 0.70 if current < 71 else 0.40
        if beat_index % 2 == 0:
            tone(current, beat * 1.72, bass[(beat_index // 2) % len(bass)], 0.17 * scene_gain, "triangle", 0.055, 0.55)
        if current > 8 and beat_index % 4 in (0, 1):
            note = motif[(beat_index // 4) % len(motif)]
            tone(current + beat * 0.18, beat * 1.25, note, 0.080 * scene_gain, "sine", 0.12, 0.48)
            tone(current + beat * 0.18, beat * 1.25, note * 0.5, 0.038 * scene_gain, "triangle", 0.12, 0.48)
        beat_index += 1
        current += beat

    # Low cinematic pad.
    for start, stop, root in ((0, 15, 73.42), (15, 40, 65.41), (40, 51, 73.42), (51, 63, 58.27), (63, 71, 55.00), (71, 79, 73.42)):
        length = stop - start
        for ratio, gain in ((1.0, 0.075), (1.5, 0.050), (2.0, 0.030)):
            tone(start, length, root * ratio, gain, "sine", 1.2, 1.4)

    # Restrained cinematic drums: low toms on broad beats, no hats/ticking.
    current = 7.0
    beat_number = 0
    while current < 71:
        if beat_number % 2 == 0:
            idx, shape = env(current, 0.54, 0.004, 0.45)
            local = (idx / sample_rate) - current
            tom = np.sin(math.tau * (68 - 21 * local) * local) * np.exp(-local * 7.5)
            music[idx] += (0.20 if current < 51 else 0.31) * shape * tom
        if beat_number % 8 == 6:
            idx2, shape2 = env(current, 0.55, 0.003, 0.45)
            noise = rng.standard_normal(len(idx2))
            # Smoothed noise becomes a distant cinematic snare wash.
            kernel = np.ones(24) / 24
            wash = np.convolve(noise, kernel, mode="same")
            music[idx2] += 0.12 * shape2 * wash
        beat_number += 1
        current += beat

    # Radar pings and interface beeps.
    for start in (15.4, 21.2, 27.5, 34.0, 40.5):
        tone(float(start), 0.72, 740.0, 0.050, "sine", 0.025, 0.62)
        tone(float(start) + 0.06, 0.58, 1110.0, 0.024, "sine", 0.025, 0.50)
    for start in (51.0, 52.1, 54.2, 57.0, 60.0):
        tone(start, 0.18, 660.0, 0.11, "square" if False else "saw", 0.003, 0.14)
        tone(start + 0.22, 0.16, 880.0, 0.08, "sine", 0.003, 0.12)

    # Impacts at the cuts.
    for start, gain in ((0.7, 0.38), (7.0, 0.42), (15.0, 0.44), (27.0, 0.48), (40.0, 0.48), (51.0, 0.72), (63.0, 0.65), (71.0, 0.54)):
        idx, shape = env(start, 1.5, 0.004, 1.3)
        local = (idx / sample_rate) - start
        boom = np.sin(math.tau * (47 - 11 * local) * local) * np.exp(-local * 2.4)
        music[idx] += gain * shape * boom
        noise = rng.standard_normal(len(idx))
        music[idx] += gain * 0.10 * shape * noise * np.exp(-local * 4)

    # Rising tension into the detection and intercept beats.
    for start, length, low, high in ((47.0, 4.0, 180.0, 950.0), (59.5, 3.5, 220.0, 1300.0)):
        idx, shape = env(start, length, 0.7, 0.15)
        local = (idx / sample_rate) - start
        frac = local / length
        phase = math.tau * (low * local + (high - low) * local * frac * 0.5)
        music[idx] += 0.065 * shape * np.sin(phase)
        music[idx] += 0.025 * shape * rng.standard_normal(len(idx))

    # Brass-like hero statement and a wide final resolve.
    for start, freq in ((63.0, 146.83), (64.3, 174.61), (66.0, 220.0), (68.0, 261.63)):
        tone(start, 2.4, freq, 0.12, "saw", 0.18, 1.0)
        tone(start, 2.4, freq * 1.5, 0.050, "sine", 0.18, 1.0)
    for start, freq in ((71.0, 293.66), (72.1, 349.23), (73.3, 440.0), (74.6, 587.33)):
        tone(start, 4.1, freq, 0.090, "sine", 0.22, 2.1)

    # Mild stereo width and mastering limiter.
    delay = int(sample_rate * 0.011)
    left = music.copy()
    right = np.roll(music, delay) * 0.90 + music * 0.16
    right[:delay] = music[:delay] * 0.16
    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo))) or 1.0
    stereo = np.tanh(stereo * (0.94 / peak) * 1.8) / np.tanh(1.8)
    pcm = np.clip(stereo * 32767, -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def render_stills(film: Film, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    times = (2.7, 10.4, 20.2, 32.5, 45.2, 57.0, 67.0, 75.0)
    for index, timestamp in enumerate(times, 1):
        path = output_dir / f"storyboard_v2_{index:02d}_{timestamp:04.1f}s.jpg"
        cv2.imwrite(str(path), film.render(timestamp), [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(path)


def render_video(film: Film, path: Path, duration: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, film.fps, (film.w, film.h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}")
    frame_count = round(duration * film.fps)
    for index in range(frame_count):
        writer.write(film.render(index / film.fps))
        if index % max(1, film.fps * 5) == 0:
            print(f"rendered {index / film.fps:5.1f}/{duration:.1f}s", flush=True)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=79.0)
    parser.add_argument("--stills-only", action="store_true")
    args = parser.parse_args()
    film = Film(args.width, args.height, args.fps)
    render_stills(film, OUT / "storyboard")
    write_soundtrack(OUT / "audio" / "atlas_v2_orchestral_spy_score.wav", args.duration)
    if not args.stills_only:
        render_video(film, OUT / "work" / "atlas_spy_demo_v2_silent.mp4", args.duration)


if __name__ == "__main__":
    main()

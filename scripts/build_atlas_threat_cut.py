#!/usr/bin/env python3
"""Render the two-minute ATLAS threat cut requested for sponsor presentation.

The film combines real project UI captures and several recorded flight/NEO
sources.  The guarded-intercept sequence is explicitly marked as a cinematic
simulation; no claim is made that it is a recording of a physical intercept.
"""

from __future__ import annotations

import math
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
    NAVY,
    RED,
    TYPE,
    WHITE,
    VideoSource,
    alpha_rect,
    bgr,
    clamp,
    contain,
    cover,
    paste,
    pulse,
    smooth,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "spy_demo"
CAPTURES = OUT / "captures"
BRANDING = ROOT / "viewer" / "public" / "branding"
WORK = OUT / "work" / "threat_cut"
MUSIC = Path("/Users/yamromano/Downloads/41 Minutes of Spy Music - Instrumental Spy Themes.mp3")
SILENT = WORK / "atlas_threat_cut_2min_silent.mp4"
IMPACT = WORK / "atlas_impact_sfx.wav"
FINAL = OUT / "ATLAS_THREAT_RESPONSE_CINEMATIC_2MIN.mp4"

WIDTH, HEIGHT, FPS, DURATION = 1920, 1080, 24, 120.0


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def prepare_sources() -> None:
    """Create edit-friendly 24 fps copies of the two portrait NEO recordings."""
    WORK.mkdir(parents=True, exist_ok=True)
    sources = (
        (
            ROOT / "viewer/public/enemy_drones/enemy_20260802_104805_b94cef/videos/enemy_calib_211d7230.MOV",
            WORK / "neo_a_24fps.mp4",
            "0",
            "24",
        ),
        (
            ROOT / "viewer/public/enemy_drones/enemy_20260802_104805_b94cef/videos/enemy_calib_3f046629.MOV",
            WORK / "neo_b_24fps.mp4",
            "6",
            "30",
        ),
    )
    for source, target, start, duration in sources:
        if target.exists():
            continue
        run(
            [
                "/opt/homebrew/bin/ffmpeg",
                "-y",
                "-ss",
                start,
                "-i",
                str(source),
                "-t",
                duration,
                "-an",
                "-vf",
                "fps=24,scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2:black",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                str(target),
            ]
        )


def line_text(frame: np.ndarray, text: str, xy: tuple[int, int], size: int, color: tuple[int, int, int], anchor: str = "la") -> None:
    TYPE.draw(frame, xy, text, size, color, "tech", anchor=anchor, stroke=1, stroke_color=(0, 0, 0))


class ThreatFilm:
    def __init__(self) -> None:
        self.w, self.h, self.fps = WIDTH, HEIGHT, FPS
        self.root = bgr(CAPTURES / "atlas_root.png")
        self.monitor = bgr(CAPTURES / "atlas_monitor.png")
        self.enemy_lab = bgr(CAPTURES / "atlas_enemy_lab.png")
        self.patrol_top = bgr(CAPTURES / "atlas_patrol_top.png")
        self.patrol_angle = bgr(CAPTURES / "atlas_patrol_angle.png")
        self.poster = bgr(BRANDING / "atlas-space-poster.png")
        self.wordmark = bgr(BRANDING / "atlas-space-wordmark.png")
        self.symbol = bgr(BRANDING / "atlas-space-symbol.png")

        self.path_a = VideoSource(ROOT / "runtime/manual_path_videos/Live_ATLAS_11-07-56_20260729.mp4")
        self.path_b = VideoSource(ROOT / "data/app_uploads/drone_upload_1699b1a3.MP4")
        self.path_c = VideoSource(ROOT / "data/app_uploads/drone_upload_5afa5460.MP4")
        self.neo_a = VideoSource(WORK / "neo_a_24fps.mp4")
        self.neo_b = VideoSource(WORK / "neo_b_24fps.mp4")

        y, x = np.ogrid[: self.h, : self.w]
        nx = (x - self.w / 2) / (self.w / 2)
        ny = (y - self.h / 2) / (self.h / 2)
        self.vignette = np.clip(1.05 - 0.27 * (nx * nx + ny * ny), 0.68, 1.0)[..., None]

    def base(self) -> np.ndarray:
        return np.full((self.h, self.w, 3), (11, 9, 6), np.uint8)

    def fit_capture(self, image: np.ndarray, margin: int = 54) -> np.ndarray:
        """Show the complete UI capture; never slice off navigation or panels."""
        frame = self.base()
        backdrop = cover(image, self.w, self.h)
        backdrop = cv2.GaussianBlur(backdrop, (0, 0), 20)
        backdrop = cv2.convertScaleAbs(backdrop, alpha=0.42, beta=-18)
        paste(frame, backdrop, 0, 0)
        panel = contain(image, self.w - 2 * margin, self.h - 2 * margin, (5, 7, 8))
        paste(frame, panel, margin, margin)
        cv2.rectangle(frame, (margin, margin), (self.w - margin, self.h - margin), CYAN_SOFT, 2, cv2.LINE_AA)
        return frame

    def camera_full(self, image: np.ndarray) -> np.ndarray:
        # Recorded patrol sources are native or effectively 16:9; no visual
        # content is cropped at this fit.
        return cover(image, self.w, self.h)

    def title(self, frame: np.ndarray, eyebrow: str, title: str, subtitle: str, alert: bool = False, x: int = 96, y: int = 120) -> None:
        accent = RED if alert else CYAN
        cv2.line(frame, (x, y), (x + 230, y), accent, 4, cv2.LINE_AA)
        line_text(frame, eyebrow.upper(), (x, y + 22), 23, accent)
        TYPE.draw(frame, (x, y + 65), title.upper(), 62, WHITE, "condensed", stroke=1, stroke_color=(0, 0, 0))
        TYPE.draw(frame, (x, y + 145 + 62 * title.count("\n")), subtitle, 25, MUTED, "regular", stroke=1, stroke_color=(0, 0, 0))

    def box(self, frame: np.ndarray, xywh: tuple[int, int, int, int], label: str, value: str, alert: bool = False) -> None:
        x, y, w, h = xywh
        accent = RED if alert else CYAN
        alpha_rect(frame, (x, y, x + w, y + h), (7, 8, 9), 0.84)
        cv2.rectangle(frame, (x, y), (x + w, y + h), accent, 1, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x + w // 3, y), accent, 5, cv2.LINE_AA)
        line_text(frame, label.upper(), (x + 17, y + 18), 18, accent)
        TYPE.draw(frame, (x + 17, y + h - 16), value, 27, WHITE, "condensed", anchor="ls")

    def hud(self, frame: np.ndarray, t: float, scene: str, alert: bool = False, simulation: bool = False) -> None:
        accent = RED if alert else CYAN_SOFT
        alpha_rect(frame, (0, 0, self.w, 58), (4, 6, 8), 0.90)
        alpha_rect(frame, (0, self.h - 45, self.w, self.h), (4, 6, 8), 0.92)
        cv2.line(frame, (0, 58), (self.w, 58), accent, 1, cv2.LINE_AA)
        line_text(frame, "ATLAS // SECURE AUTONOMOUS MISSION", (48, 30), 17, CYAN, "lm")
        line_text(frame, f"{scene.upper()}  •  T+{t:05.1f}", (self.w - 48, 30), 17, WHITE, "rm")
        footer = "CINEMATIC INTERCEPT SIMULATION • RECORDED PROJECT SOURCES" if simulation else "RECORDED PROJECT SOURCES • CINEMATIC PRESENTATION"
        line_text(frame, footer, (48, self.h - 22), 14, RED if simulation else MUTED, "lm")
        cv2.circle(frame, (self.w - 55, self.h - 22), 5, GREEN, -1, cv2.LINE_AA)
        line_text(frame, "SYSTEM ONLINE", (self.w - 72, self.h - 22), 14, GREEN, "rm")

    def scan(self, frame: np.ndarray, t: float, box: tuple[int, int, int, int], red: bool = False) -> None:
        x0, y0, x1, y1 = box
        accent = RED if red else CYAN
        y = round(y0 + ((t * 0.26) % 1.0) * (y1 - y0))
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, max(y0, y - 18)), (x1, min(y1, y + 18)), accent, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.line(frame, (x0, y), (x1, y), accent, 2, cv2.LINE_AA)

    def route(self, frame: np.ndarray, box: tuple[int, int, int, int], p: float, lap: int = 1) -> None:
        x0, y0, x1, y1 = box
        alpha_rect(frame, box, (4, 10, 13), 0.86)
        cv2.rectangle(frame, (x0, y0), (x1, y1), CYAN_SOFT, 1, cv2.LINE_AA)
        pts = np.array([
            [x0 + 40, y0 + 42], [x1 - 48, y0 + 42], [x1 - 48, y1 - 45], [x0 + 40, y1 - 45], [x0 + 40, y0 + 42]
        ], np.int32)
        cv2.polylines(frame, [pts], False, (44, 69, 72), 3, cv2.LINE_AA)
        segments = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(np.sum(segments))
        distance = clamp(p) * total
        drawn: list[np.ndarray] = [pts[0]]
        for idx, length in enumerate(segments):
            if distance >= length:
                drawn.append(pts[idx + 1])
                distance -= length
            else:
                part = pts[idx] + (pts[idx + 1] - pts[idx]) * (distance / max(length, 1e-6))
                drawn.append(part.astype(np.int32))
                break
        if len(drawn) > 1:
            cv2.polylines(frame, [np.array(drawn, np.int32)], False, CYAN, 6, cv2.LINE_AA)
        for idx, point in enumerate(pts[:-1], 1):
            cv2.circle(frame, tuple(point), 13, (7, 9, 10), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(point), 13, CYAN, 2, cv2.LINE_AA)
            line_text(frame, str(idx), tuple(point), 14, WHITE, "mm")
        line_text(frame, f"LAP {lap:02d} // 02", (x0 + 18, y1 - 14), 17, CYAN, "ls")

    def tracking_points(self, frame: np.ndarray, t: float, box: tuple[int, int, int, int]) -> None:
        rng = np.random.default_rng(1927)
        x0, y0, x1, y1 = box
        for i in range(40):
            x = round(x0 + (0.08 + rng.random() * 0.84) * (x1 - x0))
            y = round(y0 + (0.12 + rng.random() * 0.76) * (y1 - y0))
            if (i + int(t * 9)) % 5 == 0:
                cv2.circle(frame, (x, y), 4, CYAN, 1, cv2.LINE_AA)
                cv2.line(frame, (x - 7, y), (x + 7, y), CYAN_SOFT, 1, cv2.LINE_AA)

    def neo_sprite(self, frame: np.ndarray, center: tuple[int, int], scale: float, t: float) -> None:
        """Intentionally low-resolution NEO proxy, enlarged with nearest-neighbor."""
        sw, sh = 184, 112
        sprite = np.zeros((sh, sw, 4), np.uint8)
        # Shadow and four protected propeller ducts.
        for cx, cy in ((45, 34), (139, 34), (45, 79), (139, 79)):
            cv2.circle(sprite, (cx, cy), 29, (35, 39, 42, 230), -1, cv2.LINE_AA)
            cv2.circle(sprite, (cx, cy), 25, (221, 225, 222, 255), 5, cv2.LINE_AA)
            cv2.circle(sprite, (cx, cy), 17, (95, 103, 105, 190), 2, cv2.LINE_AA)
        body = np.array([[92, 17], [124, 42], [117, 83], [92, 99], [67, 83], [60, 42]], np.int32)
        cv2.fillConvexPoly(sprite, body, (225, 228, 223, 255), cv2.LINE_AA)
        cv2.polylines(sprite, [body], True, (94, 104, 105, 255), 2, cv2.LINE_AA)
        cv2.rectangle(sprite, (81, 18), (103, 34), (34, 40, 43, 255), -1)
        cv2.rectangle(sprite, (83, 38), (101, 45), (18, 192, 247, 255), -1)
        # Pixelated presentation is deliberate and requested.
        target_w, target_h = max(28, round(sw * scale)), max(18, round(sh * scale))
        sprite = cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        x0, y0 = center[0] - target_w // 2, center[1] - target_h // 2
        if x0 < 0 or y0 < 0 or x0 + target_w > frame.shape[1] or y0 + target_h > frame.shape[0]:
            return
        alpha = sprite[:, :, 3:4].astype(np.float32) / 255.0
        roi = frame[y0 : y0 + target_h, x0 : x0 + target_w]
        roi[:] = (sprite[:, :, :3] * alpha + roi * (1.0 - alpha)).astype(np.uint8)
        radius = round(max(target_w, target_h) * (0.56 + 0.03 * pulse(t, 2.2)))
        cv2.circle(frame, center, radius, RED, 2, cv2.LINE_AA)
        cv2.line(frame, (center[0] - radius - 15, center[1]), (center[0] + radius + 15, center[1]), RED, 1, cv2.LINE_AA)
        cv2.line(frame, (center[0], center[1] - radius - 15), (center[0], center[1] + radius + 15), RED, 1, cv2.LINE_AA)

    def drone_icon(self, frame: np.ndarray, center: tuple[int, int], scale: float, heading: float = 0.0) -> None:
        cx, cy = center
        forward = np.array([math.cos(heading), math.sin(heading)])
        side = np.array([-forward[1], forward[0]])
        def point(f: float, s: float) -> tuple[int, int]:
            p = np.array([cx, cy]) + scale * (forward * f + side * s)
            return int(p[0]), int(p[1])
        body = np.array([point(27, 0), point(3, 12), point(-29, 9), point(-29, -9), point(3, -12)], np.int32)
        cv2.fillConvexPoly(frame, body, CYAN, cv2.LINE_AA)
        for f, s in ((12, 26), (12, -26), (-18, 25), (-18, -25)):
            rotor = point(f, s)
            cv2.line(frame, point(f * 0.70, s * 0.65), rotor, CYAN, 3, cv2.LINE_AA)
            cv2.circle(frame, rotor, max(7, round(scale * 8)), CYAN, 2, cv2.LINE_AA)

    def scene_threat_open(self, local: float) -> np.ndarray:
        # Alternate the two real NEO recordings from the first frame of the film.
        if local < 3.2:
            # Keep the edit inside the verified interval where NEO is visible.
            source = self.neo_a.frame(0.35 + local * 0.72)
            feed = contain(source, 690, 900, (3, 4, 5))
        else:
            source = self.neo_b.frame(0.35 + ((local - 3.2) * 0.68) % 4.4)
            feed = contain(source, 690, 900, (3, 4, 5))
        frame = self.fit_capture(self.enemy_lab, 38)
        alpha_rect(frame, (0, 0, self.w, self.h), (3, 4, 8), 0.42)
        # Give the new narrative copy a clean field over the captured UI text.
        alpha_rect(frame, (52, 85, 930, 640), (3, 5, 8), 0.94)
        paste(frame, feed, self.w - 760, 92)
        cv2.rectangle(frame, (self.w - 760, 92), (self.w - 70, 992), RED, 3, cv2.LINE_AA)
        self.title(frame, "PROXIMITY EVENT", "UNKNOWN AIRCRAFT\nENTERED THE ROOM.", "Two independent NEO recordings confirm the target profile.", True, 90, 130)
        self.box(frame, (90, 675, 250, 118), "CLASS", "NEO1", True)
        self.box(frame, (365, 675, 250, 118), "TRACK", "LOCKED", True)
        self.box(frame, (90, 815, 525, 118), "MISSION STATE", "THREAT RESPONSE ARMED", True)
        self.scan(frame, local, (self.w - 760, 92, self.w - 70, 992), True)
        return frame

    def scene_title(self, local: float) -> np.ndarray:
        frame = self.base()
        poster = contain(self.poster, self.w, self.h, (4, 6, 8))
        paste(frame, poster, 0, 0)
        alpha_rect(frame, (0, 0, self.w, self.h), (4, 6, 8), 0.22)
        logo_scale = 0.88 + 0.12 * smooth(local / 2.4)
        symbol_w, symbol_h = round(420 * logo_scale), round(260 * logo_scale)
        symbol = contain(self.symbol, symbol_w, symbol_h, (4, 6, 8))
        paste(frame, symbol, self.w // 2 - symbol_w // 2, 105 + (260 - symbol_h) // 2)
        # Moving telemetry rings and a progressive copy reveal keep the brand
        # card alive without turning it into a busy UI screen.
        for index, radius in enumerate((165, 238, 315)):
            arc = round(120 + 160 * ((local * 0.12 + index * 0.21) % 1.0))
            cv2.ellipse(frame, (self.w // 2, 300), (radius, radius // 2), 0, arc, arc + 92, CYAN_SOFT, 1, cv2.LINE_AA)
        if local > 0.6:
            TYPE.draw(frame, (self.w // 2, 435), "ATLAS", 135, WHITE, "tech", anchor="mm")
        line_half = round(350 * smooth((local - 1.0) / 1.5))
        cv2.line(frame, (self.w // 2 - line_half, 520), (self.w // 2 + line_half, 520), CYAN, 2, cv2.LINE_AA)
        if local > 1.8:
            TYPE.draw(frame, (self.w // 2, 615), "THE ROOM IS DARK.\nTHE MISSION IS NOT.", 58, WHITE, "condensed", anchor="mm", spacing=8)
        if local > 3.4:
            TYPE.draw(frame, (self.w // 2, 765), "AUTONOMOUS TRACKING AND LOCALIZATION USING AERIAL SENSING", 25, CYAN, "tech", anchor="mm")
        self.scan(frame, local, (470, 110, 1450, 820), False)
        return frame

    def scene_system(self, local: float) -> np.ndarray:
        source = self.root if local < 6.5 else self.monitor
        frame = self.fit_capture(source, 46)
        alpha_rect(frame, (62, 100, 1040, 410), (4, 8, 11), 0.92)
        self.title(frame, "MISSION PREPARATION", "BUILD THE WORLD\nBEFORE THE MISSION BEGINS.", "COLMAP + SIFT map • TSolve pose • optical-flow continuity", False, 92, 128)
        for idx, (label, value) in enumerate((("MAP POINTS", "440,249"), ("CAMERAS", "3,235"), ("FEATURES", "SIFT"))):
            self.box(frame, (92 + idx * 258, 770, 230, 110), label, value)
        # Live ingest window prevents the setup section from reading as a
        # frozen screenshot while preserving the complete UI background.
        live = contain(self.path_b.frame(8.0 + local * 1.7), 520, 292, (3, 5, 7))
        paste(frame, live, 1325, 655)
        cv2.rectangle(frame, (1325, 655), (1845, 947), CYAN_SOFT, 2, cv2.LINE_AA)
        alpha_rect(frame, (1325, 875, 1845, 947), (3, 6, 8), 0.88)
        line_text(frame, "LIVE MAP INGEST", (1344, 900), 18, WHITE)
        line_text(frame, f"FRAME {round(local * 17):04d} // ACTIVE", (1825, 900), 15, GREEN, "rm")
        self.tracking_points(frame, local, (1325, 655, 1845, 875))
        self.scan(frame, local, (48, 58, self.w - 48, self.h - 48), False)
        return frame

    def scene_localize(self, local: float) -> np.ndarray:
        camera = self.camera_full(self.path_a.frame(20.0 + local * 1.15))
        camera = cv2.convertScaleAbs(camera, alpha=0.90, beta=-8)
        frame = camera
        panel = contain(self.patrol_angle, 650, 650, (4, 8, 11))
        paste(frame, panel, self.w - 700, 105)
        cv2.rectangle(frame, (self.w - 700, 105), (self.w - 50, 755), CYAN_SOFT, 2, cv2.LINE_AA)
        self.title(frame, "LIVE SELF-LOCALIZATION", "SEE THE ROOM.\nKNOW THE POSITION.", "Global registration and fast visual continuity at mission speed.", False, 75, 110)
        self.box(frame, (75, 775, 235, 105), "TSOLVE", "38 ms")
        self.box(frame, (330, 775, 235, 105), "TRACKS", str(168 + int(30 * pulse(local, 0.3))))
        self.box(frame, (585, 775, 235, 105), "POSE", "CONFIRMED")
        self.tracking_points(frame, local, (40, 60, 1200, 950))
        self.scan(frame, local, (self.w - 700, 105, self.w - 50, 755), False)
        return frame

    def panel_video(self, frame: np.ndarray, source: VideoSource, seconds: float, box: tuple[int, int, int, int], title: str, state: str, color: tuple[int, int, int] = CYAN) -> None:
        x0, y0, x1, y1 = box
        shot = contain(source.frame(seconds), x1 - x0, y1 - y0, (3, 5, 7))
        paste(frame, shot, x0, y0)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        alpha_rect(frame, (x0, y1 - 72, x1, y1), (3, 5, 7), 0.86)
        line_text(frame, title, (x0 + 17, y1 - 48), 20, WHITE)
        cv2.circle(frame, (x1 - 105, y1 - 35), 5, GREEN, -1, cv2.LINE_AA)
        line_text(frame, state, (x1 - 90, y1 - 35), 15, GREEN, "lm")

    def scene_paths(self, local: float) -> np.ndarray:
        frame = self.fit_capture(self.monitor, 36)
        alpha_rect(frame, (0, 0, self.w, self.h), (4, 7, 9), 0.55)
        alpha_rect(frame, (40, 64, 1090, 370), (3, 7, 10), 0.94)
        self.title(frame, "DISTRIBUTED PATH ARCHIVE", "ONE SYSTEM.\nMULTIPLE FLIGHT HISTORIES.", "Different recorded routes remain independently localizable.", False, 65, 82)
        top = 410
        gap = 24
        width = (self.w - 2 * 65 - 2 * gap) // 3
        self.panel_video(frame, self.path_a, 72 + local * 1.6, (65, top, 65 + width, 905), "PATH A // LAB LOOP", "LOCALIZED")
        self.panel_video(frame, self.path_b, 18 + local * 1.8, (65 + width + gap, top, 65 + 2 * width + gap, 905), "PATH B // FULL ROOM", "TRACKING")
        self.panel_video(frame, self.path_c, 12 + local * 1.5, (65 + 2 * (width + gap), top, self.w - 65, 905), "PATH C // REMOTE NODE", "LOCALIZED")
        for idx, x in enumerate((65, 65 + width + gap, 65 + 2 * (width + gap))):
            line_text(frame, f"RUN 0{idx + 1}  •  VISUAL POSE STREAM", (x + 12, top - 18), 15, CYAN)
        return frame

    def scene_patrol(self, local: float) -> np.ndarray:
        # Frequent source changes keep the demonstration energetic, while each
        # shot remains long enough to read as real movement rather than noise.
        block = int(local // 4.0)
        within = local % 4.0
        if block % 3 == 0:
            source, sec, label = self.path_b, 52 + block * 17 + within * 1.45, "PATH B // LIVE ROOM"
        elif block % 3 == 1:
            source, sec, label = self.path_a, 106 + block * 16 + within * 1.40, "PATH A // VERIFIED LOOP"
        else:
            source, sec, label = self.path_b, 285 + block * 11 + within * 1.60, "PATH B // SECOND PASS"
        frame = self.camera_full(source.frame(sec))
        frame = cv2.convertScaleAbs(frame, alpha=0.92, beta=-7)
        self.route(frame, (self.w - 520, 95, self.w - 55, 425), (local % 8.0) / 8.0, 1 if local < 10 else 2)
        alpha_rect(frame, (60, 700, 720, 900), (4, 7, 9), 0.80)
        line_text(frame, "AUTONOMOUS PATROL", (88, 740), 25, CYAN)
        TYPE.draw(frame, (88, 790), label, 42, WHITE, "condensed")
        line_text(frame, "VERIFY  •  RELOCALIZE  •  CONTINUE", (88, 852), 18, MUTED)
        self.tracking_points(frame, local, (35, 70, self.w - 40, self.h - 60))
        return frame

    def scene_neo(self, local: float) -> np.ndarray:
        frame = self.fit_capture(self.enemy_lab, 34)
        alpha_rect(frame, (0, 0, self.w, self.h), (4, 4, 8), 0.52)
        alpha_rect(frame, (40, 64, 920, 625), (5, 5, 8), 0.95)
        self.title(frame, "ENEMY-DRONE BANK", "NEO1 PROFILE\nCONFIRMED TWICE.", "Independent recorded viewpoints strengthen the target signature.", True, 65, 80)
        # Each feed loops only over a verified interval with the target in view.
        left = contain(self.neo_a.frame(0.25 + (local * 0.72) % 5.0), 675, 720, (3, 4, 6))
        right = contain(self.neo_b.frame(0.25 + (local * 0.78) % 4.5), 675, 720, (3, 4, 6))
        paste(frame, left, 500, 280)
        paste(frame, right, 1200, 280)
        for x, label in ((500, "NEO FEED A"), (1200, "NEO FEED B")):
            cv2.rectangle(frame, (x, 280), (x + 675, 1000), RED, 2, cv2.LINE_AA)
            alpha_rect(frame, (x, 925, x + 675, 1000), (5, 5, 7), 0.88)
            line_text(frame, label, (x + 18, 950), 19, WHITE)
            line_text(frame, "94.7% MATCH", (x + 655, 950), 18, RED, "rm")
            self.scan(frame, local + x * 0.001, (x, 280, x + 675, 925), True)
        self.box(frame, (65, 650, 360, 110), "TARGET", "NEO1", True)
        self.box(frame, (65, 785, 360, 110), "RESPONSE", "GUARDED", True)
        return frame

    def scene_intercept(self, local: float) -> np.ndarray:
        # Source frames advance at 2x; this is stated prominently in the image.
        source_time = 8.0 + local * 2.0
        frame = self.camera_full(self.path_b.frame(source_time))
        frame = cv2.convertScaleAbs(frame, alpha=0.80, beta=-8)
        alpha_rect(frame, (0, 0, self.w, self.h), (5, 7, 10), 0.12)
        p = smooth(local / 12.0)
        center = (round(self.w * (0.55 + 0.06 * p)), round(self.h * (0.44 + 0.10 * p)))
        scale = 0.28 + 2.30 * p**1.55
        # Perspective motion streaks converge on the NEO proxy.
        rng = np.random.default_rng(3031)
        for i in range(34):
            edge_x = 0 if i % 2 == 0 else self.w
            edge_y = round(rng.random() * self.h)
            frac = 0.30 + 0.45 * pulse(local + i * 0.02, 0.7)
            end = (round(edge_x + (center[0] - edge_x) * frac), round(edge_y + (center[1] - edge_y) * frac))
            cv2.line(frame, (edge_x, edge_y), end, CYAN_SOFT, 1, cv2.LINE_AA)
        self.neo_sprite(frame, center, scale, local)
        alpha_rect(frame, (58, 88, 740, 255), (5, 7, 9), 0.85)
        line_text(frame, "SIMULATED GUARDED INTERCEPT", (84, 122), 23, RED)
        TYPE.draw(frame, (84, 160), "DJI APPROACH // 2.0×", 48, WHITE, "condensed")
        line_text(frame, "FAST FORWARD MOTION • TRACK LOCK MAINTAINED", (84, 220), 17, CYAN)
        distance = max(0.38, 8.7 * (1.0 - p) + 0.38)
        self.box(frame, (64, 760, 270, 110), "RANGE", f"{distance:.2f} m", True)
        self.box(frame, (355, 760, 270, 110), "SPEED", "2.0×", True)
        self.box(frame, (646, 760, 300, 110), "TARGET LOCK", "NEO1", True)
        return frame

    def scene_impact(self, local: float) -> np.ndarray:
        if local < 1.15:
            frame = self.camera_full(self.path_b.frame(32.0 + local * 2.0))
            center = (round(self.w * 0.61), round(self.h * 0.55))
            self.neo_sprite(frame, center, 2.6, local)
            expansion = smooth(local / 1.15)
            overlay = np.full_like(frame, (80, 205, 255))
            cv2.addWeighted(overlay, 0.22 + 0.66 * expansion, frame, 0.78 - 0.66 * expansion, 0, frame)
            for radius in (round(80 + 620 * expansion), round(40 + 430 * expansion)):
                cv2.circle(frame, center, radius, WHITE if radius < 500 else AMBER, max(3, round(18 * (1 - expansion))), cv2.LINE_AA)
            TYPE.draw(frame, center, "BOOM", round(80 + 80 * expansion), WHITE, "tech", anchor="mm", stroke=3, stroke_color=(15, 25, 40))
            line_text(frame, "CINEMATIC SIMULATION", (self.w // 2, round(self.h * 0.83)), 22, RED, "mm")
            return frame

        # Immediately transition into map reacquisition; there is no dead/black hold.
        frame = self.fit_capture(self.patrol_top, 42)
        alpha_rect(frame, (0, 0, self.w, self.h), (6, 10, 14), 0.36)
        phase = local - 1.15
        self.title(frame, "POST-INTERCEPT RECOVERY", "DJI RELOCALIZING...", "Global map match protects the mission from accumulated drift.", False, 85, 130)
        self.route(frame, (1040, 190, 1800, 700), min(1.0, phase / 2.1), 2)
        if phase > 1.0:
            self.drone_icon(frame, (1490, 470), 1.4, -0.5)
        self.scan(frame, phase, (1000, 150, 1840, 760), False)
        status = "GLOBAL MATCH" if phase < 1.45 else "TSOLVE CONFIRMED"
        self.box(frame, (85, 690, 340, 115), "RECOVERY", status)
        self.box(frame, (450, 690, 340, 115), "POSITION", "REACQUIRED" if phase > 1.45 else "SEARCHING")
        return frame

    def scene_resume(self, local: float) -> np.ndarray:
        frame = self.camera_full(self.path_a.frame(265.0 + local * 1.75))
        frame = cv2.convertScaleAbs(frame, alpha=0.92, beta=-5)
        self.route(frame, (self.w - 530, 95, self.w - 55, 425), min(1.0, local / 5.5), 2)
        alpha_rect(frame, (65, 665, 790, 905), (4, 8, 10), 0.82)
        line_text(frame, "POSE VERIFIED // COMMAND FLOW RESTORED", (92, 710), 21, GREEN)
        TYPE.draw(frame, (92, 765), "PATROL CONTINUES.", 58, WHITE, "condensed")
        line_text(frame, "RELOCALIZE  •  RESUME  •  COMPLETE THE MISSION", (92, 850), 18, CYAN)
        self.tracking_points(frame, local, (30, 60, self.w - 30, self.h - 55))
        return frame

    def scene_outro(self, local: float) -> np.ndarray:
        frame = self.base()
        # A subtle optical push gives the final card resolution rather than a
        # held-frame feel. The branded poster is the designed background, so
        # this does not crop any recorded UI or camera source.
        poster = cover(self.poster, self.w, self.h, 1.0 + 0.018 * smooth(local / 7.0))
        paste(frame, poster, 0, 0)
        alpha_rect(frame, (0, 0, self.w, self.h), (3, 6, 8), 0.28)
        if local > 0.3:
            TYPE.draw(frame, (self.w // 2, 320), "ATLAS", 130, WHITE, "tech", anchor="mm")
        half = round(310 * smooth((local - 0.5) / 1.5))
        cv2.line(frame, (self.w // 2 - half, 420), (self.w // 2 + half, 420), CYAN, 2, cv2.LINE_AA)
        if local > 1.2:
            TYPE.draw(frame, (self.w // 2, 545), "DETECT. LOCALIZE. RESPOND.\nRETURN TO PATROL.", 58, WHITE, "condensed", anchor="mm", spacing=8)
        if local > 2.5:
            TYPE.draw(frame, (self.w // 2, 710), "FROM INDOOR AUTONOMY TO OPERATIONAL AWARENESS.", 25, CYAN, "tech", anchor="mm")
        if local > 3.8:
            TYPE.draw(frame, (self.w // 2, 805), "RECORDED SYSTEM MATERIAL + CLEARLY LABELED CINEMATIC SIMULATION", 18, MUTED, "regular", anchor="mm")
        self.scan(frame, local, (500, 175, 1420, 865), False)
        return frame

    def render(self, t: float) -> np.ndarray:
        if t < 8:
            frame, scene, alert, sim = self.scene_threat_open(t), "threat detected", True, False
        elif t < 15:
            frame, scene, alert, sim = self.scene_title(t - 8), "mission brief", False, False
        elif t < 28:
            frame, scene, alert, sim = self.scene_system(t - 15), "build the world", False, False
        elif t < 43:
            frame, scene, alert, sim = self.scene_localize(t - 28), "live localization", False, False
        elif t < 60:
            frame, scene, alert, sim = self.scene_paths(t - 43), "path archive", False, False
        elif t < 75:
            frame, scene, alert, sim = self.scene_patrol(t - 60), "patrol execution", False, False
        elif t < 86:
            frame, scene, alert, sim = self.scene_neo(t - 75), "enemy identification", True, False
        elif t < 98:
            frame, scene, alert, sim = self.scene_intercept(t - 86), "guarded intercept", True, True
        elif t < 102:
            frame, scene, alert, sim = self.scene_impact(t - 98), "contact / recovery", True, True
        elif t < 113:
            frame, scene, alert, sim = self.scene_resume(t - 102), "patrol resumed", False, False
        else:
            frame, scene, alert, sim = self.scene_outro(t - 113), "mission complete", False, False

        self.hud(frame, t, scene, alert, sim)
        frame = np.clip(frame.astype(np.float32) * self.vignette, 0, 255).astype(np.uint8)
        for y in range(0, self.h, 6):
            frame[y : y + 1] = (frame[y : y + 1].astype(np.float32) * 0.96).astype(np.uint8)
        # Short dissolves only at structural boundaries; no long black holds.
        boundaries = (8, 15, 28, 43, 60, 75, 86, 98, 102, 113, 120)
        fade = min(1.0, t / 0.32, (DURATION - t) / 0.85)
        for boundary in boundaries[:-1]:
            distance = abs(t - boundary)
            if distance < 0.12:
                fade = min(fade, 0.78 + 0.22 * distance / 0.12)
        return (frame.astype(np.float32) * clamp(fade)).astype(np.uint8)


def make_impact_sfx(path: Path) -> None:
    rate = 48_000
    count = int(DURATION * rate)
    stereo = np.zeros((count, 2), np.float64)
    rng = np.random.default_rng(819)
    start = 98.08
    i0 = int(start * rate)
    length = int(2.2 * rate)
    local = np.arange(length) / rate
    boom = np.sin(math.tau * (54 * local - 14 * local * local)) * np.exp(-local * 2.1)
    crack = rng.standard_normal(length) * np.exp(-local * 9.0)
    signal = 0.78 * boom + 0.12 * crack
    stereo[i0 : i0 + length, 0] = signal
    stereo[i0 : i0 + length, 1] = np.roll(signal, 430) * 0.90
    pcm = np.clip(stereo * 32767, -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def render_silent() -> None:
    film = ThreatFilm()
    writer = cv2.VideoWriter(str(SILENT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer: {SILENT}")
    try:
        for index in range(round(DURATION * FPS)):
            if index % (FPS * 5) == 0:
                print(f"rendered {index / FPS:5.1f}/{DURATION:.1f}s", flush=True)
            writer.write(film.render(index / FPS))
    finally:
        writer.release()


def mux() -> None:
    if not MUSIC.exists():
        raise FileNotFoundError(MUSIC)
    make_impact_sfx(IMPACT)
    run(
        [
            "/opt/homebrew/bin/ffmpeg",
            "-y",
            "-i",
            str(SILENT),
            "-i",
            str(MUSIC),
            "-i",
            str(IMPACT),
            "-filter_complex",
            "[1:a]atrim=start=0:end=120,asetpts=PTS-STARTPTS,loudnorm=I=-16:LRA=11:TP=-1.5,afade=t=out:st=118:d=2[music];[2:a]volume=0.72[sfx];[music][sfx]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.94[a]",
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
    prepare_sources()
    render_silent()
    mux()
    print(FINAL)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the sponsor-facing ATLAS Hollywood mission cut.

Editorial contract:
- Never use an application screenshot as a background.
- Render the actual COLMAP scene as a clean rotating holographic room.
- Preserve chronological motion through approach, impact, recovery and patrol.
- Use the supplied score continuously; transitions are made by mixing, not restart.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

from build_atlas_story_demo import StoryFilm
from build_atlas_spy_demo import CYAN, CYAN_SOFT, GREEN, RED, TYPE, WHITE, VideoSource, alpha_rect, clamp, cover, smooth


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/spy_demo"
WORK = OUT / "work/hollywood_cut"
ATTACK = Path("/Users/yamromano/Downloads/2026-08-16-1-13-12-347\N{NARROW NO-BREAK SPACE}PM.MP4")
MUSIC = Path("/Users/yamromano/Downloads/41 Minutes of Spy Music - Instrumental Spy Themes.mp3")
SILENT = WORK / "atlas_hollywood_silent.mp4"
SFX = WORK / "atlas_hollywood_sfx.wav"
FINAL = OUT / "ATLAS_HOLLYWOOD_MISSION_CUT_2MIN.mp4"
WIDTH, HEIGHT, FPS, DURATION = 1920, 1080, 24, 120.0


class HologramRoom:
    """Fast orthographic renderer for the verified COLMAP scene points."""

    def __init__(self) -> None:
        payload = json.loads((ROOT / "viewer/public/scene.json").read_text(encoding="utf-8"))
        raw_points = payload["points3D"]
        values = list(raw_points.values()) if isinstance(raw_points, dict) else raw_points
        xyz = np.asarray([v["xyz"] for v in values], np.float32)
        rgb = np.asarray([v.get("rgb", [80, 180, 220]) for v in values], np.float32)
        lo, hi = np.percentile(xyz, [1.0, 99.0], axis=0)
        keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        xyz, rgb = xyz[keep], rgb[keep]
        xyz -= np.median(xyz, axis=0)
        scale = np.percentile(np.linalg.norm(xyz[:, [0, 2]], axis=1), 97)
        self.xyz = xyz / max(float(scale), 1e-6)
        lum = rgb.mean(axis=1) / 255.0
        self.brightness = np.clip(.25 + lum * .75, .25, 1.0)

    def render(self, seconds: float, *, angle_offset: float = 0.0) -> np.ndarray:
        # One unbroken 360 degree orbit in 18 seconds.
        theta = angle_offset + math.tau * (seconds / 18.0)
        c, s = math.cos(theta), math.sin(theta)
        p = self.xyz
        x = c * p[:, 0] - s * p[:, 2]
        depth = s * p[:, 0] + c * p[:, 2]
        y = p[:, 1]
        # Isometric elevation gives the scene a real architectural volume.
        sy = -.80 * y + .28 * depth
        sx = x
        px = np.rint(WIDTH * .50 + sx * WIDTH * .36).astype(np.int32)
        py = np.rint(HEIGHT * .51 + sy * HEIGHT * .37).astype(np.int32)
        valid = (px >= 40) & (px < WIDTH - 40) & (py >= 70) & (py < HEIGHT - 55)
        px, py, dep, bright = px[valid], py[valid], depth[valid], self.brightness[valid]
        order = np.argsort(dep)
        px, py, bright = px[order], py[order], bright[order]

        frame = np.full((HEIGHT, WIDTH, 3), (9, 6, 3), np.uint8)
        for gx in range(0, WIDTH, 96): cv2.line(frame, (gx, 0), (gx, HEIGHT), (18, 17, 13), 1)
        for gy in range(0, HEIGHT, 96): cv2.line(frame, (0, gy), (WIDTH, gy), (18, 17, 13), 1)
        glow = np.zeros_like(frame)
        glow[py, px] = np.stack([bright * 238, bright * 170, bright * 42], axis=1).astype(np.uint8)
        glow = cv2.GaussianBlur(glow, (0, 0), 7.0)
        frame = cv2.addWeighted(frame, 1.0, glow, .72, 0)
        colors = np.stack([bright * 255, bright * 210, bright * 72], axis=1).astype(np.uint8)
        frame[py, px] = colors
        # Architectural axis rings, deliberately subtle.
        center = (WIDTH // 2, round(HEIGHT * .76))
        cv2.ellipse(frame, center, (520, 116), 0, 0, 360, CYAN_SOFT, 1, cv2.LINE_AA)
        cv2.ellipse(frame, center, (380, 84), 0, 0, 360, (70, 105, 112), 1, cv2.LINE_AA)
        return frame


class HollywoodFilm(StoryFilm):
    def __init__(self) -> None:
        super().__init__()
        if not ATTACK.is_file():
            raise FileNotFoundError(ATTACK)
        self.attack = VideoSource(ATTACK)
        self.hologram = HologramRoom()

    def room(self, local: float, title: str, sub: str) -> np.ndarray:
        frame = self.hologram.render(local)
        alpha_rect(frame, (50, 104, 875, 378), (2, 6, 9), .72)
        cv2.line(frame, (82, 145), (282, 145), CYAN, 4, cv2.LINE_AA)
        TYPE.draw(frame, (82, 181), "SPATIAL INTELLIGENCE", 18, CYAN, "tech")
        TYPE.draw(frame, (82, 224), title, 58, WHITE, "condensed", spacing=4, stroke=1, stroke_color=(0, 0, 0))
        TYPE.draw(frame, (82, 318), sub, 21, (184, 205, 208), "regular")
        self.cinematic_bar(frame, "COLMAP ENVIRONMENT")
        self.scan(frame, local)
        return frame

    def attack_frame(self, source_seconds: float) -> np.ndarray:
        frame = cover(self.attack.frame(source_seconds), self.w, self.h)
        return cv2.convertScaleAbs(frame, alpha=1.04, beta=-5)

    def scene_approach(self, local: float) -> np.ndarray:
        # The real forward run: 39.0 -> 47.45 seconds, accelerated to impact.
        p = clamp(local / 13.0)
        source = 39.0 + 8.45 * (p ** .82)
        frame = self.attack_frame(source)
        self.cinematic_bar(frame, "AUTONOMOUS RESPONSE")
        self.motion_lines(frame, local, .55 + 1.45 * p)
        alpha_rect(frame, (52, 742, 780, 930), (3, 7, 10), .74)
        TYPE.draw(frame, (82, 790), "TARGET VECTOR LOCKED", 19, RED, "tech")
        TYPE.draw(frame, (82, 845), "CLOSING DISTANCE", 48, WHITE, "condensed")
        cv2.line(frame, (82, 892), (82 + round(620 * p), 892), RED, 7, cv2.LINE_AA)
        return frame

    def scene_impact(self, local: float) -> np.ndarray:
        frame = self.attack_frame(47.45)
        if local < .11:
            amount = 1.0 - local / .11
            frame = cv2.addWeighted(frame, 1.0 - amount * .25, np.full_like(frame, 255), amount, 0)
            return frame
        if local < .68:
            # A short signal collapse, not a long dead screen.
            frame[:] = (2, 3, 4)
            rng = np.random.default_rng(round(local * 1000))
            for _ in range(18):
                y = int(rng.integers(80, self.h - 80)); h = int(rng.integers(1, 8))
                frame[y:y+h] = rng.integers(10, 55)
            TYPE.draw(frame, (self.w // 2, self.h // 2), "SIGNAL INTERRUPTED", 29, (116, 139, 143), "tech", anchor="mm")
            return frame
        # First chronological frame after closest approach.
        frame = self.attack_frame(47.45 + (local - .68) * 1.8)
        alpha_rect(frame, (0, 0, self.w, self.h), (2, 6, 9), max(0.0, .45 - (local - .68) * .32))
        return frame

    def scene_recovery_flight(self, local: float) -> np.ndarray:
        # Continue chronologically through the real lowering/rise and return.
        p = clamp(local / 14.0)
        source = 48.0 + (88.0 - 48.0) * (p ** .90)
        frame = self.attack_frame(source)
        status = "POSITION REACQUIRED" if local > 3.0 else "RELOCALIZING"
        color = GREEN if local > 3.0 else CYAN
        self.cinematic_bar(frame, "MISSION RECOVERY")
        cv2.circle(frame, (self.w - 66, self.h - 70), 7, color, -1, cv2.LINE_AA)
        TYPE.draw(frame, (self.w - 88, self.h - 70), status, 16, color, "tech", anchor="rm")
        if local > 5.0:
            alpha_rect(frame, (54, 754, 700, 920), (3, 7, 10), .70)
            TYPE.draw(frame, (84, 801), "AUTONOMY RESTORED", 18, GREEN, "tech")
            TYPE.draw(frame, (84, 856), "PATROL CONTINUES.", 47, WHITE, "condensed")
        return frame

    def render(self, t: float) -> np.ndarray:
        # One story: intrusion -> room intelligence -> patrol -> response -> recovery.
        if t < 8: frame = self.scene_intrusion(t)
        elif t < 15: frame = self.scene_lock(t - 8)
        elif t < 27: frame = self.room(t - 15, "THE ROOM BECOMES THE MAP.", "A live spatial model, built before the mission.")
        elif t < 36: frame = self.scene_launch(t - 27)
        elif t < 47: frame = self.scene_patrol(t - 36, 126.0, "AUTONOMOUS PATROL")
        elif t < 58: frame = self.scene_patrol(t - 47, 302.0, "CONTINUOUS LOCALIZATION")
        elif t < 66: frame = self.room(t - 58, "EVERY MOVE. REGISTERED.", "The mission stays anchored inside the mapped room.")
        elif t < 73: frame = self.scene_detect(t - 66)
        elif t < 86: frame = self.scene_approach(t - 73)
        elif t < 88: frame = self.scene_impact(t - 86)
        elif t < 102: frame = self.scene_recovery_flight(t - 88)
        elif t < 112: frame = self.room(t - 102, "THE MISSION SURVIVES.", "Recovered. Re-anchored. Back on patrol.")
        else: frame = self.scene_finale(t - 112)

        frame = np.clip(frame.astype(np.float32) * self.story_vignette, 0, 255).astype(np.uint8)
        for y in range(0, self.h, 8): frame[y:y+1] = (frame[y:y+1].astype(np.float32) * .97).astype(np.uint8)
        return frame


def make_sfx(path: Path) -> None:
    rate = 48_000
    audio = np.zeros((round(DURATION * rate), 2), np.float64)
    rng = np.random.default_rng(1608)

    def put(start: float, signal: np.ndarray, pan: float = 0.0) -> None:
        i = round(start * rate); n = min(len(signal), len(audio) - i)
        if n <= 0: return
        audio[i:i+n, 0] += signal[:n] * (1.0 - max(0.0, pan))
        audio[i:i+n, 1] += signal[:n] * (1.0 + min(0.0, pan))

    # Restrained transition sweeps.
    for when in (8, 15, 27, 36, 47, 58, 66, 73, 88, 102, 112):
        x = np.arange(round(.42 * rate)) / rate
        env = np.sin(np.pi * x / .42) ** 2
        whoosh = (rng.standard_normal(len(x)) - np.roll(rng.standard_normal(len(x)), 21)) * env * .045
        put(max(0, when - .17), whoosh)

    # A layered cinematic impact: crack, body, sub and decaying debris.
    x = np.arange(round(3.2 * rate)) / rate
    sub = np.sin(math.tau * (66 * x - 12.5 * x * x)) * np.exp(-x * 1.65)
    body = np.sin(math.tau * (112 * x - 22 * x * x)) * np.exp(-x * 3.4)
    crack = rng.standard_normal(len(x)) * np.exp(-x * 17.0)
    debris = (rng.standard_normal(len(x)) - np.roll(rng.standard_normal(len(x)), 19)) * np.exp(-x * 5.0)
    put(86.02, .58 * sub + .25 * body + .13 * crack + .055 * debris)

    # Reboot/reacquisition tones after blackout.
    for when, freq in ((86.76, 620), (87.18, 880), (90.9, 1180)):
        x = np.arange(round(.18 * rate)) / rate
        tone = np.sin(math.tau * freq * x) * np.sin(np.pi * x / .18) ** 2 * .11
        put(when, tone)

    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2); out.setsampwidth(2); out.setframerate(rate); out.writeframes(pcm.tobytes())


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if "--mux-only" not in sys.argv:
        film = HollywoodFilm()
        writer = cv2.VideoWriter(str(SILENT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
        if not writer.isOpened(): raise RuntimeError(SILENT)
        try:
            for index in range(round(DURATION * FPS)):
                if index % (FPS * 5) == 0: print(f"rendered {index/FPS:5.1f}/{DURATION:.1f}s", flush=True)
                writer.write(film.render(index / FPS))
        finally:
            writer.release()
    elif not SILENT.is_file():
        raise FileNotFoundError(SILENT)
    make_sfx(SFX)
    # The score is continuous. The opening is filtered/quiet, then opens up
    # gradually; the impact ducks it without restarting or changing tracks.
    af = (
        "[1:a]atrim=0:120,asetpts=PTS-STARTPTS,loudnorm=I=-16:LRA=10:TP=-1.5,asplit=3[m0][m1][m2];"
        "[m0]atrim=0:15,lowpass=f=1150,volume='0.32+0.038*t':eval=frame,afade=t=in:st=0:d=1.2[a0];"
        "[m1]atrim=15:86,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=1.2[a1];"
        "[m2]atrim=86:120,asetpts=PTS-STARTPTS,volume='if(lt(t,1.1),0.32,if(lt(t,4),0.32+(t-1.1)*0.234,1))':eval=frame,afade=t=out:st=32:d=2[a2];"
        "[a0][a1][a2]concat=n=3:v=0:a=1[music];[2:a]volume=.92[sfx];"
        "[music][sfx]amix=inputs=2:duration=first:normalize=0,alimiter=limit=.94,volume=.80[a]"
    )
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y", "-i", str(SILENT), "-i", str(MUSIC), "-i", str(SFX),
        "-filter_complex", af, "-map", "0:v:0", "-map", "[a]", "-c:v", "libx264", "-preset", "slow",
        "-crf", "17", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1", "-c:a", "aac",
        "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", "-t", "120", str(FINAL)
    ], check=True)
    print(FINAL)


if __name__ == "__main__":
    main()

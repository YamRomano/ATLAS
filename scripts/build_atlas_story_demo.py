#!/usr/bin/env python3
"""Render a coherent, product-led ATLAS action demo.

This is one continuous mission story, not a technical montage. It uses only
moving recorded patrol/mesh/NEO material, keeps text minimal, and never uses an
application screenshot as a full-frame background.
"""

from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

from build_atlas_action_game_cut import ActionFilm
from build_atlas_spy_demo import CYAN, CYAN_SOFT, GREEN, RED, TYPE, WHITE, alpha_rect, clamp, cover, smooth


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/spy_demo"
WORK = OUT / "work/story_demo"
MUSIC = Path("/Users/yamromano/Downloads/41 Minutes of Spy Music - Instrumental Spy Themes.mp3")
SILENT = WORK / "atlas_story_demo_silent.mp4"
SFX = WORK / "atlas_story_demo_sfx.wav"
FINAL = OUT / "ATLAS_CINEMATIC_PRODUCT_STORY_2MIN.mp4"
WIDTH, HEIGHT, FPS, DURATION = 1920, 1080, 24, 120.0


class StoryFilm(ActionFilm):
    def __init__(self) -> None:
        super().__init__()
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        radius = ((xx - self.w / 2) / (self.w * .68)) ** 2 + ((yy - self.h / 2) / (self.h * .70)) ** 2
        self.story_vignette = np.clip(1.04 - radius * .26, .70, 1.0)[..., None]

    def camera(self, seconds: float, *, speed_source: str = "b", zoom: float = 1.0) -> np.ndarray:
        source = self.path_b if speed_source == "b" else self.path_a
        frame = cover(source.frame(seconds), self.w, self.h, zoom=zoom)
        return cv2.convertScaleAbs(frame, alpha=.96, beta=-4)

    def cinematic_bar(self, frame: np.ndarray, label: str, *, alert: bool = False) -> None:
        accent = RED if alert else CYAN
        alpha_rect(frame, (0, 0, self.w, 54), (2, 6, 9), .82)
        cv2.line(frame, (0, 54), (self.w, 54), accent, 1, cv2.LINE_AA)
        TYPE.draw(frame, (42, 27), "ATLAS", 19, WHITE, "tech", anchor="lm")
        TYPE.draw(frame, (self.w - 42, 27), label.upper(), 15, accent, "tech", anchor="rm")

    def story_title(self, frame: np.ndarray, kicker: str, title: str, *, alert: bool = False, lower: bool = False) -> None:
        accent = RED if alert else CYAN
        x = 72
        y = 720 if lower else 120
        height = 205 if "\n" not in title else 270
        alpha_rect(frame, (44, y - 28, 760, y + height), (2, 6, 9), .82)
        cv2.line(frame, (x, y), (x + 160, y), accent, 4, cv2.LINE_AA)
        TYPE.draw(frame, (x, y + 22), kicker.upper(), 20, accent, "tech")
        TYPE.draw(frame, (x, y + 66), title.upper(), 58, WHITE, "condensed", spacing=4, stroke=1, stroke_color=(0, 0, 0))

    def motion_lines(self, frame: np.ndarray, local: float, strength: float = 1.0) -> None:
        rng = np.random.default_rng(441)
        center = np.array([self.w * .52, self.h * .52])
        for i in range(round(24 * strength)):
            side = -1 if i % 2 == 0 else 1
            start = np.array([0 if side < 0 else self.w, rng.uniform(80, self.h - 65)])
            end = start + (center - start) * (.13 + .10 * math.sin(local * 2.1 + i) ** 2)
            cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), CYAN_SOFT, 1, cv2.LINE_AA)

    def scene_intrusion(self, local: float) -> np.ndarray:
        seconds = .6 + local * 1.45
        frame, box = self.neo_feed(self.neo_a, "neo_a", seconds, label="")
        self.target_box(frame, box, confidence=88.0 + 5.5 * smooth(local / 8.0))
        self.story_title(frame, "PROXIMITY ALERT", "SOMETHING\nENTERED THE ROOM.", alert=True)
        self.cinematic_bar(frame, "UNIDENTIFIED AIRCRAFT", alert=True)
        return frame

    def scene_lock(self, local: float) -> np.ndarray:
        seconds = 13.6 + local * 1.55
        frame, box = self.neo_feed(self.neo_a, "neo_a", seconds, label="")
        self.target_box(frame, box, confidence=94.7)
        self.story_title(frame, "VISUAL ID", "NEO1 LOCKED.", alert=True, lower=True)
        self.cinematic_bar(frame, "TARGET CONFIRMED", alert=True)
        return frame

    def scene_wake(self, local: float) -> np.ndarray:
        frame = self.mesh_frame(.8 + local * 1.0, dark=.14, zoom=1.025 + .012 * smooth(local / 7))
        self.story_title(frame, "AUTONOMOUS RESPONSE", "ATLAS IS ONLINE.")
        self.cinematic_bar(frame, "MISSION START")
        self.scan(frame, local)
        return frame

    def scene_launch(self, local: float) -> np.ndarray:
        frame = self.camera(2.0 + local * 2.25, speed_source="b")
        self.story_title(frame, "LIVE PATROL", "MOVE.", lower=True)
        self.cinematic_bar(frame, "PATROL ONLINE")
        self.motion_lines(frame, local, .45)
        return frame

    def scene_patrol(self, local: float, source_start: float, title: str) -> np.ndarray:
        frame = self.camera(source_start + local * 2.15, speed_source="b")
        self.cinematic_bar(frame, title)
        # Minimal live-localization marks keep the product identity visible
        # without turning the footage into a specifications screen.
        self.tracking_points(frame, local, 20)
        cv2.circle(frame, (self.w - 68, self.h - 68), 6, GREEN, -1, cv2.LINE_AA)
        TYPE.draw(frame, (self.w - 88, self.h - 68), "POSITION CONFIRMED", 14, GREEN, "tech", anchor="rm")
        return frame

    def scene_alignment(self, local: float) -> np.ndarray:
        frame = self.mesh_frame(22.0 + local * 1.05, dark=.10, zoom=1.035, pan_x=.05)
        self.story_title(frame, "LIVE POSITION", "THE ROOM\nBECOMES THE MAP.", lower=True)
        self.cinematic_bar(frame, "VISUAL ALIGNMENT")
        self.scan(frame, local)
        return frame

    def scene_detect(self, local: float) -> np.ndarray:
        seconds = .4 + local * 1.65
        frame, box = self.neo_b and self.neo_feed(self.neo_b, "neo_b", seconds, label="")
        self.target_box(frame, box, confidence=94.7)
        self.story_title(frame, "THREAT IN VIEW", "ATLAS HAS THE TARGET.", alert=True)
        self.cinematic_bar(frame, "NEO1 DETECTED", alert=True)
        return frame

    def scene_pursuit(self, local: float) -> np.ndarray:
        seconds = 14.2 + local * 2.05
        frame, box = self.neo_feed(self.neo_b, "neo_b", seconds, label="")
        self.target_box(frame, box, confidence=94.7)
        self.story_title(frame, "RESPONSE VECTOR", "CLOSING ON NEO1", alert=True, lower=True)
        self.cinematic_bar(frame, "CLOSING DISTANCE", alert=True)
        self.motion_lines(frame, local, 1.0)
        return frame

    def scene_contact(self, local: float) -> np.ndarray:
        seconds = 38.8 + min(local, 1.3) * 2.0
        frame, box = self.neo_feed(self.neo_b, "neo_b", seconds, label="")
        self.target_box(frame, box, confidence=94.7)
        p = smooth(local / 4.0)
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        overlay = np.full_like(frame, (75, 186, 255))
        cv2.addWeighted(overlay, .08 + .37 * p, frame, .92 - .37 * p, 0, frame)
        for radius in (round(80 + 610 * p), round(35 + 360 * p)):
            cv2.circle(frame, (cx, cy), radius, WHITE, max(2, round(11 * (1 - p))), cv2.LINE_AA)
        TYPE.draw(frame, (self.w // 2, self.h // 2), "CONTACT", 92, WHITE, "tech", anchor="mm", stroke=3, stroke_color=(16, 22, 30))
        TYPE.draw(frame, (self.w - 40, self.h - 42), "TARGET INTERCEPT", 13, RED, "tech", anchor="rm")
        self.cinematic_bar(frame, "CONTACT", alert=True)
        return frame

    def scene_recover(self, local: float) -> np.ndarray:
        frame = self.mesh_frame(61.0 + local * 1.05, dark=.15, zoom=1.04, pan_x=-.04)
        phrase = "SEARCHING..." if local < 2.2 else "POSITION REACQUIRED."
        self.story_title(frame, "GLOBAL RECOVERY", phrase)
        self.cinematic_bar(frame, "RELOCALIZING")
        color = CYAN if local < 2.2 else GREEN
        cv2.circle(frame, (self.w - 62, self.h - 62), 7, color, -1, cv2.LINE_AA)
        self.scan(frame, local, color)
        return frame

    def scene_resume(self, local: float) -> np.ndarray:
        frame = self.camera(722.0 + local * 2.2, speed_source="b")
        self.story_title(frame, "MISSION RESTORED", "PATROL CONTINUES.", lower=True)
        self.cinematic_bar(frame, "AUTONOMY RESUMED")
        self.tracking_points(frame, local + 20, 18)
        return frame

    def scene_finale(self, local: float) -> np.ndarray:
        # Keep the closing product image alive. The supplied recording becomes
        # nearly static at its tail, so use an unused moving mesh interval.
        frame = self.mesh_frame(31.0 + local * .75, dark=.30, zoom=1.05 + .015 * smooth(local / 6))
        alpha_rect(frame, (0, 0, self.w, self.h), (2, 6, 9), .12)
        TYPE.draw(frame, (self.w // 2, 365), "ATLAS", 144, WHITE, "tech", anchor="mm", stroke=2, stroke_color=(0, 0, 0))
        cv2.line(frame, (620, 490), (1300, 490), CYAN, 2, cv2.LINE_AA)
        TYPE.draw(frame, (self.w // 2, 605), "DETECT. LOCALIZE. RESPOND.", 56, WHITE, "condensed", anchor="mm", stroke=1, stroke_color=(0, 0, 0))
        TYPE.draw(frame, (self.w // 2, 705), "AUTONOMY INSIDE THE ROOM", 22, CYAN, "tech", anchor="mm")
        self.cinematic_bar(frame, "MISSION CONTINUES")
        return frame

    def render(self, t: float) -> np.ndarray:
        if t < 8: frame = self.scene_intrusion(t)
        elif t < 16: frame = self.scene_lock(t - 8)
        elif t < 23: frame = self.scene_wake(t - 16)
        elif t < 31: frame = self.scene_launch(t - 23)
        elif t < 39: frame = self.scene_patrol(t - 31, 126.0, "ROOM SECURED")
        elif t < 47: frame = self.scene_alignment(t - 39)
        elif t < 55: frame = self.scene_patrol(t - 47, 274.0, "PATROL CONTINUES")
        elif t < 63: frame = self.scene_patrol(t - 55, 418.0, "NEW SECTOR")
        elif t < 72: frame = self.scene_detect(t - 63)
        elif t < 84: frame = self.scene_pursuit(t - 72)
        elif t < 88: frame = self.scene_contact(t - 84)
        elif t < 97: frame = self.scene_recover(t - 88)
        elif t < 110: frame = self.scene_resume(t - 97)
        else: frame = self.scene_finale(t - 110)

        frame = np.clip(frame.astype(np.float32) * self.story_vignette, 0, 255).astype(np.uint8)
        for y in range(0, self.h, 8):
            frame[y:y + 1] = (frame[y:y + 1].astype(np.float32) * .97).astype(np.uint8)
        return frame


def make_sfx(path: Path) -> None:
    rate = 48_000
    audio = np.zeros((round(DURATION * rate), 2), np.float64)
    rng = np.random.default_rng(120)
    for time_sec in (8, 16, 23, 31, 39, 47, 55, 63, 72, 84, 88, 97, 110):
        i0 = round((time_sec - .18) * rate)
        length = round(.5 * rate)
        x = np.arange(length) / rate
        env = np.sin(np.pi * x / .5) ** 2
        noise = (rng.standard_normal(length) - np.roll(rng.standard_normal(length), 11)) * env * .08
        audio[i0:i0 + length, 0] += noise
        audio[i0:i0 + length, 1] += np.roll(noise, 180) * .9
    i0 = round(84.05 * rate)
    length = round(2.0 * rate)
    x = np.arange(length) / rate
    hit = .58 * np.sin(math.tau * (57 * x - 12 * x * x)) * np.exp(-x * 2.0) + .08 * rng.standard_normal(length) * np.exp(-x * 9)
    audio[i0:i0 + length, 0] += hit
    audio[i0:i0 + length, 1] += np.roll(hit, 360) * .9
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2); out.setsampwidth(2); out.setframerate(rate); out.writeframes(pcm.tobytes())


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    film = StoryFilm()
    writer = cv2.VideoWriter(str(SILENT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened(): raise RuntimeError(f"Cannot create {SILENT}")
    try:
        for index in range(round(DURATION * FPS)):
            if index % (FPS * 5) == 0: print(f"rendered {index / FPS:5.1f}/{DURATION:.1f}s", flush=True)
            writer.write(film.render(index / FPS))
    finally:
        writer.release()
    make_sfx(SFX)
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y", "-i", str(SILENT), "-i", str(MUSIC), "-i", str(SFX),
        "-filter_complex",
        "[1:a]atrim=start=0:end=120,asetpts=PTS-STARTPTS,loudnorm=I=-16:LRA=10:TP=-1.5,afade=t=out:st=118:d=2[m];"
        "[2:a]volume=.95[s];[m][s]amix=inputs=2:duration=first:normalize=0,alimiter=limit=.94[a]",
        "-map", "0:v:0", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1", "-c:a", "aac", "-b:a", "256k",
        "-ar", "48000", "-movflags", "+faststart", "-t", "120", str(FINAL)
    ], check=True)
    print(FINAL)


if __name__ == "__main__":
    main()

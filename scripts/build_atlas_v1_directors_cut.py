#!/usr/bin/env python3
"""Build the 120-second ATLAS director's cut from the locked V1 picture.

The visual edit deliberately uses *only* frames from the original V1 film.  It
opens on a short threat teaser, then returns to the original narrative order.
The soundtrack is an original, synthesized spy-action score with distinct acts
rather than a repeating clock-like loop.
"""

from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "spy_demo"
SOURCE = OUT / "ATLAS_CINEMATIC_MISSION_DEMO.mp4"
SILENT = OUT / "work" / "atlas_v1_directors_cut_120s_silent.mp4"
SCORE = OUT / "audio" / "atlas_directors_cut_spy_action_score.wav"
FINAL = OUT / "ATLAS_CINEMATIC_MISSION_DEMO_DIRECTORS_CUT_2MIN.mp4"

FPS = 24
DURATION = 120.0
SAMPLE_RATE = 48_000


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def build_picture() -> None:
    """Time-remap the V1 film into a two-minute dramatic arc.

    Source intervals are the original V1 scene boundaries.  The first eight
    seconds are a cold-open glimpse of the threat.  After that, every V1 scene
    plays in its original order.  Setup scenes are given more room; the patrol
    and confrontation retain comparatively brisk pacing.
    """

    # (source start, source end, director-cut duration)
    edit = [
        (55.0, 63.0, 8.0),   # cold open: unknown-aircraft glimpse
        (0.0, 7.0, 10.0),    # title / mission statement
        (7.0, 15.0, 14.0),   # build the world
        (15.0, 27.0, 18.0),  # live localization
        (27.0, 40.0, 18.0),  # fleet monitor
        (40.0, 51.0, 16.0),  # two-lap patrol
        (51.0, 63.0, 12.0),  # full detection sequence
        (63.0, 71.0, 12.0),  # guarded intercept
        (71.0, 79.0, 12.0),  # resolution / ATLAS brand
    ]

    chains: list[str] = []
    labels: list[str] = []
    for index, (start, end, target) in enumerate(edit):
        source_length = end - start
        ratio = target / source_length
        label = f"v{index}"
        # setpts preserves the V1 imagery exactly; fps only regularizes the
        # output clock and does not invent new graphic content.
        chains.append(
            f"[0:v]trim=start={start}:end={end},"
            f"setpts=(PTS-STARTPTS)*{ratio:.9f},fps={FPS}[{label}]"
        )
        labels.append(f"[{label}]")
    chains.append("".join(labels) + f"concat=n={len(edit)}:v=1:a=0,format=yuv420p[v]")
    filter_graph = ";".join(chains)

    SILENT.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "/opt/homebrew/bin/ffmpeg",
            "-y",
            "-i",
            str(SOURCE),
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-an",
            "-t",
            f"{DURATION:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            str(SILENT),
        ]
    )


class Score:
    """Small deterministic orchestral/electronic scoring engine."""

    def __init__(self, duration: float, rate: int) -> None:
        self.duration = duration
        self.rate = rate
        self.count = int(duration * rate)
        self.audio = np.zeros((self.count, 2), dtype=np.float64)
        self.rng = np.random.default_rng(731_042)

    @staticmethod
    def hz(midi: float) -> float:
        return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)

    def window(self, start: float, length: float, attack: float, release: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i0 = max(0, int(start * self.rate))
        i1 = min(self.count, int((start + length) * self.rate))
        idx = np.arange(i0, i1)
        local = idx / self.rate - start
        if len(idx) == 0:
            return idx, local, local
        env = np.minimum(1.0, local / max(attack, 1e-5))
        env *= np.minimum(1.0, (length - local) / max(release, 1e-5))
        return idx, local, np.clip(env, 0.0, 1.0)

    def add(self, idx: np.ndarray, signal: np.ndarray, pan: float = 0.0) -> None:
        if len(idx) == 0:
            return
        angle = (np.clip(pan, -1.0, 1.0) + 1.0) * math.pi / 4.0
        self.audio[idx, 0] += signal * math.cos(angle)
        self.audio[idx, 1] += signal * math.sin(angle)

    def tone(
        self,
        start: float,
        length: float,
        midi: float,
        gain: float,
        timbre: str = "sine",
        attack: float = 0.02,
        release: float = 0.15,
        pan: float = 0.0,
        vibrato: float = 0.0,
    ) -> None:
        idx, local, env = self.window(start, length, attack, release)
        freq = self.hz(midi)
        phase = math.tau * freq * local + vibrato * np.sin(math.tau * 5.1 * local)
        if timbre == "strings":
            signal = (
                np.sin(phase)
                + 0.42 * np.sin(2.0 * phase + 0.12)
                + 0.19 * np.sin(3.0 * phase + 0.31)
                + 0.08 * np.sin(5.0 * phase)
            ) / 1.69
        elif timbre == "brass":
            signal = (
                np.sin(phase)
                + 0.58 * np.sin(2.0 * phase)
                + 0.32 * np.sin(3.0 * phase)
                + 0.16 * np.sin(4.0 * phase)
            ) / 2.06
        elif timbre == "pulse":
            signal = np.tanh(2.4 * (np.sin(phase) + 0.22 * np.sin(2.0 * phase)))
        elif timbre == "glass":
            signal = np.sin(phase) + 0.26 * np.sin(2.71 * phase) + 0.12 * np.sin(4.37 * phase)
        else:
            signal = np.sin(phase)
        self.add(idx, gain * env * signal, pan)

    def pad(self, start: float, length: float, notes: tuple[int, ...], gain: float, rise: float = 1.2) -> None:
        spread = np.linspace(-0.65, 0.65, len(notes))
        for note, pan in zip(notes, spread):
            self.tone(start, length, note, gain / len(notes), "strings", rise, 1.4, float(pan), 0.035)
            self.tone(start, length, note - 12, gain * 0.33 / len(notes), "sine", rise, 1.7, float(-pan))

    def boom(self, start: float, gain: float = 0.55, length: float = 1.7) -> None:
        idx, local, env = self.window(start, length, 0.003, length * 0.82)
        freq_phase = math.tau * (52.0 * local - 13.5 * local * local)
        low = np.sin(freq_phase) * np.exp(-local * 2.2)
        grit = self.rng.standard_normal(len(idx)) * np.exp(-local * 6.0)
        self.add(idx, gain * env * (low + 0.075 * grit), 0.0)

    def tom(self, start: float, gain: float = 0.28, pan: float = 0.0) -> None:
        idx, local, env = self.window(start, 0.62, 0.002, 0.54)
        signal = np.sin(math.tau * (82.0 * local - 25.0 * local * local)) * np.exp(-local * 6.2)
        self.add(idx, gain * env * signal, pan)

    def snare(self, start: float, gain: float = 0.13, pan: float = 0.0) -> None:
        idx, local, env = self.window(start, 0.34, 0.002, 0.30)
        noise = self.rng.standard_normal(len(idx))
        smooth_noise = noise - np.convolve(noise, np.ones(18) / 18.0, mode="same")
        body = np.sin(math.tau * 178.0 * local) * np.exp(-local * 11.0)
        self.add(idx, gain * env * (0.72 * smooth_noise + 0.35 * body), pan)

    def riser(self, start: float, length: float, gain: float = 0.10) -> None:
        idx, local, env = self.window(start, length, 0.5, 0.08)
        frac = np.clip(local / length, 0.0, 1.0)
        phase = math.tau * (75.0 * local + 450.0 * local * frac**2)
        noise = self.rng.standard_normal(len(idx))
        signal = (0.56 * np.sin(phase) + 0.25 * noise) * frac**1.7
        self.add(idx, gain * env * signal, 0.25)

    def radar(self, start: float, midi: int = 83, pan: float = 0.0) -> None:
        self.tone(start, 0.75, midi, 0.060, "glass", 0.006, 0.70, pan)
        self.tone(start + 0.08, 0.48, midi + 7, 0.024, "sine", 0.008, 0.42, -pan)

    def ostinato(self, start: float, stop: float, bpm: float, pattern: tuple[int | None, ...], gain: float, octave: int = 0) -> None:
        step = 60.0 / bpm / 2.0
        cursor = start
        count = 0
        while cursor < stop:
            note = pattern[count % len(pattern)]
            if note is not None:
                pan = -0.48 if count % 2 == 0 else 0.48
                self.tone(cursor, step * 0.82, note + octave, gain, "strings", 0.018, step * 0.34, pan, 0.025)
            count += 1
            cursor += step

    def drums(self, start: float, stop: float, bpm: float, intensity: float, variant: int = 0) -> None:
        beat = 60.0 / bpm
        cursor = start
        bar_beat = 0
        while cursor < stop:
            # Broad, syncopated action pattern—not a clock on every beat.
            if bar_beat % 8 in ((0, 3, 6) if variant == 0 else (0, 2, 5, 7)):
                self.tom(cursor, 0.25 * intensity, -0.25 if bar_beat % 2 else 0.22)
            if bar_beat % 8 in (4, 7):
                self.snare(cursor, 0.095 * intensity, 0.18)
            if intensity > 1.05 and bar_beat % 8 == 7:
                self.tom(cursor + beat * 0.50, 0.19 * intensity, 0.45)
                self.tom(cursor + beat * 0.74, 0.17 * intensity, -0.45)
            bar_beat += 1
            cursor += beat

    def compose(self) -> None:
        # ACT I — danger before the title (0–8): one sharp clue, then silence.
        self.pad(0.0, 8.0, (38, 45, 50, 53), 0.25, 0.55)
        self.boom(0.15, 0.70)
        self.riser(3.6, 4.1, 0.12)
        for start, note, pan in ((0.9, 86, -0.6), (2.2, 82, 0.5), (5.8, 89, -0.1)):
            self.radar(start, note, pan)
        self.tone(6.15, 1.65, 50, 0.12, "brass", 0.09, 0.65)

        # ACT II — title and reconstruction (8–32): spacious espionage theme.
        self.pad(8.0, 14.0, (38, 45, 50, 53), 0.20, 1.6)
        self.pad(22.0, 10.0, (36, 43, 48, 52), 0.22, 1.0)
        for start, note in ((10.0, 74), (12.4, 77), (15.1, 81), (18.5, 79), (23.0, 72), (26.2, 76), (29.1, 79)):
            self.tone(start, 1.9, note, 0.073, "glass", 0.08, 0.85, -0.35 if int(start) % 2 else 0.35)
        self.boom(8.0, 0.52)
        self.boom(18.0, 0.43)
        self.boom(22.0, 0.47)
        for start in (24.1, 27.6, 30.4):
            self.radar(start, 85, 0.45)

        # ACT III — localization and fleet activation (32–68): momentum arrives.
        self.pad(32.0, 18.0, (38, 45, 50, 53), 0.23, 0.9)
        self.pad(50.0, 18.0, (41, 48, 53, 57), 0.25, 0.8)
        self.ostinato(33.0, 50.0, 96.0, (62, None, 65, 69, 62, 67, None, 65), 0.042)
        self.ostinato(50.0, 68.0, 104.0, (65, 69, 72, None, 65, 70, 69, 72), 0.050)
        self.drums(38.0, 50.0, 96.0, 0.65)
        self.drums(50.0, 68.0, 104.0, 0.82, 1)
        for start in (32.0, 39.5, 50.0, 59.0):
            self.boom(start, 0.48 if start < 50 else 0.56)
        for start, note in ((34.0, 74), (36.0, 77), (42.3, 81), (47.0, 84), (52.0, 77), (55.0, 81), (61.0, 84), (65.0, 86)):
            self.tone(start, 1.3, note, 0.060, "strings", 0.04, 0.55, 0.35 if int(start) % 2 else -0.35)

        # ACT IV — patrol (68–84): confident forward propulsion.
        self.pad(68.0, 16.0, (38, 45, 50, 53), 0.27, 0.5)
        self.ostinato(68.0, 84.0, 116.0, (62, 65, 69, 70, 69, 65, 67, 70), 0.063)
        self.drums(68.0, 84.0, 116.0, 1.05, 1)
        for start in (68.0, 72.0, 76.0, 80.0):
            self.boom(start, 0.55)
        for start, note in ((69.0, 74), (71.0, 77), (73.0, 81), (75.0, 86), (78.0, 84), (81.0, 89)):
            self.tone(start, 1.45, note, 0.082, "brass", 0.06, 0.58, 0.18)

        # ACT V — detection (84–96): rhythm drops away, tension climbs.
        self.pad(84.0, 12.0, (36, 43, 48, 51), 0.24, 0.35)
        self.boom(84.0, 0.75)
        self.radar(85.3, 92, -0.6)
        self.radar(87.1, 92, 0.6)
        self.radar(89.0, 95, -0.2)
        self.riser(87.5, 8.3, 0.18)
        for start, note in ((88.0, 62), (90.0, 65), (92.0, 68), (94.0, 71)):
            self.tone(start, 2.0, note, 0.078, "pulse", 0.10, 0.45, -0.25)

        # ACT VI — guarded intercept (96–108): full action climax.
        self.pad(96.0, 12.0, (38, 45, 50, 53), 0.33, 0.25)
        self.boom(96.0, 0.88)
        self.ostinato(96.0, 108.0, 128.0, (62, 65, 69, 74, 70, 69, 77, 74), 0.077)
        self.drums(96.0, 108.0, 128.0, 1.35, 1)
        for start, note in ((96.0, 62), (97.4, 65), (98.8, 69), (100.2, 74), (102.0, 77), (104.0, 81), (106.0, 86)):
            self.tone(start, 2.0, note, 0.115, "brass", 0.07, 0.52, 0.12)
        self.boom(102.0, 0.72)
        self.boom(107.1, 0.92)

        # ACT VII — mission resolved (108–120): recognizable theme, wide finish.
        self.pad(108.0, 12.0, (38, 45, 50, 53, 57), 0.31, 0.45)
        self.boom(108.0, 0.64)
        for start, note, length in (
            (108.2, 74, 2.5),
            (110.0, 77, 2.5),
            (111.8, 81, 2.8),
            (114.0, 86, 3.8),
            (116.5, 89, 3.5),
        ):
            self.tone(start, length, note, 0.105, "brass", 0.16, 1.25, 0.10)
            self.tone(start, length, note + 12, 0.040, "strings", 0.16, 1.45, -0.22)
        self.boom(114.0, 0.52)
        self.boom(118.2, 0.70)

        # Subtle air and stereo delay glue the discrete instruments together.
        air = self.rng.standard_normal(self.count)
        air = np.convolve(air, np.ones(96) / 96.0, mode="same") * 0.008
        self.audio[:, 0] += air
        self.audio[:, 1] += np.roll(air, 503) * 0.87

    def write(self, path: Path) -> None:
        self.compose()
        delay = int(0.021 * self.rate)
        ambience = np.zeros_like(self.audio)
        ambience[delay:, 0] = self.audio[:-delay, 1] * 0.10
        ambience[delay:, 1] = self.audio[:-delay, 0] * 0.10
        self.audio += ambience

        peak = float(np.max(np.abs(self.audio))) or 1.0
        self.audio = np.tanh(self.audio * (0.92 / peak) * 1.72) / np.tanh(1.72)
        # Short cinematic fade-in and full fade-out.
        fade_in = min(self.count, int(0.28 * self.rate))
        fade_out = min(self.count, int(1.65 * self.rate))
        self.audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in)[:, None]
        self.audio[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)[:, None]

        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(self.audio * 32767.0, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(self.rate)
            wav.writeframes(pcm.tobytes())


def build_score() -> None:
    Score(DURATION, SAMPLE_RATE).write(SCORE)


def mux() -> None:
    run(
        [
            "/opt/homebrew/bin/ffmpeg",
            "-y",
            "-i",
            str(SILENT),
            "-i",
            str(SCORE),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-ar",
            str(SAMPLE_RATE),
            "-movflags",
            "+faststart",
            "-shortest",
            str(FINAL),
        ]
    )


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Locked V1 film not found: {SOURCE}")
    build_picture()
    build_score()
    mux()
    print(FINAL)


if __name__ == "__main__":
    main()

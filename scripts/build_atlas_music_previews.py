#!/usr/bin/env python3
"""Create four contrasting 32-second score auditions for the ATLAS film."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

from build_atlas_v1_directors_cut import Score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "spy_demo" / "music_previews"
DURATION = 32.0
RATE = 48_000


def save(score: Score, name: str) -> None:
    """Master the hand-arranged score, then create a convenient MP3 preview."""
    delay = int(0.024 * score.rate)
    room = np.zeros_like(score.audio)
    room[delay:, 0] = score.audio[:-delay, 1] * 0.11
    room[delay:, 1] = score.audio[:-delay, 0] * 0.11
    score.audio += room

    peak = float(np.max(np.abs(score.audio))) or 1.0
    score.audio = np.tanh(score.audio * (0.91 / peak) * 1.68) / np.tanh(1.68)
    fade_in = int(0.24 * score.rate)
    fade_out = int(1.15 * score.rate)
    score.audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in)[:, None]
    score.audio[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)[:, None]

    OUT.mkdir(parents=True, exist_ok=True)
    wav_path = OUT / f"{name}.wav"
    mp3_path = OUT / f"{name}.mp3"
    pcm = np.clip(score.audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(score.rate)
        wav.writeframes(pcm.tobytes())
    subprocess.run(
        [
            "/opt/homebrew/bin/ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(wav_path),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "256k",
            str(mp3_path),
        ],
        check=True,
    )
    print(mp3_path)


def velvet_dossier() -> None:
    """Elegant classic espionage: spacious, refined, sly rather than aggressive."""
    s = Score(DURATION, RATE)
    s.pad(0.0, 8.0, (38, 45, 50, 53), 0.28, 0.8)
    s.pad(8.0, 8.0, (36, 43, 48, 52), 0.25, 0.7)
    s.pad(16.0, 8.0, (41, 48, 53, 57), 0.28, 0.6)
    s.pad(24.0, 8.0, (38, 45, 50, 54), 0.32, 0.5)
    # A memorable, wide spy motif with pauses between statements.
    melody = (
        (1.0, 74, 1.5), (3.0, 77, 0.9), (4.3, 81, 1.8),
        (8.7, 72, 1.3), (10.5, 76, 1.0), (12.0, 79, 2.0),
        (16.5, 77, 1.0), (17.9, 81, 1.0), (19.3, 84, 2.2),
        (24.4, 74, 1.0), (25.8, 77, 1.0), (27.2, 81, 1.0), (28.6, 86, 2.5),
    )
    for i, (start, note, length) in enumerate(melody):
        s.tone(start, length, note, 0.105, "glass", 0.055, 0.52, -0.42 if i % 2 else 0.42)
        s.tone(start, length, note - 12, 0.045, "strings", 0.08, 0.58, 0.0)
    for start in (0.2, 8.0, 16.0, 24.0, 30.2):
        s.boom(start, 0.47 if start < 24 else 0.60)
    # Irregular low percussion keeps the cue moving without ticking.
    for i, start in enumerate((5.8, 7.2, 12.7, 15.1, 21.0, 22.4, 26.8, 29.6)):
        s.tom(start, 0.19 + 0.018 * (i % 3), -0.35 if i % 2 else 0.35)
    for start, note, pan in ((2.2, 86, -0.7), (11.2, 83, 0.65), (18.2, 88, -0.4), (27.8, 91, 0.45)):
        s.radar(start, note, pan)
    save(s, "01_velvet_dossier_elegant_spy")


def black_circuit() -> None:
    """Modern tactical electronic: muscular synth motion and clean impacts."""
    s = Score(DURATION, RATE)
    s.pad(0.0, 8.0, (33, 40, 45), 0.22, 0.35)
    s.pad(8.0, 8.0, (35, 42, 47), 0.23, 0.30)
    s.pad(16.0, 8.0, (36, 43, 48), 0.25, 0.25)
    s.pad(24.0, 8.0, (38, 45, 50), 0.28, 0.20)
    # Two complementary syncopated patterns create motion without a metronome feel.
    s.ostinato(1.0, 16.0, 116.0, (50, None, 57, 62, 50, 60, None, 57), 0.062, -12)
    s.ostinato(8.0, 24.0, 122.0, (62, 65, None, 69, 67, 62, 70, None), 0.055)
    s.ostinato(16.0, 32.0, 126.0, (65, 69, 72, None, 70, 74, 77, 72), 0.070)
    s.drums(4.0, 16.0, 116.0, 0.82, 1)
    s.drums(16.0, 32.0, 126.0, 1.25, 1)
    for start, note in ((2.0, 62), (6.0, 65), (10.0, 69), (14.0, 67), (18.0, 69), (20.0, 72), (22.0, 77), (25.0, 81), (28.0, 84)):
        s.tone(start, 1.55, note, 0.095, "pulse", 0.025, 0.38, 0.28)
    for start, gain in ((0.0, 0.58), (8.0, 0.55), (16.0, 0.68), (24.0, 0.76), (30.4, 0.82)):
        s.boom(start, gain)
    s.riser(12.5, 3.4, 0.11)
    s.riser(27.0, 4.2, 0.17)
    save(s, "02_black_circuit_modern_tactical")


def redline_protocol() -> None:
    """Large orchestral action: heroic brass, strings, and trailer-scale drums."""
    s = Score(DURATION, RATE)
    s.pad(0.0, 8.0, (38, 45, 50, 53), 0.28, 0.5)
    s.pad(8.0, 8.0, (41, 48, 53, 57), 0.30, 0.4)
    s.pad(16.0, 8.0, (43, 50, 55, 58), 0.33, 0.3)
    s.pad(24.0, 8.0, (45, 52, 57, 61), 0.36, 0.25)
    s.ostinato(4.0, 16.0, 108.0, (62, 65, 69, 65, 67, 70, 74, 70), 0.055)
    s.ostinato(16.0, 32.0, 124.0, (65, 69, 72, 77, 74, 72, 81, 77), 0.075)
    s.drums(4.0, 16.0, 108.0, 0.85, 0)
    s.drums(16.0, 32.0, 124.0, 1.35, 1)
    theme = (
        (0.8, 62, 2.4), (3.2, 65, 2.1), (5.5, 69, 2.5),
        (8.2, 67, 1.8), (10.0, 72, 2.8), (13.0, 74, 2.4),
        (16.2, 69, 1.5), (17.7, 72, 1.5), (19.2, 77, 2.4),
        (22.0, 81, 2.0), (24.2, 84, 2.5), (27.0, 86, 3.5),
    )
    for i, (start, note, length) in enumerate(theme):
        gain = 0.095 if start < 16 else 0.135
        s.tone(start, length, note, gain, "brass", 0.10, 0.65, 0.15)
        s.tone(start, length, note + 12, gain * 0.35, "strings", 0.12, 0.78, -0.22)
    for start, gain in ((0.0, 0.66), (8.0, 0.62), (16.0, 0.78), (24.0, 0.90), (30.4, 0.96)):
        s.boom(start, gain)
    s.riser(12.0, 4.0, 0.12)
    s.riser(27.0, 4.4, 0.18)
    save(s, "03_redline_protocol_orchestral_action")


def ghost_corridor() -> None:
    """Dark covert suspense: patient unease that breaks into a final pursuit."""
    s = Score(DURATION, RATE)
    s.pad(0.0, 10.0, (35, 42, 47, 50), 0.24, 1.6)
    s.pad(10.0, 10.0, (33, 40, 45, 49), 0.26, 1.2)
    s.pad(20.0, 6.0, (36, 43, 48, 51), 0.28, 0.7)
    s.pad(26.0, 6.0, (38, 45, 50, 53), 0.32, 0.3)
    # Long unresolved figures and asymmetric calls, then a compact chase.
    for i, (start, note, length) in enumerate(((1.2, 81, 2.5), (5.4, 78, 2.2), (9.8, 84, 2.9), (14.0, 80, 2.0), (18.3, 86, 2.7), (22.2, 89, 2.6))):
        s.tone(start, length, note, 0.075, "glass", 0.10, 1.2, -0.65 if i % 2 else 0.65)
        s.tone(start + 0.08, length + 0.6, note - 24, 0.052, "pulse", 0.14, 1.1, 0.0)
    for start, note, pan in ((3.1, 92, -0.75), (7.7, 87, 0.65), (12.7, 94, -0.35), (17.1, 90, 0.55), (23.2, 96, -0.2)):
        s.radar(start, note, pan)
    for start, gain in ((0.0, 0.42), (10.0, 0.48), (20.0, 0.60), (26.0, 0.75), (31.0, 0.86)):
        s.boom(start, gain)
    s.riser(17.0, 4.0, 0.10)
    s.riser(22.0, 4.0, 0.15)
    s.ostinato(25.0, 32.0, 128.0, (62, 65, 69, None, 70, 74, 77, 74), 0.074)
    s.drums(25.0, 32.0, 128.0, 1.28, 1)
    for start, note in ((25.0, 65), (26.4, 69), (27.8, 74), (29.2, 77), (30.5, 84)):
        s.tone(start, 1.8, note, 0.12, "brass", 0.07, 0.48, 0.12)
    save(s, "04_ghost_corridor_dark_suspense")


def main() -> None:
    velvet_dossier()
    black_circuit()
    redline_protocol()
    ghost_corridor()


if __name__ == "__main__":
    main()

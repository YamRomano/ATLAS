#!/usr/bin/env python3
"""Minimal SFX bed for the untouched command-center picture master."""
import math
import wave
from pathlib import Path

import numpy as np

RATE = 48000
DURATION = 127.0
OUT = Path(__file__).resolve().parents[1] / "outputs/spy_demo/work/surgical_master_sfx.wav"

audio = np.zeros((round(RATE * DURATION), 2), np.float64)
rng = np.random.default_rng(8717)

def put(start, signal, left=1.0, right=1.0):
    offset = round(start * RATE)
    count = min(len(signal), len(audio) - offset)
    if count > 0:
        audio[offset:offset+count, 0] += signal[:count] * left
        audio[offset:offset+count, 1] += signal[:count] * right

# Restrained surveillance confirmations.
for when, freq in ((0.55, 920), (1.02, 1160), (7.12, 1480), (7.36, 1960), (74.08, 1240), (74.31, 1820)):
    x = np.arange(round(.22 * RATE)) / RATE
    put(when, np.sin(math.tau * freq * x) * np.sin(np.pi * x / .22) ** 2 * .11, .95, .80)

# Short riser into the verified first orange impact frame at 87.04 seconds.
x = np.arange(round(2.8 * RATE)) / RATE
put(91.24, (rng.standard_normal(len(x)) - np.roll(rng.standard_normal(len(x)), 31)) * (x / 2.8) ** 1.8 * .045, .82, 1.0)

# Cinematic impact with transient, body and sub tail—never before the approach.
x = np.arange(round(4.8 * RATE)) / RATE
sub = np.sin(math.tau * (73*x - 12*x*x)) * np.exp(-x*1.1)
body = np.sin(math.tau * (126*x - 25*x*x)) * np.exp(-x*2.5)
crack = rng.standard_normal(len(x)) * np.exp(-x*24)
debris = (rng.standard_normal(len(x)) - np.roll(rng.standard_normal(len(x)), 37)) * np.exp(-x*3.8)
put(94.04, .67*sub + .31*body + .15*crack + .07*debris, 1.0, .95)

OUT.parent.mkdir(parents=True, exist_ok=True)
pcm = np.clip(audio * 32767, -32768, 32767).astype('<i2')
with wave.open(str(OUT), 'wb') as handle:
    handle.setnchannels(2)
    handle.setsampwidth(2)
    handle.setframerate(RATE)
    handle.writeframes(pcm.tobytes())
print(OUT)

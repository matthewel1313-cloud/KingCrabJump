"""Every sound in the video, synthesised here with numpy -- no samples, no
third-party audio. See CREDITS.md.

    heart monitor  1 kHz sine "beep" with a soft attack, 76 bpm, quickening on "Faster!"
    clink          inharmonic metal partials + a short noise transient, one per hand-off
    music          a self-composed soft marimba-ish arpeggio + bass, C major, 96 bpm, low
    chime          a rising major arpeggio on "perfect."
    oops           a short descending "wah" on the gauze mistake
    clatter        a few damped noise taps when the blooper's scalpel skids away
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 48000


def env_exp(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def tone(freq, dur, amp=1.0, attack=0.004, tau=None, shape="sine"):
    n = int(dur * SR)
    t = np.arange(n) / SR
    if shape == "sine":
        x = np.sin(2 * np.pi * freq * t)
    elif shape == "tri":
        x = 2 / np.pi * np.arcsin(np.sin(2 * np.pi * freq * t))
    a = np.minimum(1.0, t / max(attack, 1e-4))
    e = env_exp(n, tau) if tau else np.ones(n)
    return amp * x * a * e


def beep(freq=1000.0, dur=0.11):
    x = tone(freq, dur, attack=0.006)
    rel = int(0.02 * SR)
    x[-rel:] *= np.linspace(1, 0, rel)
    return x


def clink(seed=0):
    rng = np.random.default_rng(seed)
    dur = 0.55
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    base = 2350 * (1 + 0.03 * rng.standard_normal())
    for ratio, amp, tau in ((1.0, 1.0, 0.16), (1.59, 0.6, 0.11), (2.31, 0.45, 0.08),
                            (3.07, 0.3, 0.05), (4.2, 0.18, 0.035)):
        out += amp * np.sin(2 * np.pi * base * ratio * t + rng.uniform(0, 6)) * np.exp(-t / tau)
    click = rng.standard_normal(int(0.004 * SR)) * np.linspace(1, 0, int(0.004 * SR))
    out[: len(click)] += 0.8 * click
    return 0.5 * out / np.abs(out).max()


def marimba(freq, dur=0.5, amp=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.22)
    x += 0.25 * np.sin(2 * np.pi * freq * 4.0 * t) * np.exp(-t / 0.05)
    x *= np.minimum(1.0, t / 0.003)
    return amp * x


def music(total):
    """A gentle 4-bar loop: Cmaj7 - Am7 - Fmaj7 - G6, arpeggiated, with bass."""
    bpm = 96
    beat = 60 / bpm
    n = int(total * SR)
    out = np.zeros(n + SR)
    m = lambda midi: 440 * 2 ** ((midi - 69) / 12)  # noqa: E731
    chords = [(60, 64, 67, 71), (57, 60, 64, 67), (53, 57, 60, 64), (55, 59, 62, 64)]
    pattern = [0, 1, 2, 3, 2, 1, 2, 3]
    t = 0.0
    bar = 0
    while t < total:
        ch = chords[bar % 4]
        for k, idx in enumerate(pattern):
            at = t + k * beat / 2
            if at >= total:
                break
            note = marimba(m(ch[idx] + 12), 0.45, 0.5 if k % 2 == 0 else 0.38)
            i = int(at * SR)
            out[i:i + len(note)] += note[: len(out) - i]
        bass = tone(m(ch[0] - 12), beat * 3.6, 0.55, attack=0.02, tau=1.2, shape="tri")
        i = int(t * SR)
        out[i:i + len(bass)] += bass[: len(out) - i]
        t += 4 * beat
        bar += 1
    return out[:n]


def chime(start_midi=72):
    m = lambda midi: 440 * 2 ** ((midi - 69) / 12)  # noqa: E731
    out = np.zeros(int(1.4 * SR))
    for k, d in enumerate((0, 4, 7, 12)):
        note = marimba(m(start_midi + d), 1.0, 0.8)
        note += 0.3 * tone(m(start_midi + d) * 2, 1.0, 0.5, tau=0.4)
        i = int(k * 0.09 * SR)
        out[i:i + len(note)] += note[: len(out) - i]
    return 0.5 * out / np.abs(out).max()


def oops():
    dur = 0.45
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 520 * np.exp(-t * 1.6)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.3 * np.sin(2 * ph)
    x *= np.minimum(1, t / 0.01) * np.exp(-t / 0.25)
    return 0.4 * x / np.abs(x).max()


def clatter(seed=3):
    rng = np.random.default_rng(seed)
    out = np.zeros(int(0.6 * SR))
    for k, at in enumerate((0.0, 0.09, 0.16, 0.24)):
        c = clink(seed + k)[: int(0.25 * SR)] * (0.9 - 0.18 * k)
        i = int(at * SR)
        out[i:i + len(c)] += c
    out += 0.05 * rng.standard_normal(len(out)) * env_exp(len(out), 0.08)
    return out


def place(buf, x, t, gain=1.0):
    i = int(t * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(x))
    buf[i:j] += gain * x[: j - i]


def db(v):
    return 10 ** (v / 20)


def mix(total: float, cues: dict) -> np.ndarray:
    """cues: {'clinks': [t...], 'fast': (t0, t1), 'chime': t, 'oops': t,
              'clatter': t, 'music_end': t}"""
    n = int(total * SR)
    out = np.zeros(n)
    # heart monitor, 76 bpm; quickens to 104 bpm during "Faster!"
    t = 0.35
    fa, fb = cues.get("fast", (1e9, 1e9))
    while t < total - 0.1:
        place(out, beep(), t, db(-21))
        t += 60 / (104 if fa <= t < fb else 76)
    # music bed, well under the beeps
    mus = music(min(total, cues.get("music_end", total)))
    fade = int(0.8 * SR)
    mus[-fade:] *= np.linspace(1, 0, fade)
    mus[: int(0.4 * SR)] *= np.linspace(0, 1, int(0.4 * SR))
    place(out, mus / np.abs(mus).max(), 0.0, db(-27))
    for k, tc in enumerate(cues.get("clinks", [])):
        place(out, clink(k), tc, db(-9))
    if "oops" in cues:
        place(out, oops(), cues["oops"], db(-16))
    if "chime" in cues:
        place(out, chime(), cues["chime"], db(-13))
    if "clatter" in cues:
        place(out, clatter(), cues["clatter"], db(-11))
    peak = np.abs(out).max()
    if peak > 0.89:
        out *= 0.89 / peak
    return out


def write_wav(path: Path, x: np.ndarray):
    s = np.clip(x, -1, 1)
    pcm = (s * 32767).astype(np.int16)
    stereo = np.stack([pcm, pcm], axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())

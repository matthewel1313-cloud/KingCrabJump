"""An original score and sound effects, synthesised from scratch (numpy only, no samples).

    build(outdir, cut="main" | "alt_f", duration=30.0, music=None) -> {"music", "sfx", "mix"}

The score is a short pentatonic pluck-and-bass groove written here; the effects (whoosh, claw
click, scan sweep, pen pops, a sad trombone for the F, a sparkle for the A+) are synthesised the
same way. Pass `music=` to use your own track instead of the score: it is trimmed, faded and
mixed under the same effects, and the stems are written so the edit can be remixed elsewhere.
"""
from __future__ import annotations

import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "motion"))
import timeline as T  # noqa: E402


def env(n, a=0.005, d=0.25, sustain=0.0):
    t = np.arange(n) / SR
    e = np.minimum(t / a, 1.0) * (sustain + (1 - sustain) * np.exp(-t / d))
    return e


def pluck(freq, dur, bright=0.5, seed=0):
    """Karplus-Strong string."""
    n = int(dur * SR)
    period = max(int(SR / freq), 2)
    rng = np.random.default_rng(seed)
    buf = rng.uniform(-1, 1, period)
    out = np.empty(n)
    for i in range(n):
        out[i] = buf[i % period]
        buf[i % period] = 0.5 * (buf[i % period] + buf[(i + 1) % period]) * (0.996 - 0.004 * (1 - bright))
    return out * env(n, 0.002, dur * 0.6)


def tone(freq, dur, kind="sine", a=0.005, d=0.3, sustain=0.0, vib=0.0):
    t = np.arange(int(dur * SR)) / SR
    f = freq * (1 + vib * np.sin(2 * np.pi * 5.5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    w = {"sine": np.sin(ph), "tri": 2 / np.pi * np.arcsin(np.sin(ph)),
         "saw": 2 * ((ph / (2 * np.pi)) % 1) - 1}[kind]
    return w * env(len(t), a, d, sustain)


def noise(dur, seed=1):
    return np.random.default_rng(seed).uniform(-1, 1, int(dur * SR))


def add(buf, sig, at, gain=1.0):
    i = int(at * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += gain * sig[: j - i]


def note(name):
    names = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    semis = names[name[0]] + (1 if "#" in name else -1 if "b" in name[1:] else 0)
    octave = int(name[-1])
    return 440.0 * 2 ** ((semis - 9) / 12 + (octave - 4))


def score(duration, cut):
    """The groove: 104 bpm, C major pentatonic, stops for the F, triumphant for the A+."""
    buf = np.zeros(int(duration * SR) + SR)
    beat = 60 / 104
    melody = ["E5", "G5", "A5", "G5", "E5", "D5", "C5", "D5", "E5", "G5", "C6", "A5", "G5", "E5", "G5", "D5"]
    bass = ["C3", "C3", "A2", "A2", "F2", "F2", "G2", "G2"]
    groove_end = T.RELEASE + 0.2
    t, k = 0.25, 0
    while t < groove_end:
        add(buf, pluck(note(melody[k % 16]), 0.5, seed=k), t, 0.30)
        if k % 2 == 0:
            add(buf, tone(note(bass[(k // 2) % 8]), beat * 0.9, "tri", d=0.35), t, 0.32)
            add(buf, tone(55, 0.18, "sine", a=0.001, d=0.06) * np.linspace(1, 0.2, int(0.18 * SR)), t, 0.5)
        add(buf, noise(0.03, k) * env(int(0.03 * SR), 0.001, 0.01), t + beat / 2, 0.08)
        t += beat / 2 if k % 4 == 3 else beat / 2
        k += 1
    # head shake: a ticking clock under the disappointment
    for j in range(int((T.SHAKE[1] - T.SHAKE[0]) / beat)):
        add(buf, tone(1800 if j % 2 else 1500, 0.05, "sine", a=0.001, d=0.02), T.SHAKE[0] + j * beat, 0.10)
    if cut == "main":
        # the A+: a rising arpeggio and a bright groove until the end card, then a resolved chord
        t = T.DANCE_IN
        arp = ["C5", "E5", "G5", "C6", "E6", "G6"]
        for j, n in enumerate(arp):
            add(buf, pluck(note(n), 0.6, bright=0.9, seed=40 + j), t + 0.09 * j, 0.32)
        t, k = T.DANCE_IN + 0.6, 0
        while t < 27.2:
            add(buf, pluck(note(melody[(k + 8) % 16]), 0.45, bright=0.8, seed=80 + k), t, 0.30)
            if k % 2 == 0:
                add(buf, tone(note(bass[(k // 2) % 8]), beat * 0.9, "tri", d=0.35), t, 0.32)
                add(buf, tone(55, 0.18, "sine", a=0.001, d=0.06), t, 0.5)
            add(buf, noise(0.03, 200 + k) * env(int(0.03 * SR), 0.001, 0.01), t + beat / 4, 0.09)
            t += beat / 2; k += 1
        for n in ("C4", "E4", "G4", "C5"):
            add(buf, tone(note(n), 3.0, "tri", a=0.02, d=1.4), 27.3, 0.12)
    return buf[: int(duration * SR)]


def effects(duration, cut, marks_start, mark_step, n_marks, f_at, aplus_at):
    buf = np.zeros(int(duration * SR) + SR)
    # the hook: a whoosh as the professor turns to the lens
    w = noise(0.6, 3)
    w = np.convolve(w, np.ones(40) / 40, mode="same") * np.sin(np.linspace(0, np.pi, len(w)))
    add(buf, w, T.TURN[0] - 0.05, 0.5)
    # claw click on the grasp, and on the release
    for at in (T.GRASP - 0.05, T.RELEASE):
        add(buf, tone(2400, 0.04, "saw", a=0.0005, d=0.01), at, 0.25)
        add(buf, tone(900, 0.06, "sine", a=0.0005, d=0.02), at + 0.01, 0.3)
    # the scan: sweeping beeps while reading
    t = T.READ[0]
    period = (T.SHUFFLE[0] - T.READ[0] + 0.1) / 2.5
    while t < T.SHUFFLE[0]:
        n = int(period * SR)
        f = np.linspace(600, 1500, n)
        add(buf, np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.02, period, 0.6) * 0.5, t, 0.12)
        t += period
    # red pen: a pop per mark
    for k in range(n_marks):
        add(buf, tone(700 + 120 * k, 0.09, "sine", a=0.001, d=0.04), marks_start + mark_step * k, 0.45)
    # the F: a stamp thud and a sad trombone
    add(buf, tone(70, 0.35, "sine", a=0.001, d=0.12), f_at, 0.9)
    add(buf, noise(0.08, 9) * env(int(0.08 * SR), 0.001, 0.03), f_at, 0.3)
    for j, n in enumerate(("Bb3", "A3", "Ab3")):
        add(buf, tone(note(n), 0.45, "saw", a=0.03, d=0.5, sustain=0.6, vib=0.006), f_at + 0.35 + 0.48 * j, 0.10)
    add(buf, tone(note("G3"), 1.3, "saw", a=0.03, d=1.0, sustain=0.5, vib=0.02), f_at + 0.35 + 1.44, 0.10)
    if cut == "main":
        # the A+: a stamp and a sparkle that twinkles with the crown
        add(buf, tone(90, 0.25, "sine", a=0.001, d=0.1), aplus_at + 0.25, 0.6)
        rng = np.random.default_rng(5)
        for j in range(18):
            add(buf, tone(rng.uniform(2500, 4200), 0.25, "sine", a=0.001, d=0.08), aplus_at + 0.2 + 0.18 * j, 0.10)
    return buf[: int(duration * SR)]


def write_wav(path, mono):
    st = np.clip(np.stack([mono, mono], 1), -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((st * 32767).astype("<i2").tobytes())


def read_any(path, duration):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-t", f"{duration:.3f}", "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype="<f4").astype(float)
    out = np.zeros(int(duration * SR)); out[: len(x)] = x[: len(out)]
    return out


def build(outdir, *, cut="main", duration=30.0, music=None):
    import post  # the edit's own beat times, so the effects land on the overlays
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    m = read_any(music, duration) if music else score(duration, cut)
    fade = int(1.2 * SR)
    m[-fade:] *= np.linspace(1, 0, fade)
    s = effects(duration, cut, post.MARKS_START, post.MARK_STEP, len(post.ANSWERS), post.F_AT, post.APLUS_AT)
    mix = 0.8 * m / max(1e-6, np.abs(m).max()) * 0.7 + s
    mix /= max(1.0, np.abs(mix).max() / 0.95)
    tag = "main" if cut == "main" else "alt_F"
    paths = {"music": outdir / f"{tag}_music.wav", "sfx": outdir / f"{tag}_sfx.wav", "mix": outdir / f"{tag}_mix.wav"}
    write_wav(paths["music"], m); write_wav(paths["sfx"], s); write_wav(paths["mix"], mix)
    return paths

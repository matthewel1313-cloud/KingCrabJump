#!/usr/bin/env python3
"""Original music and sound effects, synthesised from scratch with numpy.

Nothing here is sampled or derived from existing recordings, so the soundtrack
is wholly our own work (the contest requires owned or royalty-free music).

    track = beach_track(duration=22.0, drop_at=2.85)   # float32 stereo, 44.1 kHz
"""

from __future__ import annotations

import numpy as np

SR = 44100
RNG = np.random.default_rng(7)


def t_axis(sec):
    return np.arange(int(sec * SR)) / SR


def env_adsr(n, a=0.005, d=0.1, s=0.6, r=0.1, hold=None):
    hold = n / SR if hold is None else hold
    t = np.arange(n) / SR
    e = np.where(t < a, t / a, np.where(t < a + d, 1 - (1 - s) * (t - a) / d, s))
    rel = np.clip((t - hold) / r, 0, 1)
    return e * (1 - rel)


def lowpass(x, cutoff):
    # one-pole, applied twice
    a = np.exp(-2 * np.pi * cutoff / SR)
    for _ in range(2):
        y = np.empty_like(x)
        acc = 0.0
        for i in range(len(x)):
            acc = (1 - a) * x[i] + a * acc
            y[i] = acc
        x = y
    return x


def fft_filter(x, lo=None, hi=None):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    m = np.ones_like(f)
    if lo:
        m *= 1 / (1 + (lo / np.maximum(f, 1)) ** 4)
    if hi:
        m *= 1 / (1 + (f / hi) ** 4)
    return np.fft.irfft(X * m, len(x))


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


# ── instruments ───────────────────────────────────────────────────────────

def steelpan(freq, dur, vel=1.0):
    t = t_axis(dur + 0.6)
    partials = [(1, 1.0, 3.0), (2, 0.55, 4.5), (3.01, 0.25, 6), (4.2, 0.12, 9), (5.03, 0.06, 12)]
    x = sum(a * np.sin(2 * np.pi * freq * k * t) * np.exp(-t * dcy) for k, a, dcy in partials)
    x *= np.minimum(t / 0.003, 1)
    return vel * 0.35 * x


def marimba(freq, dur, vel=1.0):
    t = t_axis(dur + 0.4)
    x = np.sin(2 * np.pi * freq * t) * np.exp(-t * 7) + 0.3 * np.sin(2 * np.pi * freq * 3.93 * t) * np.exp(-t * 30)
    x *= np.minimum(t / 0.002, 1)
    return vel * 0.3 * x


def bass(freq, dur, vel=1.0):
    t = t_axis(dur + 0.05)
    ph = 2 * np.pi * freq * t
    x = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
    return vel * 0.42 * x * env_adsr(len(t), 0.004, 0.12, 0.55, 0.05, hold=dur)


def pad(freqs, dur, vel=1.0):
    t = t_axis(dur + 0.3)
    x = np.zeros_like(t)
    for f in freqs:
        for det in (-0.12, 0.0, 0.12):
            ff = f * 2 ** (det / 12)
            x += np.sin(2 * np.pi * ff * t + RNG.uniform(0, 6)) + 0.3 * np.sin(4 * np.pi * ff * t)
    return vel * 0.03 * x * env_adsr(len(t), 0.25, 0.2, 0.8, 0.3, hold=dur)


def kick(vel=1.0):
    t = t_axis(0.35)
    f = 48 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return vel * 0.9 * np.sin(ph) * np.exp(-t * 9) * np.minimum(t / 0.001, 1)


def clap(vel=1.0):
    t = t_axis(0.25)
    n = RNG.standard_normal(len(t))
    e = np.zeros_like(t)
    for off in (0, 0.011, 0.022):
        e += np.where(t >= off, np.exp(-(t - off) * 60), 0)
    e += 0.6 * np.exp(-t * 18)
    return vel * 0.35 * fft_filter(n * e, lo=900, hi=5000)


def hat(vel=1.0, open_=False):
    t = t_axis(0.25 if open_ else 0.06)
    n = RNG.standard_normal(len(t))
    return vel * 0.12 * fft_filter(n, lo=7000) * np.exp(-t * (14 if open_ else 70))


def shaker(vel=1.0):
    t = t_axis(0.08)
    n = RNG.standard_normal(len(t))
    e = np.sin(np.pi * np.clip(t / 0.08, 0, 1)) ** 2
    return vel * 0.07 * fft_filter(n, lo=4500, hi=11000) * e


# ── sound effects ─────────────────────────────────────────────────────────

def whoosh(dur=0.6, rise=True, vel=1.0):
    t = t_axis(dur)
    n = RNG.standard_normal(len(t))
    shape = (t / dur) ** 2 if rise else np.exp(-t / dur * 4)
    x = fft_filter(n, lo=400, hi=6000) * shape
    return vel * 0.25 * x / (np.abs(x).max() + 1e-9)


def riser(dur=2.5, vel=1.0):
    t = t_axis(dur)
    f = 200 * 2 ** (3 * t / dur)
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = np.sin(ph) * 0.3 + 0.15 * np.sign(np.sin(ph))
    noise = fft_filter(RNG.standard_normal(len(t)), lo=1500) * 0.4
    e = (t / dur) ** 2.2
    return vel * 0.22 * (tone + noise) * e


def thwack(vel=1.0):
    """Ball hit: a tight thump plus a slap of noise."""
    t = t_axis(0.3)
    body = np.sin(2 * np.pi * (180 + 300 * np.exp(-t * 60)) * t) * np.exp(-t * 25)
    slap = fft_filter(RNG.standard_normal(len(t)), lo=1200, hi=7000) * np.exp(-t * 55)
    return vel * (0.8 * body + 0.6 * slap)


def boing(vel=1.0):
    t = t_axis(0.45)
    f = 220 + 260 * np.sin(np.pi * np.clip(t / 0.45, 0, 1)) * np.exp(-t * 2)
    ph = 2 * np.pi * np.cumsum(f) / SR
    wob = 1 + 0.25 * np.sin(2 * np.pi * 18 * t)
    return vel * 0.22 * np.sin(ph * wob) * np.exp(-t * 6)


def sand_thud(vel=1.0):
    t = t_axis(0.35)
    n = fft_filter(RNG.standard_normal(len(t)), lo=80, hi=1400)
    return vel * 0.45 * n / (np.abs(n).max() + 1e-9) * np.exp(-t * 14)


def crowd_cheer(dur=2.5, vel=1.0):
    """A small crowd 'yay': many detuned, vibrato'd formant voices plus noise."""
    t = t_axis(dur)
    x = np.zeros_like(t)
    for _ in range(26):
        f0 = RNG.uniform(180, 420)
        vib = 1 + 0.03 * np.sin(2 * np.pi * RNG.uniform(4, 7) * t + RNG.uniform(0, 6))
        glide = 1 + 0.25 * np.exp(-t * 3)
        ph = 2 * np.pi * np.cumsum(f0 * vib * glide) / SR
        v = sum(np.sin(k * ph) / k for k in range(1, 8))
        start = RNG.uniform(0, 0.25)
        e = np.clip((t - start) / 0.08, 0, 1) * np.exp(-np.clip(t - start, 0, None) * RNG.uniform(0.8, 1.6))
        x += v * e
    x = fft_filter(x, lo=300, hi=3500)
    noise = fft_filter(RNG.standard_normal(len(t)), lo=600, hi=5000) * np.exp(-t * 1.2) * 0.5
    x = x / (np.abs(x).max() + 1e-9) + noise
    return vel * 0.22 * x * np.minimum(t / 0.05, 1)


def ding(vel=1.0):
    t = t_axis(1.0)
    x = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, d in ((1568, 3), (2349, 4), (3136, 6)))
    return vel * 0.12 * x


# ── mixing ────────────────────────────────────────────────────────────────

class Mix:
    def __init__(self, sec):
        self.buf = np.zeros((int(sec * SR) + SR, 2), dtype=np.float64)
        self.sec = sec

    def add(self, at, x, pan=0.0, gain=1.0):
        i = int(at * SR)
        if i >= len(self.buf) or i + len(x) <= 0:
            return
        if i < 0:
            x, i = x[-i:], 0
        x = x[: len(self.buf) - i]
        lg, rg = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        self.buf[i : i + len(x), 0] += x * gain * lg * 1.414
        self.buf[i : i + len(x), 1] += x * gain * rg * 1.414

    def master(self, fade_out=0.8):
        y = self.buf[: int(self.sec * SR)]
        n = int(fade_out * SR)
        y[-n:] *= np.linspace(1, 0, n)[:, None]
        y = np.tanh(y * 1.1) / np.tanh(1.1)
        peak = np.abs(y).max()
        return (y / peak * 0.89).astype(np.float32)


# ── the song ──────────────────────────────────────────────────────────────

# I–V–vi–IV in C, one chord per bar. Roots (midi) and triad tones.
PROG = [(48, [60, 64, 67]), (43, [59, 62, 67]), (45, [57, 60, 64]), (41, [57, 60, 65])]
# Hook melody, 2 bars, in eighth notes (None = rest). Steelpan, C major pentatonic.
MELODY_A = [76, None, 79, 76, 74, None, 72, 74, 76, None, 79, None, 81, 79, 76, None]
MELODY_B = [72, None, 74, 76, 79, None, 76, 74, 72, None, 69, 72, 74, None, None, None]


def beach_track(duration=22.0, drop_at=2.85, bpm=120.0, stops=(), outro_at=None,
                melody_seed_shift=0):
    """`drop_at`: where bar 1 of the groove lands (the hook plays a riser before it).
    `stops`: times (s) where the band cuts out for a beat (comic pause)."""
    beat = 60.0 / bpm
    bar = 4 * beat
    mix = Mix(duration)
    mix.add(max(drop_at - 2.4, 0), riser(min(2.4, drop_at)), gain=0.8)

    nbars = int((duration - drop_at) / bar) + 1
    for b in range(nbars):
        t0 = drop_at + b * bar
        root, triad = PROG[(b + melody_seed_shift) % 4]
        last = outro_at is not None and t0 >= outro_at
        mix.add(t0, pad([midi(n) for n in triad], bar * 0.98), gain=1.0)
        for s in range(8):  # eighth notes
            ts = t0 + s * beat / 2
            if any(abs(ts - st) < beat * 0.55 for st in stops):
                continue
            if s % 2 == 0:
                mix.add(ts, kick(), gain=1.0 if not last else 0.7)
            if s in (2, 6):
                mix.add(ts, clap(), pan=0.1)
            mix.add(ts, hat(vel=0.9 if s % 2 else 0.5, open_=(s == 7)), pan=0.35)
            mix.add(ts + beat / 4, shaker(), pan=-0.4)
            # syncopated bass: root on 1, octave pops on the "and"s
            if s in (0, 3, 5):
                f = midi(root) * (2 if s == 5 else 1)
                mix.add(ts, bass(f, beat * (0.9 if s == 0 else 0.4)))
            # marimba comping on off-beats
            if s % 2 == 1:
                for k, n in enumerate(triad):
                    mix.add(ts + k * 0.008, marimba(midi(n), beat * 0.4), pan=-0.3 + 0.3 * k, gain=0.55)
        mel = MELODY_A if (b // 2) % 2 == 0 else MELODY_B
        half = (b % 2) * 8
        if b >= 1 and not last:
            for s in range(8):
                n = mel[half + s]
                ts = t0 + s * beat / 2
                if n is not None and not any(abs(ts - st) < beat * 0.55 for st in stops):
                    mix.add(ts, steelpan(midi(n), beat / 2), pan=0.15, gain=0.9)
                    mix.add(ts + beat * 0.75, steelpan(midi(n), beat / 2), pan=-0.5, gain=0.18)  # echo
    return mix


def write_wav(path, y):
    import wave

    y16 = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y16.tobytes())


if __name__ == "__main__":
    import sys

    m = beach_track(22.0)
    write_wav(sys.argv[1] if len(sys.argv) > 1 else "beach_track.wav", m.master())

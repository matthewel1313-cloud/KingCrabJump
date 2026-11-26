#!/usr/bin/env python3
"""Cut a finished video from shots: render in parallel, caption, score, mux.

    python tools/edit.py volleyball            # -> out/volleyball_vertical.mp4
    python tools/edit.py volleyball --preview   # quarter resolution, fast

A video spec (see `videos.py`) lists shots (render.py shot dicts), captions on
the global timeline, flashes, sound effects and music settings.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = Path(os.environ.get("OUT_DIR", ROOT / "out"))

FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
FPS = 30


# ── timing helpers ────────────────────────────────────────────────────────

def segments(shot):
    return shot.get("timeline") or [[shot["t0"], shot["t1"], shot.get("speed", 1.0)]]


def shot_frames(shot):
    return sum(max(1, int(round((b - a) / s * FPS))) for a, b, s in segments(shot))


def shot_duration(shot):
    return shot_frames(shot) / FPS


def sim_to_out(shot, t_sim):
    """Output seconds (from shot start) at which simulated time t_sim is shown."""
    acc = 0.0
    for a, b, s in segments(shot):
        if t_sim <= b:
            return acc + max(t_sim - a, 0) / s
        acc += max(1, int(round((b - a) / s * FPS))) / FPS
    return acc


# ── captions ──────────────────────────────────────────────────────────────

def _is_emoji(ch):
    o = ord(ch)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or 0x2B00 <= o <= 0x2BFF


_font_cache = {}


def _font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def render_text(text, size, fill=(255, 255, 255), stroke=(15, 15, 25), sw=None, box=None):
    """One line of text (with colour emoji) as an RGBA image."""
    sw = sw if sw is not None else max(3, size // 9)
    font = _font(FONT, size)
    efont = _font(EMOJI, 109)
    runs, cur, cur_e = [], "", None
    for ch in text:
        if ch == "️":
            continue
        e = _is_emoji(ch)
        if cur and e != cur_e:
            runs.append((cur, cur_e))
            cur = ""
        cur, cur_e = cur + ch, e
    if cur:
        runs.append((cur, cur_e))
    pieces = []
    for s, e in runs:
        if e:
            for ch in s:
                im = Image.new("RGBA", (136, 128), (0, 0, 0, 0))
                ImageDraw.Draw(im).text((0, 0), ch, font=efont, embedded_color=True)
                bb = im.getbbox() or (0, 0, 1, 1)
                im = im.crop(bb)
                h = int(size * 1.0)
                im = im.resize((max(1, int(im.width * h / im.height)), h), Image.LANCZOS)
                pieces.append((im, int(size * 0.08)))
        else:
            bb = font.getbbox(s, stroke_width=sw)
            im = Image.new("RGBA", (bb[2] - bb[0] + 4, int(size * 1.25) + 2 * sw), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((-bb[0] + 2, sw), s, font=font, fill=fill,
                                    stroke_width=sw, stroke_fill=stroke)
            pieces.append((im, 0))
    pad = int(size * 0.12)
    W = sum(p.width for p, _ in pieces) + pad * (len(pieces) - 1)
    H = max(p.height for p, _ in pieces)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    x = 0
    for p, dy in pieces:
        out.alpha_composite(p, (x, (H - p.height) // 2 + dy))
        x += p.width + pad
    if box:
        bx = Image.new("RGBA", (W + 2 * pad * 3, H + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(bx).rounded_rectangle((0, 0, bx.width - 1, bx.height - 1),
                                             radius=int(size * 0.3), fill=box)
        bx.alpha_composite(out, (pad * 3, pad))
        out = bx
    return out


_text_cache = {}


def composite_captions(frame, t, captions, scale=1.0):
    active = [c for c in captions if c["t"] - 0.01 <= t < c["t"] + c["dur"]]
    if not active:
        return frame
    img = Image.fromarray(frame).convert("RGBA")
    W, H = img.size
    for c in active:
        lines = c["text"].split("\n")
        size = int(c.get("size", 92) * scale)
        u = t - c["t"]
        # pop in with overshoot, fade out
        if c.get("anim", "pop") == "pop":
            k = min(u / 0.2, 1.0)
            s = 0.55 + 0.45 * (1 + 1.6 * math.sin(math.pi * k) * (1 - k)) if k < 1 else 1.0
        else:
            s = 1.0
        a = min(1.0, (c["t"] + c["dur"] - t) / 0.15, u / 0.06 if c.get("anim") == "fade" else 1.0)
        y = c.get("y", 0.18) * H
        for li, line in enumerate(lines):
            key = (line, size, c.get("color", "w"), c.get("box"))
            if key not in _text_cache:
                fill = {"w": (255, 255, 255), "y": (255, 214, 0), "c": (90, 230, 255)}[c.get("color", "w")]
                _text_cache[key] = render_text(line, size, fill=fill, box=c.get("box"))
            im = _text_cache[key]
            if abs(s - 1) > 1e-3:
                im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
            if a < 1:
                im = im.copy()
                im.putalpha(Image.fromarray((np.asarray(im)[:, :, 3] * max(a, 0)).astype(np.uint8)))
            ly = y + li * size * 1.22 - im.height / 2
            img.alpha_composite(im, (int(W / 2 - im.width / 2), int(ly)))
    return np.asarray(img.convert("RGB"))


def apply_flash(frame, t, flashes):
    for f in flashes:
        if f["t"] <= t < f["t"] + f["dur"]:
            k = 1 - (t - f["t"]) / f["dur"]
            frame = (frame.astype(np.float32) * (1 - k * f.get("amount", 0.8))
                     + 255 * k * f.get("amount", 0.8)).clip(0, 255).astype(np.uint8)
    return frame


# ── rendering a shot (worker) ────────────────────────────────────────────

def _render_shot(job):
    idx, shot, start, size, captions, flashes, path, cap_scale = job
    import render
    import imageio_ffmpeg

    W, H = size
    film = render.Film(W, H)
    wr = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, quality=None,
                                     output_params=["-crf", "16", "-preset", "medium",
                                                    "-pix_fmt", "yuv420p"],
                                     macro_block_size=8)
    wr.send(None)
    for i, fr in enumerate(film.frames(shot, fps=FPS)):
        t = start + i / FPS
        fr = apply_flash(fr, t, flashes)
        fr = composite_captions(fr, t, captions, cap_scale)
        wr.send(np.ascontiguousarray(fr))
    wr.close()
    return idx


# ── audio ────────────────────────────────────────────────────────────────

def build_audio(spec, total, path):
    import music

    m = music.beach_track(duration=total, **spec.get("music", {}))
    for ev in spec.get("sfx", []):
        fn = getattr(music, ev["fx"])
        m.add(ev["t"], fn(**ev.get("args", {})), pan=ev.get("pan", 0.0), gain=ev.get("gain", 1.0))
    y = m.master()
    music.write_wav(path, y)


# ── main ─────────────────────────────────────────────────────────────────

def build(name, preview=False, aspect="vertical", only=None):
    import videos

    spec = videos.get(name, aspect)
    size = spec["size"]
    cap_scale = min(size) / 1080  # captions sized to the short side of the frame
    if preview:
        size = (size[0] // 2 // 8 * 8, size[1] // 2 // 8 * 8)
        cap_scale /= 2
    shots = spec["shots"]
    starts, acc = [], 0.0
    for s in shots:
        starts.append(acc)
        acc += shot_duration(s)
    total = acc
    print(f"[edit] {name}/{aspect}: {len(shots)} shots, {total:.2f}s at {size[0]}x{size[1]}")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"edit_{name}_"))
    jobs = [(i, s, starts[i], size, spec.get("captions", []), spec.get("flashes", []),
             tmp / f"shot_{i:02d}.mp4", cap_scale) for i, s in enumerate(shots)
            if only is None or i in only]
    order = sorted(jobs, key=lambda j: -shot_frames(j[1]))  # longest first
    with get_context("spawn").Pool(min(4, len(jobs))) as pool:
        for idx in pool.imap_unordered(_render_shot, order):
            print(f"[edit]   shot {idx} done")
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{tmp / f'shot_{i:02d}.mp4'}'\n" for i in range(len(shots))))
    video = tmp / "video.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", str(video)], check=True)
    wav = tmp / "audio.wav"
    build_audio(spec, total, wav)
    suffix = "_preview" if preview else ""
    final = OUT / f"{name}_{aspect}{suffix}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-i", str(wav),
                    "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart",
                    str(final)], check=True)
    print(f"[edit] wrote {final} ({total:.2f}s)")
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--aspect", default="vertical", choices=["vertical", "horizontal"])
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    build(a.name, a.preview, a.aspect)

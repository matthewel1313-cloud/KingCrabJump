"""Captions, end card, soundtrack and the final H.264/AAC mux.

    python compose.py --run build/run --frames build/frames --out out_final/king_crab_purpose.mp4

Caption timing is not typed in by hand: every caption and every hand-off "tap"
is placed from the sequencer's markers (sim seconds), mapped through the same
edit decision list the renderer used, so text lands on the action it names.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from render import FPS, edl  # noqa: E402

BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
GOLD = (255, 205, 90)


# ------------------------------------------------------------- time mapping

class Clock:
    def __init__(self, markers):
        self.M = markers
        self.shots = []
        t = 0.0
        for name, cam, a, b, dur, _, _ in edl(markers):
            n = int(round(dur * FPS))
            self.shots.append((name, t, t + n / FPS, a, b))
            t += n / FPS
        self.duration = t

    def shot(self, name):
        for s in self.shots:
            if s[0] == name:
                return s[1], s[2]
        raise KeyError(name)

    def video(self, sim_t, prefer=None):
        """Video time at which sim time `sim_t` is shown (first shot containing it)."""
        for name, v0, v1, a, b in self.shots:
            if prefer and name != prefer:
                continue
            if a <= sim_t <= b:
                return v0 + (sim_t - a) / (b - a) * (v1 - v0)
        return None


# ------------------------------------------------------------------ text

_emoji_cache = {}


def emoji_img(ch, size):
    key = (ch, size)
    if key not in _emoji_cache:
        f = ImageFont.truetype(EMOJI, 109)
        im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((8, 8), ch, font=f, embedded_color=True)
        im = im.crop(im.getbbox())
        s = size / max(im.size)
        _emoji_cache[key] = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    return _emoji_cache[key]


def is_emoji(ch):
    o = ord(ch)
    return o >= 0x1F300 or 0x2600 <= o <= 0x27BF


def runs(line):
    out, buf = [], ""
    for ch in line:
        if ch == "️":
            continue
        if is_emoji(ch):
            if buf:
                out.append(("t", buf))
                buf = ""
            out.append(("e", ch))
        else:
            buf += ch
    if buf:
        out.append(("t", buf))
    return out


def draw_line(canvas, line, cx, cy, size, fill=(255, 255, 255), stroke=None, alpha=1.0, scale=1.0):
    """Centered text line with black outline and inline color emoji."""
    size = max(8, int(size * scale))
    stroke = stroke if stroke is not None else max(2, size // 9)
    font = ImageFont.truetype(BOLD, size)
    parts = runs(line)
    widths = []
    for kind, s in parts:
        if kind == "t":
            widths.append(font.getlength(s) + 2 * stroke)
        else:
            widths.append(emoji_img(s, int(size * 1.05)).width + size * 0.12)
    total = sum(widths)
    x = cx - total / 2
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for (kind, s), w in zip(parts, widths):
        if kind == "t":
            col = fill
            d.text((x + stroke, cy), s, font=font, fill=col + (255,), anchor="lm",
                   stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        else:
            em = emoji_img(s, int(size * 1.05))
            layer.alpha_composite(em, (int(x + size * 0.06), int(cy - em.height / 2)))
        x += w
    if alpha < 1.0:
        a = layer.getchannel("A").point(lambda v: int(v * alpha))
        layer.putalpha(a)
    canvas.alpha_composite(layer)


def pop(t, t0, t1, rise=0.14, fall=0.12):
    """(alpha, scale) for a caption shown t0..t1."""
    if t < t0 or t > t1:
        return 0.0, 1.0
    u = (t - t0) / rise
    a = min(1.0, u) * min(1.0, (t1 - t) / fall)
    s = 1.0 + 0.18 * math.exp(-u * 3.2) * math.cos(u * 4.0) if u < 2.5 else 1.0
    s = 0.82 + 0.18 * min(1.0, u) if u < 1 else s
    return max(0.0, a), s


# ------------------------------------------------------------------ plan

def plan(markers):
    C = Clock(markers)
    vt = C.video
    caps = []
    caps.append(("what is my purpose?", 0.25, 1.62, {}))
    caps.append(("you pass butter.", 1.66, C.shot("intro")[1] - 0.02, {"fill": (255, 236, 160)}))
    oh = vt(markers["handoff_butter_dad"], "butter_pass")
    caps.append(("oh.", oh + 0.12, C.shot("butter_pass")[1] - 0.02, {"size": 120}))
    for name, text in (("rolls", "rolls?"), ("gravy", "gravy?"), ("cranberry", "cranberry?")):
        a, b = C.shot(name)
        if name == "cranberry":
            b = C.shot("tired")[1]
        caps.append((text, a + 0.03, b - 0.02, {"size": 110}))
    a, _ = C.shot("phone_grab")
    _, b = C.shot("phone_pass")
    caps.append(("grandma can't reach\nher phone", a + 0.1, b - 0.05, {}))
    a, _ = C.shot("drawing_in")
    _, b = C.shot("drawing_pass")
    caps.append(("kid made a drawing", a + 0.1, b - 0.05, {}))
    a, _ = C.shot("napkin_grab")
    _, b = C.shot("napkin_pass")
    caps.append(("someone needs\na napkin", a + 0.1, b - 0.05, {}))
    a, b = C.shot("wide")
    caps.append(("my purpose…", a + 0.5, b - 0.05, {"size": 96}))
    caps.append(("is the table. 🫶", a + 1.6, b - 0.05, {"size": 96, "dy": 138}))
    d0, d1 = C.shot("dance")
    end_card = (d0 + 0.55, C.duration)
    taps = [vt(markers[k]) for k in markers if k.startswith("handoff_")]
    taps = [t for t in taps if t is not None]
    cues = {"duration": C.duration, "turn": C.shot("phone_grab")[0], "rapid": C.shot("rolls")[0],
            "sparkle": d0 + 0.3, "final_chord": d0 + 0.45, "taps": sorted(taps)}
    return C, caps, end_card, cues


def overlay(img, t, caps, end_card, W, H):
    canvas = img.convert("RGBA")
    k = W / 1080.0
    # captions: upper third safe zone
    for text, t0, t1, opt in caps:
        a, s = pop(t, t0, t1)
        if a <= 0:
            continue
        size = opt.get("size", 84) * k
        lines = text.split("\n")
        y0 = H * 0.215 + opt.get("dy", 0) * k
        for i, line in enumerate(lines):
            draw_line(canvas, line, W / 2, y0 + i * size * 1.22, size, fill=opt.get("fill", (255, 255, 255)),
                      alpha=a, scale=s)
    # end card
    e0, e1 = end_card
    if t >= e0:
        u = min(1.0, (t - e0) / 0.35)
        shade = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        top, bot = int(H * 0.60), int(H * 0.86)
        for y in range(top - 120, bot + 120, 4):
            edge = min(1.0, (y - (top - 120)) / 120, ((bot + 120) - y) / 120)
            sd.rectangle([0, y, W, y + 4], fill=(18, 8, 4, int(150 * u * max(0, edge))))
        canvas.alpha_composite(shade)
        a1, s1 = pop(t, e0, e1 + 1, rise=0.2)
        draw_line(canvas, "Happy Thanksgiving", W / 2, H * 0.655, 92 * k, alpha=a1, scale=s1)
        a2, s2 = pop(t, e0 + 0.25, e1 + 1, rise=0.2)
        draw_line(canvas, "from King Crab 👑🦀", W / 2, H * 0.715, 80 * k, alpha=a2, scale=s2)
        a3, s3 = pop(t, e0 + 0.55, e1 + 1, rise=0.2)
        draw_line(canvas, "#Jumper", W / 2, H * 0.79, 76 * k, fill=GOLD, alpha=a3, scale=s3)
    # the small line at the bottom, always on
    draw_line(canvas, "real physics sim · open-source Jumper robot", W / 2, H - 205 * k, 34 * k,
              fill=(255, 255, 255), stroke=max(2, int(3 * k)), alpha=0.88)
    return canvas.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--preview-only", action="store_true")
    a = ap.parse_args()
    meta = json.loads((a.run / "run.json").read_text())
    C, caps, end_card, cues = plan(meta["markers"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    work = a.out.parent / "_work"
    work.mkdir(exist_ok=True)
    (work / "cues.json").write_text(json.dumps(cues, indent=1))
    (work / "captions.json").write_text(json.dumps(
        [{"text": c[0], "in": round(c[1], 3), "out": round(c[2], 3)} for c in caps] +
        [{"text": "END CARD", "in": round(end_card[0], 3), "out": round(end_card[1], 3)}], indent=1))
    subprocess.run([sys.executable, str(HERE / "audio.py"), "--cues", str(work / "cues.json"),
                    "--out", str(work / "soundtrack.wav")], check=True)
    frames = sorted(a.frames.glob("f*.jpg"))
    n = int(round(C.duration * FPS))
    if len(frames) != n:
        raise SystemExit(f"expected {n} frames, found {len(frames)}")
    W, H = Image.open(frames[0]).size
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", str(work / "soundtrack.wav"),
           "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-level", "4.2", "-movflags", "+faststart",
           # the mix is already ~-14.5 LUFS; the limiter leaves headroom for the AAC
           # encoder's overshoot on the plucks' sharp attacks (measured: TP -2.0 dBFS)
           "-af", "lowpass=f=15000:poles=2,alimiter=limit=0.6:attack=2:release=50:level=false", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", str(a.out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        t = (i + 0.5) / FPS
        img = overlay(Image.open(f), t, caps, end_card, W, H)
        p.stdin.write(np.asarray(img).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

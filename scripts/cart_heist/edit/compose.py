"""Captions, watermark, end card, soundtrack and the final H.264/AAC mux.

    python compose.py --run build/run --frames build/frames --out ../../renders/cart_heist.mp4

Nothing is hand-timed: every caption and sound cue is placed from the sequencer's markers (sim
seconds), mapped through the same edit decision list the renderer used, so text and sound land
on the action they belong to.
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
from render import FPS, video_times  # noqa: E402

BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
GOLD = (255, 205, 60)


class Clock:
    def __init__(self, markers):
        self.M = markers
        self.shots = video_times(markers)
        self.duration = max(v["end"] for v in self.shots.values())

    def shot(self, name):
        v = self.shots[name]
        return v["start"], v["end"]

    def video(self, sim_t, shot):
        v = self.shots[shot]
        a, b = v["sim"]
        if not (a - 1e-6 <= sim_t <= b + 1e-6):
            raise ValueError(f"sim {sim_t:.2f} not inside shot {shot} {a:.2f}..{b:.2f}")
        return v["start"] + (sim_t - a) / (b - a) * (v["end"] - v["start"])


# ------------------------------------------------------------------ text

_emoji = {}


def emoji_img(ch, size):
    key = (ch, size)
    if key not in _emoji:
        f = ImageFont.truetype(EMOJI, 109)
        im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((8, 8), ch, font=f, embedded_color=True)
        im = im.crop(im.getbbox())
        s = size / max(im.size)
        _emoji[key] = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    return _emoji[key]


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
    """Centred text with a black outline and inline colour emoji."""
    size = max(8, int(size * scale))
    stroke = stroke if stroke is not None else max(2, size // 9)
    font = ImageFont.truetype(BOLD, size)
    parts = runs(line)
    widths = [font.getlength(s) + 2 * stroke if k == "t" else emoji_img(s, int(size * 1.05)).width + size * 0.14
              for k, s in parts]
    x = cx - sum(widths) / 2
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for (kind, s), w in zip(parts, widths):
        if kind == "t":
            d.text((x + stroke, cy), s, font=font, fill=fill + (255,), anchor="lm", stroke_width=stroke,
                   stroke_fill=(0, 0, 0, 255))
        else:
            em = emoji_img(s, int(size * 1.05))
            layer.alpha_composite(em, (int(x + size * 0.07), int(cy - em.height / 2)))
        x += w
    if alpha < 1.0:
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * alpha)))
    canvas.alpha_composite(layer)


def pop(t, t0, t1, rise=0.14, fall=0.12):
    """(alpha, scale) for an overlay shown t0..t1, with a little pop on entry."""
    if t < t0 or t > t1:
        return 0.0, 1.0
    u = (t - t0) / rise
    a = min(1.0, u) * min(1.0, (t1 - t) / fall)
    s = 0.82 + 0.18 * u if u < 1 else (1.0 + 0.10 * math.exp(-(u - 1) * 3.0) * math.cos((u - 1) * 4.0))
    return max(0.0, a), s


# ------------------------------------------------------------------ plan

def plan(markers):
    C = Clock(markers)
    M = markers
    caps = []
    h0, h1 = C.shot("hook")
    caps.append(("Black Friday be like…", h0 + 0.12, h1 - 0.04, {}))
    e0, e1 = C.shot("escape")
    caps.append(("they didn't notice 👀", e0 + 0.45, e1 - 0.05, {}))
    p0, _ = C.shot("place")
    _, b1 = C.shot("bow")
    caps.append(("Meanwhile, I'm a gentleman 👑", p0 + 0.35, b1 - 0.05, {"size": 64}))
    n0, n1 = C.shot("notice")
    turn = C.video(M["bow_on"] + 1.0, "notice")          # the figures' "notice" cue (sequencer)
    double_take = C.video(M["bow_on"] + 2.4, "notice")
    caps.append(("wait…", turn + 0.05, n1 - 0.04, {"size": 120}))
    s0, s1 = C.shot("salute")
    end_card = (s0 - 0.45, C.duration)
    cues = {
        "duration": C.duration,
        "sneak": C.shot("sneak")[0],
        "pinch_shot": C.shot("pinch")[0],
        "escape": e0,
        "place": p0,
        "snaps": [C.video(M["snap0"], "claw_up"), C.video(M["snap1"], "claw_up")],
        "pinch": C.video(M["pinch"], "pinch"),
        "lift": C.video(M["pinch"] + 0.25, "pinch"),
        "release": C.video(M["release"], "place"),
        "ding": C.video(M["landed"], "place"),
        "bow": C.shot("bow")[0],
        "notice": n0,
        "turn": turn,
        "double_take": double_take,
        "dance": C.shot("dance")[0],
        "tada": end_card[0],
    }
    return C, caps, end_card, cues


def overlay(img, t, caps, end_card, W, H):
    canvas = img.convert("RGBA")
    k = W / 1080.0
    for text, t0, t1, opt in caps:
        a, s = pop(t, t0, t1)
        if a <= 0:
            continue
        size = opt.get("size", 84) * k
        for i, line in enumerate(text.split("\n")):
            draw_line(canvas, line, W / 2, H * 0.205 + i * size * 1.22, size, fill=opt.get("fill", (255, 255, 255)),
                      alpha=a, scale=s)
    e0, e1 = end_card
    if t >= e0:
        u = min(1.0, (t - e0) / 0.35)
        shade = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        top, bot = int(H * 0.11), int(H * 0.36)
        for y in range(top - 120, bot + 120, 4):
            edge = min(1.0, (y - (top - 120)) / 120, ((bot + 120) - y) / 120)
            sd.rectangle([0, y, W, y + 4], fill=(10, 6, 4, int(160 * u * max(0, edge))))
        canvas.alpha_composite(shade)
        a1, s1 = pop(t, e0, e1 + 1, rise=0.2)
        draw_line(canvas, "Happy Black Friday", W / 2, H * 0.165, 94 * k, alpha=a1, scale=s1)
        a2, s2 = pop(t, e0 + 0.2, e1 + 1, rise=0.2)
        draw_line(canvas, "from King Crab 👑🦀", W / 2, H * 0.228, 82 * k, alpha=a2, scale=s2)
        a3, s3 = pop(t, e0 + 0.45, e1 + 1, rise=0.2)
        draw_line(canvas, "#Jumper", W / 2, H * 0.300, 78 * k, fill=GOLD, alpha=a3, scale=s3)
    draw_line(canvas, "real physics sim · open-source Jumper robot", W / 2, H - 205 * k, 34 * k,
              stroke=max(2, int(3 * k)), alpha=0.88)
    return canvas.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--crf", type=int, default=21)
    ap.add_argument("--stills", default=None, help="only write overlay stills at these video times")
    a = ap.parse_args()
    meta = json.loads((a.run / "run.json").read_text())
    C, caps, end_card, cues = plan(meta["markers"])
    work = a.work or (a.out.parent / "_work")
    work.mkdir(parents=True, exist_ok=True)
    (work / "cues.json").write_text(json.dumps(cues, indent=1))
    (work / "captions.json").write_text(json.dumps(
        [{"text": c[0], "in": round(c[1], 3), "out": round(c[2], 3)} for c in caps] +
        [{"text": "END CARD: Happy Black Friday / from King Crab 👑🦀 / #Jumper", "in": round(end_card[0], 3),
          "out": round(end_card[1], 3)}], indent=1, ensure_ascii=False))
    frames = sorted(a.frames.glob("f*.jpg"))
    if a.stills:
        for s in a.stills.split(","):
            i = min(len(frames) - 1, int(float(s) * FPS))
            img = overlay(Image.open(frames[i]), (i + 0.5) / FPS, caps, end_card, *Image.open(frames[i]).size)
            img.save(work / f"overlay_{float(s):05.2f}.jpg", quality=90)
        return
    subprocess.run([sys.executable, str(HERE / "audio.py"), "--cues", str(work / "cues.json"),
                    "--out", str(work / "soundtrack.wav")], check=True)
    n = int(round(C.duration * FPS))
    if len(frames) != n:
        raise SystemExit(f"expected {n} frames, found {len(frames)}")
    W, H = Image.open(frames[0]).size
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", str(work / "soundtrack.wav"),
           "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-level", "4.2", "-movflags", "+faststart",
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.66:attack=2:release=60:level=false", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-shortest", str(a.out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        img = overlay(Image.open(f), (i + 0.5) / FPS, caps, end_card, W, H)
        p.stdin.write(np.asarray(img).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

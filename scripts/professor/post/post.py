"""Post-production: overlays, captions and audio onto the raw render, encoded for vertical posts.

    python post/post.py --raw render/out/raw_main.mp4 --out output/king_crab_main.mp4 --cut main
    python post/post.py --raw render/out/raw_main.mp4 --out output/king_crab_alt_F.mp4 --cut alt_f

Overlays are anchored to the simulation: `<raw>.json` (written by render_shots.py) holds, per
frame, the projected crown position and every paper's projected corners. Marks and grades are
drawn in the paper's texture space (900 x 1200, the same layout as scene/assets/midterm.png)
and warped onto the sheet with a perspective transform, so they sit on the paper as it moves.

Audio: `--music FILE` mixes your own track under the effects; without it the original score
from post/audio.py is used. Stems are written beside the output so the music can be swapped.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "motion"))
import timeline as T  # noqa: E402  (the beats)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SERIF_BI = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf"
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
PAGE = (900, 1200)           # texture size of the midterm sheet
RED = (214, 28, 38)
GOLD = (246, 196, 52)

# Where things are on the sheet (texture pixels, origin top-left): see scene/make_assets.py.
GRADE_BOX = (750, 180)
ANSWERS = [(800, 324), (800, 456), (800, 588), (800, 720), (800, 852)]   # beside each answer line
VERDICT = ["check", "cross", "cross", "cross", "cross"]

CAPTIONS = [  # (start, end, text) in video seconds; nothing after 26 s: the end card is clean
    (0.15, 6.0, "POV: your professor is a king crab 👑🦀"),
    (6.0, 14.0, "grading midterms…"),
    (17.45, 22.0, "F."),
    (23.0, 26.0, "A+!"),
]
CLEAN_FROM = 26.0
MARKS_START = T.RELEASE + 0.4         # first mark pops once the sheet has settled on the desk
MARK_STEP = 0.2
F_AT = MARKS_START + MARK_STEP * len(ANSWERS) + 0.15
APLUS_AT = 22.6


def ease_out_back(x):
    x = min(max(x, 0.0), 1.0)
    c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def font(path, size):
    return ImageFont.truetype(path, size)


# ---------- drawing in the sheet's texture space ----------------------------------------------

def draw_check(d, x, y, s, color=RED, w=20):
    d.line([(x - 0.5 * s, y), (x - 0.15 * s, y + 0.4 * s), (x + 0.6 * s, y - 0.55 * s)],
           fill=color + (255,), width=w, joint="curve")


def draw_cross(d, x, y, s, color=RED, w=20):
    d.line([(x - 0.45 * s, y - 0.45 * s), (x + 0.45 * s, y + 0.45 * s)], fill=color + (255,), width=w)
    d.line([(x + 0.45 * s, y - 0.45 * s), (x - 0.45 * s, y + 0.45 * s)], fill=color + (255,), width=w)


def ink_layer(t, grade, start, items):
    """The red-pen layer for a sheet at time t: marks popping in, then the grade."""
    layer = Image.new("RGBA", PAGE, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for k, ((x, y), kind) in enumerate(items):
        k_t = (t - (start + MARK_STEP * k)) / 0.22
        if k_t <= 0:
            continue
        s = 105 * ease_out_back(k_t)
        (draw_check if kind == "check" else draw_cross)(d, x, y, s)
    if grade is not None:
        g_text, g_at, g_color = grade
        k_t = (t - g_at) / 0.25
        if k_t > 0:
            size = int(150 * ease_out_back(k_t))
            if size > 4:
                d.text(GRADE_BOX, g_text, font=font(SERIF_BI, size), fill=g_color + (255,), anchor="mm")
                r = int(size * 0.62)
                d.ellipse([GRADE_BOX[0] - r - 20, GRADE_BOX[1] - r, GRADE_BOX[0] + r + 20, GRADE_BOX[1] + r],
                          outline=g_color + (230,), width=10)
    return layer


def warp_onto(frame, layer, corners):
    """Paste a texture-space layer onto the sheet whose projected corners are given.

    Corners come in the paper's local order (-x,-y), (+x,-y), (+x,+y), (-x,+y). The front face's
    texture has its top (v = 1) at -x and its left (u = 0) at -y (scene/make_assets.py), so the
    texture's top-left is the (-x,-y) corner, top-right (-x,+y), bottom-right (+x,+y),
    bottom-left (+x,-y).
    """
    if any(c is None for c in corners):
        return
    tl, tr, br, bl = corners[0][:2], corners[3][:2], corners[2][:2], corners[1][:2]
    dst = [tl, tr, br, bl]
    src = [(0, 0), (PAGE[0], 0), PAGE, (0, PAGE[1])]
    coeffs = _perspective(dst, src)
    W, H = frame.size
    warped = layer.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    frame.alpha_composite(warped)


def _perspective(dst, src):
    """Coefficients mapping output (frame) pixels back to input (layer) pixels."""
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    return np.linalg.solve(np.asarray(A, float), np.asarray(b, float)).tolist()


# ---------- screen-space effects ---------------------------------------------------------------

def scan_line(frame, corners, t, t0, t1):
    """A laser scan sweeping across the held sheet, twice, with a soft trail."""
    if any(c is None for c in corners):
        return
    p = [np.asarray(c[:2]) for c in corners]
    period = (t1 - t0) / 2.5
    k = ((t - t0) % period) / period
    k = 0.5 - 0.5 * math.cos(math.pi * k)
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # sweep from the (-x) edge to the (+x) edge of the sheet
    a0, a1 = p[0], p[3]
    b0, b1 = p[1], p[2]
    for j, alpha in ((0.0, 255), (-0.04, 110), (-0.08, 60), (-0.12, 30)):
        kk = min(max(k + j, 0), 1)
        s, e = a0 + (b0 - a0) * kk, a1 + (b1 - a1) * kk
        d.line([tuple(s), tuple(e)], fill=(60, 255, 200, alpha), width=7 if j == 0 else 14)
    glow = layer.filter(ImageFilter.GaussianBlur(9))
    frame.alpha_composite(glow)
    frame.alpha_composite(layer)
    # a faint tint over the sheet while it is being read
    poly = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    ImageDraw.Draw(poly).polygon([tuple(q) for q in p], fill=(60, 255, 200, 28))
    frame.alpha_composite(poly)


def sparkle(frame, center, t, t0, seed=3, n=7, radius=150, scale=1.0):
    if center is None:
        return
    rng = np.random.default_rng(seed)
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for k in range(n):
        phase = rng.uniform(0, 1); speed = rng.uniform(1.2, 2.2)
        ang = rng.uniform(0, 2 * math.pi); r = rng.uniform(0.25, 1.0) * radius * scale
        tw = math.sin(2 * math.pi * (speed * (t - t0) + phase))
        if tw <= 0:
            continue
        cx = center[0] + r * math.cos(ang); cy = center[1] - abs(r * math.sin(ang)) * 0.8 - 10 * scale
        s = (20 + 40 * tw) * scale
        pts = []
        for q in range(8):
            rr = s if q % 2 == 0 else s * 0.22
            a = math.pi / 4 * q
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.polygon(pts, fill=(255, 245, 190, int(255 * tw)))
    frame.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1.5)))
    frame.alpha_composite(layer)


def wobble_marks(frame, center, t, t0, scale=1.0):
    """Little motion arcs either side of the crown while the head shakes."""
    if center is None:
        return
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    side = 1 if math.sin(2 * math.pi * 0.55 * (t - t0)) > 0 else -1
    for k in range(2):
        r = (60 + 22 * k) * scale
        box = [center[0] - r, center[1] - r * 0.7, center[0] + r, center[1] + r * 0.7]
        start = -60 if side > 0 else 200
        d.arc(box, start, start + 40, fill=(255, 235, 160, 240), width=int(10 * scale))
    frame.alpha_composite(layer)


def stamp(frame, text, t, t_at, color, center, size):
    """A big screen-space grade that slams in and holds."""
    k = (t - t_at) / 0.28
    if k <= 0:
        return
    s = max(int(size * (1.6 - 0.6 * min(ease_out_back(k), 1.15))), 8)
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font(SERIF_BI, s)
    d.text((center[0] + 6, center[1] + 8), text, font=f, fill=(0, 0, 0, 120), anchor="mm")
    d.text(center, text, font=f, fill=color + (255,), anchor="mm", stroke_width=max(s // 22, 3),
           stroke_fill=(255, 255, 255, 255))
    rot = layer.rotate(-8, center=center, resample=Image.BICUBIC)
    frame.alpha_composite(rot)


# ---------- captions ---------------------------------------------------------------------------

_emoji_cache: dict = {}


def emoji_image(ch, size):
    key = (ch, size)
    if key not in _emoji_cache:
        f = ImageFont.truetype(EMOJI, 109)
        im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
        im = im.crop(im.getbbox())
        _emoji_cache[key] = im.resize((int(im.width * size / im.height), size), Image.LANCZOS)
    return _emoji_cache[key]


def is_emoji(ch):
    return ord(ch) >= 0x1F000 or ch in "✨"


def caption(frame, text, t, t0, t1, W, H):
    """Big centred caption, white with a dark outline, word-wrapped; pops in, fades out."""
    k_in = min(max((t - t0) / 0.18, 0), 1)
    k_out = min(max((t1 - t) / 0.18, 0), 1)
    alpha = min(k_in, k_out)
    size = (100 if len(text) > 6 else 220) * W / 1080
    size = int(size * (0.85 + 0.15 * ease_out_back(k_in)))
    f = font(FONT, size)
    words = text.split(" ")
    lines, cur = [], ""
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    for w in words:
        trial = (cur + " " + w).strip()
        if probe.textlength(trial, font=f) > W * 0.84 and cur:
            lines.append(cur); cur = w
        else:
            cur = trial
    lines.append(cur)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    line_h = int(size * 1.18)
    y = int(H * 0.135) - line_h * len(lines) // 2 + line_h // 2
    for line in lines:
        # measure with emoji as boxes of `size`
        parts, width = [], 0
        for ch in line:
            if is_emoji(ch):
                parts.append(("e", ch)); width += size * 1.05
            else:
                if parts and parts[-1][0] == "t":
                    parts[-1] = ("t", parts[-1][1] + ch)
                else:
                    parts.append(("t", ch))
        width += sum(d.textlength(p, font=f) for kind, p in parts if kind == "t")
        x = (W - width) / 2
        for kind, p in parts:
            if kind == "t":
                d.text((x, y), p, font=f, fill=(255, 255, 255, 255), stroke_width=max(size // 12, 4),
                       stroke_fill=(20, 16, 12, 255))
                x += d.textlength(p, font=f)
            else:
                em = emoji_image(p, int(size * 0.95))
                layer.alpha_composite(em, (int(x + size * 0.05), int(y + size * 0.12)))
                x += size * 1.05
        y += line_h
    if alpha < 1:
        a = layer.getchannel("A").point(lambda v: int(v * alpha))
        layer.putalpha(a)
    frame.alpha_composite(layer)


# ---------- the edit ---------------------------------------------------------------------------

def overlay(img, meta, t, W, H):
    frame = img.convert("RGBA")
    if t >= CLEAN_FROM:
        return frame
    scale = W / 1080
    papers = meta["papers"]
    # scan while reading
    if T.READ[0] - 0.1 <= t < T.SHUFFLE[0]:
        scan_line(frame, papers["paper_5"]["corners"], t, T.READ[0] - 0.1, T.SHUFFLE[0])
    # red pen on the placed sheet, then the F
    if t >= MARKS_START - 0.05 and t < CLEAN_FROM:
        ink = ink_layer(t, ("F", F_AT, RED), MARKS_START, list(zip(ANSWERS, VERDICT)))
        p5 = papers["paper_5"]
        if p5["faces_camera"]:
            warp_onto(frame, ink, p5["corners"])
    if F_AT <= t < T.SHAKE[0]:
        c = papers["paper_5"]["center"]
        if c:
            stamp(frame, "F", t, F_AT, RED, (int(c[0] + 0.16 * W), int(c[1] - 0.10 * H)), int(300 * scale))
    if T.SHAKE[0] <= t < T.SHAKE[1]:
        c = meta.get("crown_top")
        wobble_marks(frame, c[:2] if c else None, t, T.SHAKE[0], scale)
    # A+ on the next sheet, crown sparkles
    if t >= APLUS_AT - 0.05:
        p4 = papers["paper_4"]
        ink = ink_layer(t, ("A+", APLUS_AT, (18, 140, 60)), APLUS_AT, [])
        if p4["faces_camera"]:
            warp_onto(frame, ink, p4["corners"])
        c4 = p4["center"]
        if c4:
            stamp(frame, "A+", t, APLUS_AT + 0.25, GOLD, (int(c4[0] + 0.05 * W), int(c4[1] - 0.12 * H)),
                  int(260 * scale))
        c = meta.get("crown_top")
        sparkle(frame, c[:2] if c else None, t, APLUS_AT, n=9, radius=170, scale=scale)
    for t0, t1, text in CAPTIONS:
        if t0 <= t < t1:
            caption(frame, text, t, t0, t1, W, H)
    return frame


def read_frames(path, W, H):
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = W * H * 3
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        yield Image.frombuffer("RGB", (W, H), buf, "raw", "RGB", 0, 1)
    proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cut", choices=["main", "alt_f"], default="main")
    ap.add_argument("--music", type=Path, default=None, help="your own track (wav/mp3); replaces the score")
    ap.add_argument("--preview", type=str, default=None, help="comma list of times: write PNGs only")
    a = ap.parse_args()

    info = json.loads(a.raw.with_suffix(".json").read_text())
    W, H, fps, start = info["width"], info["height"], info["fps"], info["start"]
    metas = info["frames"]
    end = 30.0 if a.cut == "main" else T.SHAKE[1] + 0.2   # the F cut ends after the head shake
    hold = 0.0 if a.cut == "main" else 2.4                 # ...then a clean 2.4 s end frame

    if a.preview:
        want = [float(x) for x in a.preview.split(",")]
        a.out.mkdir(parents=True, exist_ok=True)
        for k, img in enumerate(read_frames(a.raw, W, H)):
            t = start + k / fps
            for w in want:
                if abs(t - w) < 0.5 / fps:
                    overlay(img, metas[k], t, W, H).convert("RGB").save(a.out / f"post_{w:05.2f}.png")
        return

    tmp = a.out.with_suffix(".video.mp4")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high",
         "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    last = None
    count = 0
    for k, img in enumerate(read_frames(a.raw, W, H)):
        t = start + k / fps
        if t >= end:
            break
        last = img
        enc.stdin.write(overlay(img, metas[k], t, W, H).convert("RGB").tobytes())
        count += 1
    if hold > 0 and last is not None:
        # the alt cut's end card: the last frame, clean (no captions or marks), easing in slightly
        for j in range(int(hold * fps)):
            z = 1 + 0.04 * (j / (hold * fps))
            im = last.resize((int(W * z), int(H * z)), Image.LANCZOS)
            ox, oy = (im.width - W) // 2, (im.height - H) // 2
            enc.stdin.write(im.crop((ox, oy, ox + W, oy + H)).tobytes())
            count += 1
    enc.stdin.close(); enc.wait()
    duration = count / fps

    import audio
    stems = audio.build(a.out.parent.parent / "audio", cut=a.cut, duration=duration, music=a.music)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-i", str(stems["mix"]),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-ar", "48000", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", str(a.out)], check=True)
    tmp.unlink()
    for key in ("music", "sfx"):   # lossless stems for remixing; the WAVs are working files
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(stems[key]),
                        str(stems[key].with_suffix(".flac"))], check=True)
    print(f"[post] wrote {a.out} ({duration:.2f} s, {count} frames) with audio {stems['mix'].name}")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()

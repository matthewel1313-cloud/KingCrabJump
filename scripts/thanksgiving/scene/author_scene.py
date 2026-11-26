"""Author the Thanksgiving dinner-table scene as an mjscene `scene.json` (+ textures).

    python author_scene.py OUTDIR

Then jumper-design packages it:

    shellflow export-map OUTDIR/scene.json --default-skin king-crab.skin --output king-crab-table.map

World frame (metres, Z up, +X toward the camera end of the table):

* floor at z=0, table top surface at z=TABLE_Z (0.76);
* the crab works on the table top near the origin, facing +X;
* family members sit along the long sides (+Y is the crab's left = screen right
  when the camera looks down -X), grandma at the far corner;
* every family member is one `hand_*` body: a sleeve, a cuff and a hand built
  from primitives. They are authored `static` and parked outside the table; the
  sequencer turns them into kinematic (mocap) bodies and flies them in.

All textures are generated here procedurally, so the scene has no third-party
assets. The dining-room backdrop is pre-blurred to read as shallow depth of field.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

TABLE_Z = 0.76
TABLE_HX, TABLE_HY, TABLE_HZ = 1.15, 0.52, 0.022
SEED = 1126

# --------------------------------------------------------------------- textures


def _font(size, bold=True):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tex_backdrop(path: Path):
    """A warm dining room at golden hour, then blurred hard (out-of-focus background)."""
    W, H = 2048, 1024
    im = Image.new("RGB", (W, H), (92, 54, 36))
    d = ImageDraw.Draw(im)
    # wallpaper: warm stripes
    for x in range(0, W, 48):
        d.rectangle([x, 0, x + 22, H], fill=(104, 62, 40))
    # wainscot
    d.rectangle([0, int(H * 0.68), W, H], fill=(70, 40, 26))
    d.rectangle([0, int(H * 0.68), W, int(H * 0.68) + 14], fill=(120, 80, 50))
    # big window with low sun
    wx0, wx1, wy0, wy1 = 1180, 1700, 120, 640
    for y in range(wy0, wy1):
        t = (y - wy0) / (wy1 - wy0)
        c = (int(255), int(196 - 60 * t), int(110 - 70 * t))
        d.line([wx0, y, wx1, y], fill=c)
    d.ellipse([1360, 360, 1560, 560], fill=(255, 236, 190))  # the sun, low
    for x in (wx0, (wx0 + wx1) // 2, wx1):
        d.rectangle([x - 10, wy0, x + 10, wy1], fill=(60, 34, 22))
    d.rectangle([wx0, (wy0 + wy1) // 2 - 8, wx1, (wy0 + wy1) // 2 + 8], fill=(60, 34, 22))
    # curtains
    d.rectangle([wx0 - 150, 60, wx0 - 10, H * 0.72], fill=(130, 30, 28))
    d.rectangle([wx1 + 10, 60, wx1 + 150, H * 0.72], fill=(130, 30, 28))
    # framed family pictures
    rnd = random.Random(SEED)
    for (x, y, w, h) in ((180, 200, 220, 160), (460, 240, 140, 180), (680, 180, 260, 190)):
        d.rectangle([x - 14, y - 14, x + w + 14, y + h + 14], fill=(170, 130, 60))
        d.rectangle([x, y, x + w, y + h], fill=tuple(rnd.randint(120, 200) for _ in range(3)))
        for _ in range(3):
            cx, cy = x + rnd.randint(20, w - 20), y + rnd.randint(30, h - 10)
            d.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=(230, 190, 160))
    # sideboard with lamps
    d.rectangle([120, int(H * 0.60), 1000, int(H * 0.86)], fill=(58, 30, 18))
    for lx in (230, 880):
        d.polygon([(lx - 60, 420), (lx + 60, 420), (lx + 40, 330), (lx - 40, 330)], fill=(255, 214, 150))
        d.rectangle([lx - 6, 420, lx + 6, int(H * 0.60)], fill=(150, 110, 60))
    im = im.filter(ImageFilter.GaussianBlur(22))
    # bokeh: soft discs of warm light on top of the blur
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    g = ImageDraw.Draw(glow)
    for _ in range(70):
        x, y, r = rnd.randint(0, W), rnd.randint(40, int(H * 0.7)), rnd.randint(14, 46)
        c = rnd.choice([(255, 190, 100), (255, 160, 70), (255, 220, 150)])
        g.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v * 0.45) for v in c))
    for lx in (230, 880):
        g.ellipse([lx - 120, 260, lx + 120, 500], fill=(150, 110, 60))
    glow = glow.filter(ImageFilter.GaussianBlur(10))
    im = Image.fromarray(__import__("numpy").clip(
        __import__("numpy").asarray(im, dtype="int16") + __import__("numpy").asarray(glow, dtype="int16"),
        0, 255).astype("uint8"))
    im.save(path)


def tex_wood(path: Path, base=(120, 72, 40), seed=3):
    import numpy as np
    W, H = 1024, 1024
    rnd = np.random.default_rng(seed)
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    plank = (np.floor(y * 6) % 2) * 0.06
    grain = 0.10 * np.sin(2 * np.pi * (x * 3 + 8 * y + 0.4 * np.sin(2 * np.pi * x * 2)))
    grain += 0.05 * np.sin(2 * np.pi * (x * 37 + 3 * y))
    noise = rnd.normal(0, 0.03, (H, W))
    v = 1.0 + grain + noise - plank
    seams = (np.abs((y * 6) % 1) < 0.006).astype(float) * 0.35
    v = v - seams
    img = np.clip(np.stack([base[0] * v, base[1] * v, base[2] * v], -1), 0, 255).astype("uint8")
    Image.fromarray(img).filter(ImageFilter.GaussianBlur(0.8)).save(path)


def tex_runner(path: Path):
    W, H = 256, 1024
    im = Image.new("RGB", (W, H), (34, 66, 44))
    d = ImageDraw.Draw(im)
    for yy in range(0, H, 64):
        d.line([0, yy, W, yy], fill=(44, 82, 54), width=6)
        for xx in range(16, W, 48):
            d.polygon([(xx, yy + 22), (xx + 10, yy + 32), (xx, yy + 42), (xx - 10, yy + 32)], fill=(212, 160, 60))
    d.rectangle([0, 0, 18, H], fill=(212, 160, 60))
    d.rectangle([W - 18, 0, W, H], fill=(212, 160, 60))
    im.filter(ImageFilter.GaussianBlur(0.6)).save(path)


def tex_drawing(path: Path):
    """A kid's crayon 'hand turkey' with THANKFUL written underneath."""
    W, H = 512, 512
    im = Image.new("RGB", (W, H), (250, 246, 232))
    d = ImageDraw.Draw(im)
    rnd = random.Random(7)
    cx, cy = 250, 240
    feathers = [(255, 120, 0), (230, 40, 40), (250, 200, 0), (60, 160, 60), (160, 70, 200)]
    for i, col in enumerate(feathers):
        a = math.radians(200 + i * 32)
        ex, ey = cx + 150 * math.cos(a), cy + 150 * math.sin(a)
        for k in range(10):
            jitter = rnd.uniform(-6, 6)
            d.line([cx + jitter, cy, ex + jitter, ey + jitter], fill=col, width=12)
        d.ellipse([ex - 30, ey - 30, ex + 30, ey + 30], fill=col)
    d.ellipse([cx - 80, cy - 70, cx + 80, cy + 90], fill=(140, 82, 40))
    d.ellipse([cx + 20, cy - 110, cx + 110, cy - 20], fill=(150, 90, 45))
    d.ellipse([cx + 70, cy - 82, cx + 84, cy - 68], fill=(20, 20, 20))
    d.polygon([(cx + 108, cy - 66), (cx + 140, cy - 56), (cx + 106, cy - 46)], fill=(250, 170, 0))
    d.line([cx + 100, cy - 44, cx + 96, cy - 10], fill=(220, 30, 30), width=10)
    for lx in (cx - 30, cx + 30):
        d.line([lx, cy + 88, lx, cy + 130], fill=(240, 150, 0), width=8)
    f = _font(64)
    d.text((W // 2, 440), "THANKFUL", font=f, fill=(40, 90, 200), anchor="mm")
    for heart_x in (60, 450):
        d.ellipse([heart_x - 22, 60, heart_x, 82], fill=(230, 50, 80))
        d.ellipse([heart_x, 60, heart_x + 22, 82], fill=(230, 50, 80))
        d.polygon([(heart_x - 22, 74), (heart_x + 22, 74), (heart_x, 104)], fill=(230, 50, 80))
    im.save(path)


def tex_phone(path: Path):
    """Lock screen: warm gradient, a clock and a little family photo."""
    W, H = 256, 512
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    for y in range(H):
        t = y / H
        d.line([0, y, W, y], fill=(int(250 - 90 * t), int(150 - 80 * t), int(90 + 60 * t)))
    d.text((W // 2, 110), "4:12", font=_font(70), fill=(255, 255, 255), anchor="mm")
    d.text((W // 2, 165), "Thursday, November 26", font=_font(16, False), fill=(255, 240, 230), anchor="mm")
    d.rounded_rectangle([40, 240, 216, 400], radius=14, fill=(255, 230, 200))
    for i, c in enumerate([(200, 120, 90), (150, 90, 60), (230, 170, 140)]):
        x = 80 + i * 48
        d.ellipse([x - 20, 280, x + 20, 320], fill=c)
        d.rectangle([x - 22, 322, x + 22, 390], fill=(c[0] - 40, c[1] - 20, c[2]))
    d.rounded_rectangle([90, 470, 166, 478], radius=4, fill=(255, 255, 255))
    im.save(path)


def tex_plaid(path: Path, a, b, c):
    W = 256
    im = Image.new("RGB", (W, W), a)
    d = ImageDraw.Draw(im)
    for k in range(0, W, 64):
        d.rectangle([k, 0, k + 24, W], fill=b)
        d.rectangle([0, k, W, k + 24], fill=b)
        d.rectangle([k + 40, 0, k + 44, W], fill=c)
        d.rectangle([0, k + 40, W, k + 44], fill=c)
    im.save(path)


def tex_knit(path: Path, a, b):
    W = 256
    im = Image.new("RGB", (W, W), a)
    d = ImageDraw.Draw(im)
    for y in range(0, W, 16):
        for x in range(0, W, 16):
            d.arc([x, y, x + 16, y + 16], 200, 340, fill=b, width=3)
    for x in range(0, W, 64):
        d.line([x + 32, 0, x + 32, W], fill=b, width=6)
    im.filter(ImageFilter.GaussianBlur(0.5)).save(path)


def tex_wicker(path: Path):
    W = 256
    im = Image.new("RGB", (W, W), (176, 122, 64))
    d = ImageDraw.Draw(im)
    for y in range(0, W, 16):
        off = 8 if (y // 16) % 2 else 0
        for x in range(-16, W, 16):
            d.ellipse([x + off, y + 2, x + off + 14, y + 14], fill=(150, 98, 48))
    im.save(path)


# ------------------------------------------------------------------- scene JSON

def box(name, size, pos, material=None, rgba=None, kind="static", collision=True, **kw):
    o = {"name": name, "shape": "box", "size": size, "pos": pos, "kind": kind}
    if material:
        o["material"] = material
    if rgba:
        o["rgba"] = rgba
    if not collision:
        o["collision"] = False
    o.update(kw)
    return o


def geom(name, shape, size, pos, material=None, rgba=None, collision=False, **kw):
    o = {"name": name, "shape": shape, "size": size, "pos": pos, "kind": "static"}
    if material:
        o["material"] = material
    if rgba:
        o["rgba"] = rgba
    if not collision:
        o["collision"] = False
    o.update(kw)
    return o


#: The family: sleeve material, skin tone, where each one sits (x on the table,
#: side +1 = +Y / crab's left / screen right), and a hand scale (the kid is small).
FAMILY = {
    "dad":     dict(sleeve="flannel", cuff=[0.10, 0.08, 0.09, 1], skin=[0.86, 0.66, 0.52, 1], x=0.02, side=+1, scale=1.0),
    "mom":     dict(sleeve="sweater", cuff=[0.70, 0.52, 0.18, 1], skin=[0.93, 0.76, 0.64, 1], x=0.04, side=-1, scale=0.92),
    "aunt":    dict(sleeve="olive", cuff=[0.30, 0.34, 0.18, 1], skin=[0.55, 0.38, 0.27, 1], x=0.40, side=+1, scale=0.94),
    "kid":     dict(sleeve="hoodie", cuff=[0.06, 0.38, 0.42, 1], skin=[0.80, 0.58, 0.44, 1], x=0.40, side=-1, scale=0.72),
    "grandma": dict(sleeve="cardigan", cuff=[0.62, 0.42, 0.56, 1], skin=[0.95, 0.82, 0.74, 1], x=-0.55, side=-1, scale=0.9),
}

#: hand local frame: +X from wrist toward the fingertips, +Z out of the palm
#: (palm up when +Z is world up). The forearm runs back along -X from the wrist.


def hand_children(name, f):
    s = f["scale"]
    skin = f["skin"]
    ch = []
    sleeve_mat = "sleeve_" + f["sleeve"]
    # forearm sleeve: a long capsule from the wrist back to off-table
    ch.append(geom(f"{name}_sleeve", "capsule", [0.040 * s], [0, 0, 0], material=sleeve_mat,
                   fromto=[-0.07 * s, 0, -0.012 * s, -0.55, 0, -0.03]))
    ch.append(geom(f"{name}_cuff", "capsule", [0.043 * s], [0, 0, 0], rgba=f["cuff"],
                   fromto=[-0.075 * s, 0, -0.012 * s, -0.10 * s, 0, -0.014 * s]))
    ch.append(geom(f"{name}_wrist", "capsule", [0.026 * s], [0, 0, 0], rgba=skin,
                   fromto=[-0.07 * s, 0, -0.010 * s, -0.005 * s, 0, -0.008 * s]))
    ch.append(geom(f"{name}_palm", "ellipsoid", [0.052 * s, 0.042 * s, 0.014 * s],
                   [0.030 * s, 0, -0.004 * s], rgba=skin))
    # four fingers, gently cupped (last segment tilted up out of the palm)
    for i, (dy, ln) in enumerate(((-0.027, 0.066), (-0.009, 0.074), (0.009, 0.070), (0.026, 0.058))):
        y = dy * s
        x0 = 0.070 * s
        x1 = x0 + ln * 0.55 * s
        x2 = x1 + ln * 0.45 * s * 0.85
        ch.append(geom(f"{name}_f{i}a", "capsule", [0.0085 * s], [0, 0, 0], rgba=skin,
                       fromto=[x0, y, -0.002 * s, x1, y * 1.05, 0.002 * s]))
        ch.append(geom(f"{name}_f{i}b", "capsule", [0.0080 * s], [0, 0, 0], rgba=skin,
                       fromto=[x1, y * 1.05, 0.002 * s, x2, y * 1.08, 0.016 * s]))
    # thumb, out to the side the hand's thumb is on (+Y in hand frame)
    t = -1 if f["side"] > 0 else 1
    ch.append(geom(f"{name}_thumb_a", "capsule", [0.0105 * s], [0, 0, 0], rgba=skin,
                   fromto=[0.012 * s, t * 0.030 * s, -0.002 * s, 0.040 * s, t * 0.058 * s, 0.006 * s]))
    ch.append(geom(f"{name}_thumb_b", "capsule", [0.0095 * s], [0, 0, 0], rgba=skin,
                   fromto=[0.040 * s, t * 0.058 * s, 0.006 * s, 0.068 * s, t * 0.070 * s, 0.016 * s]))
    if name == "grandma":  # a little pearl bracelet
        for k in range(8):
            a = 2 * math.pi * k / 8
            ch.append(geom(f"{name}_pearl{k}", "sphere", [0.0055], [-0.052 * s, 0.028 * s * math.cos(a),
                           -0.010 * s + 0.028 * s * math.sin(a)], rgba=[0.96, 0.94, 0.88, 1]))
    if name == "dad":  # watch
        ch.append(geom(f"{name}_watch", "cylinder", [0.019, 0.007], [-0.045, 0, 0.018],
                       rgba=[0.15, 0.15, 0.17, 1]))
    return ch


def parked_pose(f):
    """Where a hand waits: off the table edge, below the top, out of shot."""
    return [f["x"], f["side"] * (TABLE_HY + 0.45), TABLE_Z - 0.25]


def build(out: Path):
    tex = out / "textures"
    tex.mkdir(parents=True, exist_ok=True)
    tex_backdrop(tex / "backdrop.png")
    tex_wood(tex / "table_wood.png", base=(104, 60, 34), seed=3)
    tex_wood(tex / "floor_wood.png", base=(70, 40, 24), seed=9)
    tex_runner(tex / "runner.png")
    tex_drawing(tex / "drawing.png")
    tex_phone(tex / "phone.png")
    tex_plaid(tex / "flannel.png", (150, 26, 28), (40, 20, 22), (220, 200, 160))
    tex_knit(tex / "sweater.png", (214, 170, 80), (180, 136, 56))
    tex_knit(tex / "olive.png", (98, 110, 58), (78, 88, 44))
    tex_knit(tex / "cardigan.png", (172, 128, 160), (148, 104, 136))
    tex_knit(tex / "hoodie.png", (24, 120, 132), (18, 98, 108))
    tex_wicker(tex / "wicker.png")

    T = "textures/"
    materials = [
        {"name": "table_wood", "texture": {"file": T + "table_wood.png"}, "texrepeat": [2, 1],
         "shininess": 0.55, "specular": 0.35, "reflectance": 0.06},
        {"name": "floor_wood", "texture": {"file": T + "floor_wood.png"}, "texrepeat": [4, 4],
         "texuniform": True, "shininess": 0.3, "specular": 0.2},
        {"name": "backdrop", "texture": {"file": T + "backdrop.png"}, "emission": 0.55, "specular": 0.0,
         "shininess": 0.0},
        {"name": "runner", "texture": {"file": T + "runner.png"}, "texrepeat": [1, 2], "specular": 0.05},
        {"name": "drawing", "texture": {"file": T + "drawing.png"}, "specular": 0.05},
        {"name": "phone_screen", "texture": {"file": T + "phone.png"}, "emission": 0.6, "specular": 0.5},
        {"name": "sleeve_flannel", "texture": {"file": T + "flannel.png"}, "texrepeat": [3, 3], "specular": 0.05},
        {"name": "sleeve_sweater", "texture": {"file": T + "sweater.png"}, "texrepeat": [3, 3], "specular": 0.05},
        {"name": "sleeve_olive", "texture": {"file": T + "olive.png"}, "texrepeat": [3, 3], "specular": 0.05},
        {"name": "sleeve_cardigan", "texture": {"file": T + "cardigan.png"}, "texrepeat": [3, 3], "specular": 0.05},
        {"name": "sleeve_hoodie", "texture": {"file": T + "hoodie.png"}, "texrepeat": [3, 3], "specular": 0.05},
        {"name": "wicker", "texture": {"file": T + "wicker.png"}, "texrepeat": [2, 2], "specular": 0.1},
        {"name": "china", "rgba": [0.93, 0.93, 0.92, 1], "shininess": 0.8, "specular": 0.6},
        {"name": "gold_trim", "rgba": [0.85, 0.66, 0.25, 1], "shininess": 0.9, "specular": 0.8},
        {"name": "glass", "rgba": [0.85, 0.92, 0.95, 0.30], "shininess": 1.0, "specular": 1.0},
        {"name": "silver", "rgba": [0.78, 0.78, 0.80, 1], "shininess": 0.9, "specular": 0.9},
        {"name": "flame", "rgba": [1.0, 0.78, 0.35, 1], "emission": 1.0},
        {"name": "roast", "rgba": [0.62, 0.32, 0.12, 1], "shininess": 0.7, "specular": 0.6},
        {"name": "jelly", "rgba": [0.62, 0.02, 0.10, 0.92], "shininess": 0.9, "specular": 0.9},
        {"name": "wax", "rgba": [0.96, 0.92, 0.82, 1], "emission": 0.15},
    ]

    Z = TABLE_Z
    objs = []
    # room ---------------------------------------------------------------
    objs.append(box("backdrop_wall", [0.05, 3.2, 1.6], [-2.9, 0, 1.4], material="backdrop", collision=False))
    objs.append(box("front_wall", [0.05, 3.2, 1.6], [2.9, 0, 1.4], material="backdrop", collision=False))
    objs.append(box("side_wall_l", [3.2, 0.05, 1.6], [0, -3.0, 1.4], material="backdrop", collision=False))
    objs.append(box("side_wall_r", [3.2, 0.05, 1.6], [0, 3.0, 1.4], material="backdrop", collision=False))
    # table --------------------------------------------------------------
    objs.append(box("table_top", [TABLE_HX, TABLE_HY, TABLE_HZ], [0, 0, Z - TABLE_HZ], material="table_wood",
                    friction=[1.0, 0.01, 0.001], contype=1, conaffinity=1 | 2048))
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        objs.append(box(f"table_leg{i}", [0.035, 0.035, (Z - 2 * TABLE_HZ) / 2],
                        [sx * (TABLE_HX - 0.08), sy * (TABLE_HY - 0.08), (Z - 2 * TABLE_HZ) / 2],
                        material="table_wood", collision=False))
    objs.append(box("runner", [TABLE_HX - 0.08, 0.17, 0.0008], [0, 0, Z + 0.0008], material="runner",
                    collision=False))
    # place settings at each seat ------------------------------------------
    for who, f in FAMILY.items():
        if who == "grandma":
            px, py = -0.58, -0.42
        else:
            px, py = f["x"] - 0.02, f["side"] * 0.42
        objs.append(geom(f"plate_{who}", "cylinder", [0.105, 0.004], [px, py, Z + 0.004], material="china"))
        objs.append(geom(f"plate_rim_{who}", "cylinder", [0.108, 0.0015], [px, py, Z + 0.0085],
                         material="gold_trim"))
        objs.append(geom(f"plate_in_{who}", "cylinder", [0.084, 0.0015], [px, py, Z + 0.009], material="china"))
        objs.append(geom(f"glass_{who}", "cylinder", [0.028, 0.055], [px - 0.17, py - f["side"] * 0.02, Z + 0.045],
                         material="glass"))
        objs.append(geom(f"cider_{who}", "cylinder", [0.025, 0.030], [px - 0.17, py - f["side"] * 0.02, Z + 0.026],
                         rgba=[0.86, 0.48, 0.12, 0.85]))
        for k, dx in enumerate((-0.135, 0.135)):
            objs.append(box(f"cutlery_{who}_{k}", [0.008, 0.06, 0.0015], [px + dx, py, Z + 0.0015],
                            material="silver", collision=False))
    # centerpiece far down the table: pumpkins, gourds, candles --------------
    for i, (x, y, r, col) in enumerate(((-0.95, 0.05, 0.075, [0.92, 0.45, 0.10, 1]),
                                        (-1.02, -0.10, 0.055, [0.95, 0.62, 0.20, 1]),
                                        (-0.86, -0.06, 0.045, [0.96, 0.90, 0.78, 1]))):
        objs.append(geom(f"pumpkin{i}", "ellipsoid", [r, r, r * 0.72], [x, y, Z + r * 0.70], rgba=col))
        objs.append(geom(f"pumpkin_stem{i}", "cylinder", [0.008, 0.014], [x, y, Z + r * 1.42],
                         rgba=[0.35, 0.25, 0.10, 1]))
    # roast turkey on a platter at the head of the table
    objs.append(geom("platter", "ellipsoid", [0.19, 0.13, 0.012], [-0.55, 0.02, Z + 0.010], material="china"))
    objs.append(geom("turkey", "ellipsoid", [0.12, 0.085, 0.065], [-0.55, 0.02, Z + 0.065],
                     material="roast"))
    for s in (+1, -1):
        objs.append(geom(f"drum{s}", "capsule", [0.026], [0, 0, 0], rgba=[0.58, 0.29, 0.11, 1],
                         fromto=[-0.47, s * 0.07, Z + 0.06, -0.41, s * 0.09, Z + 0.075]))
        objs.append(geom(f"drum_bone{s}", "capsule", [0.008], [0, 0, 0], rgba=[0.95, 0.92, 0.85, 1],
                         fromto=[-0.41, s * 0.09, Z + 0.075, -0.385, s * 0.095, Z + 0.08]))
    candles = [(-0.25, 0.10), (-0.25, -0.12), (0.64, 0.15), (0.64, -0.14), (-0.80, 0.18)]
    for i, (x, y) in enumerate(candles):
        h = 0.09 + 0.02 * (i % 2)
        objs.append(geom(f"candle_holder{i}", "cylinder", [0.026, 0.008], [x, y, Z + 0.008], material="gold_trim"))
        objs.append(geom(f"candle{i}", "cylinder", [0.014, h / 2], [x, y, Z + 0.016 + h / 2], material="wax"))
        objs.append(geom(f"flame{i}", "ellipsoid", [0.006, 0.006, 0.013], [x, y, Z + 0.016 + h + 0.012],
                         material="flame"))
    # gravy & co. -- the props the crab passes (dynamic, light, real-ish size) --
    props = []

    def prop(name, pos, children, mass, friction=(0.6, 0.005, 0.0001)):
        props.append({"name": name, "shape": "none", "pos": pos, "kind": "dynamic", "mass": mass,
                      "children": children, "tags": ["prop"]})

    def part(name, shape, size, pos, mass=None, **kw):
        o = {"name": name, "shape": shape, "size": size, "pos": pos, "kind": "dynamic",
             "friction": [0.6, 0.005, 0.0001], "contype": 2048, "conaffinity": 2048, "condim": 3}
        if mass:
            o["mass"] = mass
        o.update(kw)
        return o

    # positions are each prop's body origin, on the table top (z = TABLE_Z)
    prop("butter", [0.255, 0.035, Z], [
        part("butter_tray", "box", [0.055, 0.032, 0.005], [0, 0, 0.005], mass=0.03, material="china"),
        part("butter_rim", "box", [0.058, 0.035, 0.0015], [0, 0, 0.0095], mass=0.005, material="gold_trim"),
        part("butter_stick", "box", [0.034, 0.014, 0.012], [0, 0, 0.022], mass=0.02, rgba=[0.99, 0.86, 0.42, 1]),
        part("butter_pat", "box", [0.008, 0.006, 0.0025], [0.040, 0.012, 0.0125], mass=0.002,
             rgba=[0.99, 0.86, 0.42, 1]),
    ], 0.05)
    prop("rolls", [0.245, 0.19, Z], [
        part("rolls_basket", "box", [0.050, 0.036, 0.016], [0, 0, 0.016], mass=0.03, material="wicker"),
        part("rolls_cloth", "box", [0.052, 0.038, 0.002], [0, 0, 0.031], mass=0.003, rgba=[0.95, 0.92, 0.85, 1]),
        part("roll_a", "ellipsoid", [0.022, 0.017, 0.013], [-0.020, -0.006, 0.040], mass=0.008,
             rgba=[0.80, 0.52, 0.24, 1]),
        part("roll_b", "ellipsoid", [0.022, 0.017, 0.013], [0.020, 0.008, 0.041], mass=0.008,
             rgba=[0.83, 0.55, 0.26, 1]),
        part("roll_c", "ellipsoid", [0.020, 0.016, 0.012], [0.000, -0.012, 0.050], mass=0.006,
             rgba=[0.78, 0.50, 0.22, 1]),
    ], 0.05)
    prop("gravy", [0.265, -0.12, Z], [
        part("gravy_foot", "cylinder", [0.022, 0.004], [0, 0, 0.004], mass=0.01, material="china"),
        part("gravy_bowl", "ellipsoid", [0.050, 0.028, 0.022], [0, 0, 0.026], mass=0.04, material="china"),
        part("gravy_sauce", "ellipsoid", [0.042, 0.022, 0.004], [0, 0, 0.044], mass=0.002,
             rgba=[0.45, 0.25, 0.10, 1]),
        part("gravy_spout", "capsule", [0.007], [0, 0, 0], mass=0.002, material="china",
             fromto=[0.045, 0, 0.040, 0.064, 0, 0.050]),
        part("gravy_handle", "capsule", [0.0045], [0, 0, 0], mass=0.002, material="china",
             fromto=[-0.046, 0, 0.040, -0.064, 0, 0.034]),
    ], 0.05)
    prop("cranberry", [0.36, 0.14, Z], [
        part("cran_dish", "cylinder", [0.036, 0.011], [0, 0, 0.011], mass=0.04, material="china"),
        part("cran_rim", "cylinder", [0.0375, 0.0015], [0, 0, 0.0215], mass=0.002, material="gold_trim"),
        part("cran_jelly", "cylinder", [0.026, 0.012], [0, 0, 0.030], mass=0.01, material="jelly"),
    ], 0.05)
    prop("phone", [0.40, -0.10, Z], [
        part("phone_body", "box", [0.060, 0.029, 0.0042], [0, 0, 0.0042], mass=0.05,
             rgba=[0.10, 0.10, 0.12, 1]),
        part("phone_screen", "box", [0.056, 0.026, 0.0004], [0, 0, 0.0086], mass=0.001, material="phone_screen"),
    ], 0.05)
    # a folded card standing as a tent: two panels meeting at a ridge
    prop("drawing", [0.40, -0.80, Z + 0.30], [
        part("drawing_panel_a", "box", [0.002, 0.055, 0.030], [0.013, 0, 0.027], mass=0.006, material="drawing",
             euler=[0, -25, 0]),
        part("drawing_panel_b", "box", [0.002, 0.055, 0.030], [-0.013, 0, 0.027], mass=0.006, material="drawing",
             euler=[0, 25, 0]),
    ], 0.012)
    prop("napkin", [0.37, 0.03, Z], [
        part("napkin_fold", "box", [0.040, 0.040, 0.006], [0, 0, 0.006], mass=0.012,
             rgba=[0.96, 0.93, 0.84, 1], solref=[0.05, 1.0]),
        part("napkin_band", "box", [0.008, 0.041, 0.0065], [0, 0, 0.0065], mass=0.002,
             rgba=[0.66, 0.08, 0.12, 1]),
        part("napkin_leaf", "ellipsoid", [0.012, 0.007, 0.002], [0, 0.0, 0.0135], mass=0.001,
             rgba=[0.86, 0.42, 0.08, 1]),
    ], 0.015)
    objs.extend(props)

    # family hands, parked ------------------------------------------------
    for who, f in FAMILY.items():
        p = parked_pose(f)
        objs.append({"name": f"hand_{who}", "shape": "none", "pos": p, "kind": "static",
                     "children": hand_children(who, f), "tags": ["family", f["sleeve"]]})

    scene = {
        "name": "king-crab-table",
        "title": "Thanksgiving Table for King Crab",
        "description": ("A long wooden Thanksgiving dinner table at golden hour with candle glow. "
                        "Seven light dynamic props (butter dish, bread rolls, gravy boat, cranberry dish, "
                        "phone, a kid's folded drawing, a napkin) sit near the robot's spawn on the table top; "
                        "five family members' sleeved hands are parked off the table edges as static bodies "
                        "for a host to animate kinematically. Robot spawn is on the table top."),
        "tags": ["indoor", "tabletop", "thanksgiving", "hand-off"],
        "physics": {"timestep": 0.001, "integrator": "implicitfast", "cone": "pyramidal", "iterations": 10},
        "sky": {"preset": "indoor", "top": [0.20, 0.12, 0.08], "bottom": [0.30, 0.18, 0.10]},
        "lighting": {
            "headlight": False,
            "shadows": True,
            "shadow_quality": "high",
            "ambient": [0.11, 0.095, 0.085],
            "lights": [
                {"name": "golden_sun", "type": "spot", "pos": [-1.6, 2.2, 2.0], "target": [0.1, 0, Z],
                 "diffuse": [0.78, 0.55, 0.33], "specular": [0.35, 0.26, 0.16], "cutoff": 34, "exponent": 3,
                 "castshadow": True},
                {"name": "warm_fill", "type": "spot", "pos": [1.9, -1.0, 1.8], "target": [0, 0, Z],
                 "diffuse": [0.30, 0.27, 0.25], "specular": [0.08, 0.07, 0.06], "cutoff": 50,
                 "castshadow": False},
                {"name": "rim_back", "type": "spot", "pos": [-2.0, -0.4, 1.4], "target": [0, 0, Z + 0.05],
                 "diffuse": [0.40, 0.29, 0.18], "specular": [0.45, 0.32, 0.18], "cutoff": 30,
                 "castshadow": False},
            ] + [
                {"name": f"candle_glow{i}", "type": "point", "pos": [x, y, Z + 0.14],
                 "diffuse": [0.22, 0.13, 0.05], "specular": [0.10, 0.06, 0.02],
                 "attenuation": [0.2, 2.0, 6.0], "castshadow": False}
                for i, (x, y) in enumerate(candles[:4])
            ],
        },
        "ground": {"type": "plane", "size": [4, 4], "material": "floor_wood", "friction": "wood"},
        "materials": materials,
        "objects": objs,
        "cameras": [
            {"name": "closeup", "pos": [0.55, 0.04, Z + 0.10], "target": [0.05, 0.0, Z + 0.07], "fovy": 38},
            {"name": "wide_overhead", "pos": [0.95, 0.0, Z + 1.05], "target": [-0.15, 0.0, Z], "fovy": 52},
        ],
        "spawn": {"position": [0.0, 0.0, Z], "yaw": 0, "clearance": 0.12,
                  "bounds": [-1.0, -0.45, Z, 1.0, 0.45, Z + 0.5]},
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    return scene


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/scene")
    out.mkdir(parents=True, exist_ok=True)
    build(out)
    print("wrote", out / "scene.json")

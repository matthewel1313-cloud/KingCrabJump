"""Author the Black Friday store aisle as an mjscene `scene.json` (+ textures).

    python author_scene.py OUTDIR

jumper-design then packages it:

    shellflow export-map OUTDIR/scene.json --default-skin king-crab-gold-1.skin \
        --output black_friday_store.map

World frame (metres, Z up). The floor is z = 0 and the crab works on it.

* The aisle runs along X. Gondola shelves line both sides (y = +-1.75), with
  "BLACK FRIDAY" header signs and hanging banners overhead.
* **Cart A** stands at the origin, long axis on X, handle at -X. Its low bottom
  rack (top at RACK_Z) holds the deal: a small phone box with a "70% OFF" tag,
  a free rigid body (DEAL_MASS kg) near the rack's -Y edge, where the crab can
  reach it.
* Two adult shopper figures stand at either end of Cart A, facing each other.
  A small shopper (a kid) stands at the checkout lane (x ~ 1.9) with an empty
  basket on the floor. Every figure is a static body here, split into a body
  and a head; the sequencer turns them into kinematic (mocap) bodies for a
  light scripted sway. Nothing about them is simulated.
* Cart B (empty) dresses the far end of the aisle.

Collision layout: the deal box lives on its own collision bit (2048). The cart
rack, the basket and a floor slab carry that bit, so the box rests on them and
falls into the basket under physics. The robot's collision bits never include
2048, so the crab and the cart/basket geometry do not snag on each other.

Every texture is generated here; the scene uses no third-party assets.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SEED = 1128

#: top surface of Cart A's bottom rack (set from the measured claw reach, see NOTES.md)
RACK_Z = 0.060
#: the deal: a phone box, half sizes (m) and mass (kg)
DEAL_HALF = (0.070, 0.040, 0.016)
DEAL_MASS = 0.15
#: where it sits on the rack (centre, world); the crab reaches it from -Y
DEAL_POS = (0.02, -0.150)
#: the kid's basket on the floor at the checkout
BASKET_POS = (-1.90, 0.20)
BASKET_HALF = (0.15, 0.105)
BASKET_H = 0.055
PROP_BIT = 2048

# ---------------------------------------------------------------- fonts / text


def font(size, bold=True):
    for p in ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_c(d, xy, s, f, fill, stroke=0, stroke_fill=None):
    x, y = xy
    b = d.textbbox((0, 0), s, font=f, stroke_width=stroke)
    d.text((x - (b[2] - b[0]) / 2 - b[0], y - (b[3] - b[1]) / 2 - b[1]), s, font=f, fill=fill,
           stroke_width=stroke, stroke_fill=stroke_fill)


def fit_font(d, s, maxw, start):
    size = start
    while size > 8:
        f = font(size)
        b = d.textbbox((0, 0), s, font=f)
        if b[2] - b[0] <= maxw:
            return f
        size -= 4
    return font(8)


# --------------------------------------------------------------------- textures

def tex_banner(path, W=2048, H=512, line1="BLACK FRIDAY", line2="UP TO 70% OFF EVERYTHING"):
    im = Image.new("RGB", (W, H), (12, 10, 12))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 26], fill=(226, 30, 38))
    d.rectangle([0, H - 26, W, H], fill=(226, 30, 38))
    for x in range(-H, W, 90):  # diagonal accent stripes at the ends
        if x < 260 or x > W - 260 - H:
            d.polygon([(x, 26), (x + 40, 26), (x + 40 + H - 52, H - 26), (x + H - 52, H - 26)], fill=(36, 32, 36))
    f1 = fit_font(d, line1, W - 380, 260)
    text_c(d, (W / 2, H * 0.42), line1, f1, (255, 214, 0), stroke=6, stroke_fill=(120, 20, 10))
    f2 = fit_font(d, line2, W - 600, 84)
    text_c(d, (W / 2, H * 0.80), line2, f2, (255, 255, 255))
    im.save(path)


def tex_header(path, s, bg=(226, 30, 38), fg=(255, 255, 255), W=1024, H=256):
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([10, 10, W - 10, H - 10], outline=(255, 214, 0), width=10)
    text_c(d, (W / 2, H / 2), s, fit_font(d, s, W - 80, 150), fg, stroke=3, stroke_fill=(60, 0, 0))
    im.save(path)


def tex_tag(path, W=256, H=256):
    """A yellow starburst price tag: 70% OFF."""
    im = Image.new("RGB", (W, H), (226, 30, 38))
    d = ImageDraw.Draw(im)
    cx, cy, n = W / 2, H / 2, 18
    pts = []
    for i in range(2 * n):
        r = (W * 0.48) if i % 2 == 0 else (W * 0.38)
        a = math.pi * i / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    d.polygon(pts, fill=(255, 214, 0))
    text_c(d, (cx, cy - 26), "70%", font(84), (200, 16, 24))
    text_c(d, (cx, cy + 46), "OFF", font(62), (20, 16, 18))
    im.save(path)


def tex_phone_box(path, W=512, H=512):
    """The deal box: glossy black, a phone silhouette and DEAL OF THE DAY."""
    im = Image.new("RGB", (W, H), (16, 16, 20))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([150, 60, 362, 452], radius=34, fill=(40, 44, 56), outline=(150, 156, 170), width=6)
    for y in range(80, 432):  # screen glow gradient
        t = (y - 80) / 352
        d.line([164, y, 348, y], fill=(int(70 + 120 * t), int(40 + 30 * t), int(170 - 40 * t)))
    d.ellipse([240, 74, 272, 92], fill=(10, 10, 12))
    d.rectangle([0, 0, W, 34], fill=(226, 30, 38))
    d.rectangle([0, H - 34, W, H], fill=(226, 30, 38))
    text_c(d, (W / 2, 17), "DEAL OF THE DAY", font(26), (255, 255, 255))
    text_c(d, (W / 2, H - 17), "PHONE  X-70", font(26), (255, 255, 255))
    im.save(path)


def tex_products(path, seed, W=1024, H=512):
    """A stocked shelf face: rows of product boxes with little labels."""
    rnd = random.Random(seed)
    im = Image.new("RGB", (W, H), (40, 36, 34))
    d = ImageDraw.Draw(im)
    palette = [(226, 30, 38), (255, 196, 0), (30, 110, 200), (240, 240, 236), (30, 160, 90),
               (250, 120, 30), (120, 60, 170), (20, 20, 24), (0, 170, 190), (230, 70, 140)]
    x = 4
    while x < W:
        w = rnd.randint(44, 110)
        h = rnd.randint(int(H * 0.62), int(H * 0.95))
        c = rnd.choice(palette)
        d.rectangle([x, H - h, x + w - 4, H], fill=c)
        shade = tuple(max(0, v - 50) for v in c)
        d.rectangle([x + w - 12, H - h, x + w - 4, H], fill=shade)
        lc = (255, 255, 255) if sum(c) < 400 else (20, 20, 24)
        d.rectangle([x + 8, H - h + 20, x + w - 18, H - h + 20 + rnd.randint(14, 30)], fill=lc)
        if rnd.random() < 0.5:
            d.ellipse([x + 10, H - h * 0.55, x + w - 20, H - h * 0.55 + (w - 30)], fill=tuple(255 - v for v in c))
        x += w
    im.filter(ImageFilter.GaussianBlur(1.2)).save(path)


def tex_price_strip(path, W=1024, H=64):
    im = Image.new("RGB", (W, H), (245, 245, 240))
    d = ImageDraw.Draw(im)
    rnd = random.Random(5)
    for x in range(20, W, 128):
        d.rectangle([x, 6, x + 70, H - 6], fill=(255, 214, 0))
        d.text((x + 8, 16), f"${rnd.randint(2, 99)}", font=font(26), fill=(200, 16, 24))
    im.save(path)


def tex_floor(path, W=1024):
    """Glossy store vinyl tile: cream squares with a faint grout and speckle."""
    rng = np.random.default_rng(3)
    n = 4
    s = W // n
    img = np.zeros((W, W, 3), np.float32)
    for i in range(n):
        for j in range(n):
            base = np.array([226, 218, 202]) if (i + j) % 2 == 0 else np.array([214, 205, 188])
            img[i * s:(i + 1) * s, j * s:(j + 1) * s] = base
    img += rng.normal(0, 5, img.shape)
    img[::s, :] = (150, 142, 130)
    img[:, ::s] = (150, 142, 130)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7)).save(path)


def tex_checkout(path, W=512, H=256):
    im = Image.new("RGB", (W, H), (20, 20, 26))
    d = ImageDraw.Draw(im)
    d.ellipse([20, 40, 196, 216], fill=(255, 214, 0))
    text_c(d, (108, 128), "3", font(140), (20, 20, 26))
    text_c(d, (350, 100), "CHECK", font(70), (255, 255, 255))
    text_c(d, (350, 170), "OUT", font(70), (255, 255, 255))
    im.save(path)


def tex_wall(path, W=2048, H=1024):
    """Far wall: warm store wall with a giant DOORBUSTERS poster and string lights."""
    im = Image.new("RGB", (W, H), (222, 204, 176))
    d = ImageDraw.Draw(im)
    for x in range(0, W, 64):
        d.rectangle([x, 0, x + 30, H], fill=(230, 214, 188))
    d.rectangle([200, 140, W - 200, 700], fill=(14, 12, 14))
    d.rectangle([200, 140, W - 200, 700], outline=(255, 214, 0), width=14)
    text_c(d, (W / 2, 330), "BLACK FRIDAY", fit_font(d, "BLACK FRIDAY", W - 560, 260), (255, 214, 0),
           stroke=6, stroke_fill=(120, 20, 10))
    text_c(d, (W / 2, 570), "DOORBUSTERS  •  TODAY ONLY", font(92), (255, 255, 255))
    rnd = random.Random(SEED)
    for k in range(70):  # warm bulbs
        x = 40 + k * 29
        y = 70 + 26 * math.sin(k * 0.45)
        d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=rnd.choice([(255, 220, 140), (255, 190, 90), (255, 240, 200)]))
    im.save(path)


def tex_ceiling_panel(path):
    im = Image.new("RGB", (64, 64), (255, 244, 220))
    im.save(path)


# ------------------------------------------------------------------- scene JSON

def face_quat(n):
    """Quaternion [w,x,y,z] for a flat panel whose textured face points along horizontal `n`.

    MuJoCo maps a 2D texture onto a box's local +-Z faces (x across, y up), so every sign is a
    box thin in local z, turned so local z = n, local y = world up, local x = y cross z (the text then
    reads left to right for a viewer looking at it)."""
    z = np.array([n[0], n[1], 0.0], float)
    z /= np.linalg.norm(z)
    y = np.array([0.0, 0.0, 1.0])
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:  # Shepperd's method: pick the best-conditioned branch
        w = math.sqrt(1 + tr) / 2
        q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        r = math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        v = [0.0, 0.0, 0.0]
        v[i] = r / 2
        v[j] = (R[j, i] + R[i, j]) / (2 * r)
        v[k] = (R[k, i] + R[i, k]) / (2 * r)
        q = [(R[k, j] - R[j, k]) / (2 * r)] + v
    q = np.array(q) / np.linalg.norm(q)
    return [round(float(v), 6) for v in q]


def panel(name, w, h, pos, n, material, child=False, thick=0.004):
    """A textured sign/face panel of half-width w, half-height h at pos, facing n."""
    f = ch if child else obj
    return f(name, "box", [w, h, thick], pos, quat=face_quat(n), material=material)

def obj(name, shape, size, pos, *, kind="static", collision=False, **kw):
    o = {"name": name, "shape": shape, "size": size, "pos": list(pos), "kind": kind}
    if not collision:
        o["collision"] = False
    o.update(kw)
    return o


def ch(name, shape, size, pos=(0, 0, 0), **kw):
    """A visual-only child geom (no collision)."""
    o = {"name": name, "shape": shape, "size": size, "pos": list(pos), "kind": "static", "collision": False}
    o.update(kw)
    return o


def rod(name, a, b, r, **kw):
    return ch(name, "capsule", [r], (0, 0, 0), fromto=list(a) + list(b), **kw)


def cart(name, cx, cy, yaw_deg, with_rack_collision):
    """A shopping cart at robot-scale-honest human size: one static body of primitives.

    Local frame: +X toward the cart's front, handle at -X. The bottom rack top is
    at RACK_Z. Only the rack plate collides, and only with props (bit 2048)."""
    c = []
    chrome = {"material": "chrome"}
    red = {"material": "cart_red"}
    # wheels + casters
    for i, (wx, wy) in enumerate(((0.33, 0.20), (0.33, -0.20), (-0.33, 0.20), (-0.33, -0.20))):
        c.append(ch(f"{name}_wheel{i}", "cylinder", [0.040, 0.012], (wx, wy, 0.040), euler=[90, 0, 0],
                    material="rubber_black"))
        c.append(ch(f"{name}_hub{i}", "cylinder", [0.016, 0.014], (wx, wy, 0.040), euler=[90, 0, 0], **chrome))
        c.append(rod(f"{name}_fork{i}", (wx, wy, 0.040), (wx, wy, RACK_Z - 0.004), 0.007, **chrome))
    # bottom rack: a wire tray, its plate (props collide with it) + wires
    hx, hy = 0.37, 0.215
    plate = ch(f"{name}_rack", "box", [hx, hy, 0.003], (0, 0, RACK_Z - 0.003), material="rack_grey")
    if with_rack_collision:
        plate.update({"collision": True, "contype": PROP_BIT, "conaffinity": PROP_BIT,
                      "friction": [0.9, 0.005, 0.0001]})
    c.append(plate)
    for k in range(-5, 6):
        x = k * hx / 5.5
        c.append(rod(f"{name}_rw{k + 5}", (x, -hy, RACK_Z + 0.001), (x, hy, RACK_Z + 0.001), 0.0025, **chrome))
    for s in (-1, 1):
        c.append(rod(f"{name}_rl{s}", (-hx, s * hy, RACK_Z + 0.004), (hx, s * hy, RACK_Z + 0.004), 0.005, **chrome))
    for s in (-1, 1):
        c.append(rod(f"{name}_re{s}", (s * hx, -hy, RACK_Z + 0.004), (s * hx, hy, RACK_Z + 0.004), 0.005, **chrome))
    # frame: rear posts up to the handle, front posts up to the basket nose
    for s in (-1, 1):
        c.append(rod(f"{name}_post_r{s}", (-0.36, s * 0.205, RACK_Z), (-0.46, s * 0.25, 0.98), 0.011, **chrome))
        c.append(rod(f"{name}_post_f{s}", (0.36, s * 0.205, RACK_Z), (0.30, s * 0.20, 0.58), 0.010, **chrome))
        c.append(rod(f"{name}_brace{s}", (-0.36, s * 0.205, RACK_Z + 0.01), (0.36, s * 0.205, RACK_Z + 0.01),
                     0.009, **chrome))
    # basket: tapered wire box, z 0.56..0.98, x -0.44..0.42
    zb, zt = 0.56, 0.98
    xb0, xb1, xt0, xt1 = -0.36, 0.36, -0.44, 0.44
    yb, yt = 0.22, 0.27
    for k in range(13):  # side wires (both sides), slanted
        u = k / 12
        for s in (-1, 1):
            c.append(rod(f"{name}_sw{s}_{k}", (xb0 + (xb1 - xb0) * u, s * yb, zb),
                         (xt0 + (xt1 - xt0) * u, s * yt, zt), 0.0028, **chrome))
    for k in range(5):  # side horizontals
        u = k / 4
        z = zb + (zt - zb) * u
        xa, xb_ = xb0 + (xt0 - xb0) * u, xb1 + (xt1 - xb1) * u
        y = yb + (yt - yb) * u
        for s in (-1, 1):
            c.append(rod(f"{name}_sh{s}_{k}", (xa, s * y, z), (xb_, s * y, z), 0.0032, **chrome))
        c.append(rod(f"{name}_fh{k}", (xb_, -y, z), (xb_, y, z), 0.0032, **chrome))
        c.append(rod(f"{name}_bh{k}", (xa, -y, z), (xa, y, z), 0.0032, **chrome))
    for k in range(7):  # front/back verticals, floor of basket
        v = -1 + 2 * k / 6
        c.append(rod(f"{name}_fv{k}", (xb1, v * yb, zb), (xt1, v * yt, zt), 0.0028, **chrome))
        c.append(rod(f"{name}_bv{k}", (xb0, v * yb, zb), (xt0, v * yt, zt), 0.0028, **chrome))
        c.append(rod(f"{name}_fl{k}", (xb0, v * yb, zb), (xb1, v * yb, zb), 0.0028, **chrome))
    c.append(rod(f"{name}_rim_l", (xt0, 0.27, zt), (xt1, 0.27, zt), 0.006, **red))
    c.append(rod(f"{name}_rim_r", (xt0, -0.27, zt), (xt1, -0.27, zt), 0.006, **red))
    c.append(rod(f"{name}_rim_f", (xt1, -0.27, zt), (xt1, 0.27, zt), 0.006, **red))
    # the handle with its red grip
    c.append(rod(f"{name}_handle", (-0.47, -0.27, 1.00), (-0.47, 0.27, 1.00), 0.016, **red))
    for s in (-1, 1):
        c.append(rod(f"{name}_hbar{s}", (-0.44, s * 0.27, 0.98), (-0.47, s * 0.27, 1.00), 0.009, **chrome))
    return {"name": name, "shape": "none", "pos": [cx, cy, 0.0], "euler": [0, 0, yaw_deg], "kind": "static",
            "children": c, "tags": ["cart"]}


#: figures: clothing, size, where they stand and which way they face (deg)
FIGURES = {
    # the shopper at Cart A's handle (-X end), facing +X
    "shopper_a": dict(pos=(-0.80, 0.00), yaw=0, scale=1.00, coat="coat_red", pants=[0.12, 0.13, 0.20, 1],
                      hair=[0.30, 0.16, 0.08, 1], skin=[0.93, 0.76, 0.64, 1], hat=None, pose="handle_point"),
    # the shopper at Cart A's front (+X end), facing -X
    "shopper_b": dict(pos=(0.74, 0.00), yaw=180, scale=1.04, coat="coat_green", pants=[0.22, 0.20, 0.18, 1],
                      hair=[0.10, 0.08, 0.07, 1], skin=[0.62, 0.43, 0.30, 1], hat=[0.96, 0.62, 0.10, 1],
                      pose="grab_wave"),
    # the kid at the checkout, facing -Y (toward the aisle)
    "kid": dict(pos=(-1.92, 0.52), yaw=-90, scale=0.66, coat="coat_teal", pants=[0.20, 0.26, 0.42, 1],
                hair=[0.18, 0.10, 0.06, 1], skin=[0.80, 0.58, 0.44, 1], hat=None, pose="kid"),
}


def figure_bodies(name, f):
    """A stylised mannequin, as two bodies: `name` (legs, torso, arms) and `name_head`.

    Local frame: +X is where the figure faces; origin on the floor between the feet."""
    s = f["scale"]
    coat = {"material": f["coat"]}
    skin = {"rgba": f["skin"]}
    pants = {"rgba": f["pants"]}
    shoe = {"material": "shoe"}
    b = []
    for side in (-1, 1):
        y = side * 0.095 * s
        b.append(ch(f"{name}_shoe{side}", "ellipsoid", [0.13 * s, 0.055 * s, 0.045 * s], (0.04 * s, y, 0.045 * s), **shoe))
        b.append(rod(f"{name}_leg{side}", (0, y, 0.09 * s), (0, y * 0.95, 0.86 * s), 0.065 * s, **pants))
    b.append(ch(f"{name}_hips", "ellipsoid", [0.14 * s, 0.19 * s, 0.12 * s], (0, 0, 0.90 * s), **pants))
    b.append(ch(f"{name}_torso", "ellipsoid", [0.16 * s, 0.23 * s, 0.33 * s], (0, 0, 1.20 * s), **coat))
    b.append(ch(f"{name}_coat_hem", "ellipsoid", [0.17 * s, 0.22 * s, 0.10 * s], (0, 0, 0.98 * s), **coat))
    b.append(rod(f"{name}_neck", (0, 0, 1.45 * s), (0, 0, 1.54 * s), 0.05 * s, **skin))
    sh_z = 1.40 * s
    sh = {1: (0, 0.235 * s, sh_z), -1: (0, -0.235 * s, sh_z)}

    def arm(side, elbow, hand, sleeve=True):
        a = sh[side]
        b.append(rod(f"{name}_uarm{side}", a, elbow, 0.055 * s, **coat))
        b.append(rod(f"{name}_farm{side}", elbow, hand, 0.048 * s, **coat))
        b.append(ch(f"{name}_hand{side}", "sphere", [0.050 * s], hand, **skin))

    pose = f["pose"]
    if pose == "handle_point":  # right hand on the handle, left arm up and jabbing a finger
        arm(-1, (0.15 * s, -0.27 * s, 1.13 * s), (0.33, -0.15, 1.00))
        arm(1, (0.15 * s, 0.27 * s, 1.13 * s), (0.33, 0.15, 1.00))
        b.append(ch(f"{name}_bag", "box", [0.11, 0.045, 0.14], (-0.02, -0.34, 0.66), euler=[0, 8, 0],
                    material="bag_yellow"))
        b.append(rod(f"{name}_bag_strap", (-0.02, -0.34, 0.80), (0.02, -0.27, 1.02), 0.006, material="bag_yellow"))
    elif pose == "grab_wave":  # left hand gripping the cart nose, right hand thrown up
        arm(1, (0.18 * s, 0.30 * s, 1.08 * s), (0.30, 0.18, 0.96))
        arm(-1, (0.10 * s, -0.40 * s, 1.40 * s), (0.18 * s, -0.42 * s, 1.68 * s))
    elif pose == "kid":  # arms down, holding nothing
        arm(1, (0.04 * s, 0.27 * s, 1.12 * s), (0.10 * s, 0.25 * s, 0.86 * s))
        arm(-1, (0.04 * s, -0.27 * s, 1.12 * s), (0.10 * s, -0.25 * s, 0.86 * s))
    head = []
    hz = 1.66 * s
    head.append(ch(f"{name}_skull", "ellipsoid", [0.105 * s, 0.095 * s, 0.125 * s], (0, 0, 0), **skin))
    head.append(ch(f"{name}_nose", "ellipsoid", [0.03 * s, 0.018 * s, 0.03 * s], (0.10 * s, 0, -0.01 * s), **skin))
    head.append(ch(f"{name}_hair", "ellipsoid", [0.112 * s, 0.10 * s, 0.09 * s], (-0.015 * s, 0, 0.05 * s),
                   rgba=f["hair"]))
    for side in (-1, 1):  # mannequin eyes: two small dark dots
        head.append(ch(f"{name}_eye{side}", "sphere", [0.012 * s], (0.095 * s, side * 0.038 * s, 0.012 * s),
                       rgba=[0.08, 0.07, 0.07, 1]))
    if f["hat"] is not None:
        head.append(ch(f"{name}_beanie", "ellipsoid", [0.118 * s, 0.108 * s, 0.10 * s], (-0.01 * s, 0, 0.07 * s),
                       rgba=f["hat"]))
        head.append(ch(f"{name}_pompom", "sphere", [0.035 * s], (-0.01 * s, 0, 0.17 * s), rgba=[0.98, 0.96, 0.9, 1]))
    if name == "shopper_a":
        head.append(ch(f"{name}_bun", "sphere", [0.05], (-0.10, 0, 0.08), rgba=f["hair"]))
    x, y = f["pos"]
    return [
        {"name": name, "shape": "none", "pos": [x, y, 0.0], "euler": [0, 0, f["yaw"]], "kind": "static",
         "children": b, "tags": ["figure"]},
        {"name": f"{name}_head", "shape": "none", "pos": [x, y, hz], "euler": [0, 0, f["yaw"]], "kind": "static",
         "children": head, "tags": ["figure", "head"]},
    ]


def gondola(name, x0, x1, y, facing, seed):
    """A store shelf run: back panel, product faces on 4 shelves, price strips, header sign."""
    L = (x1 - x0) / 2
    xc = (x0 + x1) / 2
    depth = 0.45
    c = []
    c.append(ch(f"{name}_back", "box", [L, 0.02, 0.95], (0, -facing * depth / 2, 0.95), material="shelf_metal"))
    c.append(ch(f"{name}_base", "box", [L, depth / 2, 0.06], (0, 0, 0.06), material="shelf_metal"))
    for k, z in enumerate((0.12, 0.52, 0.92, 1.32)):
        c.append(panel(f"{name}_prod{k}", L, 0.17, (0, -facing * 0.10, z + 0.17), (0, facing),
                       f"products{(seed + k) % 3}", child=True))
        c.append(ch(f"{name}_shelf{k}", "box", [L, depth / 2, 0.012], (0, 0, z), material="shelf_metal"))
        c.append(panel(f"{name}_strip{k}", L, 0.022, (0, facing * depth / 2, z), (0, facing), "price_strip",
                       child=True))
    c.append(ch(f"{name}_header_board", "box", [L * 0.9, 0.012, 0.18], (0, 0, 2.08), material="shelf_metal"))
    c.append(panel(f"{name}_header", L * 0.9 - 0.01, 0.17, (0, facing * 0.016, 2.08), (0, facing), "header_bf",
                   child=True))
    return {"name": name, "shape": "none", "pos": [xc, y, 0], "kind": "static", "children": c,
            "tags": ["shelf"]}


def build(out: Path):
    tex = out / "textures"
    tex.mkdir(parents=True, exist_ok=True)
    tex_banner(tex / "banner_bf.png")
    tex_banner(tex / "banner_deals.png", line1="DOORBUSTERS", line2="LIMIT ONE PER CUSTOMER")
    tex_header(tex / "header_bf.png", "BLACK FRIDAY DEALS")
    tex_tag(tex / "tag70.png")
    tex_phone_box(tex / "phone_box.png")
    for k in range(3):
        tex_products(tex / f"products{k}.png", seed=10 + k)
    tex_price_strip(tex / "price_strip.png")
    tex_floor(tex / "floor.png")
    tex_checkout(tex / "checkout.png")
    tex_wall(tex / "wall.png")
    tex_ceiling_panel(tex / "panel.png")

    T = "textures/"
    materials = [
        {"name": "floor_tile", "texture": {"file": T + "floor.png"}, "texrepeat": [1.2, 1.2], "texuniform": True,
         "shininess": 0.7, "specular": 0.35, "reflectance": 0.12},
        {"name": "banner_bf", "texture": {"file": T + "banner_bf.png"}, "emission": 0.35, "specular": 0.05},
        {"name": "banner_deals", "texture": {"file": T + "banner_deals.png"}, "emission": 0.35, "specular": 0.05},
        {"name": "header_bf", "texture": {"file": T + "header_bf.png"}, "emission": 0.4, "specular": 0.05},
        {"name": "tag70", "texture": {"file": T + "tag70.png"}, "emission": 0.25, "specular": 0.1},
        {"name": "phone_box", "texture": {"file": T + "phone_box.png"}, "shininess": 0.8, "specular": 0.5},
        {"name": "products0", "texture": {"file": T + "products0.png"}, "texrepeat": [3, 1], "specular": 0.15},
        {"name": "products1", "texture": {"file": T + "products1.png"}, "texrepeat": [3, 1], "specular": 0.15},
        {"name": "products2", "texture": {"file": T + "products2.png"}, "texrepeat": [3, 1], "specular": 0.15},
        {"name": "price_strip", "texture": {"file": T + "price_strip.png"}, "texrepeat": [3, 1], "emission": 0.1},
        {"name": "checkout_sign", "texture": {"file": T + "checkout.png"}, "emission": 0.8},
        {"name": "back_wall", "texture": {"file": T + "wall.png"}, "emission": 0.30, "specular": 0.0},
        {"name": "side_wall", "rgba": [0.86, 0.80, 0.70, 1], "specular": 0.0},
        {"name": "ceiling_panel", "texture": {"file": T + "panel.png"}, "emission": 1.0},
        {"name": "chrome", "rgba": [0.80, 0.81, 0.84, 1], "shininess": 0.95, "specular": 0.9, "reflectance": 0.1},
        {"name": "cart_red", "rgba": [0.86, 0.10, 0.12, 1], "shininess": 0.7, "specular": 0.5},
        {"name": "rack_grey", "rgba": [0.55, 0.56, 0.58, 1], "shininess": 0.6, "specular": 0.4},
        {"name": "rubber_black", "rgba": [0.08, 0.08, 0.09, 1], "shininess": 0.2, "specular": 0.1},
        {"name": "shelf_metal", "rgba": [0.86, 0.86, 0.84, 1], "shininess": 0.4, "specular": 0.3},
        {"name": "ceiling", "rgba": [0.70, 0.66, 0.60, 1], "specular": 0.0},
        {"name": "coat_red", "rgba": [0.74, 0.08, 0.12, 1], "shininess": 0.3, "specular": 0.2},
        {"name": "coat_green", "rgba": [0.10, 0.40, 0.30, 1], "shininess": 0.6, "specular": 0.4},
        {"name": "coat_teal", "rgba": [0.05, 0.55, 0.62, 1], "shininess": 0.2, "specular": 0.1},
        {"name": "shoe", "rgba": [0.10, 0.09, 0.09, 1], "shininess": 0.6, "specular": 0.4},
        {"name": "bag_yellow", "rgba": [0.98, 0.80, 0.10, 1], "shininess": 0.5, "specular": 0.3},
        {"name": "basket_red", "rgba": [0.88, 0.14, 0.16, 1], "shininess": 0.6, "specular": 0.4},
        {"name": "counter", "rgba": [0.24, 0.22, 0.24, 1], "shininess": 0.5, "specular": 0.3},
        {"name": "belt", "rgba": [0.05, 0.05, 0.06, 1], "shininess": 0.3, "specular": 0.2},
        {"name": "steel", "rgba": [0.70, 0.71, 0.74, 1], "shininess": 0.8, "specular": 0.7},
        {"name": "screen", "rgba": [0.30, 0.75, 0.95, 1], "emission": 0.8},
    ]

    objs = []
    # ---- room: back walls, ceiling with light panels, hanging banners ----
    objs.append(obj("wall_far", "box", [0.05, 2.4, 1.6], (4.4, 0, 1.6), material="side_wall"))
    objs.append(obj("wall_near", "box", [0.05, 2.4, 1.6], (-3.9, 0, 1.6), material="side_wall"))
    objs.append(panel("poster_far", 2.3, 1.15, (4.34, 0, 1.75), (-1, 0), "back_wall"))
    objs.append(panel("poster_near", 2.3, 1.15, (-3.84, 0, 1.75), (1, 0), "back_wall"))
    objs.append(obj("wall_l", "box", [4.2, 0.05, 1.6], (0.25, 2.3, 1.6), material="side_wall"))
    objs.append(obj("wall_r", "box", [4.2, 0.05, 1.6], (0.25, -2.3, 1.6), material="side_wall"))
    objs.append(obj("ceiling", "box", [4.2, 2.4, 0.02], (0.25, 0, 3.22), material="ceiling"))
    for i, x in enumerate(np.arange(-3.0, 4.01, 1.4)):
        for j, y in enumerate((-0.9, 0.9)):
            objs.append(obj(f"light_panel_{i}_{j}", "box", [0.55, 0.16, 0.01], (float(x), y, 3.19),
                            material="ceiling_panel"))
    for i, (x, mat) in enumerate(((-1.7, "banner_bf"), (1.0, "banner_deals"), (3.2, "banner_bf"))):
        objs.append(panel(f"banner{i}_a", 1.10, 0.28, (x + 0.005, 0, 2.55), (1, 0), mat))
        objs.append(panel(f"banner{i}_b", 1.10, 0.28, (x - 0.005, 0, 2.55), (-1, 0), mat))
        for s in (-1, 1):
            objs.append(obj(f"banner{i}_wire{s}", "capsule", [0.003], (0, 0, 0),
                            fromto=[x, s * 1.0, 2.83, x, s * 1.0, 3.2], material="steel"))
    # ---- shelving both sides ----
    objs.append(gondola("shelf_l", -1.2, 3.6, 1.75, -1, 0))
    objs.append(gondola("shelf_r", -1.2, 3.6, -1.75, 1, 1))
    objs.append(gondola("shelf_r2", -3.8, -2.4, -1.75, 1, 2))
    # endcap stack of deal boxes behind the hook shot
    for k in range(9):
        i, j = k % 3, k // 3
        objs.append(obj(f"endcap_box{k}", "box", [0.16, 0.12, 0.10], (2.0 + 0.33 * i, -1.1, 0.10 + 0.205 * j),
                        material="phone_box"))
    objs.append(panel("endcap_sign", 0.28, 0.28, (1.85, -1.06, 0.98), (-0.30, 0.95),
                    material="tag70"))
    # ---- Cart A (the deal), Cart B (empty, scenery) ----
    objs.append(cart("cart_a", 0.0, 0.0, 0, with_rack_collision=True))
    objs.append(cart("cart_b", 2.75, 0.75, 152, with_rack_collision=False))
    # ---- checkout lane: counter, belt, register, lane sign ----
    cx0, cx1, cy = -3.1, -1.45, 1.00
    objs.append(obj("counter", "box", [(cx1 - cx0) / 2, 0.30, 0.42], ((cx0 + cx1) / 2, cy, 0.42), material="counter"))
    objs.append(obj("belt", "box", [(cx1 - cx0) / 2 - 0.05, 0.22, 0.01], ((cx0 + cx1) / 2, cy, 0.85),
                    material="belt"))
    objs.append(obj("register", "box", [0.16, 0.14, 0.12], (-2.85, cy + 0.05, 0.98), material="steel"))
    objs.append(obj("register_screen", "box", [0.01, 0.11, 0.08], (-2.70, cy + 0.05, 1.18), euler=[0, 20, 0],
                    material="screen"))
    objs.append(obj("lane_pole", "cylinder", [0.025, 1.0], (cx1 - 0.05, cy - 0.25, 1.0), material="steel"))
    objs.append(panel("lane_sign", 0.26, 0.13, (cx1 - 0.025, cy - 0.25, 2.1), (1, 0), "checkout_sign"))
    objs.append(panel("lane_sign_back", 0.26, 0.13, (cx1 - 0.075, cy - 0.25, 2.1), (-1, 0), "checkout_sign"))
    for k in range(4):  # impulse-buy rack
        objs.append(obj(f"candy{k}", "box", [0.10, 0.05, 0.07], (-1.75 - 0.22 * k, cy - 0.33, 0.35),
                        material=f"products{k % 3}"))
    # ---- the deal: a free rigid phone box with a 70% OFF tag ----
    hx, hy, hz = DEAL_HALF
    pcol = {"kind": "dynamic", "contype": PROP_BIT, "conaffinity": PROP_BIT, "condim": 3,
            "friction": [0.9, 0.005, 0.0001]}
    deal = {"name": "deal", "shape": "none", "pos": [DEAL_POS[0], DEAL_POS[1], RACK_Z], "kind": "dynamic",
            "mass": DEAL_MASS, "tags": ["prop", "deal"], "children": [
                dict({"name": "deal_box", "shape": "box", "size": [hx, hy, hz], "pos": [0, 0, hz],
                      "material": "phone_box", "mass": DEAL_MASS * 0.97}, **pcol),
                dict({"name": "deal_tag", "shape": "box", "size": [0.026, 0.026, 0.0008],
                      "pos": [0.030, 0.0, 2 * hz + 0.0008], "euler": [0, 0, 18], "material": "tag70",
                      "mass": DEAL_MASS * 0.03}, **pcol),
            ]}
    objs.append(deal)
    # ---- the kid's basket on the floor (static; the box lands in it) ----
    bx, by = BASKET_POS
    bhx, bhy = BASKET_HALF
    t = 0.006
    wallc = {"collision": True, "contype": PROP_BIT, "conaffinity": PROP_BIT, "material": "basket_red"}
    bk = [
        ch("basket_floor", "box", [bhx, bhy, 0.004], (0, 0, 0.004), **wallc),
        ch("basket_wall_xp", "box", [t, bhy, BASKET_H / 2], (bhx - t, 0, BASKET_H / 2), **wallc),
        ch("basket_wall_xn", "box", [t, bhy, BASKET_H / 2], (-bhx + t, 0, BASKET_H / 2), **wallc),
        ch("basket_wall_yp", "box", [bhx, t, BASKET_H / 2], (0, bhy - t, BASKET_H / 2), **wallc),
        ch("basket_wall_yn", "box", [bhx, t, BASKET_H / 2], (0, -bhy + t, BASKET_H / 2), **wallc),
    ]
    for s in (-1, 1):  # two folded-down handles
        bk.append(rod(f"basket_handle{s}", (-bhx * 0.6, s * (bhy + 0.004), BASKET_H),
                      (bhx * 0.6, s * (bhy + 0.004), BASKET_H), 0.006, material="basket_red"))
    for k in range(-3, 4):  # slots on the long walls
        for s in (-1, 1):
            bk.append(ch(f"basket_slot{s}_{k + 3}", "box", [0.012, 0.0008, BASKET_H * 0.28],
                         (k * 0.04, s * (bhy + 0.0002), BASKET_H * 0.5), rgba=[0.55, 0.06, 0.08, 1]))
    objs.append({"name": "basket", "shape": "none", "pos": [bx, by, 0.0], "kind": "static", "children": bk,
                 "tags": ["basket"]})
    # an invisible floor slab on the prop bit, so a dropped box would land on the floor too
    objs.append(obj("prop_floor", "box", [4.0, 2.2, 0.005], (0.25, 0, -0.005), collision=True, contype=0,
                    conaffinity=PROP_BIT, rgba=[0, 0, 0, 0], group=3))
    # ---- figures ----
    for name, f in FIGURES.items():
        objs.extend(figure_bodies(name, f))

    scene = {
        "name": "black_friday_store",
        "title": "Black Friday Store Aisle",
        "description": ("A store aisle on Black Friday, at full human scale: gondola shelves with BLACK FRIDAY "
                        "header signs, hanging banners, warm ceiling light, two shopping carts and a checkout "
                        "lane. Cart A's low bottom rack (top 6 cm off the floor) holds one free rigid deal box "
                        "(a 14 x 8 x 3.2 cm phone box with a 70% OFF tag, 0.15 kg) where Jumper's claw can reach it. "
                        "Two adult shopper figures stand at either end of Cart A, and a small shopper stands at "
                        "the checkout with an empty basket on the floor. The figures are static bodies a host "
                        "may animate kinematically. Robot spawn is on the floor behind Cart A."),
        "tags": ["indoor", "store", "black-friday", "grasp", "pick-and-place"],
        "physics": {"timestep": 0.001, "integrator": "implicitfast", "cone": "pyramidal", "iterations": 10},
        "sky": {"preset": "indoor", "top": [0.20, 0.14, 0.10], "bottom": [0.30, 0.20, 0.14]},
        "lighting": {
            "headlight": False, "shadows": True, "shadow_quality": "high",
            "ambient": [0.22, 0.19, 0.16],
            "lights": [
                {"name": "key_cart", "type": "spot", "pos": [-0.6, 1.2, 2.9], "target": [0.0, -0.1, 0.0],
                 "diffuse": [0.85, 0.72, 0.55], "specular": [0.3, 0.25, 0.2], "cutoff": 48, "exponent": 2,
                 "castshadow": True},
                {"name": "key_checkout", "type": "spot", "pos": [-1.6, -0.6, 2.9], "target": [-1.95, 0.3, 0.0],
                 "diffuse": [0.80, 0.68, 0.50], "specular": [0.3, 0.25, 0.2], "cutoff": 45, "exponent": 2,
                 "castshadow": True},
                {"name": "aisle_fill_a", "type": "spot", "pos": [-2.4, 0.0, 3.0], "target": [-1.6, 0.0, 0.0],
                 "diffuse": [0.40, 0.34, 0.27], "specular": [0.1, 0.1, 0.1], "cutoff": 70, "exponent": 1,
                 "castshadow": False},
                {"name": "aisle_fill_b", "type": "spot", "pos": [3.0, 0.0, 3.0], "target": [3.0, 0.0, 0.0],
                 "diffuse": [0.40, 0.34, 0.27], "specular": [0.1, 0.1, 0.1], "cutoff": 70, "exponent": 1,
                 "castshadow": False},
                {"name": "warm_bounce", "type": "directional", "pos": [0, 0, 3], "dir": [0.3, 0.4, -1.0],
                 "diffuse": [0.22, 0.18, 0.14], "specular": [0.05, 0.05, 0.05], "castshadow": False},
            ],
        },
        "ground": {"type": "plane", "size": [4.6, 2.6], "material": "floor_tile", "friction": "polished_floor"},
        "materials": materials,
        "objects": objs,
        "cameras": [
            {"name": "hook", "pos": [2.6, 1.3, 1.45], "target": [0.0, 0.0, 0.95], "fovy": 50},
            {"name": "crab_low", "pos": [-0.35, -0.33, 0.08], "target": [0.02, -0.19, 0.08], "fovy": 45},
            {"name": "checkout", "pos": [-1.2, -0.8, 0.45], "target": [-1.9, 0.2, 0.15], "fovy": 45},
        ],
        "spawn": {"position": [1.60, -0.55, 0.0], "yaw": 180, "clearance": 0.12,
                  "bounds": [-3.5, -1.4, 0.0, 4.2, 1.4, 0.6]},
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    return scene


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/scene")
    out.mkdir(parents=True, exist_ok=True)
    build(out)
    print("wrote", out / "scene.json")

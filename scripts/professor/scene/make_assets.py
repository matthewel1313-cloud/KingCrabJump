"""Generate the classroom diorama's original textures and small meshes (all procedural).

    python scene/make_assets.py scene/assets

Writes PNG textures (chalkboard, midterm paper, desk wood, book covers) and OBJ meshes with
UVs (textured quads, the mug handle, the lamp shade). Fonts are DejaVu (Bitstream Vera
licence, freely redistributable); every drawing is made here, in code.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = Path("/usr/share/fonts/truetype/dejavu")


def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def chalk_text(draw, xy, text, f, fill=(236, 234, 222), jitter=2, passes=3, anchor="la"):
    rng = random.Random(hash(text) & 0xFFFF)
    for _ in range(passes):
        dx, dy = rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)
        a = rng.randint(170, 235)
        draw.text((xy[0] + dx, xy[1] + dy), text, font=f, fill=fill + (a,), anchor=anchor)


def chalkboard(path: Path):
    W, H = 1640, 800
    rng = np.random.default_rng(3)
    base = np.zeros((H, W, 3), np.float32)
    base[:] = (38, 64, 50)
    base += rng.normal(0, 4.0, (H, W, 1))
    yy, xx = np.ogrid[:H, :W]
    for _ in range(12):   # old eraser smudges
        cx, cy, rx, ry = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(80, 260), rng.uniform(30, 80)
        base += np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))[..., None] * rng.uniform(6, 14)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    chalk_text(d, (W // 2, 40), "King Crab's Class", font("DejaVuSerif-BoldItalic.ttf", 168), anchor="mt")
    d.line([(220, 250), (W - 220, 250)], fill=(236, 234, 222, 170), width=6)
    chalk_text(d, (90, 300), "MIDTERM TODAY", font("DejaVuSans-Bold.ttf", 84), fill=(250, 214, 120))
    chalk_text(d, (90, 420), "No pinching your", font("DejaVuSerif-Italic.ttf", 58))
    chalk_text(d, (90, 490), "neighbour's answers.", font("DejaVuSerif-Italic.ttf", 58))
    chalk_text(d, (90, 590), "Office hours: low tide", font("DejaVuSerif-Italic.ttf", 58))
    chalk_text(d, (90, 690), "Q1. Sideways is a direction. T / F", font("DejaVuSans.ttf", 46))
    cx, cy = 1330, 520   # a small crab doodle wearing a crown, drawn from primitives
    c = (236, 234, 222, 200)
    d.ellipse([cx - 120, cy - 60, cx + 120, cy + 60], outline=c, width=8)
    for s in (-1, 1):
        for k in range(3):
            x0 = cx + s * (80 + k * 18)
            d.line([(x0, cy + 30), (x0 + s * 60, cy + 110)], fill=c, width=7)
        d.line([(cx + s * 100, cy - 30), (cx + s * 175, cy - 110)], fill=c, width=8)
        d.ellipse([cx + s * 175 - 38, cy - 160, cx + s * 175 + 38, cy - 90], outline=c, width=8)
        d.ellipse([cx + s * 38 - 14, cy - 92, cx + s * 38 + 14, cy - 64], outline=c, width=6)
    d.line([(cx - 60, cy - 60), (cx - 60, cy - 120), (cx - 30, cy - 90), (cx, cy - 130), (cx + 30, cy - 90),
            (cx + 60, cy - 120), (cx + 60, cy - 60)], fill=(250, 214, 120, 230), width=8)
    layer = layer.filter(ImageFilter.GaussianBlur(0.8))
    Image.alpha_composite(img, layer).convert("RGB").save(path)


def paper(path: Path):
    W, H = 900, 1200   # portrait sheet; the quad maps it so the header faces the reader
    img = Image.new("RGB", (W, H), (248, 246, 238))
    d = ImageDraw.Draw(img)
    for y in range(250, H - 60, 44):
        d.line([(60, y), (W - 60, y)], fill=(185, 205, 230), width=2)
    d.line([(120, 0), (120, H)], fill=(230, 160, 160), width=3)
    d.text((W // 2, 50), "MIDTERM EXAM", font=font("DejaVuSans-Bold.ttf", 72), fill=(30, 30, 40), anchor="mt")
    d.text((140, 150), "Name: ____________", font=font("DejaVuSans.ttf", 40), fill=(40, 40, 50))
    d.rectangle([W - 230, 130, W - 70, 230], outline=(40, 40, 50), width=4)
    d.text((W - 150, 136), "Grade", font=font("DejaVuSans.ttf", 26), fill=(80, 80, 90), anchor="mt")
    hand = font("DejaVuSerif-Italic.ttf", 34)
    lines = ["1. Why do crabs walk sideways?", "   Because forward is overrated.",
             "2. Define 'exoskeleton'.", "   A skeleton that went outside.",
             "3. Solve for x:  2x + 3 = 11", "   x = crab", "4. Name three tides.",
             "   high, low, and medium", "5. Show your work.", "   (drew a fish)"]
    y = 262
    for i, t in enumerate(lines):
        d.text((140, y), t, font=hand if t.startswith("   ") else font("DejaVuSans.ttf", 32),
               fill=(40, 60, 140) if t.startswith("   ") else (40, 40, 50))
        y += 88 if t.startswith("   ") else 44
    img.save(path)


def paper_back(path: Path):
    """Page 2, printed on the back of every midterm: what the professor reads while holding it up."""
    W, H = 900, 1200
    img = Image.new("RGB", (W, H), (248, 246, 238))
    d = ImageDraw.Draw(img)
    for y in range(170, H - 60, 44):
        d.line([(60, y), (W - 60, y)], fill=(185, 205, 230), width=2)
    d.line([(120, 0), (120, H)], fill=(230, 160, 160), width=3)
    d.text((W // 2, 50), "PAGE 2", font=font("DejaVuSans-Bold.ttf", 64), fill=(30, 30, 40), anchor="mt")
    hand = font("DejaVuSerif-Italic.ttf", 34)
    lines = ["6. Why does the professor", "   wear a crown?", "   ...he's the king?",
             "7. Explain molting.", "   new year, new shell", "8. Draw a crab.", ""]
    y = 182
    for t in lines:
        d.text((140, y), t, font=hand if t.startswith("   ") else font("DejaVuSans.ttf", 32),
               fill=(40, 60, 140) if t.startswith("   ") else (40, 40, 50))
        y += 66 if t.startswith("   ") else 48
    cx, cy = 450, 860   # a student's wobbly crab drawing
    c = (40, 60, 140)
    d.ellipse([cx - 110, cy - 55, cx + 110, cy + 55], outline=c, width=6)
    for s in (-1, 1):
        for k in range(3):
            d.line([(cx + s * (70 + 20 * k), cy + 35), (cx + s * (125 + 20 * k), cy + 110)], fill=c, width=5)
        d.line([(cx + s * 90, cy - 35), (cx + s * 160, cy - 110)], fill=c, width=6)
        d.ellipse([cx + s * 160 - 30, cy - 150, cx + s * 160 + 30, cy - 95], outline=c, width=6)
    d.text((cx + 150, cy + 150), "(it's you)", font=hand, fill=c)
    img.save(path)


def desk_wood(path: Path):
    """Straight planks running along the image's x, each with its own tone and fine grain."""
    W, H = 2048, 1024
    rng = np.random.default_rng(11)
    img = np.zeros((H, W, 3), np.float32)
    rows = 6
    ph = H // rows
    for k in range(rows):
        y0, y1 = k * ph, (k + 1) * ph
        tone = np.array([150, 98, 58], np.float32) * rng.uniform(0.9, 1.08)
        yy = np.arange(y1 - y0)[:, None]
        xx = np.arange(W)[None, :]
        freq = rng.uniform(0.18, 0.3)
        warp = 6 * np.sin(xx / rng.uniform(180, 320) + rng.uniform(0, 6))
        grain = np.sin((yy + warp) * freq + rng.normal(0, 0.15, (y1 - y0, W)))
        streak = rng.normal(0, 1, (y1 - y0, 1)) * 4
        img[y0:y1] = tone + grain[..., None] * 7 + streak[..., None] + rng.normal(0, 2, (y1 - y0, W, 1))
        img[y0:y0 + 3] *= 0.55          # the seam between planks
        cut = rng.integers(200, W - 200)  # a butt joint along the plank
        img[y0:y1, cut:cut + 3] *= 0.6
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


def book_cover(path: Path, rgb, title):
    W, H = 512, 384
    img = Image.new("RGB", (W, H), rgb)
    d = ImageDraw.Draw(img)
    d.rectangle([24, 24, W - 24, H - 24], outline=(225, 200, 120), width=6)
    d.text((W // 2, H // 2), title, font=font("DejaVuSerif-Bold.ttf", 54), fill=(232, 210, 140), anchor="mm")
    img.save(path)


def write_obj(path: Path, verts, uvs, faces):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# generated by make_assets.py\n")
        for v in verts:
            fh.write("v %.6f %.6f %.6f\n" % tuple(v))
        for t in uvs:
            fh.write("vt %.6f %.6f\n" % tuple(t))
        for f in faces:
            fh.write("f " + " ".join(f"{i + 1}/{i + 1}" for i in f) + "\n")


def slab(path: Path, corners_uv, normal, thickness=0.0004):
    """A thin closed slab: the face `corners_uv` (4 x (xyz, uv), counter-clockwise seen from
    `normal`) carries the texture; the back and the rims are UV (0, 0). Closed, so MuJoCo can
    compute its (tiny) volume."""
    import numpy as np
    n = np.asarray(normal, float) * thickness
    front = [np.asarray(p, float) for p, _ in corners_uv]
    back = [p - n for p in front]
    verts, uvs, faces = [], [], []
    def add(ps, uv):
        base = len(verts)
        verts.extend(tuple(p) for p in ps); uvs.extend(uv)
        return base
    b = add(front, [uv for _, uv in corners_uv]); faces += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    b = add(back[::-1], [(0, 0)] * 4); faces += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    for i in range(4):
        j = (i + 1) % 4
        b = add([front[i], back[i], back[j], front[j]], [(0, 0)] * 4)
        faces += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    write_obj(path, verts, uvs, faces)


def paper_quad(path: Path, sx, sy):
    """A sheet face facing +Z. Read from a camera on +X looking back: the page's top (v = 1)
    is at -X and its left edge (u = 0) at -Y."""
    slab(path, [((-sx, -sy, 0), (0, 1)), ((sx, -sy, 0), (0, 0)), ((sx, sy, 0), (1, 0)),
                ((-sx, sy, 0), (1, 1))], (0, 0, 1))


def paper_back_quad(path: Path, sx, sy):
    """The sheet's back face, facing -Z. Seen while the claw holds the sheet up to read, the
    page's top (v = 1) is along +X and its right edge (u = 1) along +Y."""
    slab(path, [((-sx, -sy, 0), (0, 0)), ((-sx, sy, 0), (1, 0)), ((sx, sy, 0), (1, 1)),
                ((sx, -sy, 0), (0, 1))], (0, 0, -1))


def board_quad(path: Path, sy, sz):
    """A board face facing +X: u along +Y (the camera's right), v along +Z."""
    slab(path, [((0, -sy, -sz), (0, 0)), ((0, sy, -sz), (1, 0)), ((0, sy, sz), (1, 1)),
                ((0, -sy, sz), (0, 1))], (1, 0, 0))


def torus_handle(path: Path, R=0.022, r=0.0055, n=40, m=12):
    """Half a torus in the XZ plane (a mug handle), opening toward -X."""
    verts, uvs, faces = [], [], []
    for i in range(n + 1):
        a = -math.pi / 2 + math.pi * i / n
        for j in range(m):
            b = 2 * math.pi * j / m
            x = (R + r * math.cos(b)) * math.cos(a)
            z = (R + r * math.cos(b)) * math.sin(a)
            verts.append((x, r * math.sin(b), z)); uvs.append((i / n, j / m))
    for i in range(n):
        for j in range(m):
            a0, a1 = i * m + j, i * m + (j + 1) % m
            b0, b1 = (i + 1) * m + j, (i + 1) * m + (j + 1) % m
            faces += [(a0, b0, b1), (a0, b1, a1)]
    write_obj(path, verts, uvs, faces)


def lamp_shade(path: Path, r0=0.022, r1=0.055, h=0.07, n=48, wall=0.0015):
    """An open frustum (lamp shade), narrow end at +Z, opening toward -Z. Outer and inner walls
    have their own vertices so each shades with its own normals, and a rim joins them."""
    verts, uvs, faces = [], [], []
    for (ra, rb) in ((r0, r1), (r0 - wall, r1 - wall)):
        for k in range(n):
            a = 2 * math.pi * k / n
            verts += [(ra * math.cos(a), ra * math.sin(a), h / 2), (rb * math.cos(a), rb * math.sin(a), -h / 2)]
            uvs += [(k / n, 1), (k / n, 0)]
    inner = 2 * n
    for k in range(n):
        i0, i1 = 2 * k, 2 * ((k + 1) % n)
        faces += [(i0, i0 + 1, i1 + 1), (i0, i1 + 1, i1)]                          # outer, outward
        j0, j1 = inner + i0, inner + i1
        faces += [(j0, j1 + 1, j0 + 1), (j0, j1, j1 + 1)]                          # inner, inward
        faces += [(i0 + 1, j0 + 1, j1 + 1), (i0 + 1, j1 + 1, i1 + 1)]              # bottom rim
        faces += [(i0, i1, j1), (i0, j1, j0)]                                      # top rim
    write_obj(path, verts, uvs, faces)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    chalkboard(out / "chalkboard.png")
    paper(out / "midterm.png")
    paper_back(out / "midterm_back.png")
    desk_wood(out / "desk_wood.png")
    book_cover(out / "book_red.png", (120, 32, 36), "TIDES")
    book_cover(out / "book_navy.png", (30, 44, 86), "SHELLS 101")
    board_quad(out / "board_face.obj", 0.28, 0.1365)
    paper_quad(out / "paper_face.obj", 0.0605, 0.0455)
    paper_back_quad(out / "paper_back_face.obj", 0.0605, 0.0455)
    torus_handle(out / "mug_handle.obj")
    lamp_shade(out / "lamp_shade.obj")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()

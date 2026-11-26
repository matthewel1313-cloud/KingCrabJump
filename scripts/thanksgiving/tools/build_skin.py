"""Build the King Crab appearance as a protocol-valid `.skin/3` with jumper-design.

Run inside a jumper-design checkout with its own venv (`pip install -e ".[sim,dev]"`):

    python build_skin.py --design /path/to/jumper-design --work OUTDIR

What it makes, all display-only:

* an upper-shell STL in **CAD millimetres** = the original Jumper upper shell
  (inverted exactly from the profile's `cad_mm_to_parent_m`, so the seam check
  passes untouched) + a tiny gold crown on top + a tucked napkin bib under the
  display;
* a Bambu-style AMS 3MF whose triangles match the STL one-for-one, painted with
  four palette slots (orange-red shell, gold crown, cream napkin, ruby jewels) --
  the input jumper-design's `assemble` reads shell colours from;
* `visual_overrides.json`: warm orange-red / peach recolours of the lower body
  and limb visual geoms (never collision, never physics);

then runs jumper-design's own `assemble` -> `export-skin` -> `verify-package
--mujoco`. Collision, inertia and joints remain the baseline's.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import trimesh

SHELL_RGB = "#E2461E"     # warm orange-red
GOLD = "#E9B43A"          # crown gold
NAPKIN = "#F3ECDA"        # cream napkin
RUBY = "#A3122A"          # jewels and napkin stripe
PALETTE = [SHELL_RGB, GOLD, NAPKIN, RUBY]
PAINT = ["4", "8", "0C", "1C"]  # Bambu whole-face paint codes, slots 0..3

LIMB_RED = [0.886, 0.275, 0.118, 1.0]    # #E2461E
LIMB_PEACH = [0.949, 0.741, 0.584, 1.0]  # #F2BD95 (was the grey anodised parts)
BASE_RED = [0.62, 0.17, 0.08, 1.0]       # lower body: deep roasted red


def link_to_cad(points_m: np.ndarray, tf: dict) -> np.ndarray:
    r = np.asarray(tf["rotation_matrix"], float)
    t = np.asarray(tf["translation_m"], float)
    return ((points_m - t) @ r) / tf["scale"]  # R^T (p - t) / s, row vectors


def crown_parts_link_m(top_z: float, cx: float, cy: float):
    """Crown and jewels in the upper-shell link frame (metres)."""
    gold, ruby = [], []
    r_out, r_in, band_h = 0.020, 0.017, 0.010
    band = trimesh.creation.annulus(r_min=r_in, r_max=r_out, height=band_h, sections=48)
    band.apply_translation([0, 0, band_h / 2])
    gold.append(band)
    rim = trimesh.creation.annulus(r_min=r_in - 0.0005, r_max=r_out + 0.0015, height=0.0022, sections=48)
    rim.apply_translation([0, 0, 0.0011])
    gold.append(rim)
    n = 5
    for k in range(n):
        a = 2 * np.pi * k / n
        spike = trimesh.creation.cone(radius=0.0055, height=0.012, sections=16)
        spike.apply_translation([(r_out - 0.0015) * np.cos(a), (r_out - 0.0015) * np.sin(a), band_h - 0.001])
        gold.append(spike)
        ball = trimesh.creation.icosphere(subdivisions=2, radius=0.0022)
        ball.apply_translation([(r_out - 0.0015) * np.cos(a), (r_out - 0.0015) * np.sin(a), band_h + 0.0115])
        gold.append(ball)
        gem = trimesh.creation.icosphere(subdivisions=2, radius=0.0024)
        a2 = a + np.pi / n
        gem.apply_translation([(r_out + 0.0006) * np.cos(a2), (r_out + 0.0006) * np.sin(a2), band_h * 0.5])
        ruby.append(gem)
    # a little velvet cap inside the band
    cap = trimesh.creation.icosphere(subdivisions=3, radius=r_in * 0.98)
    cap.vertices[:, 2] = np.clip(cap.vertices[:, 2], 0, None) * 0.55
    cap.apply_translation([0, 0, 0.003])
    ruby.append(cap)
    tilt = trimesh.transformations.rotation_matrix(np.radians(-9), [1, 0, 0])
    grow = 1.3
    for m in gold + ruby:
        m.apply_scale(grow)
        m.apply_transform(tilt)
        m.apply_translation([cx, cy, top_z - 0.003])
    return gold, ruby


def bib_parts_link_m(front_x: float, top_z: float):
    """A napkin tucked under the display, hanging in front of the chest."""
    w, h, t = 0.060, 0.058, 0.0018
    cloth = trimesh.creation.box(extents=[t, w, h])
    # taper to a soft point: squeeze the lower vertices toward the centre line
    v = cloth.vertices
    s = (v[:, 2] + h / 2) / h  # 0 bottom .. 1 top
    v[:, 1] *= 0.45 + 0.55 * s
    cloth.vertices = v
    stripe = trimesh.creation.box(extents=[t * 1.6, w * 0.80, 0.004])
    stripe.apply_translation([0, 0, h / 2 - 0.010])
    tilt = trimesh.transformations.rotation_matrix(np.radians(24), [0, 1, 0])
    out = []
    for m in (cloth, stripe):
        m.apply_transform(tilt)
        m.apply_translation([front_x + 0.018, 0, top_z - h / 2 + 0.006])
        out.append(m)
    return [out[0]], [out[1]]


def write_ams(path: Path, mesh: trimesh.Trimesh, face_slots: np.ndarray) -> None:
    core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    prod = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    lines = [f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xmlns="{core}">'
             "<resources><object id=\"1\" type=\"model\"><mesh><vertices>"]
    lines += [f'<vertex x="{repr(float(a))}" y="{repr(float(b))}" z="{repr(float(c))}"/>' for a, b, c in verts]
    lines.append("</vertices><triangles>")
    lines += [f'<triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}" paint_color="{PAINT[s]}"/>'
              for f, s in zip(mesh.faces, face_slots)]
    lines.append("</triangles></mesh></object></resources><build/></model>")
    obj = "".join(lines)
    main = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xmlns="{core}" '
            f'xmlns:p="{prod}"><resources><object id="2" type="model"><components>'
            f'<component p:path="/3D/Objects/object_1.model" objectid="1" '
            f'transform="1 0 0 0 1 0 0 0 1 0 0 0"/></components></object></resources>'
            f'<build><item objectid="2"/></build></model>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("3D/3dmodel.model", main)
        z.writestr("3D/Objects/object_1.model", obj)
        z.writestr("Metadata/project_settings.config", json.dumps({"filament_colour": PALETTE}))


def visual_overrides(baseline: Path) -> list:
    root = ET.parse(baseline).getroot()
    out = []
    for body in root.iter("body"):
        for g in body.findall("geom"):
            name, rgba = g.get("name", ""), g.get("rgba")
            if not name.endswith("_visual") or rgba is None or body.get("name") == "upper_shell_link":
                continue
            c = [float(x) for x in rgba.split()]
            if body.get("name") == "base_link":
                new = BASE_RED
            elif c[0] > 0.6 and c[1] < 0.2:          # the original red parts
                new = LIMB_RED
            elif abs(c[0] - 0.68) < 0.02 and abs(c[2] - 0.72) < 0.02:  # grey anodised
                new = LIMB_PEACH
            else:
                continue  # pads, display, sensors keep their baseline look
            out.append({"link": body.get("name"), "geom": name, "rgba": new})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output .skin path")
    ap.add_argument("--author", default="King Crab (poly-videos) contributors")
    ap.add_argument("--license", default="CC0-1.0")
    a = ap.parse_args()
    design = a.design.resolve()
    work = a.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    profile_path = design / "robots/jumper/profile.json"
    profile = json.loads(profile_path.read_text())
    tf = profile["cad_mm_to_parent_m"]

    shell = trimesh.load(design / "robots/jumper/meshes/visual/upper_shell_link.stl", process=False)
    # top of the shell at the crown's footprint, by ray
    cx, cy = -0.012, 0.0
    hit, _, _ = shell.ray.intersects_location([[cx, cy, 0.2]], [[0, 0, -1]])
    top_z = float(hit[:, 2].max())
    front_x = float(shell.bounds[1][0])
    gold, ruby = crown_parts_link_m(top_z, cx, cy)
    napkin, stripe = bib_parts_link_m(front_x - 0.006, 0.004)

    parts = [(shell, 0)] + [(m, 1) for m in gold] + [(m, 3) for m in ruby] + \
            [(m, 2) for m in napkin] + [(m, 3) for m in stripe]
    meshes, slots = [], []
    for m, s in parts:
        mm = m.copy()
        mm.vertices = link_to_cad(np.asarray(mm.vertices), tf)
        meshes.append(mm)
        slots.append(np.full(len(mm.faces), s, dtype=np.uint8))
    combined = trimesh.util.concatenate(meshes)
    combined_slots = np.concatenate(slots)
    stl = work / "king_crab_shell.stl"
    combined.export(stl)

    # Re-read exactly as `assemble` will (trimesh, process=True) and paint that.
    reread = trimesh.load_mesh(stl, process=True)
    if len(reread.faces) != len(combined.faces):
        raise SystemExit("face count changed on reload; cannot paint one-for-one")
    from scipy.spatial import cKDTree
    _, idx = cKDTree(combined.triangles_center).query(reread.triangles_center)
    write_ams(work / "ams.3mf", reread, combined_slots[idx])
    overrides = visual_overrides(design / "robots/jumper/baseline.xml")
    (work / "visual_overrides.json").write_text(json.dumps(overrides, indent=1))
    print(f"shell faces {len(reread.faces)}, crown top {top_z:.4f} m, {len(overrides)} visual overrides")

    sim_dir = work / "simulation"
    py = sys.executable
    run = lambda *args: subprocess.run([py, *map(str, args)], cwd=design, check=True,
                                       capture_output=True, text=True).stdout
    # The exporter `shellflow assemble` calls; it reads `ams.3mf` beside the shell.
    rep = json.loads(run("scripts/export_simulation.py", "--shell", stl, "--output", sim_dir,
                         "--visual-overrides", work / "visual_overrides.json",
                         "--profile", profile_path).strip().splitlines()[-1])
    report = json.loads((sim_dir / "simulation-report.json").read_text())
    mode = json.dumps(report).split('"mode": "')[1].split('"')[0] if '"mode"' in json.dumps(report) else "?"
    colors = report.get("colors") or report.get("color_report") or {}
    print("assemble ok:", rep.get("ok", True), "| color mode:", colors.get("mode", mode))
    if "palette" not in json.dumps(colors):
        raise SystemExit("AMS colours were NOT applied (monochrome fallback): " + json.dumps(colors)[:400])
    out = a.out.resolve()
    if out.exists():
        out.unlink()
    print(run("scripts/shellflow.py", "export-skin", "--simulation", sim_dir, "--profile", profile_path,
              "--id", "king-crab", "--title", "King Crab", "--version", "1.0.0",
              "--author", a.author, "--license", a.license, "--output", out))
    print(run("scripts/shellflow.py", "verify-package", out, "--profile", profile_path, "--mujoco"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

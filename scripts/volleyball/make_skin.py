#!/usr/bin/env python3
"""Build the King Crab (Gold Team #1) display skin as a real `.skin` (kk-skin-package/3).

Same design as the video's render-time skin (`render.py` SKINS["gold_team"] +
`draw_skin`): yellow jersey shell, blue piping and #1, gold crown with red/blue
jewels, blue lower body, red legs and claws. Here it is real geometry, produced
through the jumper-design toolkit's own pipeline:

1. Start from the robot's original upper-shell display mesh, converted to the
   profile's CAD millimetre frame (so the shared seam registers exactly).
2. Add the crown, piping and number as meshes placed on the shell's real surface.
3. Write a matching Bambu-style AMS 3MF with one whole-face paint code per
   triangle (4 palette slots), which the exporter splits into coloured parts.
4. Recolour lower body and limbs through `--visual-overrides` (the protocol's
   only allowed recolour route), run `scripts/export_simulation.py`, then
   `shellflow export-skin` and `verify-package --mujoco`.

    DESIGN=<jumper-design checkout> python tools/make_skin.py OUT_DIR

The added parts overlap the shell; this is a display skin, not a printable shell.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import trimesh

DESIGN = Path(os.environ["DESIGN"]).resolve()
PROFILE = DESIGN / "robots/jumper/profile.json"

PALETTE = ["#FFD11A", "#0D40CC", "#FFC72E", "#D90D1F"]  # jersey, team blue, crown gold, ruby
JERSEY, BLUE, GOLD, RED = range(4)
SHORTS = [0.05, 0.2, 0.65, 1.0]
LIMBS = [0.88, 0.12, 0.08, 1.0]
CLAWS = [0.95, 0.2, 0.12, 1.0]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shell_mesh_m():
    """Original upper-shell display mesh in the upper_shell_link frame (metres)."""
    return trimesh.load_mesh(DESIGN / "robots/jumper/meshes/visual/upper_shell_link.stl", process=False)


def to_cad_mm(points_m, profile):
    T = profile["cad_mm_to_parent_m"]
    R = np.asarray(T["rotation_matrix"], float)
    t = np.asarray(T["translation_m"], float)
    return (np.asarray(points_m, float) - t) @ R / T["scale"]  # R^T (p - t) / s


class Surface:
    def __init__(self, mesh):
        self.v = np.asarray(mesh.vertices, float)

    def top(self, x, y, r=0.012):
        near = self.v[(abs(self.v[:, 0] - x) < r) & (abs(self.v[:, 1] - y) < r)]
        return float(near[:, 2].max()) if len(near) else float(self.v[:, 2].max())


def frame_along(d):
    z = np.asarray(d, float) / (np.linalg.norm(d) + 1e-12)
    x = np.cross([0, 0, 1.0], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, :3] = np.stack([x, y, z], axis=1)
    return M


def capsule(p0, p1, r):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = max(np.linalg.norm(p1 - p0), 1e-4)
    m = trimesh.creation.capsule(height=L, radius=r, count=[12, 12])
    M = frame_along(p1 - p0)
    M[:3, 3] = (p0 + p1) / 2
    m.apply_transform(M)
    return m


def sphere(c, r, sub=2):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    m.apply_translation(c)
    return m


def ellipsoid(c, axes, direction):
    m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    m.apply_scale(axes)
    M = frame_along(direction)
    M[:3, 3] = c
    m.apply_transform(M)
    return m


def cylinder(c, r, half_h, sections=48):
    m = trimesh.creation.cylinder(radius=r, height=2 * half_h, sections=sections)
    m.apply_translation(c)
    return m


def king_crab_parts(surf):
    """(mesh, palette slot) list in the shell link frame, metres. Mirrors render.draw_skin."""
    parts = []

    def s(x, y, lift):
        return [x, y, surf.top(x, y) + lift]

    for y in (-0.064, 0.064):  # piping
        xs = np.linspace(-0.096, 0.042, 13)
        pts = [s(x, y, 0.0025) for x in xs]
        for a, b in zip(pts, pts[1:]):
            parts.append((capsule(a, b, 0.0042), BLUE))
    cx, size = -0.064, 0.05  # jersey number "1", upright seen from behind
    top_, bot = cx + size * 0.5, cx - size * 0.5
    for a, b in ((s(top_, 0, 0.003), s(bot, 0, 0.003)),
                 (s(top_, 0, 0.003), s(top_ - 0.012, 0.012, 0.003)),
                 (s(bot, -0.011, 0.003), s(bot, 0.011, 0.003))):
        parts.append((capsule(a, b, 0.0055), BLUE))
    ccx = -0.012  # crown on the crest
    base_z = surf.top(ccx, 0.0) - 0.002
    r, h = 0.026, 0.009
    parts.append((cylinder([ccx, 0, base_z + h], r, h), GOLD))
    parts.append((cylinder([ccx, 0, base_z + 0.003], r * 1.06, 0.002), GOLD))
    for k in range(5):
        a = k * 2 * math.pi / 5
        out = np.array([math.cos(a), math.sin(a), 0])
        base = np.array([ccx, 0, base_z + 2 * h]) + out * r * 0.9
        tip = base + out * 0.004 + np.array([0, 0, 0.017])
        dv = tip - base
        parts.append((ellipsoid((base + tip) / 2, [0.0065, 0.0045, np.linalg.norm(dv) / 2 + 0.004], dv),
                      GOLD))
        parts.append((sphere(tip, 0.0035, 1), GOLD))
        jewel = np.array([ccx, 0, base_z + h]) + out * (r + 0.0005)
        parts.append((sphere(jewel, 0.0038, 1), RED if k % 2 == 0 else BLUE))
    return parts


def write_ams(path, mesh, labels):
    """Bambu-style 3MF: one external object mesh, identity component, whole-face paint codes."""
    codes = ["4", "8", "0C", "1C"]
    ns = 'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"'
    pns = 'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"'
    verts = "".join(f'<vertex x="{x!r}" y="{y!r}" z="{z!r}"/>'
                    for x, y, z in (tuple(float(c) for c in v) for v in mesh.vertices))
    tris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}" paint_color="{codes[l]}"/>'
                   for (a, b, c), l in zip(mesh.faces.tolist(), labels.tolist()))
    obj = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" {ns}><resources>'
           f'<object id="1" type="model"><mesh><vertices>{verts}</vertices><triangles>{tris}'
           f'</triangles></mesh></object></resources><build/></model>')
    main = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" {ns} {pns}><resources>'
            f'<object id="2" type="model"><components><component p:path="/3D/Objects/object_1.model" '
            f'objectid="1" transform="1 0 0 0 1 0 0 0 1 0 0 0"/></components></object></resources>'
            f'<build><item objectid="2"/></build></model>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("3D/3dmodel.model", main)
        z.writestr("3D/Objects/object_1.model", obj)
        z.writestr("Metadata/project_settings.config", json.dumps({"filament_colour": PALETTE}))


def visual_overrides():
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(DESIGN / "robots/jumper/baseline.xml"))
    out = []
    for i in range(m.ngeom):
        g = m.geom(i).name
        if not g.endswith("_visual") or m.geom_contype[i] or m.geom_conaffinity[i]:
            continue
        link = m.body(m.geom_bodyid[i]).name
        if link == "upper_shell_link" or any(k in link for k in ("display", "tof", "camera")):
            continue
        if link == "base_link":
            rgba = SHORTS
        elif m.geom_rgba[i][:3].mean() <= 0.15:
            continue  # keep dark trim (grip inserts etc.) as the baseline has it
        elif "finger" in link or "palm" in link:
            rgba = CLAWS
        else:
            rgba = LIMBS
        out.append({"link": link, "geom": g, "rgba": [float(c) for c in rgba]})
    return out


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile = json.loads(PROFILE.read_text())
    base = shell_mesh_m()
    surf = Surface(base)
    meshes, labels = [base], [np.full(len(base.faces), JERSEY)]
    for m, slot in king_crab_parts(surf):
        meshes.append(m)
        labels.append(np.full(len(m.faces), slot))
    combo = trimesh.util.concatenate(meshes)
    lab = np.concatenate(labels)
    combo.vertices = to_cad_mm(combo.vertices, profile)
    work = out / "work"
    work.mkdir(exist_ok=True)
    stl = work / "shell.stl"
    combo.export(stl)
    # The exporter loads with process=True; build the AMS from exactly that mesh.
    loaded = trimesh.load_mesh(stl, process=True)
    tree_src = trimesh.Trimesh(combo.vertices, combo.faces, process=False)
    from scipy.spatial import cKDTree

    _, idx = cKDTree(tree_src.triangles_center).query(loaded.triangles_center)
    write_ams(work / "ams.3mf", loaded, lab[idx])
    ov = work / "visual-overrides.json"
    ov.write_text(json.dumps(visual_overrides(), indent=1))
    print(f"[skin] shell {len(loaded.faces)} faces, palette {PALETTE}, "
          f"{len(json.loads(ov.read_text()))} visual overrides")

    py = sys.executable
    sim = out / "simulation"
    subprocess.run([py, str(DESIGN / "scripts/export_simulation.py"), "--shell", str(stl),
                    "--output", str(sim), "--profile", str(PROFILE),
                    "--visual-overrides", str(ov)], check=True, cwd=DESIGN)
    skin = out / "king-crab-gold-1.skin"
    cmd = [py, str(DESIGN / "scripts/shellflow.py"), "export-skin", "--simulation", str(sim),
           "--profile", str(PROFILE), "--id", "king-crab-gold-1",
           "--title", "King Crab (Gold Team #1)", "--version", "1.0.0",
           "--author", os.environ.get("SKIN_AUTHOR", ""),
           "--license", os.environ.get("SKIN_LICENSE", ""), "--output", str(skin)]
    if os.environ.get("SKIN_PREVIEW"):
        cmd += ["--preview", os.environ["SKIN_PREVIEW"]]
    subprocess.run(cmd, check=True, cwd=DESIGN)
    subprocess.run([py, str(DESIGN / "scripts/shellflow.py"), "verify-package", str(skin),
                    "--profile", str(PROFILE), "--mujoco"], check=True, cwd=DESIGN)
    print(f"[skin] wrote {skin} sha256={sha(skin)}")


if __name__ == "__main__":
    main()

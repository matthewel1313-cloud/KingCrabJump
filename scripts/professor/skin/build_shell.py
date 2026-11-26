"""Build the King Crab Professor upper-shell appearance: shell + crown + glasses + tweed paint.

Run in the jumper-design environment, from anywhere:

    python skin/build_shell.py --design ../kingkongrobotics/jumper-design --out skin/work

Writes, in the profile's original CAD millimetre frame (the frame `assemble` expects):

* ``shell.stl``  -- the stock upper shell (subdivided once, surface unchanged) with a crown
  and a pair of round glasses added as extra display-only geometry;
* ``ams.3mf``    -- the same triangles, in the same order, each painted with one of four
  palette slots (the whole-face paint codes ``shellflow.simulation.colors_from_ams`` reads);
* ``visual-overrides.json`` -- limb / lower-body RGBA overrides for ``.skin/3``.

Everything here is original procedural geometry; nothing is imported from a third party.
CAD frame (from robots/jumper/profile.json): robot forward = -Y, robot left = +X, up = +Z.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import trimesh

# Palette slots -> paint codes understood by shellflow's AMS adapter.
PALETTE = ["#4E3524", "#8C6A47", "#D9AE2F", "#A4482A"]  # chocolate, camel, gold, rust
CODES = ["4", "8", "0C", "1C"]
DARK, LIGHT, GOLD, RUST = range(4)

# Placement, measured from the stock shell (see README): the top ridge peaks at ~78.7 mm
# around y = -70..-50, just behind the display window (y -105..-80, z 38..72).
CROWN_CENTER = np.array([0.0, -52.0, 73.0])   # base sunk ~4 mm into the shell top
CROWN_TILT_DEG = 12.0                         # jaunty tilt, rolled toward robot's left
CROWN_R = 19.0
CROWN_BAND = 10.0
CROWN_SPIKE = 15.0
CROWN_POINTS = 5


def rot(axis, deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), axis)


def crown_mesh() -> trimesh.Trimesh:
    """A five-pointed crown: a thick wall of revolution whose top edge zig-zags, plus orbs."""
    n = CROWN_POINTS * 24
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    phase = (theta * CROWN_POINTS / (2 * np.pi)) % 1.0
    top = CROWN_BAND + CROWN_SPIKE * (1 - np.abs(phase - 0.5) * 2)  # peaks at phase 0.5
    wall = 2.2
    verts, faces = [], []
    rings = []
    for radius, flare in ((CROWN_R, 1.0), (CROWN_R - wall, 1.0)):
        bottom = np.c_[radius * np.cos(theta), radius * np.sin(theta), np.zeros(n)]
        # flare the spikes outward slightly, like a real crown
        r_top = radius + 2.5 * (top - CROWN_BAND) / CROWN_SPIKE
        upper = np.c_[r_top * np.cos(theta), r_top * np.sin(theta), top]
        rings.append((bottom, upper))
    (ob, ot), (ib, it) = rings
    verts = np.vstack([ob, ot, ib, it])
    OB, OT, IB, IT = 0, n, 2 * n, 3 * n
    for i in range(n):
        j = (i + 1) % n
        faces += [[OB + i, OB + j, OT + j], [OB + i, OT + j, OT + i]]      # outer wall
        faces += [[IB + i, IT + j, IB + j], [IB + i, IT + i, IT + j]]      # inner wall
        faces += [[OT + i, OT + j, IT + j], [OT + i, IT + j, IT + i]]      # top rim
        faces += [[OB + i, IB + j, OB + j], [OB + i, IB + i, IB + j]]      # bottom rim
    band = trimesh.Trimesh(verts, np.asarray(faces), process=False)
    parts = [band]
    for k in range(CROWN_POINTS):
        a = 2 * np.pi * (k + 0.5) / CROWN_POINTS
        r = CROWN_R + 2.5 - wall / 2
        orb = trimesh.creation.icosphere(subdivisions=2, radius=2.6)
        orb.apply_translation([r * np.cos(a), r * np.sin(a), CROWN_BAND + CROWN_SPIKE + 1.8])
        parts.append(orb)
    # A jewel boss on the front of the band (robot forward is -Y).
    gem = trimesh.creation.icosphere(subdivisions=2, radius=3.0)
    gem.apply_translation([0, -(CROWN_R + 0.8), CROWN_BAND * 0.5])
    parts.append(gem)
    crown = trimesh.util.concatenate(parts)
    crown.apply_transform(rot([0, 1, 0], CROWN_TILT_DEG))   # roll about the forward axis
    crown.apply_transform(rot([1, 0, 0], -6.0))             # and a touch back
    crown.apply_translation(CROWN_CENTER)
    return crown


def glasses_mesh(design: Path) -> trimesh.Trimesh:
    """Two round rims + bridge + temples, a few mm proud of the display's front face."""
    profile = json.loads((design / "robots/jumper/profile.json").read_text())
    t = profile["cad_mm_to_parent_m"]
    R, T, s = np.asarray(t["rotation_matrix"]), np.asarray(t["translation_m"]), t["scale"]
    disp = trimesh.load_mesh(design / "robots/jumper/meshes/visual/display_module_link.stl")
    # display frame -> base -> upper shell frame -> CAD mm (positions from baseline.xml)
    v = disp.vertices + np.array([0.0802718, 0, 0.0495238]) - np.array([-0.0073137, 0, 0.0315])
    cad = ((v - T) @ R) / s
    # The window's face slopes up and back; its outward normal points forward and up.
    slope = np.array([0.0, 24.6, 33.6]); slope /= np.linalg.norm(slope)
    normal = np.array([0.0, -slope[2], slope[1]])
    center = np.array([0.0, (cad[:, 1].min() + cad[:, 1].max()) / 2, (cad[:, 2].min() + cad[:, 2].max()) / 2])
    front = float(((cad - center) @ normal).max())
    origin = center + normal * (front + 3.0)
    # local frame on the window face: u = robot-left (+X), w = up the slope, n = out
    u, w, nrm = np.array([1.0, 0, 0]), slope, normal
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = u, w, nrm, origin
    parts = []
    rim_r, tube = 13.0, 1.6
    for side in (-1, 1):
        rim = trimesh.creation.torus(major_radius=rim_r, minor_radius=tube, major_sections=64, minor_sections=12)
        rim.apply_translation([side * 21.5, -1.0, 0])
        parts.append(rim)
        # temple: from the rim's outer edge back into the shell side
        temple = trimesh.creation.cylinder(radius=1.2, segment=[[side * 34.0, 0, 0], [side * 44.0, 6.0, -14.0]])
        parts.append(temple)
    bridge = trimesh.creation.cylinder(radius=1.4, segment=[[-8.6, 1.5, 0.5], [8.6, 1.5, 0.5]])
    parts.append(bridge)
    g = trimesh.util.concatenate(parts)
    g.apply_transform(M)
    return g


def tweed_colors(mesh: trimesh.Trimesh, seed: int = 7) -> np.ndarray:
    """Herringbone tweed with a rust windowpane overcheck, by face centre (mm)."""
    rng = np.random.default_rng(seed)
    c = mesh.triangles_center
    nz = np.abs(mesh.face_normals)
    # Triplanar: use the plane the face mostly lies in.
    a = np.where(nz[:, 2] > 0.6, c[:, 0], np.where(nz[:, 0] > nz[:, 1], c[:, 1], c[:, 0]))
    b = np.where(nz[:, 2] > 0.6, c[:, 1], c[:, 2])
    col = np.floor(a / 12.0).astype(int)
    u = np.where(col % 2 == 0, a + b, a - b)
    stripe = np.floor(u / 4.4).astype(int) % 2
    out = np.where(stripe == 1, LIGHT, DARK).astype(np.uint8)
    flecks = rng.random(len(out))
    out[(flecks < 0.06)] = DARK
    out[(flecks > 0.96)] = LIGHT
    # windowpane overcheck: thin rust lines every 24 mm both ways
    over = (np.abs(((a + 18) % 36) - 18) < 0.9) | (np.abs(((b + 18) % 36) - 18) < 0.9)
    out[over] = RUST
    return out


def write_3mf(path: Path, mesh: trimesh.Trimesh, slots: np.ndarray) -> None:
    v, f = np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces)
    obj = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
           '<resources><object id="1" type="model"><mesh><vertices>']
    obj += [f'<vertex x="{x!r}" y="{y!r}" z="{z!r}"/>' for x, y, z in v.tolist()]
    obj.append('</vertices><triangles>')
    obj += [f'<triangle v1="{a}" v2="{b}" v3="{c}" paint_color="{CODES[k]}"/>'
            for (a, b, c), k in zip(f.tolist(), slots.tolist())]
    obj.append('</triangles></mesh></object></resources><build/></model>')
    main = ('<?xml version="1.0" encoding="UTF-8"?>'
            '<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
            'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">'
            '<resources><object id="2" type="model"><components>'
            '<component p:path="/3D/Objects/shell.model" objectid="1"/>'
            '</components></object></resources><build><item objectid="2"/></build></model>')
    settings = {"filament_colour": PALETTE}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("3D/3dmodel.model", main)
        z.writestr("3D/Objects/shell.model", "\n".join(obj))
        z.writestr("Metadata/project_settings.config", json.dumps(settings))


def rgba(hexcode):
    return [round(int(hexcode[i:i + 2], 16) / 255, 6) for i in (1, 3, 5)] + [1.0]


def visual_overrides():
    cream, gold, brown, deep, pen = "#EDE2C6", "#C9A227", "#5A3E2B", "#2E2019", "#C8102E"
    out = [{"link": "base_link", "geom": "base_link_visual", "rgba": rgba(deep)}]
    for side in ("LF", "RF"):
        for part, colour in (("shoulder", cream), ("upper_arm", cream), ("forearm", gold),
                             ("palm", cream), ("palm_pad_f", brown), ("palm_pad_b", brown),
                             ("finger", cream), ("finger_tip", pen if side == "RF" else gold),
                             ("finger_grip_insert", brown), ("palm_grip_insert", brown)):
            out.append({"link": f"{side}_{part}_link", "geom": f"{side}_{part}_link_visual",
                        "rgba": rgba(colour)})
    for side in ("LM", "RM", "LR", "RR"):
        for part, colour in (("hip", cream), ("thigh", gold), ("calf", cream), ("foot_tip", deep)):
            out.append({"link": f"{side}_{part}_link", "geom": f"{side}_{part}_link_visual",
                        "rgba": rgba(colour)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--stock", type=Path, required=True, help="stock upper shell in CAD mm")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    shell = trimesh.load_mesh(a.stock, process=True)
    # Uniform-ish triangles so the woven pattern reads; 4 mm keeps the total under the
    # exporter's 100k display-face budget, so no decimation re-mixes the paint.
    sv, sf = trimesh.remesh.subdivide_to_size(shell.vertices, shell.faces, max_edge=4.0)
    shell = trimesh.Trimesh(sv, sf, process=False)
    crown, glasses = crown_mesh(), glasses_mesh(a.design)
    slots = np.concatenate([tweed_colors(shell),
                            np.full(len(crown.faces), GOLD, np.uint8),
                            np.full(len(glasses.faces), GOLD, np.uint8)])
    combined = trimesh.util.concatenate([shell, crown, glasses])
    stl = a.out / "shell.stl"
    combined.export(stl)
    # Reload exactly as `assemble` will (process=True) so the 3MF matches triangle for triangle.
    reloaded = trimesh.load_mesh(stl, process=True)
    assert len(reloaded.faces) == len(slots), (len(reloaded.faces), len(slots))
    write_3mf(a.out / "ams.3mf", reloaded, slots)
    (a.out / "visual-overrides.json").write_text(json.dumps(visual_overrides(), indent=1))
    counts = {PALETTE[i]: int((slots == i).sum()) for i in range(4)}
    print(json.dumps({"faces": len(slots), "shell_faces": len(shell.faces), "crown_faces": len(crown.faces),
                      "glasses_faces": len(glasses.faces), "paint_counts": counts,
                      "extents_mm": reloaded.extents.round(2).tolist()}))


if __name__ == "__main__":
    main()

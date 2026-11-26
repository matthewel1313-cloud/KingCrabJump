#!/usr/bin/env python3
"""Rebuild the Surgical Tech film's packages as real, validated Jumper packages.

The film was made with a prototype appearance overlay (`kingcrab-skin-overlay/1`,
not a `.skin/3`), a `kk-scene-package/1` operating room, and Jumper's shipped
example `.app`. This script produces proper packages for the same design and
scene, through jumper-design's own pipeline:

* `king-crab-surgeon.skin` (`kk-skin-package/3`): the overlay's crown, cap and
  mask, at the overlay's exact positions, built into the upper-shell display
  mesh with a 4-slot AMS paint (white, gold, cap blue, ruby). The overlay's
  recolours become `.skin/3` `visual_overrides`. Everything the overlay put on
  `base_link` is rigid with the shell (fixed offset, no rotation), so it maps
  1:1 into the shell frame.
* `king-crab-operating-room.map` (`kk-scene-package/2`): the film's own
  `scene.xml` (tray, drape, Mayo stand, four free-body instruments, mocap
  glove), with be-web/1 appearance, the film's spawn on the tray, and the skin
  above as its default robot.

    DESIGN=<jumper-design checkout> python build_packages.py <surgeon king_crab dir> OUT_DIR

The `.app` is built separately with Jumper's `scripts/deploy.py` from `app/manifests.json`.
"""

from __future__ import annotations

import json
import math
import os
import sys
import xml.etree.ElementTree as E
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "volleyball"))
import make_skin as ms  # noqa: E402  (shared, tested helpers: meshes, CAD frame, AMS writer)

DESIGN = ms.DESIGN
PROFILE = ms.PROFILE
sys.path[:0] = [str(DESIGN / "src"), str(DESIGN / "scripts")]

PALETTE = ["#E6E8ED", "#F5BD29", "#5C9EEB", "#CC0D40"]  # white, gold, cap blue, ruby
WHITE, GOLD, BLUE, RUBY = range(4)
SHELL_IN_BASE = np.array([-0.0073137, 0.0, 0.0315])  # upper_shell_link origin in base_link

# the overlay's recolours (palm and finger stay the baseline red)
WHITE_RGBA = [0.90, 0.91, 0.93, 1.0]
LIGHT_GRAY = [0.74, 0.76, 0.80, 1.0]
MID_GRAY = [0.62, 0.65, 0.69, 1.0]
SPAWN = [-0.001, 0.153, 0.0]  # the film's spawn: HANDLE_X - 0.03 - 0.161, scalpel y - 0.047


def overrides():
    out = [{"link": "base_link", "geom": "base_link_visual", "rgba": MID_GRAY}]
    for side in ("LF", "RF"):
        out += [{"link": f"{side}_shoulder_link", "geom": f"{side}_shoulder_link_visual", "rgba": WHITE_RGBA},
                {"link": f"{side}_upper_arm_link", "geom": f"{side}_upper_arm_link_visual", "rgba": LIGHT_GRAY}]
    for leg in ("LM", "RM", "LR", "RR"):
        out += [{"link": f"{leg}_hip_link", "geom": f"{leg}_hip_link_visual", "rgba": WHITE_RGBA},
                {"link": f"{leg}_thigh_link", "geom": f"{leg}_thigh_link_visual", "rgba": LIGHT_GRAY},
                {"link": f"{leg}_calf_link", "geom": f"{leg}_calf_link_visual", "rgba": WHITE_RGBA}]
    return out


def S(p):
    """base_link frame -> upper_shell_link frame."""
    return np.asarray(p, float) - SHELL_IN_BASE


def surgeon_parts():
    """The overlay's accessories (positions in base_link, as in its build_skin.py)."""
    parts = []
    cap = ms.ellipsoid(S([-0.030, 0.0, 0.066]), [0.080, 0.078, 0.026], [0, 0, 1])
    band = ms.ellipsoid(S([-0.030, 0.0, 0.060]), [0.084, 0.082, 0.010], [0, 0, 1])
    parts += [(cap, BLUE), (band, BLUE)]
    cx, cy, cz = -0.028, 0.0, 0.095
    crown = trimesh.creation.cylinder(radius=0.024, height=0.014, sections=48)
    M = trimesh.transformations.rotation_matrix(math.radians(-8), [0, 1, 0])
    M[:3, 3] = S([cx, cy, cz])
    crown.apply_transform(M)
    parts.append((crown, GOLD))
    for k in range(5):
        a = 2 * math.pi * k / 5
        p0 = [cx + 0.021 * math.cos(a), cy + 0.021 * math.sin(a), cz + 0.004]
        p1 = [cx + 0.023 * math.cos(a), cy + 0.023 * math.sin(a), cz + 0.021]
        parts.append((ms.capsule(S(p0), S(p1), 0.0042), GOLD))
        gem = [cx + 0.023 * math.cos(a), cy + 0.023 * math.sin(a), cz + 0.023]
        parts.append((ms.sphere(S(gem), 0.0045, 1), RUBY if k % 2 == 0 else GOLD))
    mask = trimesh.creation.box(extents=[0.008, 0.080, 0.028])
    mask.apply_translation(S([0.097, 0.007, 0.017]))
    parts.append((mask, WHITE))
    for z in (0.010, 0.017, 0.024):
        pleat = trimesh.creation.box(extents=[0.0084, 0.078, 0.0018])
        pleat.apply_translation(S([0.0972, 0.007, z]))
        parts.append((pleat, BLUE))  # pleat seams in cap blue (the 4-slot palette has no 5th tone)
    for y in (0.047, -0.033):
        end = [0.050, y + (0.04 if y > 0 else -0.05), 0.040]
        parts.append((ms.capsule(S([0.096, y, 0.017]), S(end), 0.0016), WHITE))
    return parts


def build_skin(out: Path, preview: Path | None):
    import subprocess

    from scipy.spatial import cKDTree

    profile = json.loads(PROFILE.read_text())
    base = ms.shell_mesh_m()
    meshes, labels = [base], [np.full(len(base.faces), WHITE)]
    for m, slot in surgeon_parts():
        meshes.append(m)
        labels.append(np.full(len(m.faces), slot))
    combo = trimesh.util.concatenate(meshes)
    lab = np.concatenate(labels)
    combo.vertices = ms.to_cad_mm(combo.vertices, profile)
    work = out / "skin_work"
    work.mkdir(parents=True, exist_ok=True)
    stl = work / "shell.stl"
    combo.export(stl)
    loaded = trimesh.load_mesh(stl, process=True)
    _, idx = cKDTree(combo.triangles_center).query(loaded.triangles_center)
    ms.PALETTE[:] = PALETTE
    ms.write_ams(work / "ams.3mf", loaded, lab[idx])
    ov = work / "visual-overrides.json"
    ov.write_text(json.dumps(overrides(), indent=1))
    py = sys.executable
    sim = work / "simulation"
    subprocess.run([py, str(DESIGN / "scripts/export_simulation.py"), "--shell", str(stl), "--output", str(sim),
                    "--profile", str(PROFILE), "--visual-overrides", str(ov)], check=True, cwd=DESIGN,
                   stdout=subprocess.DEVNULL)
    skin = out / "king-crab-surgeon.skin"
    cmd = [py, str(DESIGN / "scripts/shellflow.py"), "export-skin", "--simulation", str(sim), "--profile",
           str(PROFILE), "--id", "king-crab-surgeon", "--title", "King Crab, Surgical Tech", "--version", "1.0.0",
           "--author", "matthewel1313-cloud", "--license", "Apache-2.0", "--output", str(skin)]
    if preview:
        cmd += ["--preview", str(preview)]
    subprocess.run(cmd, check=True, cwd=DESIGN, stdout=subprocess.DEVNULL)
    return skin, sim


def build_map(src: Path, skin: Path, out: Path):
    from shellflow.package_io import file_records
    from shellflow.map_package import export_environment
    from shellflow.web_appearance import complete_appearance

    root = E.fromstring((src / "scene/scene.xml").read_bytes())
    files = {"scene.xml": E.tostring(root, encoding="utf-8", xml_declaration=True),
             "README.md": README_MAP.encode()}
    world = root.find("worldbody")
    props = []
    for b in world.iter("body"):
        f = b.find("freejoint")
        if f is not None:
            props.append(dict(name=b.get("name"), body=b.get("name"), freeJoint=True, freeJointName=f.get("name"),
                              geoms=[g.get("name") for g in b.iter("geom")],
                              position=list(map(float, b.get("pos").split()))))
    asset = root.find("asset")
    counts = {k: len(list((asset if k in ("textures", "materials", "meshes") else world).iter(t)))
              for k, t in [("geoms", "geom"), ("bodies", "body"), ("lights", "light"), ("textures", "texture"),
                           ("materials", "material"), ("meshes", "mesh")]}
    manifest = dict(schema="kk-scene-package/1", id="king-crab-operating-room", title="King Crab: OR Instrument Tray",
                    version="1.1.0", use="watching", exportedBy=dict(repo="KingCrabJump", tool="build_packages.py"),
                    source=dict(repository="KingCrabJump", authoredGround=True), requiredCapabilities=["rigid"],
                    world=dict(file="scene.xml", assetDir="", terrainType="none", ground=dict(geom="tray_top"),
                               counts=counts, attach=dict(prefix="or_")),
                    props=props, flex=[], lights=[dict(name=l.get("name")) for l in world.iter("light") if l.get("name")],
                    cameras=[], spawn=dict(position=SPAWN, yaw=0), files=file_records(files))
    files["scene-package.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    files = complete_appearance(files)
    thumb = out / "map_thumbnail.png"
    render_thumbnail(files, thumb)
    archive = out / "king-crab-operating-room.map"
    if archive.exists():
        archive.unlink()
    export_environment(files, archive, skin, PROFILE, preview=thumb)
    return archive


def render_thumbnail(files, output):
    import mujoco
    from preview_simulation import save_png

    manifest = json.loads(files["scene-package.json"])
    world = E.fromstring(files[manifest["world"]["file"]])
    assets = {n: b for n, b in files.items() if not n.endswith(".json") and n != "README.md"}
    m = mujoco.MjModel.from_xml_string(E.tostring(world, encoding="unicode"), assets=assets)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1200, 900
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.15, 0.0, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 1.3, 205, -38
    with mujoco.Renderer(m, height=900, width=1200) as r:
        r.update_scene(d, cam)
        save_png(output, r.render())


README_MAP = """# King Crab: OR Instrument Tray

The operating-room scene from "King Crab, Surgical Tech" (#Jumper #CrabRobot). Units are metres; X
points forward and Z up. The stainless tray top is the ground at z = 0. A blue sterile drape, a
Mayo stand and four free-body instruments (scalpel, forceps, scissors, gauze) sit at claw height.
`glove_hand` is a kinematic mocap body with no collision; in the film the control script moves it
and hands tools to it with weld constraints, which are app/script behaviour and not part of this
static map. The robot spawns at (-0.001, 0.153) facing +X, as in the film. Default robot: the
King Crab Surgical Tech skin. No controller or scoring is included.
"""


def main():
    src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    preview = os.environ.get("SKIN_PREVIEW")  # a render of this skin (see README); none on the first pass
    skin, sim = build_skin(out, Path(preview) if preview else None)
    print("[surgeon] skin", skin)
    print("[surgeon] map", build_map(src, skin, out))


if __name__ == "__main__":
    main()

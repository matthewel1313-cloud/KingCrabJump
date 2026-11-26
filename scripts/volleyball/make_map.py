#!/usr/bin/env python3
"""Build the beach-volleyball `.map` packages (kk-scene-package/2) with jumper-design.

Two maps, one per video build:

* `beach-volleyball-court`      -- the court from video 1: net 0.9 m in front of
                                   the spawn, top 0.30 m.
* `beach-volleyball-real-ball`  -- the court from video 2: net 0.47 m in front,
                                   top 0.27 m (where the simulated bump really
                                   clears it), plus a physical 9 cm / 50 g
                                   volleyball prop.

Both bundle the King Crab skin as the default robot. Follows the toolkit's own
authored-map pattern (`scripts/build_authored_challenges.py`): authored MJCF ->
`complete_appearance` (be-web/1 visual-appearance.json) -> `export_environment`
with the default skin -> `verify_package --mujoco` -> `compose_default`.

    DESIGN=<jumper-design checkout> python tools/make_map.py SKIN OUT_DIR
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as E
from pathlib import Path

DESIGN = Path(os.environ["DESIGN"]).resolve()
sys.path[:0] = [str(DESIGN / "src"), str(DESIGN / "scripts")]

from shellflow.package_io import file_records, read_archive  # noqa: E402
from shellflow.map_package import export_environment, compose_default, MAP_LIMITS  # noqa: E402
from shellflow.packages import verify_package  # noqa: E402
from shellflow.web_appearance import complete_appearance  # noqa: E402

PROFILE = DESIGN / "robots/jumper/profile.json"
SAND = (0.82, 0.69, 0.50, 1)
LINE = (0.95, 0.95, 0.92, 1)
NET = (0.08, 0.08, 0.10, 1)
POST = (0.85, 0.85, 0.88, 1)
GROUND_FRICTION = "1 .005 .0001"  # same as the toolkit's authored maps


def vec(v):
    return " ".join(f"{x:.6g}" for x in v)


class Court:
    def __init__(self, ident, title, net_x, net_top, ball=None):
        self.ident, self.title, self.net_x = ident, title, net_x
        self.root = E.Element("mujoco", model=ident)
        E.SubElement(self.root, "compiler", angle="radian")
        self.asset = E.SubElement(self.root, "asset")
        self.world = E.SubElement(self.root, "worldbody")
        E.SubElement(self.world, "geom", name="ground", type="plane", size="20 20 .1",
                     rgba=vec(SAND), friction=GROUND_FRICTION)
        E.SubElement(self.world, "light", name="sun", pos="0 0 7", dir="0 0 -1",
                     directional="true", diffuse=".25 .25 .25")
        self.build(net_x, net_top)
        if ball:
            self.add_ball(ball)

    def box(self, name, pos, size, rgba, visual=False):
        a = dict(name=name, type="box", pos=vec(pos), size=vec(size), rgba=vec(rgba),
                 friction=GROUND_FRICTION)
        if visual:
            a.update(contype="0", conaffinity="0", mass="0")
        return E.SubElement(self.world, "geom", **a)

    def build(self, net_x, net_top):
        half_w, length = 0.6, 1.4
        for i, y in enumerate((-half_w, half_w)):  # sidelines
            self.box(f"line_side_{i}", (net_x, y, 0.001), (length, 0.012, 0.001), LINE, True)
        for i, x in enumerate((net_x - length, net_x + length)):  # end lines
            self.box(f"line_end_{i}", (x, 0, 0.001), (0.012, half_w, 0.001), LINE, True)
        self.box("line_center", (net_x, 0, 0.001), (0.006, half_w, 0.001), LINE, True)
        net_h = 0.11
        for i, y in enumerate((-half_w - 0.05, half_w + 0.05)):  # solid posts
            E.SubElement(self.world, "geom", name=f"net_post_{i}", type="cylinder",
                         pos=vec((net_x, y, net_top / 2)), size=vec((0.012, net_top / 2 + 0.01)),
                         rgba=vec(POST), friction=GROUND_FRICTION)
        self.box("net_tape_top", (net_x, 0, net_top - 0.012), (0.006, half_w + 0.05, 0.012), LINE)
        self.box("net_tape_bottom", (net_x, 0, net_top - net_h), (0.004, half_w + 0.05, 0.006),
                 LINE, True)
        n = 26
        for i in range(n + 1):  # visual mesh strings
            y = -half_w - 0.05 + i * (2 * half_w + 0.1) / n
            self.box(f"net_v_{i}", (net_x, y, net_top - net_h / 2 - 0.012),
                     (0.002, 0.0015, net_h / 2), NET, True)
        for j in range(1, 5):
            self.box(f"net_h_{j}", (net_x, 0, net_top - 0.012 - j * net_h / 5),
                     (0.002, half_w + 0.05, 0.0015), NET, True)

    def add_ball(self, pos):
        b = E.SubElement(self.world, "body", name="prop:volleyball", pos=vec(pos))
        E.SubElement(b, "freejoint", name="prop:volleyball:free")
        E.SubElement(b, "geom", name="volleyball", type="sphere", size="0.045", mass="0.05",
                     rgba="0.97 0.97 0.95 1", condim="6", friction="0.6 0.005 0.01",
                     solref="0.01 0.15")

    def files(self, readme):
        files = {"scene.xml": E.tostring(self.root, encoding="utf-8", xml_declaration=True),
                 "README.md": readme.encode("utf-8")}
        props = []
        for b in self.world.iter("body"):
            f = b.find("freejoint")
            if f is not None:
                props.append(dict(name=b.get("name"), body=b.get("name"), freeJoint=True,
                                  freeJointName=f.get("name"),
                                  geoms=[g.get("name") for g in b.iter("geom")],
                                  position=list(map(float, b.get("pos").split()))))
        counts = {k: len(list((self.asset if k in ("textures", "materials", "meshes") else self.world).iter(t)))
                  for k, t in [("geoms", "geom"), ("bodies", "body"), ("lights", "light"),
                               ("textures", "texture"), ("materials", "material"), ("meshes", "mesh")]}
        manifest = dict(
            schema="kk-scene-package/1", id=self.ident, title=self.title, version="1.0.0",
            use="watching",
            exportedBy=dict(repo="poly-videos", tool="tools/make_map.py"),
            source=dict(repository="poly-videos", authoredGround=True),
            requiredCapabilities=["rigid"],
            world=dict(file="scene.xml", assetDir="", terrainType="none", ground=dict(geom="ground"),
                       counts=counts, attach=dict(prefix="scn_")),
            props=props, flex=[], lights=[dict(name="sun")], cameras=[],
            spawn=dict(position=[0.0, 0.0, 0.0], yaw=0), files=file_records(files))
        files["scene-package.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
        return complete_appearance(files)


README_COURT = """# Beach Volleyball Court

A beach-volleyball court sized to Jumper, from the King Crab volleyball video (#Jumper #CrabRobot).
Units are metres; X points forward from the spawn and Z points up. The robot spawns at the origin
facing +X. The court is 2.8 m x 1.2 m with the net across it 0.9 m in front of the spawn; the net
top is 0.30 m high. Posts and the top tape have collision; court lines and the net mesh are visual
only (zero mass, no collision). Ground friction matches the toolkit's authored maps.

Default robot: King Crab (Gold Team #1) skin. This package has no controller, ball or scoring; in
the video the ball was an animated visual prop. See the real-ball map for a physical ball.
"""

README_REAL = """# Beach Volleyball Court (Real Ball)

The court from the King Crab real-physics volleyball video (#Jumper #CrabRobot). Units are metres;
X points forward from the spawn and Z points up; the robot spawns at the origin facing +X. The net
crosses the court 0.47 m in front of the spawn with its top at 0.27 m. Posts and the top tape have
collision; court lines and the net mesh are visual only.

`prop:volleyball` is a physical ball: 0.045 m radius, 0.05 kg, condim 6, friction 0.6/0.005/0.01,
solref 0.01/0.15. It rests on the sand at (0.30, 0.35). A static map cannot launch it, so the toss
is the app's job. The video's toss, applied at each reset, was: position (0.19, 0.15, 0.16) m in
map coordinates, velocity (0, -0.3846, 1.913) m/s, with jumper.jump starting at the same moment.
In simulation the jump's claws then strike the ball at about 0.42 s, it peaks about 0.37 m high
and lands about 0.75 m in front (three attempts within 2 cm). No scoring or controller is included.
"""


def render_thumbnail(files, output, net_x):
    """Environment-only thumbnail (no robot), court framed with margins.

    Presentation-only edits on a copy, never written into the map: shadows off
    (a 20 m plane under a directional light shows acne at grazing angles), a
    moderate headlight, and a steep enough camera that the sand fills the frame.
    """
    import mujoco
    import numpy as np
    from preview_simulation import save_png

    manifest = json.loads(files["scene-package.json"])
    world = E.fromstring(files[manifest["world"]["file"]])
    for light in world.iter("light"):
        light.set("castshadow", "false")
    vis = E.SubElement(world, "visual")
    E.SubElement(vis, "headlight", ambient=".42 .40 .38", diffuse=".45 .44 .42", specular="0 0 0")
    E.SubElement(vis, "global", offwidth="1200", offheight="900")
    assets = {n: b for n, b in files.items() if not n.endswith(".json") and n != "README.md"}
    m = mujoco.MjModel.from_xml_string(E.tostring(world, encoding="unicode"), assets=assets)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [net_x, 0, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 4.3, 215, -40
    with mujoco.Renderer(m, height=900, width=1200) as r:
        r.update_scene(d, cam)
        save_png(output, r.render())


def main():
    skin = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    import mujoco
    from preview_simulation import save_png

    builds = [
        (Court("beach-volleyball-court", "Beach Volleyball Court", 0.9, 0.30), README_COURT),
        (Court("beach-volleyball-real-ball", "Beach Volleyball Court (Real Ball)", 0.47, 0.27,
               ball=(0.30, 0.35, 0.046)), README_REAL),
    ]
    for court, readme in builds:
        files = court.files(readme)
        thumb = out / f"{court.ident}.thumbnail.png"
        render_thumbnail(files, thumb, court.net_x)
        archive = out / f"{court.ident}.map"
        if archive.exists():
            archive.unlink()
        export_environment(files, archive, skin, PROFILE, preview=thumb)
        check = verify_package(archive, platform_profile=PROFILE, capabilities={"rigid", "jumper"},
                               mujoco=True)
        m, d, comp = compose_default(read_archive(archive, limits=MAP_LIMITS), profile=PROFILE)
        m.vis.global_.offwidth, m.vis.global_.offheight = 1000, 750
        c = mujoco.MjvCamera()
        c.lookat[:] = d.xpos[m.body("base_link").id] + [0.35, 0, 0.05]
        c.distance, c.azimuth, c.elevation = 1.9, 215, -25
        with mujoco.Renderer(m, height=750, width=1000) as r:
            r.update_scene(d, c)
            save_png(out / f"{court.ident}.robot.png", r.render())
        (out / f"{court.ident}.verification.json").write_text(
            json.dumps(dict(validation=check, composition=comp), indent=2, default=str))
        print(f"[map] {archive.name}: ok={check.get('ok', True)} sha256={check.get('sha256')}")


if __name__ == "__main__":
    main()

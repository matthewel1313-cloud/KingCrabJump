"""Compose the playable world: training robot + `black_friday_store.map` + the map's King Crab skin.

* **Physics robot**: the compiled `jumper.posture` training model (`robot_dump.xml`, from
  `tools/dump_robot_model.py`), so the shipped policies see the dynamics they were trained in.
  Its terrain plane and contact sensors are dropped.
* **World**: the `.map`'s `scene.xml`, mounted with `MjSpec.attach` (prefix `scn_`).
* **Look**: the map's default robot is the King Crab gold-crown skin (`king-crab-gold-1`). Its
  display layer is applied to the physics robot: every visual colour from the skin's
  `robot.xml`, and its upper-shell meshes (with the crown) in place of the stock shell visual.
  Visual geoms only; collision, inertia and joints stay the training model's.
* **Host-side additions** (not part of either package):
  - the figures (`shopper_a`, `shopper_b`, `kid`, each a body and a head) become mocap
    (kinematic) bodies, for a light scripted sway;
  - one inactive weld between the deal box and the left claw's palm, switched on while the box
    is held;
  - eyes on the robot's display module (what its screen shows);
  - a sparkle pool for the crown during the dance.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

PROPS = ["deal"]
FIGURES = ["shopper_a", "shopper_b", "kid"]
CLAW_BODY = "robot/LF_palm_link"
N_SPARKLES = 40


def unzip(pkg: Path, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(pkg) as z:
        z.extractall(into)
    return into


@dataclass
class World:
    m: mujoco.MjModel
    d: mujoco.MjData
    prop_body: dict = field(default_factory=dict)
    prop_qadr: dict = field(default_factory=dict)
    fig_mocap: dict = field(default_factory=dict)   # "shopper_a" / "shopper_a_head" -> mocap id
    fig_body: dict = field(default_factory=dict)
    fig_home: dict = field(default_factory=dict)    # authored (pos, quat)
    weld: dict = field(default_factory=dict)        # (prop, holder) -> eq id
    eyes: dict = field(default_factory=dict)
    sparkles: list = field(default_factory=list)

    def holder_body(self, holder: str) -> int:
        assert holder == "claw"
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, CLAW_BODY)

    def attach(self, prop: str, holder: str = "claw") -> None:
        """Weld `prop` to `holder` exactly where it is now (no snap)."""
        m, d = self.m, self.d
        eq = self.weld[(prop, holder)]
        b1, b2 = self.prop_body[prop], self.holder_body(holder)
        p1, q1 = d.xpos[b1].copy(), d.xquat[b1].copy()
        p2, q2 = d.xpos[b2].copy(), d.xquat[b2].copy()
        q1inv = np.zeros(4)
        mujoco.mju_negQuat(q1inv, q1)
        rel_p = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_p, p2 - p1, q1inv)
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, q1inv, q2)
        m.eq_data[eq, 0:3] = 0.0
        m.eq_data[eq, 3:6] = rel_p
        m.eq_data[eq, 6:10] = rel_q
        m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1

    def release(self, prop: str) -> None:
        for (p, h), e in self.weld.items():
            if p == prop:
                self.d.eq_active[e] = 0

    def holder_of(self, prop: str) -> str | None:
        for (p, h), e in self.weld.items():
            if p == prop and self.d.eq_active[e]:
                return h
        return None

    def prop_pos(self, prop: str) -> np.ndarray:
        return self.d.xpos[self.prop_body[prop]].copy()


def _skin_visuals(spec: mujoco.MjSpec, skin_root: Path) -> None:
    """Apply the skin's display layer to the physics robot (prefix `robot/`)."""
    xml_path = next(skin_root.glob("assembled/*/robot.xml"))
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    meshdir = xml_path.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    files = {m.get("name"): m.get("file") for m in root.find("asset").findall("mesh")}
    geoms = {g.name: g for g in spec.geoms}
    for body in root.iter("body"):
        for g in body.findall("geom"):
            name = g.get("name", "")
            if g.get("group") != "2" or g.get("rgba") is None:
                continue
            rgba = [float(v) for v in g.get("rgba").split()]
            if name.startswith("shellflow_upper_"):
                mesh_name = "robot/" + name
                spec.add_mesh(name=mesh_name, file=str((meshdir / files[g.get("mesh")]).resolve()))
                parent = spec.body("robot/" + body.get("name"))
                parent.add_geom(name=mesh_name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name,
                                contype=0, conaffinity=0, group=2, density=0, rgba=rgba)
            elif ("robot/" + name) in geoms:
                geoms["robot/" + name].rgba = rgba
    stock = geoms.get("robot/upper_shell_link_visual")
    if stock is not None:
        spec.delete(stock)


def _eyes(spec: mujoco.MjSpec) -> None:
    disp = spec.body("robot/display_module_link")
    spec.add_material(name="eye_glow", rgba=[0.75, 0.95, 1.0, 1], emission=1.0)
    spec.add_material(name="eye_pupil", rgba=[0.05, 0.10, 0.16, 1], emission=0.0)
    for side, y in (("l", 0.024), ("r", -0.024)):
        disp.add_geom(name=f"eye_{side}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      size=[0.0012, 0.0095, 0.0115], pos=[0.0118, y, -0.001],
                      material="eye_glow", contype=0, conaffinity=0, group=2)
        disp.add_geom(name=f"pupil_{side}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      size=[0.0012, 0.0042, 0.0050], pos=[0.0124, y, -0.0015],
                      material="eye_pupil", contype=0, conaffinity=0, group=2)
        disp.add_geom(name=f"shine_{side}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[0.0016, 0, 0], pos=[0.0130, y + 0.0025, 0.0022],
                      material="eye_glow", contype=0, conaffinity=0, group=2)
        disp.add_geom(name=f"lid_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.0010, 0.0105, 0.0016], pos=[0.0128, y, 0.010],
                      rgba=[0.02, 0.02, 0.03, 0], contype=0, conaffinity=0, group=2)
    for side, y in (("l", 0.040), ("r", -0.040)):
        disp.add_geom(name=f"cheek_{side}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      size=[0.0008, 0.0060, 0.0035], pos=[0.0120, y, -0.010],
                      rgba=[1.0, 0.45, 0.55, 0], contype=0, conaffinity=0, group=2)


def _sparkles(spec: mujoco.MjSpec) -> None:
    spec.add_material(name="sparkle", rgba=[1.0, 0.92, 0.55, 1], emission=1.0)
    for i in range(N_SPARKLES):
        b = spec.worldbody.add_body(name=f"sparkle{i}", mocap=True, pos=[0, 0, -5])
        b.add_geom(name=f"sparkle{i}_a", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0012, 0.0012, 0.0055],
                   material="sparkle", contype=0, conaffinity=0, group=0)
        b.add_geom(name=f"sparkle{i}_b", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0055, 0.0012, 0.0012],
                   material="sparkle", contype=0, conaffinity=0, group=0)


def build_world(robot_dump: Path, jumper_assets: Path, map_pkg: Path, workdir: Path,
                width: int = 1080, height: int = 1920) -> World:
    workdir = Path(workdir)
    map_root = unzip(map_pkg, workdir / "map")
    skin_root = map_root / "robot"

    spec = mujoco.MjSpec.from_file(str(robot_dump))
    spec.meshdir = str(Path(jumper_assets).resolve()) + "/"
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 10
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.10, 0.09, 0.08]
    spec.visual.headlight.diffuse = [0.16, 0.14, 0.12]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.map.znear = 0.004
    spec.visual.map.zfar = 30
    spec.stat.extent = 2.0
    for s in list(spec.sensors):
        spec.delete(s)
    terrain = spec.body("terrain")
    if terrain is not None:
        spec.delete(terrain)
    for light in list(spec.lights):
        spec.delete(light)

    _skin_visuals(spec, skin_root)
    _eyes(spec)

    scene = mujoco.MjSpec.from_file(str(map_root / "scene.xml"))
    frame = spec.worldbody.add_frame()
    spec.attach(scene, prefix="scn_", frame=frame)

    for f in FIGURES:
        for part in (f, f + "_head"):
            spec.body(f"scn_{part}").mocap = True
    _sparkles(spec)

    for p in PROPS:
        spec.add_equality(name=f"weld_{p}_claw", type=mujoco.mjtEq.mjEQ_WELD,
                          name1=f"scn_{p}", name2=CLAW_BODY, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          active=False, solref=[0.004, 1.0])

    m = spec.compile()
    d = mujoco.MjData(m)
    w = World(m=m, d=d)
    for p in PROPS:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"scn_{p}")
        w.prop_body[p] = b
        jnt = m.body_jntadr[b]
        w.prop_qadr[p] = m.jnt_qposadr[jnt]
        w.weld[(p, "claw")] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_{p}_claw")
    for f in FIGURES:
        for part in (f, f + "_head"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"scn_{part}")
            w.fig_body[part] = b
            w.fig_mocap[part] = m.body_mocapid[b]
            w.fig_home[part] = (m.body_pos[b].copy(), m.body_quat[b].copy())
    for n in ("eye_l", "eye_r", "pupil_l", "pupil_r", "shine_l", "shine_r", "lid_l", "lid_r",
              "cheek_l", "cheek_r"):
        w.eyes[n] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    w.sparkles = [m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"sparkle{i}")]
                  for i in range(N_SPARKLES)]
    mujoco.mj_forward(m, d)
    return w

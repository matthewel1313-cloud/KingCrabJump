"""King Crab harness: the shipped Jumper app, driven by a scripted gamepad.

Builds the same world `scripts/play.py --app` builds (native MuJoCo backend,
one environment), with three additions made through the scene's `spec_fn`
hook, after the robot is attached:

* the operating-room `.map` (a kk-scene-package/1), loaded by Jumper's own
  importer (`scenes.package.load_package`);
* the `.skin` overlay: recolours of the baseline visual geoms and the
  visual-only accessories (crown, cap, mask), all contype=0/conaffinity=0 and
  density=0, so the robot's mass, inertia and contacts are untouched;
* weld equality constraints, created inactive, between every instrument and
  (a) the robot's claw link and (b) the gloved hand's mocap body. The control
  script switches them on and off at runtime.

The robot itself is driven by `mjrl.app_play.AppPlayer`: the bundle's own
controller (`controller.so`) and ONNX policies, exactly as `play --app` runs
them. The only thing replaced is the person: `ScriptedPad` stands in for a
gamepad and is fed by the control script.

Run with the Jumper checkout's venv:
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python king_crab/sim/shoot.py
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

JUMPER = Path(os.environ.get("JUMPER_ROOT", "../jumper"))
if str(JUMPER / "scripts") not in sys.path:
    sys.path.insert(0, str(JUMPER / "scripts"))

INSTRUMENTS = ("scalpel", "forceps", "scissors", "gauze")
CLAW_BODY = "LF_palm_link"  # the left claw's fixed jaw: what claw_left carries
CLAW_LINKS = ("LF_forearm_link", "LF_palm_link", "LF_palm_pad_f_link", "LF_palm_pad_b_link",
              "LF_finger_link", "LF_finger_tip_link", "LF_finger_grip_insert_link",
              "LF_palm_grip_insert_link")


# --------------------------------------------------------------------------
# The person, scripted
# --------------------------------------------------------------------------
@dataclass
class ScriptedPad:
    """Quacks like `controller.open()`'s pad: `state()` and `connected`.

    Sticks -1..1, triggers 0..1. `ly`/`ry` keep the kernel's sign: positive is
    DOWN, so walking forward is `ly < 0`.
    """

    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    hat_x: int = 0
    hat_y: int = 0
    buttons: set = field(default_factory=set)
    connected: bool = True

    def state(self):
        from controller.xbox import GamepadState

        return GamepadState(self.lx, self.ly, self.rx, self.ry, self.lt, self.rt,
                            self.hat_x, self.hat_y, frozenset(self.buttons))

    def neutral(self):
        self.lx = self.ly = self.rx = self.ry = self.lt = self.rt = 0.0
        self.hat_x = self.hat_y = 0
        self.buttons = set()


# --------------------------------------------------------------------------
# The .skin overlay
# --------------------------------------------------------------------------
def read_skin(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("skin-package.json"))


def apply_skin(spec, skin: dict, robot_prefix: str) -> None:
    """Recolour baseline visual geoms and add the visual-only accessories."""
    for ov in skin.get("visual_overrides", []):
        g = spec.geom(robot_prefix + ov["geom"])
        if g is None:
            raise KeyError(f"skin recolours {ov['geom']!r}, which the robot lacks")
        g.rgba = ov["rgba"]
    for acc in skin.get("accessories", []):
        body = spec.body(robot_prefix + acc["link"])
        if body is None:
            raise KeyError(f"skin attaches to {acc['link']!r}, which the robot lacks")
        kw = dict(
            name=robot_prefix + "skin_" + acc["name"],
            type=getattr(mujoco.mjtGeom, "mjGEOM_" + acc["type"].upper()),
            size=(list(acc["size"]) + [0, 0, 0])[:3],
            pos=acc.get("pos", [0, 0, 0]),
            rgba=acc["rgba"],
            contype=0, conaffinity=0, density=0, group=2,
        )
        if "quat" in acc:
            kw["quat"] = acc["quat"]
        if "euler" in acc:
            q = np.zeros(4)
            mujoco.mju_euler2Quat(q, np.radians(acc["euler"]), "xyz")
            kw["quat"] = q.tolist()
        if acc.get("fromto"):
            kw["fromto"] = acc["fromto"]
            kw.pop("pos")
        body.add_geom(**kw)


# --------------------------------------------------------------------------
# The world
# --------------------------------------------------------------------------
@dataclass
class World:
    env: object
    wrapped: object
    player: object
    pad: ScriptedPad
    model: mujoco.MjModel
    data: mujoco.MjData
    sim: object
    obs: object
    step_dt: float
    t: float = 0.0
    prefix_scene: str = "or_"
    prefix_robot: str = "robot/"

    # ids
    def bid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise KeyError(name)
        return i

    def eqid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if i < 0:
            raise KeyError(name)
        return i

    def tool_body(self, tool: str) -> int:
        return self.bid(f"{self.prefix_scene}{tool}")

    def claw_body(self) -> int:
        return self.bid(self.prefix_robot + CLAW_BODY)

    def grip_point(self) -> np.ndarray:
        """Midpoint of the two grip inserts: where a held tool sits."""
        a = self.data.xpos[self.bid(self.prefix_robot + "LF_palm_grip_insert_link")]
        b = self.data.xpos[self.bid(self.prefix_robot + "LF_finger_grip_insert_link")]
        return 0.5 * (a + b)

    def base_body(self) -> int:
        return self.bid(self.prefix_robot + "base_link")

    def hand_body(self) -> int:
        return self.bid(f"{self.prefix_scene}glove_hand")

    # welds -----------------------------------------------------------------
    def weld(self, tool: str, to: str, on: bool) -> None:
        """Switch the `tool`->`to` weld ('claw' or 'hand'), freezing the
        current relative pose as the weld's target so nothing jumps."""
        e = self.eqid(f"weld_{tool}_{to}")
        if on:
            b1 = self.tool_body(tool)
            b2 = self.claw_body() if to == "claw" else self.hand_body()
            d = self.data
            # relpose (MuJoCo): the pose of body2 relative to body1
            q1inv = np.zeros(4)
            mujoco.mju_negQuat(q1inv, d.xquat[b1])
            dp = d.xpos[b2] - d.xpos[b1]
            rel_p = np.zeros(3)
            mujoco.mju_rotVecQuat(rel_p, dp, q1inv)
            rel_q = np.zeros(4)
            mujoco.mju_mulQuat(rel_q, q1inv, d.xquat[b2])
            # eq_data for weld: anchor(3) relpos(3) relquat(4) torquescale(1)
            self.model.eq_data[e, 0:3] = 0.0
            self.model.eq_data[e, 3:6] = rel_p
            self.model.eq_data[e, 6:10] = rel_q
            self.model.eq_data[e, 10] = 1.0
        # `mujoco.rollout` (the native backend's stepper) re-initialises
        # eq_active from the model's eq_active0 on every roll, so a weld
        # switched on in data alone is gone after one step. Set both.
        self.model.eq_active0[e] = 1 if on else 0
        self.data.eq_active[e] = 1 if on else 0
        # A held tool does not collide: otherwise the table it is lifted off
        # holds it down against the weld (measured: the gauze rose 1 cm of the
        # claw's 3). Collisions come back when nothing holds it any more.
        held = self.weld_on(tool, "claw") or self.weld_on(tool, "hand")
        self.tool_collisions(tool, not held)

    def tool_collisions(self, tool: str, on: bool) -> None:
        b = self.tool_body(tool)
        if not hasattr(self, "_coll"):
            self._coll = {}
        for g in range(self.model.body_geomadr[b],
                       self.model.body_geomadr[b] + self.model.body_geomnum[b]):
            if g not in self._coll:
                self._coll[g] = (int(self.model.geom_contype[g]),
                                 int(self.model.geom_conaffinity[g]))
            ct, ca = self._coll[g] if on else (0, 0)
            self.model.geom_contype[g] = ct
            self.model.geom_conaffinity[g] = ca

    def weld_on(self, tool: str, to: str) -> bool:
        return bool(self.data.eq_active[self.eqid(f"weld_{tool}_{to}")])

    # mocap -------------------------------------------------------------------
    def set_hand(self, pos, quat=None) -> None:
        mid = self.model.body_mocapid[self.hand_body()]
        buf_p = self.sim.data.numpy("mocap_pos")
        buf_p[0, mid] = pos
        self.data.mocap_pos[mid] = pos
        if quat is not None:
            buf_q = self.sim.data.numpy("mocap_quat")
            buf_q[0, mid] = quat
            self.data.mocap_quat[mid] = quat

    # stepping ---------------------------------------------------------------
    def step(self) -> None:
        import torch

        with torch.inference_mode():
            action = self.player(self.obs)
            self.obs, *_ = self.wrapped.step(action)
        self.t += self.step_dt

    @property
    def mode(self) -> str:
        return self.player.fsm.mode


def make_world(map_path: Path, skin_path: Path | None, app_path: Path,
               task: str | None = None, quiet: bool = True,
               spawn: tuple[float, float, float] = (0.0, 0.0, 0.0),
               weld_grasp: bool = True) -> World:
    import contextlib
    import io

    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    res = resolve(backend="native", device="cpu", num_envs=1, training=False)
    use_backend(res)

    import tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjrl.app_play import AppPlayer, leave_falls_to_the_app, open_app, step_the_world_at
    from scenes.package import load_package
    from scenes.registry import apply_scene

    app = open_app(app_path)
    task = task or app.default_task
    env_cfg = tasks.load_env_cfg(task, play=True)
    # Deterministic takes: the task's reset scatters the robot over +-0.5 m and
    # a random heading, and its startup events randomise friction, encoder bias
    # and COM. A film wants the same take every time, so all are pinned.
    x0, y0, yaw0 = spawn
    env_cfg.events["reset_base"].params["pose_range"] = {
        "x": (x0, x0), "y": (y0, y0), "z": (0.0, 0.0), "yaw": (yaw0, yaw0)}
    for name in ("encoder_bias", "base_com"):
        env_cfg.events.pop(name, None)
    if "foot_friction" in env_cfg.events:
        env_cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
    env_cfg.seed = 7
    step_the_world_at(env_cfg, app.control_hz)
    leave_falls_to_the_app(env_cfg, app)
    scene = load_package(map_path)
    apply_scene(env_cfg, scene, label=str(map_path))
    env_cfg.scene.num_envs = 1

    skin = read_skin(skin_path) if skin_path else None
    previous = env_cfg.scene.spec_fn

    def spec_fn(spec):
        if previous is not None:
            previous(spec)
        if skin is not None:
            apply_skin(spec, skin, "robot/")
        # The held tool and the left claw do not collide: the grasp is a weld
        # (see NOTES.md, "hybrid grasp"), and jaw contacts fighting a weld make
        # the tool buzz. Everything else -- the stand, the tray, the other
        # tools, the rest of the robot -- still collides with every tool.
        for tool in (INSTRUMENTS if weld_grasp else ()):
            for link in CLAW_LINKS:
                ex = spec.add_exclude()
                ex.bodyname1 = f"or_{tool}"
                ex.bodyname2 = "robot/" + link
        # Inactive welds, switched by the control script.
        for tool in INSTRUMENTS:
            for to, other in (("claw", "robot/" + CLAW_BODY), ("hand", "or_glove_hand")):
                eq = spec.add_equality()
                eq.name = f"weld_{tool}_{to}"
                eq.type = mujoco.mjtEq.mjEQ_WELD
                eq.objtype = mujoco.mjtObj.mjOBJ_BODY
                eq.name1 = f"or_{tool}"
                eq.name2 = other
                eq.active = False
                eq.solref = [0.02, 1.0]

    env_cfg.scene.spec_fn = spec_fn

    sink = io.StringIO() if quiet else None
    with (contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()):
        env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
        from mjrl.replay import skip_rewards

        skip_rewards(env)
        agent_cfg = tasks.load_agent_cfg(task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        pad = ScriptedPad()
        player = AppPlayer(app, wrapped, pad=pad)
        obs = wrapped.get_observations()
    sim = env.sim
    model, data = sim.env_mjdata(0)
    w = World(env=env, wrapped=wrapped, player=player, pad=pad, model=model,
              data=data, sim=sim, obs=obs, step_dt=env.step_dt)
    # mjlab's mocap buffer starts at zeros, which would put the hand at the
    # world origin (inside the robot). Park it off-set explicitly.
    w.set_hand(HAND_PARK, [1.0, 0.0, 0.0, 0.0])
    return w


HAND_PARK = (2.6, 2.6, 1.5)

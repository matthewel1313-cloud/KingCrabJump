"""King Crab, Surgical Tech -- the control script (the show's ".app" logic).

What drives what:

* **Policy** (shipped, unmodified, from the Jumper example `.app`): every joint
  of the robot, every step. Walking, sideways scuttling, turning, raising and
  lowering the claw arm, opening/closing the claw (`claw_left` mode, the
  `jumper.five_foot` policy and its deploy hook), the look-up in the hook
  (`Ry` nose-up) and the finale (`dance_dream_wings`).
* **Scripted operator** (this file): a virtual gamepad. It closes the loop
  the way a person with a pad would -- reads where the claw is, pushes the
  sticks toward the next instrument, holds LT to close the claw, presses the
  d-pad for the arm presets, and the bumper/menu chords to switch modes.
* **Scripted physics glue** (this file): the grasp and the hand-off are
  welds. When the claw has closed on a tool, a weld (tool -> LF_palm_link) is
  switched on at the current relative pose; at hand-off a weld
  (tool -> glove_hand) switches on and the claw weld off. The glove is a
  kinematic mocap body on a scripted path. Used tools are moved off-set to a
  back table once the hand has left the frame.

Writes `out/sim_<take>.npz` (per-step qpos / mocap / weld state) and
`out/sim_<take>.json` (marks: beat times, clinks), plus the compiled model
(`out/model_<take>.mjb`) for `render.py`.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python king_crab/sim/choreography.py [main|blooper]
"""

from __future__ import annotations

import os

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from harness import HAND_PARK, INSTRUMENTS, make_world  # noqa: E402

APP = Path(os.environ.get("APP", Path(os.environ.get("JUMPER_ROOT", "../jumper")) / "out/bundle_example/jumper.app"))
OUT = ROOT / "out"

# Where things are on the Mayo stand (world frame; see scene/scene.xml).
TOOL_Y = {"scalpel": 0.20, "forceps": 0.07, "gauze": -0.06, "scissors": -0.19}
HANDLE_X = 0.19          # handle ends overhang the stand edge (x=0.215) to here
TOOL_LEN_FWD = {"scalpel": 0.045, "forceps": 0.045, "scissors": 0.05, "gauze": 0.03}
TOOL_LEN_BACK = {"scalpel": 0.0425, "forceps": 0.043, "scissors": 0.041, "gauze": 0.03}
BACK_TABLE = np.array([2.6, -2.6, 0.04])


def yaw_of(q) -> float:
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def quat_yaw_pitch(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """q = Rz(yaw) * Ry(pitch) * Rx(roll): yaw applied last, about world z."""
    q = np.zeros(4)
    qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    qy = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
    qx = np.array([math.cos(roll / 2), math.sin(roll / 2), 0, 0])
    a = np.zeros(4)
    mujoco.mju_mulQuat(a, qz, qy)
    mujoco.mju_mulQuat(q, a, qx)
    return q


def smooth(a: float) -> float:
    a = min(max(a, 0.0), 1.0)
    return a * a * (3 - 2 * a)


class Show:
    def __init__(self, w):
        self.w = w
        self.tasks = []
        self.marks: list[dict] = []
        self.flags: dict[str, bool] = {}
        self.rec = {k: [] for k in ("t", "qpos", "mocap_pos", "mocap_quat", "eq_active")}
        self.modes: list[tuple[float, str]] = []

    # --- infrastructure ---------------------------------------------------
    def mark(self, name: str, **kw):
        m = {"t": round(self.w.t, 4), "name": name, **kw}
        self.marks.append(m)
        print(f"  [{self.w.t:6.2f}] {name} {kw if kw else ''}", flush=True)

    def spawn(self, gen, clears: str | None = None):
        if clears:
            self.flags[clears] = False
        self.tasks.append(gen)

    def record(self):
        d = self.w.data
        self.rec["t"].append(self.w.t)
        self.rec["qpos"].append(d.qpos.copy())
        self.rec["mocap_pos"].append(d.mocap_pos.copy())
        self.rec["mocap_quat"].append(d.mocap_quat.copy())
        self.rec["eq_active"].append(d.eq_active.copy())
        if not self.modes or self.modes[-1][1] != self.w.mode:
            self.modes.append((self.w.t, self.w.mode))

    def run(self, main, t_end: float):
        self.spawn(main)
        self.record()
        while self.w.t < t_end - 1e-9:
            for g in list(self.tasks):
                try:
                    next(g)
                except StopIteration:
                    self.tasks.remove(g)
            self.w.step()
            self.record()

    def save(self, take: str):
        OUT.mkdir(exist_ok=True)
        np.savez_compressed(OUT / f"sim_{take}.npz",
                            **{k: np.asarray(v) for k, v in self.rec.items()})
        (OUT / f"sim_{take}.json").write_text(json.dumps(
            {"take": take, "step_dt": self.w.step_dt, "marks": self.marks,
             "modes": self.modes}, indent=1))
        mujoco.mj_saveModel(self.w.model, str(OUT / f"model_{take}.mjb"), None)

    # --- state helpers ----------------------------------------------------
    @property
    def pad(self):
        return self.w.pad

    def base_pose(self):
        d = self.w.data
        b = self.w.base_body()
        return d.xpos[b].copy(), yaw_of(d.xquat[b])

    def tool_end(self, tool: str) -> np.ndarray:
        """World point at the tool's far (blade) end."""
        d = self.w.data
        b = self.w.tool_body(tool)
        R = d.xmat[b].reshape(3, 3)
        return d.xpos[b] + R @ np.array([TOOL_LEN_FWD[tool], 0, 0])

    # --- primitives (generators; one yield == one control step) -----------
    def wait(self, dur: float):
        for _ in range(max(1, int(round(dur / self.w.step_dt)))):
            yield

    def wait_flag(self, name: str, timeout: float = 5.0):
        n = int(timeout / self.w.step_dt)
        for _ in range(n):
            if self.flags.get(name):
                return
            yield

    def tap(self, *buttons, hat=(0, 0), dur: float = 0.12):
        self.pad.buttons = set(buttons)
        self.pad.hat_x, self.pad.hat_y = hat
        yield from self.wait(dur)
        self.pad.buttons = set()
        self.pad.hat_x = self.pad.hat_y = 0

    def ramp(self, field: str, to: float, dur: float):
        start = getattr(self.pad, field)
        n = max(1, int(round(dur / self.w.step_dt)))
        for i in range(1, n + 1):
            setattr(self.pad, field, start + (to - start) * i / n)
            yield

    def drive(self, x=None, y=None, yaw=None, point="grip", tol=0.008, yaw_tol=0.09,
              timeout=3.0, vmax=0.8, k=9.0, kyaw=2.2, smin=0.35, settle=0.06):
        """Closed-loop stick control: move the grip point (or the base) to
        (x, y) and the heading to `yaw`. Returns when within tolerance for
        `settle` seconds, or at `timeout`."""
        w = self.w
        ok_for = 0.0
        for _ in range(int(timeout / w.step_dt)):
            p = w.grip_point() if point == "grip" else self.base_pose()[0]
            _, th = self.base_pose()
            ex = (x - p[0]) if x is not None else 0.0
            ey = (y - p[1]) if y is not None else 0.0
            c, s = math.cos(th), math.sin(th)
            fx, fy = c * ex + s * ey, -s * ex + c * ey   # error in robot frame
            def stick(e, tl):
                if abs(e) < tl:
                    return 0.0
                v = min(vmax, max(smin, k * abs(e)))
                return math.copysign(v, e)
            self.pad.ly = -stick(fx, tol)
            self.pad.lx = -stick(fy, tol)
            if yaw is not None:
                eyaw = (yaw - th + math.pi) % (2 * math.pi) - math.pi
                # Rx's first half twists the body in place; turning needs > 0.5
                self.pad.rx = 0.0 if abs(eyaw) < yaw_tol else -math.copysign(
                    min(1.0, max(0.62, kyaw * abs(eyaw))), eyaw)
            else:
                self.pad.rx = 0.0
                eyaw = 0.0
            done = abs(fx) < tol and abs(fy) < tol and abs(eyaw) < yaw_tol
            ok_for = ok_for + w.step_dt if done else 0.0
            if ok_for >= settle:
                break
            yield
        else:
            self.mark("drive_timeout", fx=round(fx, 4), fy=round(fy, 4), eyaw=round(eyaw, 3))
        self.pad.lx = self.pad.ly = self.pad.rx = 0.0

    # --- the glove ---------------------------------------------------------
    def hand_take(self, tool: str, approach=0.75, hold=0.18, retreat=0.65,
                  side=(1.0, 0.35), lift=0.22, flag="taken"):
        """The surgeon's gloved hand reaches in from the surgeon's side (+x),
        closes on the tool's far end, takes it (weld swap), and withdraws."""
        w = self.w
        self.flags[flag] = False
        dirv = np.array([side[0], side[1], 0.0])
        dirv /= np.linalg.norm(dirv)
        yaw = math.atan2(dirv[1], dirv[0])        # hand +x points back toward the surgeon
        quat = quat_yaw_pitch(yaw, pitch=-0.30)   # fingers (-x) tipped slightly down

        def target():
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, quat)
            R = R.reshape(3, 3)
            # fingertips (~0.075 m along -x of the palm) on the tool's far end
            return self.tool_end(tool) + R @ np.array([0.070, 0.0, 0.010])

        start = target() + dirv * 0.55 + np.array([0, 0, lift])
        n = int(approach / w.step_dt)
        for i in range(1, n + 1):
            a = i / n
            e = 1 - (1 - a) ** 3              # ease-out: reaches in, slows to a stop
            tgt = target()
            arc = np.array([0, 0, 0.05 * math.sin(math.pi * a)])
            w.set_hand(start + e * (tgt - start) + arc, quat)
            yield
        # take it
        w.set_hand(target(), quat)
        w.weld(tool, "hand", True)
        w.weld(tool, "claw", False)
        self.mark("handoff", tool=tool)
        self.flags[flag] = True
        held = w.data.mocap_pos[w.model.body_mocapid[w.hand_body()]].copy()
        yield from self.wait(hold)
        n = int(retreat / w.step_dt)
        away = held + dirv * 0.9 + np.array([0, 0, 0.35])
        for i in range(1, n + 1):
            a = smooth(i / n)
            w.set_hand(held + a * (away - held), quat)
            yield
        # off-set: the circulating nurse takes it; park the glove
        w.weld(tool, "hand", False)
        self.park_tool(tool)
        w.set_hand(HAND_PARK, [1, 0, 0, 0])

    def park_tool(self, tool: str):
        w = self.w
        j = w.model.body_jntadr[w.tool_body(tool)]
        qa, va = w.model.jnt_qposadr[j], w.model.jnt_dofadr[j]
        k = INSTRUMENTS.index(tool)
        w.data.qpos[qa:qa + 3] = BACK_TABLE + np.array([0.12 * k - 0.18, 0, 0.0])
        w.data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        w.data.qvel[va:va + 6] = 0

    # --- tool beats ----------------------------------------------------------
    def handle_end(self, tool: str) -> np.ndarray:
        """World point at the tool's near (handle) end -- where it is now, not
        where it started: a tool that was put back down is wherever it landed."""
        d = self.w.data
        b = self.w.tool_body(tool)
        return d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([-TOOL_LEN_BACK[tool], 0, 0])

    def grasp(self, tool: str, fast=False, close=0.85):
        """Line the open claw up with the handle, step in, close, weld."""
        h = self.handle_end(tool)
        vmax = 1.0 if fast else 0.8
        yield from self.drive(x=h[0] - 0.03, y=h[1], yaw=0.0, tol=0.015,
                              timeout=1.4, vmax=vmax, yaw_tol=0.13)
        self.mark("aligned", tool=tool)
        h = self.handle_end(tool)
        # the jaws reach ~4 cm past the grip point: stop just short of the
        # handle end so they close on the overhang without hitting the table
        # (the claw tip meets the table's front face here; that is the stop)
        yield from self.drive(x=h[0] - 0.016, y=h[1], yaw=0.0, tol=0.011,
                              timeout=0.45 if not fast else 0.35, vmax=0.5, yaw_tol=0.12)
        b = self.w.tool_body(tool)
        self._origin = getattr(self, "_origin", {})
        self._origin[tool] = (self.w.data.xpos[b].copy(), self.w.data.xquat[b].copy())
        for _ in range(int(0.8 / self.w.step_dt)):   # arm fully carried before closing
            if self.w.grip_point()[2] > 0.089:
                break
            yield
        yield from self.ramp("lt", close, 0.12 if fast else 0.2)
        yield from self.wait(0.05)
        self.w.weld(tool, "claw", True)
        yield from self.seat(tool, 0.12 if fast else 0.16)
        g = self.w.grip_point()
        h = self.w.data.xpos[self.w.tool_body(tool)]
        self.mark("grasp", tool=tool, grip=[round(float(v), 3) for v in g],
                  tool_center=[round(float(v), 3) for v in h])

    def seat(self, tool: str, dur: float):
        """Slide the welded tool into the jaws: move the weld's target so a
        point 1.5 cm in from the handle end ends on the grip point, over `dur`
        seconds, orientation unchanged. (The grip inserts sit across the
        crab's front; the handle overhang is ~3 cm off them when the jaws
        close.) The weld is a soft constraint, so the tool is pulled there,
        not teleported."""
        w = self.w
        e = w.eqid(f"weld_{tool}_claw")
        b = w.tool_body(tool)
        d = w.data
        R1 = d.xmat[b].reshape(3, 3)
        p = d.xpos[b] + R1 @ np.array([-TOOL_LEN_BACK[tool] + 0.015, 0, 0])
        shift = R1.T @ (w.grip_point() - p)
        r0 = w.model.eq_data[e, 3:6].copy()
        r1 = r0 - shift
        n = max(1, int(dur / w.step_dt))
        for i in range(1, n + 1):
            w.model.eq_data[e, 3:6] = r0 + smooth(i / n) * (r1 - r0)
            yield

    def place(self, tool: str, dur: float, lift: float = 0.006):
        """Put a held tool back where it was picked up (world pose, `lift`
        higher), by steering the claw weld's target there over `dur` seconds.
        Recomputed every step from the claw's live pose, so the crab's own
        sway does not move the spot."""
        w = self.w
        e = w.eqid(f"weld_{tool}_claw")
        x1, q1 = self._origin[tool]
        x1 = x1 + np.array([0, 0, lift])
        q1inv = np.zeros(4)
        mujoco.mju_negQuat(q1inv, q1)
        p0 = w.model.eq_data[e, 3:6].copy()
        r0 = w.model.eq_data[e, 6:10].copy()
        n = max(1, int(dur / w.step_dt))
        for i in range(1, n + 1):
            a = smooth(i / n)
            cb = w.claw_body()
            pt = np.zeros(3)
            mujoco.mju_rotVecQuat(pt, w.data.xpos[cb] - x1, q1inv)
            qt = np.zeros(4)
            mujoco.mju_mulQuat(qt, q1inv, w.data.xquat[cb])
            if np.dot(qt, r0) < 0:
                qt = -qt
            q = r0 + a * (qt - r0)
            w.model.eq_data[e, 3:6] = p0 + a * (pt - p0)
            w.model.eq_data[e, 6:10] = q / np.linalg.norm(q)
            yield
        for _ in range(int(0.08 / w.step_dt)):   # hold there a moment
            cb = w.claw_body()
            mujoco.mju_rotVecQuat(pt, w.data.xpos[cb] - x1, q1inv)
            mujoco.mju_mulQuat(qt, q1inv, w.data.xquat[cb])
            w.model.eq_data[e, 3:6] = pt
            w.model.eq_data[e, 6:10] = qt
            yield

    def present(self, tool: str, fast=False, turn=0.30):
        """Back off the stand, raise the arm (d-pad up), turn toward the
        surgeon, and hold it out for the hand."""
        p, _ = self.base_pose()
        yield from self.drive(x=p[0] - 0.05, y=p[1], point="base", yaw=0.0, tol=0.01,
                              timeout=0.5 if fast else 0.7, vmax=0.8)
        # offer it up: a gentle nose-up lifts the carried claw (the d-pad arm
        # presets point this claw down-forward, which would aim the tool at the table)
        yield from self.ramp("ry", 0.45, 0.2 if fast else 0.25)
        # twist toward the surgeon: Rx inside its first half twists the body
        # in place (feet planted) rather than walking a turn
        yield from self.ramp("rx", -turn, 0.22 if fast else 0.3)
        yield from self.wait(0.12 if fast else 0.25)
        self.mark("present", tool=tool)

    def recover(self, fast=False):
        """Open the claw, lower the arm, face the stand again."""
        self.pad.lt = 0.0
        yield from self.wait(0.10)
        self.pad.hat_y = 0
        yield from self.ramp("ry", 0.0, 0.15)
        yield from self.ramp("rx", 0.0, 0.2)
        yield from self.wait(0.08 if fast else 0.2)


# --------------------------------------------------------------------------
# The show
# --------------------------------------------------------------------------
def main_take(show: Show):
    s, pad = show, show.pad
    # 0.0-2.0  Hook: looks up at camera (nose up), then switches to claw mode.
    s.mark("beat", beat="hook")
    yield from s.tap("LB")                        # claw_left: carry the left claw
    yield from s.wait(0.2)
    yield from s.ramp("ry", 0.9, 0.35)           # right stick down: nose up = look up
    yield from s.wait(0.8)
    yield from s.ramp("ry", 0.0, 0.3)
    yield from s.wait(0.3)

    # 2.0-6.0  Scalpel.
    s.mark("beat", beat="scalpel")
    yield from s.grasp("scalpel")
    yield from s.present("scalpel", turn=0.45)
    s.spawn(clears="taken", gen=s.hand_take("scalpel", side=(1.0, 0.45)))
    yield from s.wait_flag("taken")
    yield from s.wait(0.12)
    s.mark("check", tool="scalpel")
    yield from s.recover()

    # 6.0-10.0  Forceps. Same pattern, faster.
    s.mark("beat", beat="forceps")
    yield from s.grasp("forceps", fast=True)
    yield from s.present("forceps", fast=True, turn=0.4)
    s.spawn(clears="taken", gen=s.hand_take("forceps", approach=0.6, hold=0.12, side=(1.0, 0.35)))
    yield from s.wait_flag("taken")
    yield from s.wait(0.08)
    s.mark("check", tool="forceps")
    yield from s.recover(fast=True)

    # 10.0-14.0  Scissors... grabs the gauze first.
    s.mark("beat", beat="scissors")
    yield from s.grasp("gauze", fast=True)
    yield from s.ramp("ry", 0.7, 0.2)             # nose up: lifts it off the table, flat
    s.mark("oops", tool="gauze")
    yield from s.wait(0.15)
    yield from s.ramp("rx", -0.42, 0.16)          # ...and looks at it: "...hm?"
    yield from s.wait(0.16)
    yield from s.ramp("rx", 0.42, 0.26)
    yield from s.wait(0.14)
    yield from s.ramp("rx", 0.0, 0.16)
    yield from s.ramp("ry", 0.0, 0.25)            # back down, and lets go where it was
    yield from s.wait(0.1)
    yield from s.place("gauze", 0.25)
    pad.lt = 0.0
    yield from s.wait(0.06)
    s.w.weld("gauze", "claw", False)
    s.mark("putdown", tool="gauze")
    yield from s.wait(0.15)
    p, _ = s.base_pose()
    yield from s.drive(x=p[0] - 0.03, y=p[1], point="base", yaw=0.0, timeout=0.35)
    yield from s.grasp("scissors", fast=True)
    yield from s.present("scissors", fast=True, turn=0.4)
    s.spawn(clears="taken", gen=s.hand_take("scissors", approach=0.55, hold=0.12, side=(1.0, 0.25)))
    yield from s.wait_flag("taken")
    yield from s.wait(0.08)
    s.mark("check", tool="scissors")
    yield from s.recover(fast=True)

    # 14.0-17.0  Faster! Quick sideways scuttle, last tool.
    s.mark("beat", beat="faster")
    yield from s.grasp("gauze", fast=True)
    yield from s.present("gauze", fast=True, turn=0.4)
    s.spawn(clears="taken", gen=s.hand_take("gauze", approach=0.5, hold=0.10, side=(1.0, 0.3)))
    yield from s.wait_flag("taken")
    yield from s.wait(0.06)
    s.mark("check", tool="gauze")
    pad.lt = 0.0
    yield from s.ramp("ry", 0.0, 0.15)

    # 17.0-20.0  perfect. Victory wiggle, claws up (dance_dream_wings).
    s.mark("beat", beat="perfect")
    pad.hat_y = 0
    yield from s.drive(yaw=0.0, timeout=0.5)
    yield from s.tap("menu", hat=(1, 0))          # menu + d-pad right
    s.mark("dance")
    yield from s.wait(6.0)


def blooper_take(show: Show):
    """The outtake, and an honest one: a **pure-physics** grasp. This take is
    built with `weld_grasp=False` -- the jaws collide with the tools, no weld
    is ever switched on -- so whether the scalpel stays in the claw is left to
    contact and friction alone. It does not: it squirts out as the arm lifts."""
    s, pad = show, show.pad
    yield from s.tap("LB")
    yield from s.wait(1.4)
    s.mark("beat", beat="blooper")
    h = s.handle_end("scalpel")
    yield from s.drive(x=h[0] - 0.03, y=h[1], yaw=0.0, tol=0.011, timeout=1.0)
    yield from s.drive(x=h[0] - 0.016, y=h[1], yaw=0.0, tol=0.011, timeout=0.6)
    yield from s.ramp("lt", 1.0, 0.25)            # squeeze -- friction only
    s.mark("grasp_physics", tool="scalpel")
    yield from s.wait(0.15)
    pad.hat_y = -1                                # lift...
    yield from s.wait(1.2)
    pad.lt = 0.0
    pad.hat_y = 0
    yield from s.ramp("rx", -0.45, 0.3)           # looks at the camera, sheepish
    yield from s.wait(1.0)


if __name__ == "__main__":
    take = sys.argv[1] if len(sys.argv) > 1 else "main"
    spawn_y = TOOL_Y["scalpel"] - 0.047
    w = make_world(ROOT / "scene/operating_room.map", ROOT / "skin/king_crab_surgeon.skin",
                   APP, spawn=(HANDLE_X - 0.03 - 0.161, spawn_y, 0.0),
                   weld_grasp=(take == "main"))
    show = Show(w)
    if take == "main":
        show.run(main_take(show), t_end=float(sys.argv[2]) if len(sys.argv) > 2 else 23.0)
    else:
        show.run(blooper_take(show), t_end=6.0)
    show.save(take)
    sc = w.data.xpos[w.tool_body("scalpel")]
    print("scalpel final position:", np.round(sc, 3))
    print("modes:", show.modes)

"""King Crab -- the beat-sheet sequencer.

    python sequencer.py --out build/run      # simulate the whole story, record every tick

What is the policy and what is scripted (also in NOTES.md):

POLICY (shipped Jumper bundle, compiled controller + ONNX, unchanged):
  * all leg motion: standing, side-scuttling, walking, turning -- `locomotion`
    and `claw_left` (the `jumper.five_foot` policy: walks on five legs, left claw carried);
  * the claw's pose: nose-down aim (body_pose pitch), the arm held out (d-pad
    preset), the jaw closing on the trigger -- the controller and its deploy hook;
  * the closing victory dance -- `dance_crab`, a reference-tracking policy.
SCRIPTED (this file):
  * the "operator": pad frames computed each 5 ms tick by a closed-loop
    go-to-pose (position + heading error -> stick deflection), plus button and
    trigger timing. The policy turns those commands into joint targets;
  * grasp: when the trigger closes the jaw at the item, a weld attaches the item
    to the claw ("weld-attach on contact"); the item is carried self-levelling
    (kept upright under the grip point, like a gimballed tray);
  * the family's hands: kinematic (mocap) bodies on eased paths; hand-off swaps
    the item's weld from the claw to the hand;
  * the robot's display face (eyes), the crown sparkles.
"""

from __future__ import annotations

import argparse
import os
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sim"))
from crab_runtime import CONTROL_HZ, CrabRuntime, PadState  # noqa: E402
from world import FAMILY, PROPS, build_world, look_at_quat  # noqa: E402

Z = 0.76  # table top
DT = 1.0 / CONTROL_HZ

#: Where the jaws meet in the claw's nose-down, closed pose, in the body frame
#: (forward, left), measured with the shipped claw_left policy (see NOTES.md).
JAW_OFFSET = np.array([0.156, 0.044])

#: Grip point on each prop, from its origin (world-aligned at rest).
GRIP = {
    "butter": (-0.046, 0.0, 0.022), "rolls": (-0.044, 0.0, 0.030), "gravy": (-0.060, 0.0, 0.040),
    "cranberry": (-0.033, 0.0, 0.026), "phone": (-0.052, 0.0, 0.016), "drawing": (-0.016, 0.0, 0.026),
    "napkin": (-0.036, 0.0, 0.020),
}

#: Seats: x along the table and side (+1 = +Y = the crab's left).
SEAT = {"dad": (0.02, +1), "mom": (0.04, -1), "aunt": (0.40, +1), "kid": (0.40, -1), "grandma": (-0.55, -1)}
SCALE = {"dad": 1.0, "mom": 0.92, "aunt": 0.94, "kid": 0.72, "grandma": 0.9}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Hand:
    """A kinematic family member's hand: pose = wrist position + facing."""

    def __init__(self, world, who):
        self.w, self.who = world, who
        self.mid = world.hand_mocap[who]
        x, side = SEAT[who]
        self.side = side
        self.shoulder = np.array([x, side * 0.80, Z + 0.30])
        self.parked = np.array([x, side * 0.97, Z - 0.25])
        self.pos = self.parked.copy()
        self.fwd = self._facing(np.array([x, 0.0, Z]))
        self.tween = None
        self.legs = []
        self.t = 0.0
        self.apply()

    def _facing(self, target):
        f = target - self.shoulder
        f[2] *= 0.45  # flatter than the true line: a hand reaching keeps its palm fairly level
        return f / np.linalg.norm(f)

    def palm_to_wrist(self, palm, fwd):
        return palm - fwd * 0.030 * SCALE[self.who]

    def _edge(self):
        x, side = SEAT[self.who]
        return np.array([x, side * 0.60, Z + 0.14])

    def _below_table(self, p):
        return p[2] < Z + 0.03 and abs(p[1]) > 0.50

    def _queue(self, legs):
        """legs: [(wrist_target, fwd_target, duration)], run in order."""
        self.legs = list(legs)
        self._next_leg()

    def _next_leg(self):
        if not self.legs:
            self.tween = None
            return
        p1, f1, dur = self.legs.pop(0)
        self.tween = (self.pos.copy(), self.fwd.copy(), p1, f1, 0.0, max(dur, 1e-3))

    def go(self, palm_target, dur, look=None):
        """Move so the palm centre reaches `palm_target` in `dur` seconds,
        coming up over the table edge first if the hand is below it."""
        palm_target = np.asarray(palm_target, float)
        look = palm_target if look is None else np.asarray(look, float)
        fwd = self._facing(look)
        goal = self.palm_to_wrist(palm_target, fwd)
        if self._below_table(self.pos):
            self._queue([(self._edge(), self._facing(palm_target), 0.42 * dur), (goal, fwd, 0.58 * dur)])
        else:
            self._queue([(goal, fwd, dur)])

    def park(self, dur):
        x, side = SEAT[self.who]
        rest = self._facing(np.array([x, 0.0, Z]))
        self._queue([(self._edge(), rest, 0.45 * dur), (self.parked.copy(), rest, 0.55 * dur)])

    @property
    def busy(self):
        return self.tween is not None

    def palm(self):
        return self.pos + self.fwd * 0.030 * SCALE[self.who]

    def tick(self, dt):
        self.t += dt
        if self.tween is not None:
            p0, f0, p1, f1, u, dur = self.tween
            u += dt / dur
            e = ease(u)
            self.pos = p0 + (p1 - p0) * e
            f = f0 + (f1 - f0) * e
            self.fwd = f / np.linalg.norm(f)
            if u >= 1.0:
                self._next_leg()
            else:
                self.tween = (p0, f0, p1, f1, u, dur)
        self.apply()

    def apply(self, sway=0.0):
        d = self.w.d
        d.mocap_pos[self.mid] = self.pos + np.array([0, 0, sway])
        d.mocap_quat[self.mid] = look_at_quat(self.fwd)


class Director:
    def __init__(self, world, rt):
        self.w, self.rt = world, rt
        self.hands = {who: Hand(world, who) for who in FAMILY}
        self.pad = {}       # persistent pad fields (held trigger, d-pad, pitch)
        self.markers = {}
        self.expr = {"name": "neutral", "look": [0.0, 0.0], "since": 0.0}
        self.carry = {}     # prop -> (holder, grip point in holder frame, yaw offset)
        self.frames = []
        self.sparkle_on = None
        self.log = []
        self.gaps = {}

    # ------------------------------------------------------------ bookkeeping
    @property
    def t(self):
        return self.rt.tick_n * DT

    def mark(self, name):
        self.markers[name] = round(self.t, 4)
        self.log.append(f"{self.t:7.3f}  {name}")

    def face(self, name, look=(0.0, 0.0)):
        self.expr = {"name": name, "look": list(look), "since": self.t}

    # ------------------------------------------------------------ carrying
    def attach(self, prop, holder):
        """Weld `prop` to `holder` where it is, and keep it upright from now on."""
        w = self.w
        w.attach(prop, holder)
        hb = w.holder_body(holder)
        R = w.d.xmat[hb].reshape(3, 3)
        pb = w.prop_body[prop]
        grip_w = w.d.xpos[pb] + np.array(GRIP[prop]) if holder == "claw" else w.d.xpos[pb].copy()
        grip_local = R.T @ (grip_w - w.d.xpos[hb])
        # prop heading is kept relative to the holder's heading
        qp = w.d.xquat[pb]
        prop_yaw = math.atan2(2 * (qp[0] * qp[3] + qp[1] * qp[2]), 1 - 2 * (qp[2] ** 2 + qp[3] ** 2))
        hold_yaw = math.atan2(R[1, 0], R[0, 0])
        offset = np.array(GRIP[prop]) if holder == "claw" else np.zeros(3)
        self.carry[prop] = (holder, grip_local, wrap(prop_yaw - hold_yaw), offset)
        self.log.append(f"{self.t:7.3f}  attach {prop} -> {holder}")

    def release(self, prop):
        self.w.release(prop)
        self.carry.pop(prop, None)
        self.log.append(f"{self.t:7.3f}  release {prop}")

    def _level(self):
        """Re-aim each carried prop's weld so it hangs upright under its grip point."""
        w, m, d = self.w, self.w.m, self.w.d
        for prop, (holder, grip_local, yaw_off, offset) in self.carry.items():
            hb = w.holder_body(holder)
            R = d.xmat[hb].reshape(3, 3)
            grip_w = d.xpos[hb] + R @ grip_local
            hold_yaw = math.atan2(R[1, 0], R[0, 0])
            yaw = hold_yaw + yaw_off
            qp = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            Rp = np.zeros(9)
            mujoco.mju_quat2Mat(Rp, qp)
            Rp = Rp.reshape(3, 3)
            pp = grip_w - Rp @ offset
            # relpose = holder pose in the prop's frame
            qh = d.xquat[hb]
            qpinv = np.array([qp[0], -qp[1], -qp[2], -qp[3]])
            rel_p = Rp.T @ (d.xpos[hb] - pp)
            rel_q = np.zeros(4)
            mujoco.mju_mulQuat(rel_q, qpinv, qh)
            eq = w.weld[(prop, holder)]
            m.eq_data[eq, 3:6] = rel_p
            m.eq_data[eq, 6:10] = rel_q

    # ------------------------------------------------------------ the tick
    def step(self, **over):
        fields = dict(self.pad)
        fields.update(over)
        pad = PadState(lx=fields.get("lx", 0.0), ly=fields.get("ly", 0.0), rx=fields.get("rx", 0.0),
                       ry=fields.get("ry", 0.0), lt=fields.get("lt", 0.0), rt=fields.get("rt", 0.0),
                       buttons=set(fields.get("buttons", ())), dpad_x=fields.get("dpad_x", 0),
                       dpad_y=fields.get("dpad_y", 0))
        for h in self.hands.values():
            h.tick(DT)
        self._level()
        self.rt.step(pad)
        self._record()

    def _record(self):
        d = self.w.d
        self.frames.append({
            "t": self.t, "qpos": d.qpos.copy(), "mocap_pos": d.mocap_pos.copy(),
            "mocap_quat": d.mocap_quat.copy(), "expr": dict(self.expr), "mode": self.rt.mode,
        })

    # ------------------------------------------------------------ primitives
    def hold(self, sec, **pad):
        for _ in range(max(1, int(round(sec / DT)))):
            self.step(**pad)

    def ramp(self, key, to, sec, **pad):
        start = self.pad.get(key, 0.0)
        n = max(1, int(round(sec / DT)))
        for i in range(n):
            self.pad[key] = start + (to - start) * ease((i + 1) / n)
            self.step(**pad)
        self.pad[key] = to

    def base_target_for(self, prop, yaw=0.0):
        g = self.w.prop_pos(prop) + np.array(GRIP[prop])
        c, s = math.cos(yaw), math.sin(yaw)
        off = np.array([c * JAW_OFFSET[0] - s * JAW_OFFSET[1], s * JAW_OFFSET[0] + c * JAW_OFFSET[1]])
        return g[:2] - off

    def goto(self, xy, yaw=None, tol=0.008, yaw_tol=0.06, vmax=0.75, k=7.0, timeout=4.0, settle=0.15,
             **pad):
        """Closed-loop pad operator: drive the policy to a base pose."""
        n_ok, n = 0, 0
        while n * DT < timeout:
            b = self.rt.base_pos()
            yw = self.rt.base_yaw()
            e = np.asarray(xy) - b[:2]
            c, s = math.cos(-yw), math.sin(-yw)
            eb = np.array([c * e[0] - s * e[1], s * e[0] + c * e[1]])
            vx = float(np.clip(k * eb[0], -vmax, vmax))
            vy = float(np.clip(k * eb[1], -vmax, vmax))
            rx = 0.0
            ey = 0.0 if yaw is None else wrap(yaw - yw)
            if abs(ey) > yaw_tol * 0.6:
                rx = -math.copysign(min(0.98, 0.56 + 0.55 * abs(ey)), ey)
            done = np.linalg.norm(e) < tol and abs(ey) < yaw_tol
            n_ok = n_ok + 1 if done else 0
            if n_ok * DT >= settle:
                break
            self.step(ly=-vx, lx=-vy, rx=rx, **pad)
            n += 1
        return np.linalg.norm(np.asarray(xy) - self.rt.base_pos()[:2])

    def turn(self, yaw, **kw):
        return self.goto(self.rt.base_pos()[:2], yaw=yaw, **kw)

    def jaw_point(self):
        m, d = self.w.m, self.w.d
        a = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/LF_finger_tip_link")]
        b = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/LF_palm_grip_insert_link")]
        return (a + b) / 2

    def grab(self, prop, fast=1.0):
        """Nose down, close the jaw on the item, weld on contact, lift."""
        self.ramp("ry", -1.0, 0.45 / fast)
        self.hold(0.15 / fast)
        n = max(1, int(round(0.30 / fast / DT)))
        attached = False
        for i in range(n):
            self.pad["lt"] = ease((i + 1) / n)
            self.step()
            if not attached and self.pad["lt"] > 0.6:
                gv = self.jaw_point() - (self.w.prop_pos(prop) + np.array(GRIP[prop]))
                yw = self.rt.base_yaw()
                c, s_ = math.cos(-yw), math.sin(-yw)
                gb = (c * gv[0] - s_ * gv[1], s_ * gv[0] + c * gv[1], gv[2])
                gap = float(np.linalg.norm(gv))
                self.log.append(f"{self.t:7.3f}  jaw-to-grip gap for {prop}: {gap * 1000:.0f} mm "
                                f"(body fwd {gb[0] * 1000:+.0f}, left {gb[1] * 1000:+.0f}, up {gb[2] * 1000:+.0f})")
                self.gaps[prop] = gb
                if gap > 0.08:
                    raise RuntimeError(f"grasp of {prop} missed by {gap:.3f} m; not welding")
                self.attach(prop, "claw")
                attached = True
        self.pad["lt"] = 1.0
        self.mark(f"grab_{prop}")
        self.ramp("ry", 0.0, 0.35 / fast)

    def present(self, sec=0.35):
        self.pad["dpad_y"] = 1
        self.hold(sec)

    def handoff(self, prop, who, reach=0.55, after="hold", hold_at=None, fast=1.0):
        """The hand comes to the item; the weld moves to the hand; the claw lets go."""
        hand = self.hands[who]
        item = self.w.prop_pos(prop)
        palm_target = item + np.array([0, 0, -0.014])
        hand.go(palm_target, reach / fast)
        while hand.busy:
            self.step()
        self.attach(prop, who)
        self.mark(f"handoff_{prop}_{who}")
        self.ramp("lt", 0.0, 0.18 / fast)
        self.pad["dpad_y"] = 0
        if after == "hold":
            x, side = SEAT[who]
            target = hold_at if hold_at is not None else np.array([x + 0.02, side * 0.36, Z + 0.11])
            hand.go(target, 0.7 / fast, look=np.array([x, 0.0, Z + 0.05]))

    def set_down(self, prop, who, where, dur=0.6):
        hand = self.hands[who]
        hand.go(np.asarray(where) + np.array([0, 0, -0.012]), dur)
        while hand.busy:
            self.step()
        self.release(prop)
        hand.park(0.8)

    # ------------------------------------------------------------ the story
    def run(self):
        rt, w = self.rt, self.w
        # the kid holds their drawing, off the table, from the start
        kid = self.hands["kid"]
        q = w.prop_qadr["drawing"]
        w.d.qpos[q:q + 3] = kid.palm() + np.array([0, 0, 0.012])
        w.d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(w.m, w.d)
        self.attach("drawing", "kid")

        # ---- 0-3 s: wake up, raise the claw, look around ---------------------
        self.mark("start")
        self.face("neutral")
        self.hold(0.5)
        self.face("wow")
        self.hold(0.08, buttons={"LB"})       # claw_left: the claw comes up
        self.mark("claw_up")
        self.hold(0.9)
        self.face("look", (0.8, 0.1))
        self.ramp("rx", 0.42, 0.35)            # body twist: look left
        self.hold(0.35)
        self.face("look", (-0.8, 0.1))
        self.ramp("rx", -0.42, 0.5)            # look right
        self.hold(0.3)
        self.ramp("rx", 0.0, 0.3)
        self.face("neutral")
        self.mark("purpose_q_end")

        # ---- 3-6 s: butter -> dad, "oh." ------------------------------------
        self.face("focus")
        self.goto(self.base_target_for("butter"), yaw=0.0, vmax=0.4, timeout=1.5)
        self.mark("butter_approach_done")
        self.grab("butter")
        self.turn(1.15, timeout=1.4)
        self.present(0.3)
        self.handoff("butter", "dad", reach=0.55)
        self.face("neutral")
        self.mark("oh")
        self.hold(1.0)
        self.face("sad")
        self.mark("droop")
        self.ramp("ry", -0.55, 0.45)           # the head droop
        self.hold(0.45)
        # dad puts the butter down by his plate while we move on
        self.hands["dad"].go(np.array([0.17, 0.30, Z - 0.002]), 0.6)

        # ---- 6-12 s: rolls / gravy / cranberry, faster each time ------------
        self.ramp("ry", 0.0, 0.25)
        self.face("focus")
        self.mark("rolls_start")
        self.goto(self.base_target_for("rolls"), yaw=0.0, vmax=0.8, timeout=1.6)
        if True:  # dad's butter lands by his plate
            self.release("butter")
            self.hands["dad"].park(0.8)
        self.grab("rolls", fast=1.2)
        self.goto([0.03, -0.10], yaw=-1.2, vmax=0.9, timeout=1.6, tol=0.03, yaw_tol=0.12)
        self.present(0.2)
        self.handoff("rolls", "mom", reach=0.45, fast=1.2,
                     hold_at=np.array([0.10, -0.40, Z + 0.10]))
        self.face("neutral")
        self.mark("gravy_start")
        self.goto(self.base_target_for("gravy"), yaw=0.0, vmax=0.9, timeout=1.6, tol=0.015)
        self.grab("gravy", fast=1.4)
        self.goto([0.10, 0.10], yaw=0.95, vmax=0.9, timeout=1.6, tol=0.03, yaw_tol=0.12)
        self.present(0.18)
        self.handoff("gravy", "aunt", reach=0.4, fast=1.4,
                     hold_at=np.array([0.42, 0.40, Z + 0.10]))
        self.mark("cranberry_start")
        self.goto(self.base_target_for("cranberry"), yaw=0.0, vmax=0.9, timeout=1.6, tol=0.015)
        self.grab("cranberry", fast=1.6)
        self.goto([0.14, -0.08], yaw=-0.95, vmax=0.95, timeout=1.6, tol=0.03, yaw_tol=0.12)
        self.present(0.15)
        self.handoff("cranberry", "kid", reach=0.35, fast=1.6,
                     hold_at=np.array([0.44, -0.40, Z + 0.09]))
        self.face("tired")
        self.mark("rapid_end")
        self.hold(0.4)
        # the kid sets the cranberry down by their plate
        self.hands["kid"].go(np.array([0.52, -0.33, Z - 0.002]), 0.5)
        self.hold(0.5)
        self.release("cranberry")
        self.hands["kid"].park(0.7)

        # ---- 12-19 s: the turn -------------------------------------------------
        # grandma can't reach her phone
        self.face("neutral")
        self.mark("phone_start")
        self.goto(self.base_target_for("phone"), yaw=0.0, vmax=0.6, timeout=2.0)
        self.face("focus")
        self.grab("phone", fast=0.9)
        self.face("neutral")
        self.mark("phone_walk")
        self.goto([-0.22, -0.06], yaw=-2.2, vmax=0.7, timeout=3.0, tol=0.03, yaw_tol=0.1)
        self.present(0.3)
        self.handoff("phone", "grandma", reach=0.7, fast=0.9,
                     hold_at=np.array([-0.50, -0.40, Z + 0.13]))
        self.face("happy")
        self.hold(0.5)

        # kid made a drawing: the kid's hand sets it down, the crab carries it to dad
        self.face("neutral")
        self.mark("drawing_start")
        self.hands["kid"].go(np.array([0.36, -0.17, Z + 0.004]), 0.9)
        self.goto([0.10, -0.05], yaw=0.0, vmax=0.7, timeout=2.2, tol=0.03)
        while self.hands["kid"].busy:
            self.step()
        self.release("drawing")
        self.hands["kid"].park(0.8)
        self.hold(0.25)
        self.goto(self.base_target_for("drawing"), yaw=0.0, vmax=0.5, timeout=1.8)
        self.face("focus")
        self.grab("drawing", fast=0.9)
        self.face("wow")
        self.mark("drawing_walk")
        self.goto([0.0, 0.08], yaw=1.25, vmax=0.6, timeout=2.4, tol=0.03, yaw_tol=0.1)
        self.present(0.35)
        self.handoff("drawing", "dad", reach=0.6, fast=0.9,
                     hold_at=np.array([0.04, 0.38, Z + 0.16]))
        self.face("happy")
        self.hold(0.4)

        # someone needs a napkin -> the kid (cranberry everywhere)
        self.face("neutral")
        self.mark("napkin_start")
        self.goto(self.base_target_for("napkin"), yaw=0.0, vmax=0.6, timeout=2.0)
        self.face("focus")
        self.grab("napkin", fast=0.8)
        self.goto([0.16, -0.06], yaw=-0.9, vmax=0.5, timeout=2.4, tol=0.03, yaw_tol=0.1)
        self.present(0.45)
        self.handoff("napkin", "kid", reach=0.8, fast=0.75,
                     hold_at=np.array([0.44, -0.38, Z + 0.10]))
        self.face("happy")
        self.mark("turn_end")

        # ---- 19-24 s: wide, the table is full ------------------------------------
        # everyone brings what they're holding in toward the middle of the table
        gather = {"dad": (0.06, 0.25, 0.14), "mom": (0.13, -0.26, 0.12), "aunt": (0.36, 0.24, 0.12),
                  "kid": (0.38, -0.23, 0.11), "grandma": (-0.30, -0.22, 0.15)}
        for who, (x, y, h) in gather.items():
            self.hands[who].go(np.array([x, y, Z + h]), 1.4, look=np.array([0.08, 0.0, Z + 0.08]))
        self.goto([0.08, 0.0], yaw=0.0, vmax=0.5, timeout=2.0, tol=0.02)
        self.mark("wide")
        self.face("content")
        self.hold(4.2)

        # ---- 24-27 s: victory dance + sparkles --------------------------------------
        if self.rt.mode == "claw_left":
            self.hold(0.08, buttons={"LB"})    # leave the claw mode
        self.hold(0.5)
        self.face("joy")
        self.mark("dance")
        self.hold(0.06, buttons={"menu"})      # menu held, then d-pad up -> dance_crab
        self.hold(0.08, buttons={"menu"}, dpad_y=1)
        self.hold(0.06, buttons={"menu"})
        self.hold(4.0)
        self.mark("end")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "build" / "run")
    ap.add_argument("--jumper", type=Path, default=Path(os.environ.get("JUMPER", "../jumper")))
    ap.add_argument("--robot-dump", type=Path, default=Path(os.environ.get("ROBOT_DUMP", "build/robot_dump.xml")))
    ap.add_argument("--map", type=Path, default=HERE / "packages" / "king-crab-table.map")
    ap.add_argument("--app", type=Path, default=HERE / "packages" / "king-crab.app")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    w = build_world(a.robot_dump, a.jumper / "assets/jumper", a.map, None, a.out / "unpacked")
    from crab_runtime import open_bundle
    bundle = open_bundle(a.app if a.app.exists() else a.jumper / "out/bundle_example/jumper",
                         a.out / "unpacked" / "app")
    rt = CrabRuntime(w.m, w.d, bundle)
    rt.reset_pose(xy=(0.0, 0.0), z_ground=Z, yaw=0.0)
    director = Director(w, rt)
    t0 = time.time()
    director.run()
    print(f"simulated {director.t:.2f} s in {time.time() - t0:.1f} s wall")
    np.savez_compressed(a.out / "frames.npz",
                        t=np.array([f["t"] for f in director.frames]),
                        qpos=np.stack([f["qpos"] for f in director.frames]),
                        mocap_pos=np.stack([f["mocap_pos"] for f in director.frames]),
                        mocap_quat=np.stack([f["mocap_quat"] for f in director.frames]))
    meta = {"markers": director.markers, "log": director.log,
            "expr": [f["expr"] for f in director.frames], "mode": [f["mode"] for f in director.frames],
            "transitions": [list(map(str, e)) for e in rt.log], "inferences": rt.inferences}
    (a.out / "run.json").write_text(json.dumps(meta, indent=0))
    print("\n".join(director.log))
    print("controller transitions:", rt.log)
    print("inferences:", rt.inferences)


if __name__ == "__main__":
    main()

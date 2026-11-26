"""Cart Heist -- the beat-sheet sequencer.

    python sequencer.py --out build/run      # simulate the whole story, record every 5 ms tick

POLICY (the app's compiled controller + its ONNX policies, unchanged, nothing retrained):
  * every leg motion: the sneak (sideways crab-walk), turning, the carry -- `locomotion`
    (jumper.posture) and `claw_left` (jumper.five_foot: walks on five legs, carries the sixth
    as a claw);
  * the claw: its height (the right stick's pitch, which the claw hook turns into the shoulder),
    the jaw closing and opening (the trigger) -- `claw_left` and its deploy hook;
  * the bow, the samba and the salute -- `gesture_bow`, `dance_brazilian`, `gesture_salute`.
SCRIPTED (this file):
  * the "operator": pad frames computed every 5 ms tick by a closed loop that turns position,
    heading and jaw-height error into stick deflection, plus button and trigger timing;
  * the hold: when the trigger has closed the jaw at the box (gap check), a weld attaches the
    box to the claw where it is, and keeps it level while carried. Opening the trigger over
    the basket removes the weld; the box then falls into the basket under physics;
  * the shoppers and the kid: kinematic (mocap) figures with a light sway, head bobs and turns;
  * the robot's display face (eyes) and the crown sparkles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sim"))
sys.path.insert(0, str(HERE / "scene"))
from crab_runtime import CONTROL_HZ, CrabRuntime, PadState, open_bundle  # noqa: E402
from world import build_world  # noqa: E402
import author_scene as S  # noqa: E402

DT = 1.0 / CONTROL_HZ
UP = math.pi / 2  # facing +Y: toward Cart A's rack, and toward the kid's basket

#: the grip point on the deal box, in its body frame (origin = bottom centre):
#: on the -Y long edge, 12 mm in, at mid-height
GRIP = np.array([0.0, -S.DEAL_HALF[1] + 0.012, S.DEAL_HALF[2]])
BASKET = np.array(S.BASKET_POS)
#: where the box is set down: the near half of the basket, so the crab's legs stay clear of it
PLACE = BASKET + np.array([0.0, -0.045])
PLACE_YAW = UP
#: In claw_left the right claw walks on its finger tip about 0.20 m ahead of the body, level with the
#: stowed jaw. For the pinch and the place the arm is held straight out on the claw hook's d-pad
#: preset (jaw ~0.28 m ahead), so that walking foot stays clear of the cart rack and the basket.


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def quat_zyx(yaw=0.0, pitch=0.0, roll=0.0):
    """World quaternion for yaw (z) then pitch (about the rotated y) then roll."""
    q = np.zeros(4)
    mat = np.zeros(9)
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]])
    mat[:] = R.reshape(-1)
    mujoco.mju_mat2Quat(q, mat)
    return q, R


class Figures:
    """The three kinematic figures. Pose is a function of sim time plus a few cues."""

    def __init__(self, world):
        self.w = world
        self.cues = {}  # name -> sim time

    def cue(self, name, t):
        self.cues[name] = t

    def _since(self, name, t, dur):
        t0 = self.cues.get(name)
        return 0.0 if t0 is None else ease((t - t0) / dur)

    def apply(self, t):
        d = self.w.d
        for fig, ph in (("shopper_a", 0.0), ("shopper_b", math.pi), ("kid", 0.6)):
            p0, q0 = self.w.fig_home[fig]
            hp0, _ = self.w.fig_home[fig + "_head"]
            yaw0 = 2 * math.atan2(q0[3], q0[0])
            yaw = yaw0
            lean, roll, bob, head_yaw, head_pitch, lift = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            notice = self._since("notice", t, 0.7)
            if fig.startswith("shopper"):
                # tug-of-war over the cart: leaning back and in, out of phase, with little head bobs
                # ("mine!" "no, mine!") and a sway -- all of it stops once they notice
                k = 1.0 - 0.85 * notice
                lean = k * (-0.025 + 0.05 * math.sin(2 * math.pi * 0.85 * t + ph))
                roll = k * 0.02 * math.sin(2 * math.pi * 0.55 * t + ph)
                yaw += k * 0.05 * math.sin(2 * math.pi * 0.7 * t + 2 * ph)
                bob = k * (0.14 * math.sin(2 * math.pi * 2.4 * t + ph) ** 2 - 0.06)
                # "wait..." -- both turn toward the rack where the box was (-Y side of the cart)
                side = -1.0 if fig == "shopper_a" else 1.0
                head_pitch = 0.55 * notice
                head_yaw = side * 0.55 * notice
                yaw += side * 0.30 * notice
                lean += 0.10 * notice
                # a double-take: heads snap back to each other, then down again
                dt_ = self._since("double_take", t, 0.25) - self._since("double_take_back", t, 0.25)
                head_yaw -= side * 0.55 * dt_
                head_pitch -= 0.45 * dt_
            else:  # the kid: a small idle sway, looking down at the empty basket, then up at the crab
                happy = self._since("kid_happy", t, 0.5)
                roll = 0.015 * math.sin(2 * math.pi * 0.4 * t)
                head_pitch = 0.45 * (1 - happy) - 0.10 * happy
                head_yaw = 0.25 * happy * math.sin(2 * math.pi * 0.5 * (t - self.cues.get("kid_happy", 0)))
                if "kid_happy" in self.cues:
                    tt = t - self.cues["kid_happy"]
                    if 0 < tt < 1.6:  # two little hops of joy
                        lift = 0.035 * abs(math.sin(math.pi * tt / 0.4)) * (1 - tt / 1.6)
            q, R = quat_zyx(yaw, lean, roll)
            mid = self.w.fig_mocap[fig]
            d.mocap_pos[mid] = p0 + np.array([0, 0, lift])
            d.mocap_quat[mid] = q
            # head rides on the (rotated) neck, with its own nod and turn
            neck = np.array([0, 0, hp0[2]])
            hq, _ = quat_zyx(yaw + head_yaw, lean + head_pitch + bob, roll)
            hm = self.w.fig_mocap[fig + "_head"]
            d.mocap_pos[hm] = p0 + np.array([0, 0, lift]) + R @ neck
            d.mocap_quat[hm] = hq


class Director:
    def __init__(self, world, rt):
        self.w, self.rt = world, rt
        self.figs = Figures(world)
        self.pad = {}
        self.markers = {}
        self.expr = {"name": "neutral", "look": [0.0, 0.0], "since": 0.0}
        self.carry = None
        self.frames = []
        self.log = []
        self.gaps = {}
        m = world.m
        self.tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/LF_finger_tip_link")
        self.ins = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/LF_palm_grip_insert_link")
        self.palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/LF_palm_link")
        self.jaw_off = None

    @property
    def t(self):
        return self.rt.tick_n * DT

    def mark(self, name):
        self.markers[name] = round(self.t, 4)
        self.log.append(f"{self.t:7.3f}  {name}")

    def face(self, name, look=(0.0, 0.0)):
        self.expr = {"name": name, "look": list(look), "since": self.t}

    # ------------------------------------------------------------ holding
    def attach(self, prop):
        w = self.w
        w.attach(prop, "claw")
        hb = w.holder_body("claw")
        R = w.d.xmat[hb].reshape(3, 3)
        pb = w.prop_body[prop]
        Rp = w.d.xmat[pb].reshape(3, 3)
        grip_w = w.d.xpos[pb] + Rp @ GRIP
        grip_local = R.T @ (grip_w - w.d.xpos[hb])
        qp = w.d.xquat[pb]
        prop_yaw = math.atan2(2 * (qp[0] * qp[3] + qp[1] * qp[2]), 1 - 2 * (qp[2] ** 2 + qp[3] ** 2))
        hold_yaw = math.atan2(R[1, 0], R[0, 0])
        self.carry = (prop, grip_local, wrap(prop_yaw - hold_yaw))
        self.log.append(f"{self.t:7.3f}  attach {prop} -> claw")

    def release(self, prop):
        self.w.release(prop)
        self.carry = None
        self.log.append(f"{self.t:7.3f}  release {prop}")

    def _level(self):
        """Re-aim the held box's weld so it stays level under the grip point."""
        if self.carry is None:
            return
        w, m, d = self.w, self.w.m, self.w.d
        prop, grip_local, yaw_off = self.carry
        hb = w.holder_body("claw")
        R = d.xmat[hb].reshape(3, 3)
        grip_w = d.xpos[hb] + R @ grip_local
        yaw = math.atan2(R[1, 0], R[0, 0]) + yaw_off
        qp = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        Rp = np.zeros(9)
        mujoco.mju_quat2Mat(Rp, qp)
        Rp = Rp.reshape(3, 3)
        pp = grip_w - Rp @ GRIP
        qpinv = np.array([qp[0], -qp[1], -qp[2], -qp[3]])
        rel_p = Rp.T @ (d.xpos[hb] - pp)
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, qpinv, d.xquat[hb])
        eq = w.weld[(prop, "claw")]
        m.eq_data[eq, 3:6] = rel_p
        m.eq_data[eq, 6:10] = rel_q

    # ------------------------------------------------------------ the tick
    def step(self, **over):
        f = dict(self.pad)
        f.update(over)
        pad = PadState(lx=f.get("lx", 0.0), ly=f.get("ly", 0.0), rx=f.get("rx", 0.0), ry=f.get("ry", 0.0),
                       lt=f.get("lt", 0.0), rt=f.get("rt", 0.0), buttons=set(f.get("buttons", ())),
                       dpad_x=f.get("dpad_x", 0), dpad_y=f.get("dpad_y", 0))
        self.figs.apply(self.t)
        self._level()
        self.rt.step(pad)
        d = self.w.d
        self.frames.append({"t": self.t, "qpos": d.qpos.copy(), "mocap_pos": d.mocap_pos.copy(),
                            "mocap_quat": d.mocap_quat.copy(), "expr": dict(self.expr), "mode": self.rt.mode,
                            "held": self.carry is not None})

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

    def press(self, sec=0.08, **pad):
        """Press-and-release of a toggle control, then one tick released."""
        self.hold(sec, **pad)
        self.step()

    def wait_mode(self, mode, timeout):
        n = 0
        while self.rt.mode != mode and n * DT < timeout:
            self.step()
            n += 1
        return self.rt.mode == mode

    def jaw(self):
        """Where the jaw closes. Before calibration, the finger-tip / palm-insert midpoint; after
        `calibrate_jaw`, that midpoint as measured with the jaw shut, carried in the palm's frame
        (so it does not move when the finger opens)."""
        d = self.w.d
        if self.jaw_off is None:
            return (d.xpos[self.tip] + d.xpos[self.ins]) / 2
        pb = self.palm
        return d.xpos[pb] + d.xmat[pb].reshape(3, 3) @ self.jaw_off

    def calibrate_jaw(self):
        """Snip-snip: squeeze the trigger twice; while shut, record the closed jaw in the palm frame."""
        d = self.w.d
        for i in range(2):
            self.ramp("lt", 1.0, 0.14)
            self.mark(f"snap{i}")
            self.hold(0.10)
            if i == 0:
                mid = (d.xpos[self.tip] + d.xpos[self.ins]) / 2
                pb = self.palm
                self.jaw_off = d.xmat[pb].reshape(3, 3).T @ (mid - d.xpos[pb])
                self.log.append(f"{self.t:7.3f}  closed-jaw point in the palm frame (mm): "
                                f"{np.round(self.jaw_off * 1000, 1).tolist()}")
            self.ramp("lt", 0.0, 0.14)
            self.hold(0.06)

    def grip_world(self, prop="deal"):
        d, pb = self.w.d, self.w.prop_body[prop]
        return d.xpos[pb] + d.xmat[pb].reshape(3, 3) @ GRIP

    def goto(self, xy, yaw=None, tol=0.008, yaw_tol=0.05, vmax=0.75, k=6.0, timeout=4.0, settle=0.12,
             vmin=0.16, **pad):
        """Closed-loop operator: stick deflection from base position + heading error.

        Below `vmin` of stick the policies' gait gate (0.05 m/s) keeps the robot standing, so a
        remaining error above `tol` is always asked for with at least `vmin`."""
        n_ok, n = 0, 0
        while n * DT < timeout:
            b = self.rt.base_pos()
            yw = self.rt.base_yaw()
            e = np.asarray(xy, float) - b[:2]
            c, s = math.cos(-yw), math.sin(-yw)
            eb = np.array([c * e[0] - s * e[1], s * e[0] + c * e[1]])
            vx = float(np.clip(k * eb[0], -vmax, vmax))
            vy = float(np.clip(k * eb[1], -vmax, vmax))
            sp = math.hypot(vx, vy)
            if np.linalg.norm(e) > tol and 1e-9 < sp < vmin:
                vx, vy = vx * vmin / sp, vy * vmin / sp
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
        return float(np.linalg.norm(np.asarray(xy) - self.rt.base_pos()[:2]))

    def turn(self, yaw, **kw):
        return self.goto(self.rt.base_pos()[:2], yaw=yaw, **kw)

    def servo_jaw(self, target, yaw, rounds=3, vmax=0.25, z=True):
        """Bring the jaw point to `target` (world): base xy by goto, height by the pitch stick."""
        for r in range(rounds):
            if z:
                for _ in range(int(0.6 / DT)):  # height loop on the pitch stick
                    ez = target[2] - self.jaw()[2]
                    self.pad["ry"] = float(np.clip(self.pad.get("ry", 0.0) + 0.12 * ez, -1, 1))
                    self.step()
            e = target[:2] - self.jaw()[:2]
            b = self.rt.base_pos()[:2]
            if np.linalg.norm(e) > 0.004:
                self.goto(b + e, yaw=yaw, vmax=vmax, tol=0.004, timeout=1.2, k=5.0)
            self.log.append(f"{self.t:7.3f}  servo round {r}: jaw err (mm) {np.round((target - self.jaw()) * 1000, 1).tolist()}"
                            f" ry {self.pad.get('ry', 0):+.2f}")
        return target - self.jaw()

    # ------------------------------------------------------------ the story
    def run(self):
        F = self.figs
        # ---- 0-2 s (hook): the shoppers argue; the crab waits off-screen and turns to face the aisle
        self.mark("start")
        self.face("sneaky", (0.6, 0.0))
        self.hold(0.3)
        self.goto(self.rt.base_pos()[:2], yaw=UP, timeout=2.0, vmax=0.0)
        self.hold(0.4)

        # ---- 2-6 s: the sneak -- crab-walk sideways (-X) behind Cart A, slow
        self.mark("sneak")
        self.face("sneaky", (-0.7, 0.1))
        self.goto([0.70, -0.55], yaw=UP, vmax=0.45, timeout=6.0, tol=0.03, yaw_tol=0.08)
        self.face("sneaky", (0.7, 0.3))     # glance up at the shoppers
        self.mark("glance")
        self.hold(0.45)
        self.face("sneaky", (-0.7, 0.1))
        pre = self.grip_world()[:2] + np.array([0.045, -0.30])
        self.goto(pre, yaw=UP, vmax=0.40, timeout=5.0, tol=0.02, yaw_tol=0.05)
        self.mark("arrive_cart")
        self.hold(0.2)

        # ---- 6-11 s: the claw comes up, reaches, pinches, lifts
        self.face("focus")
        self.press(buttons={"LB"})          # claw_left
        self.wait_mode("claw_left", 3.0)
        self.mark("claw_up")
        self.hold(0.3)
        self.mark("snip")
        self.calibrate_jaw()
        self.pad["dpad_y"] = 1              # d-pad up held: the arm straight out (the claw hook's preset)
        self.hold(0.7)
        self.mark("arm_out_grab")
        g = self.grip_world()
        approach = g + np.array([0.0, -0.06, 0.0])  # jaw 6 cm short of the box edge, at its height
        self.servo_jaw(approach, UP, rounds=2)
        self.mark("reach")
        err = self.servo_jaw(g, UP, rounds=3, vmax=0.18)
        self.log.append(f"{self.t:7.3f}  servo residual (mm): {np.round(err * 1000, 1).tolist()}")
        self.hold(0.15)
        n = int(round(0.35 / DT))
        attached = False
        for i in range(n):  # the trigger squeezes the jaw shut
            self.pad["lt"] = ease((i + 1) / n)
            self.step()
            if not attached and self.pad["lt"] > 0.6:
                gv = self.jaw() - self.grip_world()
                gap = float(np.linalg.norm(gv))
                self.gaps["deal"] = np.round(gv * 1000, 1).tolist()
                self.log.append(f"{self.t:7.3f}  jaw-to-grip gap: {gap * 1000:.1f} mm {self.gaps['deal']}")
                if gap > 0.03:
                    raise RuntimeError(f"pinch missed the box by {gap:.3f} m; not welding")
                self.attach("deal")
                attached = True
        self.pad["lt"] = 1.0
        self.mark("pinch")
        self.face("wow")
        self.hold(0.25)
        self.ramp("ry", 0.45, 0.7)          # lift: the shoulder raises the box off the rack
        self.mark("lift")
        self.hold(0.35)
        b = self.rt.base_pos()[:2]
        self.goto(b + np.array([0.0, -0.12]), yaw=UP, vmax=0.3, timeout=1.2, tol=0.015)
        self.pad["dpad_y"] = 0              # arm back to the stow, box in hand, held high on the pitch
        self.hold(0.5)
        self.mark("got_it")
        self.face("sneaky", (0.7, 0.3))
        self.hold(0.3)

        # ---- 11-16 s: crab-walk away with the box, past the shoppers, to the checkout
        self.face("sneaky", (-0.8, 0.0))
        self.mark("escape")
        self.goto([-0.75, -0.50], yaw=UP, vmax=0.70, timeout=5.0, tol=0.04, yaw_tol=0.08)
        self.mark("pass_shopper")
        self.goto([-1.70, -0.48], yaw=UP, vmax=0.70, timeout=5.0, tol=0.04, yaw_tol=0.08)
        self.face("neutral")
        self.mark("near_checkout")

        # ---- 16-21 s: set the box in the kid's basket (real place), step back, bow
        F.cue("kid_notice", self.t)
        fwd = np.array([math.cos(PLACE_YAW), math.sin(PLACE_YAW)])
        self.turn(PLACE_YAW, timeout=1.5)
        for r in range(2):  # stop with the box 12 cm short of the basket
            e = PLACE - 0.12 * fwd - self.w.prop_pos("deal")[:2]
            if np.linalg.norm(e) > 0.01:
                self.goto(self.rt.base_pos()[:2] + e, yaw=PLACE_YAW, vmax=0.35, timeout=1.5, tol=0.01)
        self.mark("arm_out")
        self.pad["dpad_y"] = 1              # d-pad up held: the arm goes out straight (the claw hook's preset)
        self.hold(0.7)
        for r in range(3):  # then the box centre over the near half of the basket
            e = PLACE - self.w.prop_pos("deal")[:2]
            if np.linalg.norm(e) > 0.006:
                self.goto(self.rt.base_pos()[:2] + e, yaw=PLACE_YAW, vmax=0.3, timeout=1.5, tol=0.005)
        self.mark("over_basket")
        self.face("focus")
        target_bottom = S.BASKET_H + 0.012
        self.ramp("ry", 0.05, 0.5)          # lower on the pitch stick until the box clears the rim by ~1 cm
        for _ in range(int(1.0 / DT)):
            ez = target_bottom - self.w.prop_pos("deal")[2]
            if abs(ez) < 0.002:
                break
            self.pad["ry"] = float(np.clip(self.pad.get("ry", 0.0) + 0.25 * ez, -1, 1))
            self.step()
        self.hold(0.15)
        self.mark("lowered")
        self.log.append(f"{self.t:7.3f}  box bottom {self.w.prop_pos('deal')[2]*1000:.0f} mm, rim {S.BASKET_H*1000:.0f} mm, "
                        f"box-to-target xy {np.round((self.w.prop_pos('deal')[:2]-PLACE)*1000,1).tolist()} mm")
        n = int(round(0.30 / DT))
        released = False
        for i in range(n):  # the trigger lets go
            self.pad["lt"] = 1.0 - ease((i + 1) / n)
            self.step()
            if not released and self.pad["lt"] < 0.5:
                self.release("deal")
                released = True
                self.mark("release")
        self.pad["lt"] = 0.0
        # wait for the box to land (physics)
        landed = None
        for _ in range(int(1.0 / DT)):
            self.step()
            v = np.linalg.norm(self.w.d.cvel[self.w.prop_body["deal"]][3:])
            inside = np.all(np.abs(self.w.prop_pos("deal")[:2] - BASKET) < np.array(S.BASKET_HALF) - 0.01)
            if landed is None and inside and self.w.prop_pos("deal")[2] < 0.012 and v < 0.05:
                landed = self.t
                self.mark("landed")
        if landed is None:
            raise RuntimeError(f"the box did not land in the basket: pos {self.w.prop_pos('deal')}, "
                               f"quat {self.w.d.xquat[self.w.prop_body['deal']]}")
        F.cue("kid_happy", self.t)
        self.face("happy")
        self.pad["dpad_y"] = 0              # arm back to the stow
        self.ramp("ry", 0.0, 0.4)
        self.goto(self.rt.base_pos()[:2] - 0.18 * fwd, yaw=PLACE_YAW, vmax=0.35, timeout=1.5, tol=0.02)
        self.turn(UP, timeout=1.2)          # square up to the kid
        self.mark("stepped_back")
        self.press(buttons={"LB"})          # leave the claw: back to locomotion
        self.wait_mode("locomotion", 3.0)
        self.hold(1.2)
        self.mark("bow")
        self.face("content")
        self.press(dpad_y=-1)               # d-pad down: gesture_bow
        self.wait_mode("gesture_bow", 3.0)
        self.mark("bow_on")
        F.cue("notice", self.t + 1.0)       # meanwhile, at Cart A
        F.cue("double_take", self.t + 2.4)
        F.cue("double_take_back", self.t + 2.9)
        self.wait_mode("locomotion", 14.0)
        self.mark("bow_done")
        self.hold(0.6)

        # ---- 24-28 s: samba, then salute
        self.face("joy")
        self.mark("dance")
        self.hold(0.06, buttons={"menu"})   # Menu held, then d-pad down: dance_brazilian
        self.hold(0.08, buttons={"menu"}, dpad_y=-1)
        self.hold(0.06, buttons={"menu"})
        self.step()
        self.wait_mode("dance_brazilian", 3.0)
        self.mark("dance_on")
        self.hold(4.5)
        self.mark("leave_dance")
        self.press(buttons={"menu"})        # Menu alone leaves the motion
        self.wait_mode("locomotion", 4.0)
        self.hold(0.6)
        self.face("content")
        self.mark("salute")
        self.press(dpad_x=1)                # d-pad right: gesture_salute
        self.wait_mode("gesture_salute", 3.0)
        self.mark("salute_on")
        self.wait_mode("locomotion", 10.0)
        self.mark("salute_done")
        self.hold(0.5)
        self.mark("end")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "build" / "run")
    ap.add_argument("--jumper", type=Path, default=Path(os.environ.get("JUMPER", "../jumper")))
    ap.add_argument("--robot-dump", type=Path, default=Path(os.environ.get("ROBOT_DUMP", "build/robot_dump.xml")))
    ap.add_argument("--map", type=Path, default=HERE / "../../maps/black_friday_store.map")
    ap.add_argument("--app", type=Path, default=HERE / "../../apps/cart_heist.app")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    w = build_world(a.robot_dump, a.jumper / "assets/jumper", a.map, a.out / "unpacked")
    app = a.app.resolve()
    print("app:", app)
    bundle = open_bundle(app, a.out / "unpacked" / "app")
    rt = CrabRuntime(w.m, w.d, bundle)
    rt.reset_pose(xy=SPAWN[:2], z_ground=0.0, yaw=SPAWN[2])
    director = Director(w, rt)
    t0 = time.time()
    try:
        director.run()
    finally:
        print(f"simulated {director.t:.2f} s in {time.time() - t0:.1f} s wall")
        np.savez_compressed(a.out / "frames.npz",
                            t=np.array([f["t"] for f in director.frames]),
                            qpos=np.stack([f["qpos"] for f in director.frames]),
                            mocap_pos=np.stack([f["mocap_pos"] for f in director.frames]),
                            mocap_quat=np.stack([f["mocap_quat"] for f in director.frames]),
                            held=np.array([f["held"] for f in director.frames]))
        meta = {"markers": director.markers, "log": director.log, "gaps": director.gaps,
                "expr": [f["expr"] for f in director.frames], "mode": [f["mode"] for f in director.frames],
                "transitions": [list(map(str, e)) for e in rt.log], "inferences": rt.inferences,
                "app": str(app)}
        (a.out / "run.json").write_text(json.dumps(meta, indent=0))
        print("\n".join(director.log))
        print("controller transitions:", rt.log)
        print("inferences:", rt.inferences)


#: where the crab starts: off-screen at the near end of the aisle, behind Cart A's line (x, y, yaw)
SPAWN = (1.60, -0.55, math.pi)

if __name__ == "__main__":
    main()

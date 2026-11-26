"""Measure where the shipped claw_left policy puts its jaws, on the store floor.

For each (right-stick pitch, d-pad preset), with the trigger squeezed, record the jaw point
(midway between the finger tip and the palm's grip insert) in the body frame and its height.
These numbers set the cart rack height, the basket rim and JAW_OFFSET in the sequencer.
"""
import json, math, os, sys
from pathlib import Path
import mujoco, numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "sim"))
from crab_runtime import CrabRuntime, PadState, open_bundle  # noqa
from world import build_world  # noqa

# usage: JUMPER=... ROBOT_DUMP=... python tools/probe_claw.py [OUT_DIR]
B = Path(sys.argv[1] if len(sys.argv) > 1 else "build")
J = Path(os.environ["JUMPER"])
w = build_world(Path(os.environ["ROBOT_DUMP"]), J / "assets/jumper", HERE / "../../maps/black_friday_store.map", B / "probe")
bundle = open_bundle(HERE / "../../apps/cart_heist.app", B / "probe/app")
rt = CrabRuntime(w.m, w.d, bundle)
rt.reset_pose(xy=(-1.5, -0.6), z_ground=0.0, yaw=0.0)
m, d = w.m, w.d
tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/LF_finger_tip_link")
ins = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/LF_palm_grip_insert_link")
def jaw():
    return (d.xpos[tip] + d.xpos[ins]) / 2
def run(sec, **kw):
    for _ in range(int(sec * 200)):
        rt.step(PadState(**kw))
run(0.5)
run(0.06, buttons={"LB"})
run(1.5)
print("mode", rt.mode, "base z", rt.base_pos()[2])
res = []
for dp in (0, 1):
    for ry in (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0):
        run(1.2, ry=ry, lt=1.0, dpad_y=dp)
        b = rt.base_pos(); yw = rt.base_yaw(); j = jaw()
        e = j - b; c, s = math.cos(-yw), math.sin(-yw)
        fwd, left = c * e[0] - s * e[1], s * e[0] + c * e[1]
        r = dict(dpad_y=dp, ry=ry, fwd=round(fwd, 4), left=round(left, 4), jaw_z=round(j[2], 4), base_z=round(b[2], 4), mode=rt.mode)
        print(r); res.append(r)
    run(1.0, ry=0.0, lt=0.0)
# walking speed in claw mode (forward stick full) and sideways
p0 = rt.base_pos().copy(); run(2.0, ly=-1.0); p1 = rt.base_pos().copy()
print("claw fwd speed m/s", np.linalg.norm(p1[:2]-p0[:2]) / 2.0, "mode", rt.mode)
run(1.0)
p0 = rt.base_pos().copy(); run(2.0, lx=-1.0); p1 = rt.base_pos().copy()
print("claw side speed m/s", np.linalg.norm(p1[:2]-p0[:2]) / 2.0, p1 - p0)
json.dump(res, open(B / "probe_claw.json", "w"), indent=1)

"""Check, over a recorded run, how close the crab's legs and body come to the basket and the cart rack.

    python tools/check_clearance.py RUN_DIR ROBOT_DUMP JUMPER

In the film the robot and these props share no collision bit (by design, so nothing snags), so
this check is what shows the choreography keeps them apart. On a copy of the model the basket
walls/floor and the rack plate are made collidable with the robot, with a 6 cm detection margin,
and MuJoCo's own narrowphase (`mj_fwdPosition`, the collision code the physics uses) is run on
every recorded state, sampled every 50 ms. Contacts between the target geoms and any robot
collision geom except the carried claw arm (`LF_*`, which reaches over the rack and the basket on
purpose) are collected. A negative distance would be interpenetration."""
import sys
from pathlib import Path

import os

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "sim"))
from world import build_world  # noqa: E402

run = Path(sys.argv[1])
w = build_world(Path(sys.argv[2]), Path(sys.argv[3]) / "assets/jumper", HERE / "../../maps/black_friday_store.map",
                run / "chk")
rec = np.load(run / "frames.npz")
MARGIN = float(os.environ.get("MARGIN", "0.06"))  # detection margin (m)
m, d = w.m, w.d


def gname(g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""


def bname(g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""


robot = {g for g in range(m.ngeom) if bname(g).startswith("robot/") and m.geom_contype[g] != 0
         and not bname(g).startswith("robot/LF_")}
targets = {"basket": {g for g in range(m.ngeom) if gname(g).startswith(("scn_basket_wall", "scn_basket_floor"))},
           "cart_rack": {g for g in range(m.ngeom) if gname(g).startswith("scn_cart_a_rack")}}
assert all(targets.values()), targets
alltg = set().union(*targets.values())
# collide only robot <-> targets: everything else off
for g in range(m.ngeom):
    m.geom_contype[g] = 0
    m.geom_conaffinity[g] = 0
for g in robot:
    m.geom_contype[g] = 1
for g in alltg:
    m.geom_conaffinity[g] = 1
    m.geom_margin[g] = MARGIN
# the broadphase reads per-body masks compiled from the geoms: recompute them
for b_ in range(m.nbody):
    gs = range(m.body_geomadr[b_], m.body_geomadr[b_] + m.body_geomnum[b_]) if m.body_geomnum[b_] else []
    m.body_contype[b_] = int(np.bitwise_or.reduce([m.geom_contype[g] for g in gs])) if gs else 0
    m.body_conaffinity[b_] = int(np.bitwise_or.reduce([m.geom_conaffinity[g] for g in gs])) if gs else 0
    m.body_margin[b_] = max([m.geom_margin[g] for g in gs], default=0.0) if hasattr(m, "body_margin") else 0
m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
worst = {k: (MARGIN, None) for k in targets}
spans = {k: [] for k in targets}
for k in range(0, len(rec["t"]), 10):
    d.qpos[:] = rec["qpos"][k]
    d.mocap_pos[:] = rec["mocap_pos"][k]
    mujoco.mj_fwdPosition(m, d)
    t = round(float(rec["t"][k]), 2)
    for c in d.contact[: d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        rg, tg = (g1, g2) if g1 in robot else (g2, g1)
        if rg not in robot or tg not in alltg:
            continue
        name = next(n for n, s in targets.items() if tg in s)
        if c.dist < worst[name][0]:
            worst[name] = (float(c.dist), (t, gname(rg)))
        if c.dist < 0:
            sp = spans[name]
            if sp and t - sp[-1][1] < 0.11:
                sp[-1][1] = t
            else:
                sp.append([t, t])
for name, (dist, at) in worst.items():
    if at is None:
        print(f"{name}: no robot geom within {MARGIN * 1000:.0f} mm")
    else:
        print(f"{name}: closest {dist * 1000:.1f} mm at {at}; interpenetrating during {spans[name]}")

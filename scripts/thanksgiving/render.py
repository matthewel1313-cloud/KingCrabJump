"""Render the recorded run through the edit decision list (EDL) into frames.

    MUJOCO_GL=egl python render.py --run build/run --out build/frames [--scale 0.5] [--workers 3]

Every video frame maps to one simulation instant through the EDL (shots with a
sim-time range and a video duration, so a shot can run faster or slower than
real time, and cuts can skip time). The recorded state at that instant is
restored, the robot's display face and the crown sparkles are posed, and the
frame is rendered with MuJoCo, plus a depth-of-field pass from the depth buffer.
No physics runs here: the frames show exactly what the simulation did.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sim"))

FPS = 30
Z = 0.76


# ----------------------------------------------------------------------- EDL

def edl(M):
    """Shots: (name, camera, sim_start, sim_end, video_seconds, focus, extra).

    `M` are the sequencer's markers (sim seconds)."""
    return [
        # 0-3 s  close-up: wake, claw up, look around
        ("intro", "close", 0.0, 3.0, 3.0, ["crab"], {"dist": 0.66, "el": -8}),
        # 3-6.4 s  butter
        ("butter_grab", "medium", M["purpose_q_end"] + 0.55, M["grab_butter"] + 0.75, 1.3,
         ["crab", "prop:butter"], {"az": 200, "el": -22, "dist": 0.7}),
        ("butter_pass", "medium", M["grab_butter"] + 1.35, M["droop"] + 0.85, 2.1,
         ["crab", "hand:dad"], {"az": 238, "el": -24, "dist": 0.85}),
        # 6.4-12 s  rapid requests
        ("rolls", "track", M["rolls_start"], M["handoff_rolls_mom"] + 0.15, 2.15,
         ["crab", "carried"], {"az": 152, "el": -30, "dist": 0.95}),
        ("gravy", "track", M["gravy_start"], M["handoff_gravy_aunt"] + 0.12, 1.45,
         ["crab", "carried"], {"az": 214, "el": -32, "dist": 0.95}),
        ("cranberry", "track", M["cranberry_start"], M["handoff_cranberry_kid"] + 0.1, 1.25,
         ["crab", "carried"], {"az": 148, "el": -32, "dist": 0.95}),
        ("tired", "close", M["rapid_end"], M["rapid_end"] + 0.75, 0.75, ["crab"], {"dist": 0.58, "el": -12, "az": 140}),
        # 12-20 s  the turn (slower)
        ("phone_grab", "medium", M["grab_phone"] - 1.0, M["grab_phone"] + 0.4, 1.4,
         ["crab", "prop:phone"], {"az": 200, "el": -22, "dist": 0.7}),
        ("phone_pass", "medium", M["handoff_phone_grandma"] - 1.6, M["handoff_phone_grandma"] + 0.6, 1.5,
         ["crab", "hand:grandma"], {"az": 62, "el": -26, "dist": 0.9}),
        ("drawing_in", "medium", M["drawing_start"] + 0.2, M["drawing_start"] + 1.9, 1.1,
         ["hand:kid", "crab"], {"az": 205, "el": -30, "dist": 0.85}),
        ("drawing_pass", "medium", M["handoff_drawing_dad"] - 1.9, M["handoff_drawing_dad"] + 0.6, 1.8,
         ["crab", "hand:dad"], {"az": 240, "el": -24, "dist": 0.88}),
        ("napkin_grab", "close", M["grab_napkin"] - 0.9, M["grab_napkin"] + 0.3, 1.0,
         ["crab", "prop:napkin"], {"az": 192, "el": -12, "dist": 0.6}),
        ("napkin_pass", "medium", M["handoff_napkin_kid"] - 1.5, M["handoff_napkin_kid"] + 0.7, 1.7,
         ["crab", "hand:kid"], {"az": 132, "el": -24, "dist": 0.82}),
        # 20-24.5 s  wide, the table is full (slow push-in)
        ("wide", "wide", M["wide"] - 0.2, M["dance"] - 0.2, 4.5, ["table"], {}),
        # 24.5-27.5 s  victory dance + end card
        ("dance", "close", M["dance"] + 0.2, M["dance"] + 3.4, 2.8, ["crab"],
         {"dist": 0.80, "el": -14, "az": 182}),
    ]


def timeline(M):
    """Per video frame: (shot index, sim time, u within shot)."""
    shots = edl(M)
    frames = []
    for i, (_, _, a, b, dur, _, _) in enumerate(shots):
        n = int(round(dur * FPS))
        for k in range(n):
            u = (k + 0.5) / n
            frames.append((i, a + (b - a) * u, u))
    return shots, frames


def video_times(M):
    """Start time (video seconds) of each shot, for the caption track."""
    t, out = 0.0, {}
    for name, _, _, _, dur, _, _ in edl(M):
        out[name] = (t, t + int(round(dur * FPS)) / FPS)
        t += int(round(dur * FPS)) / FPS
    return out


# ------------------------------------------------------------------ the face

BLINKS = np.arange(1.7, 60, 3.1)


def pose_face(m, eyes, expr, t_sim):
    import mujoco  # noqa: F401
    name = expr["name"]
    lx, ly = expr.get("look", [0, 0])
    since = expr.get("since", 0.0)
    eye = np.array([0.0012, 0.0095, 0.0115])
    pup = np.array([0.0012, 0.0042, 0.0050])
    dz_eye, pup_dy, pup_dz = 0.0, 0.0, -0.0015
    lid_alpha, lid_z, lid_tilt = 0.0, 0.010, 0.0
    cheek_alpha, pupil_alpha = 0.0, 1.0
    if name == "look":
        pup_dy, pup_dz = 0.0042 * lx, -0.0015 + 0.003 * ly
    elif name == "focus":
        eye[2] = 0.0088
        pup_dz = -0.003
    elif name == "wow":
        eye *= [1, 1.12, 1.18]
        pup = pup * 0.8
    elif name == "sad":
        eye[2] = 0.0080
        dz_eye = -0.0025
        pup_dz = -0.0045
        lid_alpha, lid_z, lid_tilt = 1.0, 0.0045, 0.38
    elif name == "tired":
        eye[2] = 0.0070
        dz_eye = -0.0015
        lid_alpha, lid_z, lid_tilt = 1.0, 0.0040, 0.0
    elif name in ("happy", "content", "joy"):
        eye[2] = 0.0042 if name != "content" else 0.0055
        dz_eye = 0.0015
        pupil_alpha = 0.0
        cheek_alpha = 1.0 if name != "content" else 0.75
        if name == "joy":
            eye[1] *= 1.12
    # blinks (not while squinting)
    if pupil_alpha > 0 and np.any(np.abs(BLINKS - t_sim) < 0.06):
        eye[2] = 0.0015
        pupil_alpha = 0.0
    # little pop when an expression starts
    pop = 1.0 + 0.10 * math.exp(-((t_sim - since) / 0.08) ** 2)
    for side, y in (("l", 0.024), ("r", -0.024)):
        g = eyes[f"eye_{side}"]
        m.geom_size[g] = eye * [1, pop, pop]
        m.geom_pos[g] = [0.0118, y, -0.001 + dz_eye]
        g = eyes[f"pupil_{side}"]
        m.geom_size[g] = pup
        m.geom_pos[g] = [0.0124, y + pup_dy, pup_dz + dz_eye]
        m.geom_rgba[g][3] = pupil_alpha
        g = eyes[f"shine_{side}"]
        m.geom_pos[g] = [0.0130, y + pup_dy + 0.0025, pup_dz + dz_eye + 0.0035]
        m.geom_rgba[g][3] = pupil_alpha
        g = eyes[f"lid_{side}"]
        m.geom_rgba[g][3] = lid_alpha
        m.geom_pos[g] = [0.0128, y, lid_z + dz_eye]
        tilt = lid_tilt if side == "l" else -lid_tilt
        m.geom_quat[g] = [math.cos(tilt / 2), math.sin(tilt / 2), 0, 0]
        g = eyes[f"cheek_{side}"]
        m.geom_rgba[g][3] = cheek_alpha


# --------------------------------------------------------------- sparkles

def pose_sparkles(w, t_sim, t0, crown):
    """Crown sparkles from t0: deterministic little stars bursting up and out."""
    rng = np.random.default_rng(42)
    m, d = w.m, w.d
    for i, mid in enumerate(w.sparkles):
        birth = t0 + rng.uniform(0, 3.0)
        life = rng.uniform(0.45, 0.8)
        ang = rng.uniform(0, 2 * math.pi)
        up = rng.uniform(0.18, 0.40)
        out = rng.uniform(0.06, 0.20)
        spin = rng.uniform(-8, 8)
        age = t_sim - birth
        if t_sim < t0 or not (0 <= age <= life):
            d.mocap_pos[mid] = [0, 0, -5]
            continue
        u = age / life
        p = crown + np.array([out * math.cos(ang) * u, out * math.sin(ang) * u, up * u - 0.25 * (u * life) ** 2 + 0.01])
        d.mocap_pos[mid] = p
        a = spin * age
        d.mocap_quat[mid] = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
        s = math.sin(math.pi * u) * (0.8 + 0.4 * math.sin(30 * age + i))
        bid = m.mocap_bodyid if hasattr(m, "mocap_bodyid") else None
        for gname in (f"sparkle{i}_a", f"sparkle{i}_b"):
            import mujoco
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
            base = np.array([0.0012, 0.0012, 0.0055]) if gname.endswith("_a") else np.array([0.0055, 0.0012, 0.0012])
            m.geom_size[g] = base * max(s, 0.05) * 2.6


# --------------------------------------------------------------- cameras

def focus_point(w, M, tag, frame_state):
    import mujoco
    m, d = w.m, w.d
    if tag == "crab":
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
        return d.xpos[b] + np.array([0, 0, -0.02])
    if tag.startswith("prop:"):
        return w.prop_pos(tag[5:]) + np.array([0, 0, 0.02])
    if tag.startswith("hand:"):
        b = w.hand_body[tag[5:]]
        p = d.xpos[b].copy()
        if p[2] < Z + 0.02 or abs(p[1]) > 0.45:  # parked / still coming: aim at the table edge by the seat
            p = np.array([p[0], math.copysign(0.33, p[1]), Z + 0.05])
        return p
    if tag == "carried":
        # whatever the claw holds, else the crab
        for p in w.prop_body:
            if w.holder_of(p) == "claw":
                return w.prop_pos(p)
        return focus_point(w, M, "crab", frame_state)
    if tag == "table":
        return np.array([0.0, 0.0, Z])
    raise ValueError(tag)


def camera_for(shot, w, M, u, raw_lookat):
    kind, extra = shot[1], shot[6]
    if kind == "close":
        return dict(lookat=raw_lookat + np.array([0.0, 0.0, 0.05]), dist=extra.get("dist", 0.62),
                    az=extra.get("az", 188) + 6 * math.sin(u * 2.0), el=extra.get("el", -10), fovy=40)
    if kind in ("medium", "track"):
        return dict(lookat=raw_lookat + np.array([0, 0, 0.03]), dist=extra.get("dist", 0.85),
                    az=extra.get("az", 190), el=extra.get("el", -30), fovy=44)
    if kind == "wide":
        # high and wide, slow push-in toward the crab on the final line
        e = u * u * (3 - 2 * u)
        crab = focus_point(w, None, "crab", None)
        return dict(lookat=np.array([0.06, 0.0, Z + 0.02]) * (1 - e) + (crab + [0, 0, 0.03]) * e,
                    dist=1.85 - 0.95 * e, az=180, el=-62 + 22 * e, fovy=58 - 12 * e)
    raise ValueError(kind)


# ---------------------------------------------------------------- worker

_W = {}


def _init(args_dict):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from world import build_world
    a = argparse.Namespace(**args_dict)
    run = Path(a.run)
    W, H = int(1080 * a.scale) // 2 * 2, int(1920 * a.scale) // 2 * 2
    w = build_world(Path(a.robot_dump), Path(a.jumper) / "assets/jumper", Path(a.map), None,
                    Path(a.tmp) / f"w{os.getpid()}", W, H)
    rec = np.load(run / "frames.npz")
    meta = json.loads((run / "run.json").read_text())
    r = mujoco.Renderer(w.m, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
    # weld state is needed for "carried": rebuild from the log
    _W.update(w=w, rec=rec, meta=meta, r=r, opt=opt, a=a, W=W, H=H)


def _weld_state_at(meta, t):
    """Which holder each prop is welded to at sim time t, from the sequencer log."""
    holders = {}
    for line in meta["log"]:
        ts, _, rest = line.strip().partition("  ")
        tt = float(ts)
        if tt > t:
            break
        if rest.startswith("attach "):
            p, _, h = rest[7:].partition(" -> ")
            holders[p] = h
        elif rest.startswith("release "):
            holders.pop(rest[8:], None)
    return holders


def _render_range(job):
    import mujoco
    from PIL import Image, ImageFilter
    w, rec, meta, r, opt, a = _W["w"], _W["rec"], _W["meta"], _W["r"], _W["opt"], _W["a"]
    M = meta["markers"]
    shots, frames = timeline(M)
    out = Path(a.out)
    m, d = w.m, w.d
    ts = rec["t"]
    lo, hi = job
    # camera smoothing needs neighbours: compute raw lookats for the whole shot set once
    cams = _W.get("cams")
    if cams is None:
        cams = _all_cameras(w, rec, meta, shots, frames)
        _W["cams"] = cams
    crown_local = np.array([-0.012, 0.0, 0.058])
    for fi in range(lo, hi):
        si, t_sim, u = frames[fi]
        k = int(np.clip(np.searchsorted(ts, t_sim), 0, len(ts) - 1))
        d.qpos[:] = rec["qpos"][k]
        d.mocap_pos[:] = rec["mocap_pos"][k]
        d.mocap_quat[:] = rec["mocap_quat"][k]
        holders = _weld_state_at(meta, ts[k])
        for (p, h), e in w.weld.items():
            d.eq_active[e] = 1 if holders.get(p) == h else 0
        pose_face(m, w.eyes, meta["expr"][k], ts[k])
        mujoco.mj_forward(m, d)
        sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/upper_shell_link")
        crown = d.xpos[sb] + d.xmat[sb].reshape(3, 3) @ crown_local
        pose_sparkles(w, ts[k], M["dance"] + 0.5, crown)
        mujoco.mj_kinematics(m, d)
        cam = cams[fi]
        c = mujoco.MjvCamera()
        c.lookat[:] = cam["lookat"]
        c.distance, c.azimuth, c.elevation = cam["dist"], cam["az"], cam["el"]
        m.vis.global_.fovy = cam["fovy"]
        r.update_scene(d, c, opt)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(d, c, opt)
        depth = r.render().copy()
        r.disable_depth_rendering()
        img = depth_of_field(rgb, depth, cam["dist"], a.scale)
        Image.fromarray(img).save(out / f"f{fi:05d}.jpg", quality=94)
    return hi - lo


def depth_of_field(rgb, depth, focus, scale):
    """Layered blur by circle of confusion |1/z - 1/f| -- a lens with shallow focus."""
    from PIL import Image, ImageFilter
    coc = np.abs(1.0 / np.maximum(depth, 0.02) - 1.0 / focus)  # dioptres
    amt = np.clip(coc / 1.6, 0, 1)  # 0 sharp .. 1 max blur
    im = Image.fromarray(rgb)
    b1 = np.asarray(im.filter(ImageFilter.GaussianBlur(4 * scale)), dtype=np.float32)
    b2 = np.asarray(im.filter(ImageFilter.GaussianBlur(11 * scale)), dtype=np.float32)
    amt = np.asarray(Image.fromarray((amt * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3 * scale)),
                     dtype=np.float32)[..., None] / 255.0
    base = rgb.astype(np.float32)
    a1 = np.clip(amt * 2, 0, 1)
    a2 = np.clip(amt * 2 - 1, 0, 1)
    out = base * (1 - a1) + b1 * a1
    out = out * (1 - a2) + b2 * a2
    # warm grade + gentle vignette
    h, wdt = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:wdt]
    v = 1.0 - 0.32 * (((xx - wdt / 2) / (wdt / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
    out = out * v[..., None]
    out = out * np.array([1.04, 1.0, 0.94])
    return np.clip(out, 0, 255).astype(np.uint8)


def _all_cameras(w, rec, meta, shots, frames):
    """Raw camera per frame, then smoothed within each shot (no smoothing across cuts)."""
    import mujoco
    m, d = w.m, w.d
    ts = rec["t"]
    M = meta["markers"]
    raw = []
    for si, t_sim, u in frames:
        k = int(np.clip(np.searchsorted(ts, t_sim), 0, len(ts) - 1))
        d.qpos[:] = rec["qpos"][k]
        d.mocap_pos[:] = rec["mocap_pos"][k]
        d.mocap_quat[:] = rec["mocap_quat"][k]
        holders = _weld_state_at(meta, ts[k])
        for (p, h), e in w.weld.items():
            d.eq_active[e] = 1 if holders.get(p) == h else 0
        mujoco.mj_kinematics(m, d)
        pts = [focus_point(w, M, tag, None) for tag in shots[si][5]]
        raw.append((si, np.mean(pts, axis=0)))
    cams = []
    i = 0
    while i < len(raw):
        si = raw[i][0]
        j = i
        while j < len(raw) and raw[j][0] == si:
            j += 1
        pts = np.array([p for _, p in raw[i:j]])
        # forward-backward exponential smoothing
        alpha = 0.10
        f = pts.copy()
        for k in range(1, len(f)):
            f[k] = f[k - 1] + alpha * (f[k] - f[k - 1])
        for k in range(len(f) - 2, -1, -1):
            f[k] = f[k + 1] + alpha * (f[k] - f[k + 1])
        for k in range(i, j):
            u = frames[k][2]
            cams.append(camera_for(shots[si], w, M, u, f[k - i]))
        i = j
    return cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--every", type=int, default=1, help="render every Nth frame (previews)")
    ap.add_argument("--jumper", default=os.environ.get("JUMPER", "../jumper"))
    ap.add_argument("--robot-dump", default=os.environ.get("ROBOT_DUMP", "build/robot_dump.xml"))
    ap.add_argument("--map", default=str(HERE / "packages" / "king-crab-table.map"))
    ap.add_argument("--tmp", default=os.environ.get("RENDER_TMP", "build/render_tmp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((a.run / "run.json").read_text())
    shots, frames = timeline(meta["markers"])
    total = len(frames)
    print(f"{total} frames = {total / FPS:.2f} s at {FPS} fps")
    (a.out / "timeline.json").write_text(json.dumps({"shots": {k: v for k, v in video_times(meta["markers"]).items()},
                                                     "frames": total, "fps": FPS}, indent=1))
    if a.every > 1:
        idx = list(range(0, total, a.every))
        jobs = [(i, i + 1) for i in idx]
    else:
        step = 24
        jobs = [(i, min(i + step, total)) for i in range(0, total, step)]
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    with Pool(a.workers, initializer=_init, initargs=(args_dict,)) as pool:
        done = 0
        for n in pool.imap_unordered(_render_range, jobs):
            done += n
            if done % 96 < n:
                print(f"  {done}/{len(jobs) if a.every > 1 else total}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

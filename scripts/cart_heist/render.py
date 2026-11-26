"""Render the recorded run through the edit decision list (EDL) into frames.

    MUJOCO_GL=egl python render.py --run build/run --out build/frames [--scale 0.5] [--workers 4]
    MUJOCO_GL=egl python render.py --run build/run --out build/stills --stills 1.0,4.5,8.0   # previews

Each video frame maps to one simulation instant through the EDL: shots with a sim-time range
and a video duration, so a shot can run faster or slower than real time and cuts can skip
time. The recorded state at that instant is restored (no physics runs here: the frames show
exactly what the simulation did), the robot's display face and crown sparkles are posed, and
MuJoCo renders it, plus a depth-of-field pass from the depth buffer and a warm store grade.
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


# ----------------------------------------------------------------------- EDL

def edl(M):
    """Shots: (name, sim_start, sim_end, video_seconds, camera).

    `M` are the sequencer's markers (sim seconds). A camera is a dict:
      focus:  list of focus tags whose mean is the look-at point
      off:    look-at offset (m), or (start, end)
      az, el, dist, fovy: numbers, or (start, end) pairs eased over the shot
      smooth: look-at smoothing (0..1, lower = smoother)
    """
    return [
        # 0-2 s: hook -- the two shoppers fight over Cart A (the kid waits at the checkout behind)
        ("hook", 0.0, 2.0, 2.0,
         dict(focus=["point:-0.30,0.05,0.88"], az=(-160, -157), el=(-8, -9), dist=(3.35, 3.05), fovy=50)),
        # 2-6 s: low angle -- King Crab crab-walks in behind Cart A, toward a camera on the floor
        ("sneak", M["sneak"] + 0.4, M["arrive_cart"] + 0.1, 4.0,
         dict(focus=["crab"], off=(0.0, 0.0, 0.05), pos=((-0.38, -0.27, 0.06), (-0.42, -0.28, 0.06)), fovy=50,
              smooth=0.06)),
        # 6-11 s: close-up -- the claw comes up (snip snip), reaches, pinches, lifts
        ("claw_up", M["claw_up"] - 0.4, M["arm_out_grab"] + 0.2, 1.4,
         dict(focus=["crab", "prop:deal"], off=(0.0, 0.0, 0.03), az=(20, 28), el=-10, dist=(0.68, 0.60), fovy=44)),
        ("pinch", M["reach"] - 0.1, M["got_it"] + 0.1, 3.6,
         dict(focus=["jaw", "prop:deal"], off=(0.0, 0.0, 0.01), az=(22, 34), el=(-12, -8), dist=(0.50, 0.55),
              fovy=42, smooth=0.05)),
        # 11-16 s: away with the box, past the oblivious shoppers (camera leads the crab)
        ("escape", M["escape"], M["near_checkout"], 5.0,
         dict(focus=["crab"], off=(0.0, 0.0, 0.10), az=(-26, -18), el=(9, 11), dist=(0.95, 0.85), fovy=60,
              smooth=0.05)),
        # 16-21 s: the checkout -- place in the basket, step back, bow (profile: crab, basket, kid)
        ("place", M["arm_out"] - 0.3, M["landed"] + 0.8, 2.6,
         dict(focus=["point:-1.88,0.12,0.42"], pos=((-3.45, 0.02, 0.62), (-3.35, 0.02, 0.56)), fovy=62)),
        ("bow", M["bow_on"] + 0.5, M["bow_on"] + 4.5, 2.4,
         dict(focus=["point:-1.85,-0.17,0.30"], pos=((-2.85, -0.17, 0.20), (-2.78, -0.17, 0.20)), fovy=70)),
        # 21-24 s: cut back -- the shoppers turn to Cart A: empty (high angle on the rack)
        ("notice", M["bow_on"] + 0.5, M["bow_on"] + 3.5, 3.0,
         dict(focus=["point:-0.25,0.0,0.80"], az=(-149, -145), el=(-14, -17), dist=(2.95, 2.55), fovy=52)),
        # 24-28 s: samba, then salute, end card over it
        ("dance", M["dance_on"] + 0.5, M["dance_on"] + 3.0, 2.3,
         dict(focus=["crab"], off=(0.0, 0.0, 0.03), az=(-128, -118), el=(-20, -22), dist=(0.85, 0.80), fovy=48,
              smooth=0.04)),
        ("salute", M["salute_on"] + 0.6, M["salute_on"] + 2.6, 1.7,
         dict(focus=["crab"], off=(0.0, 0.0, 0.03), az=(-122, -118), el=(-20, -22), dist=(0.80, 0.74), fovy=48,
              smooth=0.04)),
    ]


def timeline(M):
    """Per video frame: (shot index, sim time, u within shot)."""
    shots = edl(M)
    frames = []
    for i, (_, a, b, dur, _) in enumerate(shots):
        n = int(round(dur * FPS))
        for k in range(n):
            u = (k + 0.5) / n
            frames.append((i, a + (b - a) * u, u))
    return shots, frames


def video_times(M):
    t, out = 0.0, {}
    for name, a, b, dur, _ in edl(M):
        n = int(round(dur * FPS))
        out[name] = {"start": t, "end": t + n / FPS, "sim": [a, b]}
        t += n / FPS
    return out


def sim_to_video(M, t_sim):
    """Video time at which sim instant t_sim is shown (first shot containing it), or None."""
    for name, v in video_times(M).items():
        a, b = v["sim"]
        if a <= t_sim <= b:
            return v["start"] + (t_sim - a) / (b - a) * (v["end"] - v["start"])
    return None


# ------------------------------------------------------------------ the face

BLINKS = np.arange(1.9, 80, 3.3)


def pose_face(m, eyes, expr, t_sim):
    name = expr["name"]
    lx, ly = expr.get("look", [0, 0])
    since = expr.get("since", 0.0)
    eye = np.array([0.0012, 0.0095, 0.0115])
    pup = np.array([0.0012, 0.0042, 0.0050])
    dz_eye, pup_dy, pup_dz = 0.0, 0.0, -0.0015
    lid_alpha, lid_z, lid_tilt = 0.0, 0.010, 0.0
    cheek_alpha, pupil_alpha = 0.0, 1.0
    if name == "sneaky":  # heavy lids, eyes darting to the side
        eye[2] = 0.0072
        dz_eye = -0.0012
        pup_dy, pup_dz = 0.0046 * lx, -0.0022 + 0.002 * ly
        lid_alpha, lid_z, lid_tilt = 1.0, 0.0050, 0.0
    elif name == "look":
        pup_dy, pup_dz = 0.0042 * lx, -0.0015 + 0.003 * ly
    elif name == "focus":
        eye[2] = 0.0088
        pup_dz = -0.003
    elif name == "wow":
        eye *= [1, 1.12, 1.18]
        pup = pup * 0.8
    elif name in ("happy", "content", "joy"):
        eye[2] = 0.0042 if name != "content" else 0.0055
        dz_eye = 0.0015
        pupil_alpha = 0.0
        cheek_alpha = 1.0 if name != "content" else 0.75
        if name == "joy":
            eye[1] *= 1.12
    if pupil_alpha > 0 and np.any(np.abs(BLINKS - t_sim) < 0.06):
        eye[2] = 0.0015
        pupil_alpha = 0.0
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


def pose_sparkles(w, t_sim, t0, t1, crown):
    """Crown sparkles between t0 and t1: deterministic little stars bursting up and out."""
    import mujoco
    rng = np.random.default_rng(42)
    m, d = w.m, w.d
    for i, mid in enumerate(w.sparkles):
        birth = t0 + rng.uniform(0, max(0.1, t1 - t0))
        life = rng.uniform(0.45, 0.8)
        ang = rng.uniform(0, 2 * math.pi)
        up = rng.uniform(0.14, 0.32)
        out = rng.uniform(0.05, 0.16)
        spin = rng.uniform(-8, 8)
        age = t_sim - birth
        if not (0 <= age <= life):
            d.mocap_pos[mid] = [0, 0, -5]
            continue
        u = age / life
        d.mocap_pos[mid] = crown + np.array([out * math.cos(ang) * u, out * math.sin(ang) * u,
                                             up * u - 0.25 * (u * life) ** 2 + 0.01])
        a = spin * age
        d.mocap_quat[mid] = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
        s = math.sin(math.pi * u) * (0.8 + 0.4 * math.sin(30 * age + i))
        for gname, base in ((f"sparkle{i}_a", np.array([0.0012, 0.0012, 0.0055])),
                            (f"sparkle{i}_b", np.array([0.0055, 0.0012, 0.0012]))):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
            m.geom_size[g] = base * max(s, 0.05) * 2.4


# --------------------------------------------------------------- cameras

def lerp_param(v, u):
    if isinstance(v, (tuple, list)) and len(v) == 2 and not isinstance(v[0], (tuple, list)):
        e = u * u * (3 - 2 * u)
        return v[0] + (v[1] - v[0]) * e
    return v


def focus_point(w, tag, jaw_ids):
    import mujoco
    m, d = w.m, w.d
    if tag == "crab":
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
        return d.xpos[b] + np.array([0, 0, -0.02])
    if tag == "jaw":
        return (d.xpos[jaw_ids[0]] + d.xpos[jaw_ids[1]]) / 2
    if tag.startswith("prop:"):
        return w.prop_pos(tag[5:]) + np.array([0, 0, 0.015])
    if tag.startswith("point:"):
        return np.array([float(v) for v in tag[6:].split(",")])
    raise ValueError(tag)


def restore(w, rec, meta, k):
    import mujoco
    d = w.d
    d.qpos[:] = rec["qpos"][k]
    d.mocap_pos[:] = rec["mocap_pos"][k]
    d.mocap_quat[:] = rec["mocap_quat"][k]
    held = bool(rec["held"][k])
    for (p, h), e in w.weld.items():
        d.eq_active[e] = 1 if held else 0
    mujoco.mj_kinematics(w.m, d)


def all_cameras(w, rec, meta, shots, frames):
    """Raw look-at per frame, smoothed within each shot (never across a cut)."""
    import mujoco
    m = w.m
    jaw_ids = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/LF_finger_tip_link"),
               mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/LF_palm_grip_insert_link"))
    ts = rec["t"]
    raw = []
    for si, t_sim, u in frames:
        k = int(np.clip(np.searchsorted(ts, t_sim), 0, len(ts) - 1))
        restore(w, rec, meta, k)
        cam = shots[si][4]
        pts = [focus_point(w, tag, jaw_ids) for tag in cam["focus"]]
        raw.append(np.mean(pts, axis=0))
    cams = []
    i = 0
    while i < len(frames):
        si = frames[i][0]
        j = i
        while j < len(frames) and frames[j][0] == si:
            j += 1
        cam = shots[si][4]
        alpha = cam.get("smooth", 0.10)
        f = np.array(raw[i:j])
        for k in range(1, len(f)):
            f[k] = f[k - 1] + alpha * (f[k] - f[k - 1])
        for k in range(len(f) - 2, -1, -1):
            f[k] = f[k + 1] + alpha * (f[k] - f[k + 1])
        for k in range(i, j):
            u = frames[k][2]
            off = np.asarray(lerp_param(cam.get("off", (0.0, 0.0, 0.0)), u)
                             if isinstance(cam.get("off", (0, 0, 0))[0], (tuple, list))
                             else cam.get("off", (0.0, 0.0, 0.0)), float)
            look = f[k - i] + off
            if "pos" in cam:  # a camera standing at a fixed (or slowly moving) place, aimed at the focus
                p0, p1 = (np.asarray(v, float) for v in cam["pos"])
                e = u * u * (3 - 2 * u)
                v = look - (p0 + (p1 - p0) * e)
                dist = float(np.linalg.norm(v))
                az = math.degrees(math.atan2(v[1], v[0]))
                el = math.degrees(math.asin(v[2] / dist))
                cams.append(dict(lookat=look, az=az, el=el, dist=dist, fovy=lerp_param(cam["fovy"], u)))
                continue
            cams.append(dict(lookat=look, az=lerp_param(cam["az"], u), el=lerp_param(cam["el"], u),
                             dist=lerp_param(cam["dist"], u), fovy=lerp_param(cam["fovy"], u)))
        i = j
    return cams


# ---------------------------------------------------------------- worker

_W = {}


def _init(args_dict):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from world import build_world
    a = argparse.Namespace(**args_dict)
    run = Path(a.run)
    W, H = int(1080 * a.scale) // 2 * 2, int(1920 * a.scale) // 2 * 2
    w = build_world(Path(a.robot_dump), Path(a.jumper) / "assets/jumper", Path(a.map),
                    Path(a.tmp) / f"w{os.getpid()}", W, H)
    rec = dict(np.load(run / "frames.npz"))
    meta = json.loads((run / "run.json").read_text())
    r = mujoco.Renderer(w.m, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
    shots, frames = timeline(meta["markers"])
    cams = all_cameras(w, rec, meta, shots, frames)
    _W.update(w=w, rec=rec, meta=meta, r=r, opt=opt, a=a, frames=frames, cams=cams)


def _render_one(fi, path):
    import mujoco
    from PIL import Image
    w, rec, meta, r, opt, a = _W["w"], _W["rec"], _W["meta"], _W["r"], _W["opt"], _W["a"]
    M = meta["markers"]
    m, d = w.m, w.d
    ts = rec["t"]
    si, t_sim, u = _W["frames"][fi]
    k = int(np.clip(np.searchsorted(ts, t_sim), 0, len(ts) - 1))
    restore(w, rec, meta, k)
    pose_face(m, w.eyes, meta["expr"][k], ts[k])
    sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/upper_shell_link")
    crown = d.xpos[sb] + d.xmat[sb].reshape(3, 3) @ np.array([-0.012, 0.0, 0.058])
    pose_sparkles(w, ts[k], M["dance_on"] + 0.4, M["salute_on"] + 3.0, crown)
    mujoco.mj_kinematics(m, d)
    cam = _W["cams"][fi]
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
    img = grade(rgb, depth, cam["dist"], a.scale)
    Image.fromarray(img).save(path, quality=94)


def _render_range(job):
    lo, hi = job
    out = Path(_W["a"].out)
    for fi in range(lo, hi):
        _render_one(fi, out / f"f{fi:05d}.jpg")
    return hi - lo


def _render_still(job):
    fi, name = job
    _render_one(fi, Path(_W["a"].out) / name)
    return 1


def grade(rgb, depth, focus, scale):
    """Depth of field by circle of confusion |1/z - 1/f|, a warm store grade, a soft vignette."""
    from PIL import Image, ImageFilter
    coc = np.abs(1.0 / np.maximum(depth, 0.02) - 1.0 / focus)
    amt = np.clip(coc / 1.4, 0, 1)
    im = Image.fromarray(rgb)
    b1 = np.asarray(im.filter(ImageFilter.GaussianBlur(4 * scale)), dtype=np.float32)
    b2 = np.asarray(im.filter(ImageFilter.GaussianBlur(10 * scale)), dtype=np.float32)
    amt = np.asarray(Image.fromarray((amt * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3 * scale)),
                     dtype=np.float32)[..., None] / 255.0
    base = rgb.astype(np.float32)
    a1 = np.clip(amt * 2, 0, 1)
    a2 = np.clip(amt * 2 - 1, 0, 1)
    out = base * (1 - a1) + b1 * a1
    out = out * (1 - a2) + b2 * a2
    h, wdt = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:wdt]
    v = 1.0 - 0.30 * (((xx - wdt / 2) / (wdt / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
    out = out * v[..., None]
    out = out * np.array([1.05, 1.0, 0.93])
    out = 255 * (out / 255) ** 0.95  # lift the mids a touch
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stills", default=None, help="comma-separated video times (s) to render as stills")
    ap.add_argument("--jumper", default=os.environ.get("JUMPER", "../jumper"))
    ap.add_argument("--robot-dump", default=os.environ.get("ROBOT_DUMP", "build/robot_dump.xml"))
    ap.add_argument("--map", default=str(HERE / "../../maps/black_friday_store.map"))
    ap.add_argument("--tmp", default=os.environ.get("RENDER_TMP", "build/render_tmp"))
    a = ap.parse_args()
    for need in (Path(a.map), Path(a.robot_dump), a.run / "frames.npz"):
        if not need.exists():  # a Pool initializer that raises would respawn forever
            raise SystemExit(f"missing {need}")
    a.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((a.run / "run.json").read_text())
    shots, frames = timeline(meta["markers"])
    total = len(frames)
    print(f"{total} frames = {total / FPS:.2f} s at {FPS} fps")
    (a.out / "timeline.json").write_text(json.dumps({"shots": video_times(meta["markers"]), "frames": total,
                                                     "fps": FPS}, indent=1))
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    if a.stills:
        jobs = []
        for s in a.stills.split(","):
            fi = min(total - 1, int(float(s) * FPS))
            jobs.append((fi, f"still_{float(s):05.2f}_{shots[frames[fi][0]][0]}.jpg"))
        with Pool(min(a.workers, len(jobs)), initializer=_init, initargs=(args_dict,)) as pool:
            list(pool.imap_unordered(_render_still, jobs))
        print("stills:", [j[1] for j in jobs])
        return
    step = 24
    jobs = [(i, min(i + step, total)) for i in range(0, total, step)]
    with Pool(a.workers, initializer=_init, initargs=(args_dict,)) as pool:
        done = 0
        for n in pool.imap_unordered(_render_range, jobs):
            done += n
            if done % 120 < n:
                print(f"  {done}/{total}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Film a capture from `capture.py`: keyframed camera, slow motion, visual props.

A *shot* is a dict:

    {
      "capture": "jump_beach",          # captures/<name>.npz + .mjb
      "t0": 0.2, "t1": 1.2,             # simulated seconds to film
      "speed": 0.35,                    # playback speed (<1 = slow motion)
      "cam": [                          # keyframes over shot progress u in [0, 1]
         {"u": 0, "lookat": [0,0,.12], "dist": .9, "az": 160, "el": -8},
         {"u": 1, "lookat": [0,0,.2],  "dist": .6, "az": 200, "el": 5},
      ],
      "fovy": 40,
      "props": "volleyball",            # name of a prop function below (optional)
      "palette": "lifeguard",           # robot recolour (optional)
    }

Props are drawn as extra visual geoms in the MuJoCo scene (no physics): they
are animated as a function of simulated time so they line up with the motion.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

CAPTURE_DIR = Path(os.environ.get("CAPTURE_DIR", Path(__file__).resolve().parents[1] / "captures"))


# ── helpers ───────────────────────────────────────────────────────────────

def smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def interp_cam(keys, u):
    keys = sorted(keys, key=lambda k: k["u"])
    if u <= keys[0]["u"]:
        return keys[0]
    for a, b in zip(keys, keys[1:]):
        if u <= b["u"]:
            w = smooth((u - a["u"]) / max(b["u"] - a["u"], 1e-9))
            out = {}
            for k in ("dist", "az", "el"):
                out[k] = a[k] + (b[k] - a[k]) * w
            out["lookat"] = list(np.add(a["lookat"], np.multiply(np.subtract(b["lookat"], a["lookat"]), w)))
            for k in ("follow",):
                if k in a:
                    out[k] = a[k]
            return out
    return keys[-1]


def add_geom(scene, gtype, size, pos, rgba, mat=None, emission=0.0, specular=0.5):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, int(gtype), np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64),
        (np.eye(3) if mat is None else np.asarray(mat)).reshape(-1).astype(np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    g.emission = emission
    g.specular = specular
    g.shininess = 0.6
    scene.ngeom += 1


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_from_axis(v):
    """Rotation whose z axis points along v."""
    z = np.asarray(v, float)
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.cross([0, 0, 1.0], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def capsule_between(scene, p0, p1, r, rgba):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    mid, d = (p0 + p1) / 2, p1 - p0
    add_geom(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, [r, np.linalg.norm(d) / 2, 0], mid,
             rgba, rot_from_axis(d))


# ── props ─────────────────────────────────────────────────────────────────

WHITE = [0.97, 0.97, 0.95, 1]
NET_Y = 0.0


def ball_geoms(scene, pos, spin, r=0.045):
    """A volleyball: white/yellow/blue panels approximated by three tilted bands."""
    add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], pos, WHITE)
    for k, col in enumerate(([1.0, 0.82, 0.05, 1], [0.05, 0.25, 0.8, 1], [1.0, 0.82, 0.05, 1])):
        m = rot_z(spin + k * 2.1) @ rot_from_axis([math.sin(spin + k), 0.3, 1])
        add_geom(scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [r * 1.004, r * 1.004, r * 0.33],
                 pos, col, m)


def court(scene, net_x, net_top=0.30):
    """Small beach-volleyball court sized to the crab: lines + a net at x=net_x."""
    line = [0.95, 0.95, 0.92, 1]
    half_w, length = 0.6, 1.4
    for y in (-half_w, half_w):
        add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, [length, 0.012, 0.001],
                 [net_x, y, 0.001], line)
    for x in (net_x - length, net_x + length):
        add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, [0.012, half_w, 0.001], [x, 0, 0.001], line)
    net_h = 0.11
    for y in (-half_w - 0.05, half_w + 0.05):
        add_geom(scene, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.012, net_top / 2 + 0.01, 0],
                 [net_x, y, net_top / 2], [0.85, 0.85, 0.88, 1])
    # top tape and mesh
    add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, [0.006, half_w + 0.05, 0.012],
             [net_x, 0, net_top - 0.012], WHITE)
    add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, [0.004, half_w + 0.05, 0.006],
             [net_x, 0, net_top - net_h], WHITE)
    n = 26
    for i in range(n + 1):
        y = -half_w - 0.05 + i * (2 * half_w + 0.1) / n
        add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, [0.002, 0.0015, net_h / 2],
                 [net_x, y, net_top - net_h / 2 - 0.012], [0.08, 0.08, 0.1, 0.9])
    for j in range(1, 5):
        add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, [0.002, half_w + 0.05, 0.0015],
                 [net_x, 0, net_top - 0.012 - j * net_h / 5], [0.08, 0.08, 0.1, 0.9])


def volleyball(scene, t, ctx):
    """Set -> spike. The ball is tossed in, meets the crab's claws at the top of
    the jump, and is driven down over the net."""
    hit_t = ctx["hit_t"]          # simulated time of contact (apex of jump)
    hit_p = np.array(ctx["hit_p"])  # contact point
    net_x = ctx.get("net_x", 0.6)
    court(scene, net_x)
    g = 9.81 * 0.55  # a touch floaty so it reads on screen
    if t <= hit_t:
        # a teammate's set, arriving from the crab's left on a high arc
        T = ctx.get("set_time", 1.0)
        s = max(t - (hit_t - T), 0.0)
        p0 = hit_p + np.array(ctx.get("set_from", [0.15, 0.75, 0.1]))
        v0 = (hit_p - p0 + 0.5 * np.array([0, 0, g]) * T * T) / T
        p = p0 + v0 * s - 0.5 * np.array([0, 0, g]) * s * s
        if t < hit_t - T:
            return
        ball_geoms(scene, p, s * 6)
    else:
        s = t - hit_t
        v = np.array(ctx.get("shot_v", [2.2, -0.1, 1.6]))
        p = hit_p + v * s - 0.5 * np.array([0, 0, 9.81]) * s * s
        floor = 0.045
        if p[2] < floor:  # bounce in the sand once, then roll
            t_land = (v[2] + math.sqrt(v[2] ** 2 + 2 * 9.81 * (hit_p[2] - floor))) / 9.81
            pl = hit_p + v * t_land - 0.5 * np.array([0, 0, 9.81]) * t_land ** 2
            u = s - t_land
            vb = np.array([v[0] * 0.35, v[1] * 0.35, 1.1])
            decay = (1 - math.exp(-2.5 * u)) / 2.5  # sand drag on the roll-out
            p = pl + np.array([vb[0] * decay, vb[1] * decay, vb[2] * u - 4.905 * u * u])
            if p[2] < floor:
                p[2] = floor
            # sand puff at landing
            if 0 <= u < 0.35:
                for k in range(7):
                    a = k * 0.9 + 0.3
                    rr = 0.02 + u * 0.35
                    add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012 + u * 0.03, 0, 0],
                             [pl[0] + rr * math.cos(a), pl[1] + rr * math.sin(a),
                              0.01 + u * 0.12 * (1 + 0.3 * math.sin(k))],
                             [0.82, 0.68, 0.48, max(0.0, 0.8 - u * 2.2)])
        ball_geoms(scene, p, 8 + s * 25)
        # impact streaks
        if s < 0.12:
            for k in range(8):
                a = k * math.pi / 4 + 0.4
                dvec = np.array([0.0, math.cos(a), math.sin(a)])
                r0, r1 = 0.06 + s * 0.6, 0.09 + s * 0.9
                capsule_between(scene, hit_p + dvec * r0, hit_p + dvec * r1, 0.003,
                                [1, 0.95, 0.6, 1 - s * 8])


def soccer_ball(scene, t, ctx):
    pass


def court_only(scene, t, ctx, d=None, m=None):
    court(scene, ctx.get("net_x", 0.6), ctx.get("net_top", 0.30))


def ball_geoms_oriented(scene, pos, R, r=0.045):
    """Volleyball panels fixed to the ball's real orientation (so its spin is the simulated spin)."""
    add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], pos, WHITE)
    axes = ([0.3, 0.2, 1.0], [1.0, 0.3, -0.2], [-0.2, 1.0, 0.3])
    for ax, col in zip(axes, ([1.0, 0.82, 0.05, 1], [0.05, 0.25, 0.8, 1], [1.0, 0.82, 0.05, 1])):
        add_geom(scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [r * 1.004, r * 1.004, r * 0.33],
                 pos, col, R @ rot_from_axis(ax))


def realball(scene, t, ctx, d, m):
    """The simulated volleyball: drawn where MuJoCo put it. Court and the hit/landing
    accents (streaks, sand puff) are decoration timed from the measured physics."""
    court(scene, ctx.get("net_x", 0.47), ctx.get("net_top", 0.27))
    bid = m.body("volleyball/volleyball").id
    pos, R = d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()
    ball_geoms_oriented(scene, pos, R)
    s = t - ctx["hit_t"]
    if 0 <= s < 0.12:
        hp = np.array(ctx["hit_p"])
        for k in range(8):
            a = k * math.pi / 4 + 0.4
            dv = np.array([0.0, math.cos(a), math.sin(a)])
            capsule_between(scene, hp + dv * (0.06 + s * 0.6), hp + dv * (0.09 + s * 0.9), 0.003,
                            [1, 0.95, 0.6, 1 - s * 8])
    u = t - ctx["land_t"]
    if 0 <= u < 0.35:
        pl = ctx["land_p"]
        for k in range(7):
            a = k * 0.9 + 0.3
            rr = 0.02 + u * 0.35
            add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012 + u * 0.03, 0, 0],
                     [pl[0] + rr * math.cos(a), pl[1] + rr * math.sin(a),
                      0.01 + u * 0.12 * (1 + 0.3 * math.sin(k))],
                     [0.82, 0.68, 0.48, max(0.0, 0.8 - u * 2.2)])


def _scripted(fn):
    return lambda scene, t, ctx, d=None, m=None: fn(scene, t, ctx)


PROPS = {"volleyball": _scripted(volleyball), "court": court_only, "realball": realball}


# ── palettes (a recolour of the robot's visual geoms only) ───────────────

PALETTES = {
    # name: (shell, limbs, accent)
    "lifeguard": ([0.95, 0.18, 0.12, 1], [0.98, 0.96, 0.92, 1], [1.0, 0.80, 0.10, 1]),
    "sunset": ([1.0, 0.45, 0.15, 1], [0.35, 0.15, 0.45, 1], [1.0, 0.85, 0.3, 1]),
    "ice": ([0.55, 0.85, 1.0, 1], [0.95, 0.97, 1.0, 1], [0.15, 0.35, 0.9, 1]),
}


def apply_palette(model, name):
    if not name:
        return
    shell, limbs, accent = PALETTES[name]
    for i in range(model.ngeom):
        b = model.body(model.geom_bodyid[i]).name
        if not b.startswith("robot/") or model.geom_contype[i] or model.geom_conaffinity[i]:
            continue
        rgba = model.geom_rgba[i]
        lum = rgba[:3].mean()
        if "shell" in b or "base" in b or "display" in b:
            model.geom_rgba[i] = shell if lum > 0.15 else rgba
        elif "finger" in b or "palm" in b:
            model.geom_rgba[i] = accent if lum > 0.15 else rgba
        else:
            model.geom_rgba[i] = limbs if lum > 0.15 else rgba
        model.geom_matid[i] = -1


# ── skins: jersey recolour + crown, drawn on the shell so they ride every motion ──

GOLD = [1.0, 0.78, 0.18, 1]
CROWN_RED = [0.85, 0.05, 0.12, 1]
CROWN_BLUE = [0.1, 0.35, 0.95, 1]

# name: jersey, stripes, number colour, shorts (lower body), limbs, claws, number
SKINS = {
    "gold_team": dict(jersey=[1.0, 0.82, 0.1, 1], stripe=[0.05, 0.25, 0.8, 1],
                      number=[0.05, 0.25, 0.8, 1], shorts=[0.05, 0.2, 0.65, 1],
                      limbs=[0.88, 0.12, 0.08, 1], claws=[0.95, 0.2, 0.12, 1], num="1"),
    "classic_red": dict(jersey=[0.88, 0.07, 0.06, 1], stripe=[0.98, 0.98, 0.96, 1],
                        number=[0.98, 0.98, 0.96, 1], shorts=[0.06, 0.06, 0.08, 1],
                        limbs=[0.85, 0.1, 0.07, 1], claws=[0.95, 0.95, 0.95, 1], num="1"),
    "ocean_blue": dict(jersey=[0.08, 0.3, 0.85, 1], stripe=[0.98, 0.98, 0.96, 1],
                       number=[1.0, 0.82, 0.1, 1], shorts=[0.98, 0.98, 0.96, 1],
                       limbs=[0.88, 0.12, 0.08, 1], claws=[0.95, 0.2, 0.12, 1], num="1"),
}


def recolour_for_skin(model, skin):
    for i in range(model.ngeom):
        b = model.body(model.geom_bodyid[i]).name
        if not b.startswith("robot/"):
            continue
        if "upper_shell" in b:
            model.geom_rgba[i] = skin["jersey"]
        elif b.endswith("base_link"):
            model.geom_rgba[i] = skin["shorts"]
        elif any(k in b for k in ("display", "tof", "camera")):
            continue
        elif any(k in b for k in ("finger", "palm")):
            if model.geom_rgba[i][:3].mean() > 0.15:
                model.geom_rgba[i] = skin["claws"]
        elif model.geom_rgba[i][:3].mean() > 0.15:
            model.geom_rgba[i] = skin["limbs"]
        model.geom_matid[i] = -1


class ShellFrame:
    """Height of the shell's top surface in its own body frame, from its mesh."""

    def __init__(self, model, data):
        self.bid = model.body("robot/upper_shell_link").id
        g = [i for i in range(model.ngeom) if model.geom_bodyid[i] == self.bid
             and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH][0]
        mid = model.geom_dataid[g]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        v = model.mesh_vert[a:a + n].astype(float)
        # mesh -> body frame
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        self.v = v @ R.reshape(3, 3).T + model.geom_pos[g]

    def top(self, x, y, r=0.012):
        near = self.v[(abs(self.v[:, 0] - x) < r) & (abs(self.v[:, 1] - y) < r)]
        return float(near[:, 2].max()) if len(near) else float(self.v[:, 2].max())


SEGMENTS = {  # 7-segment digits: a b c d e f g
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd",
    "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}


def draw_skin(scene, data, shell, skin):
    R = data.xmat[shell.bid].reshape(3, 3)
    X = data.xpos[shell.bid]

    def W(p):
        return X + R @ np.asarray(p, float)

    def box(p, half, rgba, local_rot=None):
        rot = R if local_rot is None else R @ local_rot
        add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, half, W(p), rgba, rot, specular=0.3)

    def surf(x, y, lift):
        return np.array([x, y, shell.top(x, y) + lift])

    def stroke(p0, p1, rad, rgba):
        capsule_between(scene, W(p0), W(p1), rad, rgba)

    # side stripes: smooth piping front to back, hugging the top surface
    for y in (-0.064, 0.064):
        xs = np.linspace(-0.096, 0.042, 13)
        pts = [surf(x, y, 0.0025) for x in xs]
        for p0, p1 in zip(pts, pts[1:]):
            stroke(p0, p1, 0.0042, skin["stripe"])
    # jersey number on the back of the shell, upright when seen from behind
    cx, size = -0.064, 0.05
    segs = SEGMENTS.get(skin.get("num", "1"), "bc")
    L = {"a": ((0, 1), (1, 1)), "b": ((1, 1), (1, 0)), "c": ((1, 0), (1, -1)),
         "d": ((0, -1), (1, -1)), "e": ((0, 0), (0, -1)), "f": ((0, 1), (0, 0)),
         "g": ((0, 0), (1, 0))}
    if skin.get("num", "1") == "1":  # a proper serif 1 reads better than two segments
        top_, bot = cx + size * 0.5, cx - size * 0.5
        stroke(surf(top_, 0.0, 0.003), surf(bot, 0.0, 0.003), 0.0055, skin["number"])
        stroke(surf(top_, 0.0, 0.003), surf(top_ - 0.012, 0.012, 0.003), 0.0055, skin["number"])
        stroke(surf(bot, -0.011, 0.003), surf(bot, 0.011, 0.003), 0.0055, skin["number"])
    else:
        for sgm in segs:
            (u0, v0), (u1, v1) = L[sgm]
            y0, y1 = (0.5 - u0) * size * 0.6, (0.5 - u1) * size * 0.6
            x0, x1 = cx + v0 * size * 0.5, cx + v1 * size * 0.5
            stroke(surf(x0, y0, 0.003), surf(x1, y1, 0.003), 0.0055, skin["number"])
    # the crown, on the crest of the shell
    ccx = -0.012
    base_z = shell.top(ccx, 0.0) - 0.002
    r, h = 0.026, 0.009
    add_geom(scene, mujoco.mjtGeom.mjGEOM_CYLINDER, [r, h, 0], W([ccx, 0, base_z + h]), GOLD, R,
             specular=0.9)
    add_geom(scene, mujoco.mjtGeom.mjGEOM_CYLINDER, [r * 1.06, 0.002, 0],
             W([ccx, 0, base_z + 0.003]), GOLD, R, specular=0.9)
    for k in range(5):
        a = k * 2 * math.pi / 5
        out = np.array([math.cos(a), math.sin(a), 0])
        base = np.array([ccx, 0, base_z + 2 * h]) + out * r * 0.9
        tip = base + out * 0.004 + np.array([0, 0, 0.017])
        p0, p1 = W(base), W(tip)
        mid, dv = (p0 + p1) / 2, p1 - p0
        add_geom(scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.0065, 0.0045, np.linalg.norm(dv) / 2 + 0.004],
                 mid, GOLD, rot_from_axis(dv), specular=0.9)
        add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0035, 0, 0], p1, GOLD, specular=0.9)
        jewel = np.array([ccx, 0, base_z + h]) + out * (r + 0.0005)
        add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0038, 0, 0], W(jewel),
                 CROWN_RED if k % 2 == 0 else CROWN_BLUE, specular=1.0)


# ── the renderer ──────────────────────────────────────────────────────────

class Film:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self._cache = {}

    def load(self, name, palette=None, skin=None):
        key = (name, palette, skin)
        if key not in self._cache:
            m = mujoco.MjModel.from_binary_path(str(CAPTURE_DIR / f"{name}.mjb"))
            m.vis.global_.offwidth = max(m.vis.global_.offwidth, self.w)
            m.vis.global_.offheight = max(m.vis.global_.offheight, self.h)
            m.vis.quality.shadowsize = 4096
            m.vis.quality.offsamples = 8
            apply_palette(m, palette)
            if skin:
                recolour_for_skin(m, SKINS[skin])
            for i in range(m.ngeom):
                if m.geom(i).name == "volleyball/volleyball":
                    m.geom_rgba[i, 3] = 0.0
            cap = np.load(CAPTURE_DIR / f"{name}.npz")
            r = mujoco.Renderer(m, self.h, self.w, max_geom=4000)
            d = mujoco.MjData(m)
            shell = ShellFrame(m, d) if skin else None
            self._cache[key] = (m, d, cap["qpos"], float(cap["dt"]), r, shell)
        return self._cache[key]

    def frames(self, shot, fps=30):
        m, d, qpos, dt, r, shell = self.load(shot["capture"], shot.get("palette"),
                                             shot.get("skin"))
        skin = SKINS.get(shot.get("skin"))
        segs = shot.get("timeline") or [[shot["t0"], shot["t1"], shot.get("speed", 1.0)]]
        times = []
        for a_, b_, sp in segs:
            k_ = max(1, int(round((b_ - a_) / sp * fps)))
            times += [a_ + (b_ - a_) * j / k_ for j in range(k_)]
        n = len(times)
        m.vis.global_.fovy = shot.get("fovy", 40)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        prop = PROPS.get(shot.get("props"))
        opt = mujoco.MjvOption()
        for i in range(n):
            u = i / max(n - 1, 1)
            t = times[i]
            k = t / dt
            a = int(min(math.floor(k), len(qpos) - 1))
            b = min(a + 1, len(qpos) - 1)
            w = k - a
            q = qpos[a] * (1 - w) + qpos[b] * w
            for j in np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
                a_ = m.jnt_qposadr[j] + 3
                q[a_:a_ + 4] /= np.linalg.norm(q[a_:a_ + 4])
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_camlight(m, d)
            c = interp_cam(shot["cam"], u)
            look = np.array(c["lookat"], float)
            if shot.get("follow", True):
                look[:2] += d.qpos[:2]
            cam.lookat[:] = look
            cam.distance = c["dist"]
            cam.azimuth = c["az"]
            cam.elevation = c["el"]
            r.update_scene(d, cam, opt)
            if skin:
                draw_skin(r.scene, d, shell, skin)
            if prop:
                prop(r.scene, t, shot.get("ctx", {}), d, m)
            yield r.render().copy()


def write(frames, path, fps=30, w=1080, h=1920):
    import imageio_ffmpeg

    wr = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, quality=9,
                                     macro_block_size=8, output_params=["-pix_fmt", "yuv420p"])
    wr.send(None)
    count = 0
    for f in frames:
        wr.send(np.ascontiguousarray(f))
        count += 1
    wr.close()
    return count

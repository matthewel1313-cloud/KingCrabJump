"""The shot list: a camera for every moment of the 30 s timeline (video seconds).

MuJoCo free-camera conventions: azimuth 0 looks along +X, 90 along +Y; elevation negative looks
down. The robot works facing +X, so a camera in front of it has azimuth ~180. Beats follow
motion/timeline.py.
"""
from __future__ import annotations

import math

import numpy as np

SHOTS = [  # name, start, end
    ("hook", 0.0, 2.0), ("walk", 2.0, 6.0), ("grasp", 6.0, 10.0), ("read_front", 10.0, 11.4),
    ("read_scan", 11.4, 14.0), ("place", 14.0, 18.0), ("shake", 18.0, 22.0), ("aplus", 22.0, 26.0),
    ("end", 26.0, 30.01),
]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a, b, k):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * k


def yaw_of(rep, body):
    m = rep.d.xmat[body].reshape(3, 3)
    return math.degrees(math.atan2(m[1, 0], m[0, 0]))


def from_position(eye, lookat):
    v = np.asarray(eye, float) - np.asarray(lookat, float)
    dist = float(np.linalg.norm(v))
    f = -v / dist
    return {"lookat": np.asarray(lookat, float), "distance": dist,
            "azimuth": math.degrees(math.atan2(f[1], f[0])), "elevation": math.degrees(math.asin(f[2]))}


def camera(t, rep, base):
    name = next(n for n, a, b in SHOTS if a <= t < b)
    s0, s1 = next((a, b) for n, a, b in SHOTS if n == name)
    k = (t - s0) / (s1 - s0)
    pos = rep.d.xpos[base].copy()
    if name == "hook":
        # Tight on the crown and glasses; the robot twists back to face the lens.
        look = pos + np.array([0.03, 0.0, 0.055])
        spec = {"lookat": look, "distance": 0.30 + 0.06 * ease(k), "azimuth": 168.0,
                "elevation": -8.0 - 4 * ease(k), "fovy": 38.0}
    elif name == "walk":
        # Locked off on the desk; the robot crab-walks in from frame right, the camera eases along.
        look = lerp([0.02, -0.26, 0.075], [0.07, -0.40, 0.085], ease(k))
        spec = {"lookat": look, "distance": 0.98 - 0.12 * ease(k), "azimuth": 196.0,
                "elevation": -12.0, "fovy": 40.0}
    elif name == "grasp":
        k1 = ease((t - 6.0) / 2.2)
        k2 = ease((t - 8.6) / 1.4)
        look = lerp([0.12, -0.45, 0.10], [0.08, -0.44, 0.16], k2)
        spec = {"lookat": look, "distance": 0.80 - 0.08 * k1 - 0.06 * k2, "azimuth": 208.0 - 30 * k2,
                "elevation": -17.0 - 6 * k2, "fovy": 40.0}
    elif name == "read_front":
        # Square-ish to the page held up to read (its printed back faces +X, -Y and up).
        look = pos + np.array([0.06, -0.04, 0.10])
        spec = {"lookat": look, "distance": 0.66 - 0.08 * ease(k), "azimuth": 150.0 - 10 * ease(k),
                "elevation": -24.0 - 6 * ease(k), "fovy": 40.0}
    elif name == "read_scan":
        # Closer on the page while the scan line sweeps it, framed between the page and the crown.
        p = rep.body("paper_5")
        crown = pos + rep.d.xmat[base].reshape(3, 3) @ np.array([0.04, 0.0, 0.09])
        look = 0.5 * rep.d.xpos[p] + 0.5 * crown
        spec = {"lookat": look, "distance": 0.62 - 0.06 * ease(k), "azimuth": 162.0, "elevation": -7.0,
                "fovy": 40.0}
    elif name == "place":
        # Wide enough for the professor (crown and all) and the sheet being put down.
        # From the -Y side, so the sheet being put down is in front of the stack, not behind it.
        look = lerp([0.06, -0.58, 0.10], [0.08, -0.62, 0.07], ease((t - 14.0) / 2.0))
        spec = {"lookat": look, "distance": 0.80 - 0.06 * ease((t - 15.5) / 1.5), "azimuth": 162.0,
                "elevation": -24.0 - 6 * ease(k), "fovy": 40.0}
    elif name == "shake":
        look = pos + np.array([0.04, 0.0, 0.05])
        spec = {"lookat": look, "distance": 0.40 - 0.04 * ease(k), "azimuth": 186.0, "elevation": -9.0,
                "fovy": 38.0}
    elif name == "aplus":
        look = lerp([0.10, -0.52, 0.08], [0.08, -0.50, 0.09], ease(k))
        spec = {"lookat": look, "distance": 0.92 - 0.1 * ease(k), "azimuth": 198.0, "elevation": -15.0,
                "fovy": 40.0}
    else:  # end: ease out to a calm wide frame and hold
        kk = ease((t - 26.0) / 2.0)
        look = lerp([0.08, -0.50, 0.09], [0.02, -0.42, 0.11], kk)
        spec = {"lookat": look, "distance": 0.82 + 0.5 * kk, "azimuth": 198.0, "elevation": -15.0 - 2 * kk,
                "fovy": 40.0}
    spec["name"] = name
    return spec

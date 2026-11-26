#!/usr/bin/env python3
"""Search toss parameters for a real, physically simulated bump.

Each trial runs the toolkit's jump policy with a physical volleyball (see
capture.py BALL=...) tossed so that, if nothing touched it, it would pass the
point (x, 0, z) at time tc. Whatever happens after that is MuJoCo contact
between the ball and the robot. Trials are scored on how far forward and how
high the ball actually travels after the hit.

    python tools/ball_sweep.py            # prints a ranked table
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = 9.81


def toss(x, z, tc, side=0.15, rise=0.0):
    """Launch state so the ball would pass (x, 0, z) at t=tc, tossed from the crab's left."""
    p = np.array([x, 0.0, z])
    p0 = p + np.array([0.0, side, rise])
    v = (p - p0) / tc + np.array([0, 0, 0.5 * G * tc])
    return list(p0) + list(v)


def run(name, ball, steps=320):
    env = dict(os.environ, BALL=",".join(f"{b:.4f}" for b in ball))
    ck = "tasks/jumper/jump/out/example/model_3100.pt"
    subprocess.run([sys.executable, str(HERE / "capture.py"), name, "--", "--task", "jumper.jump",
                    "--checkpoint", ck, "--scene", "beach", "--steps", str(steps)],
                   env=env, check=True, capture_output=True)
    q = np.load(Path(os.environ["CAPTURE_DIR"]) / f"{name}.npz")["qpos"]
    b = q[:, -7:-4]
    return b


def score(b, dt=0.005):
    # the hit: first step where the ball's vertical velocity jumps upward
    vz = np.diff(b[:, 2]) / dt
    jumps = np.where(np.diff(vz) / dt > 40)[0]
    i0 = int(jumps[0]) + 1 if len(jumps) else int(0.38 / dt)
    after = b[i0:]
    # first ground contact after the hit (centre at radius height)
    land = np.argmax(after[:, 2] < 0.05) if (after[:, 2] < 0.05).any() else len(after) - 1
    peak = after[: land + 1, 2].max()
    x_land = after[land, 0]
    y_land = after[land, 1]
    return dict(x_land=float(x_land), y_land=float(y_land), peak=float(peak),
                t_hit=float(i0 * dt), z_hit=float(b[i0, 2]), t_land=float((i0 + land) * dt))


def main():
    xs = [float(v) for v in os.environ.get("SWEEP_X", "0.17,0.19,0.205,0.22").split(",")]
    zs = [float(v) for v in os.environ.get("SWEEP_Z", "0.16,0.22").split(",")]
    tcs = [float(v) for v in os.environ.get("SWEEP_T", "0.39,0.42,0.45").split(",")]
    grid = list(itertools.product(xs, zs, tcs))
    results = []

    def job(k):
        x, z, tc = grid[k]
        ball = toss(x, z, tc)
        b = run(f"sweep_{k:02d}", ball)
        return k, ball, score(b)

    with ThreadPoolExecutor(4) as ex:
        for k, ball, sc in ex.map(job, range(len(grid))):
            results.append((k, grid[k], ball, sc))
            print(f"{k:02d} x={grid[k][0]:.2f} z={grid[k][1]:.2f} tc={grid[k][2]:.2f} -> "
                  f"land x={sc['x_land']:.2f} y={sc['y_land']:.2f} peak={sc['peak']:.2f} "
                  f"t_hit={sc['t_hit']:.3f} z_hit={sc['z_hit']:.2f} t_land={sc['t_land']:.2f}", flush=True)
    results.sort(key=lambda r: -(r[3]["x_land"] + (r[3]["peak"] - r[3]["z_hit"])
                                 - 2 * abs(r[3]["y_land"])))
    print("\nbest:")
    for k, g, ball, sc in results[:5]:
        print(k, g, [round(v, 4) for v in ball], sc)


if __name__ == "__main__":
    main()

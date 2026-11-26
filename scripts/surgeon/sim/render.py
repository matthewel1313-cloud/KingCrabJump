"""Render recorded takes to frames, shot by shot.

The simulation (`choreography.py`) records the full state every control step;
this replays it kinematically (qpos + mocap -> mj_forward) and draws it from
the cameras the edit asks for. No physics is stepped here, so a shot can be
re-rendered from any camera without changing what happened.

    python king_crab/sim/render.py preview            # low-res 3-camera contact video
    python king_crab/sim/render.py shots              # 1080x1920 shots per edit/edl.json
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "out"
sys.path.insert(0, str(HERE))
from camera import Cam, Renderer  # noqa: E402

FPS = 30


class Take:
    def __init__(self, name: str, model: mujoco.MjModel):
        z = np.load(OUT / f"sim_{name}.npz")
        self.t = z["t"]
        self.qpos, self.mp, self.mq, self.eq = z["qpos"], z["mocap_pos"], z["mocap_quat"], z["eq_active"]
        self.meta = json.loads((OUT / f"sim_{name}.json").read_text())
        self.model = model
        self.data = mujoco.MjData(model)

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def set(self, t: float) -> mujoco.MjData:
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        d = self.data
        d.qpos[:] = self.qpos[i]
        d.mocap_pos[:] = self.mp[i]
        d.mocap_quat[:] = self.mq[i]
        d.eq_active[:] = self.eq[i]
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_camlight(self.model, d)   # lights/cameras follow bodies
        return d

    def mark(self, name: str, n: int = 0, **match) -> float:
        hits = [m for m in self.meta["marks"] if m["name"] == name
                and all(m.get(k) == v for k, v in match.items())]
        return hits[n]["t"]


def body(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


class Shots:
    """The three cameras, as functions of the recorded state.

    Smoothed: a tracking camera follows an exponentially smoothed target so the
    robot's gait bob does not shake the frame.
    """

    def __init__(self, model):
        self.m = model
        self.base = body(model, "robot/base_link")
        self.ga = body(model, "robot/LF_palm_grip_insert_link")
        self.gb = body(model, "robot/LF_finger_grip_insert_link")
        self._s = {}

    def _smooth(self, key, v, a=0.12):
        v = np.asarray(v, float)
        if key not in self._s:
            self._s[key] = v.copy()
        self._s[key] += a * (v - self._s[key])
        return self._s[key].copy()

    def reset(self):
        self._s = {}

    def grip(self, d):
        return 0.5 * (d.xpos[self.ga] + d.xpos[self.gb])

    def wide(self, d, t, u=0.0):
        """Wide shot of the tray: from the crab's left-rear, high, so the crab,
        the stand and the surgeon's side stack vertically in a 9:16 frame."""
        b = d.xpos[self.base]
        look = self._smooth("wide", [b[0] + 0.12, b[1] + 0.02, 0.07], a=0.05)
        return Cam(look, 0.95, -145, -40, 44)

    def close(self, d, t, u=0.0):
        """Close-up on the claw, from the front-left, slightly below the arm."""
        g = self._smooth("close", self.grip(d), a=0.18)
        return Cam(g + [0.04, -0.015, -0.005], 0.36, -138, -46, 38)

    def pov(self, d, t, u=0.0):
        """The surgeon's point of view: across the stand, looking down at the
        crab; the glove enters from the bottom of frame."""
        b = d.xpos[self.base]
        look = self._smooth("pov", [b[0] + 0.12, b[1] + 0.06, 0.08], a=0.06)
        return Cam(look, 0.66, 200, -42, 46)

    def hook(self, d, t, u=0.0):
        """Slow push-in on the crab's face from in front, low (it looks up)."""
        b = d.xpos[self.base]
        far = Cam(b + [0.03, 0.0, 0.06], 0.62, 180, -22, 40)
        near = Cam(b + [0.06, 0.0, 0.07], 0.33, 180, -14, 40)
        return far.lerp(near, u)

    def blooper(self, d, t, u=0.0):
        """The outtake: front-left of the claw, so the skid reads."""
        b = d.xpos[self.base]
        look = self._smooth("blooper", [0.19, 0.5 * b[1] + 0.09, 0.09], a=0.08)
        return Cam(look, 0.52, 205, -30, 42).lerp(Cam(look, 0.46, 200, -28, 42), u)

    def finale(self, d, t, u=0.0):
        b = d.xpos[self.base]
        a = Cam(b + [0.0, 0.0, 0.07], 0.75, 160, -14, 42)
        z = Cam(b + [0.0, 0.0, 0.08], 0.62, 175, -10, 42)
        return a.lerp(z, u)


def encode(frames_dir: Path, out: Path, fps=FPS, crf=18):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i",
                    str(frames_dir / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", str(crf), str(out)], check=True)


def preview(take_name="main"):
    from PIL import Image

    model = mujoco.MjModel.from_binary_path(str(OUT / f"model_{take_name}.mjb"))
    take = Take(take_name, model)
    W, H = 270, 480
    r = Renderer(model, W, H)
    sh = Shots(model)
    fdir = OUT / f"preview_{take_name}"
    fdir.mkdir(exist_ok=True, parents=True)
    for f in fdir.glob("*.png"):
        f.unlink()
    n = int(take.duration * FPS)
    for i in range(n):
        t = i / FPS
        d = take.set(t)
        row = [r.render(d, sh.wide(d, t)), r.render(d, sh.close(d, t)), r.render(d, sh.pov(d, t))]
        im = Image.fromarray(np.concatenate(row, axis=1))
        from PIL import ImageDraw
        ImageDraw.Draw(im).text((4, 4), f"{t:5.2f}s", fill=(0, 0, 0))
        im.save(fdir / f"{i:05d}.png")
    encode(fdir, OUT / f"preview_{take_name}.mp4", crf=23)
    print("wrote", OUT / f"preview_{take_name}.mp4")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if what == "preview":
        preview(sys.argv[2] if len(sys.argv) > 2 else "main")

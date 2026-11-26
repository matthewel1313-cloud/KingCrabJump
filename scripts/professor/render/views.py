"""Render a MuJoCo model from a few named angles into PNGs (offscreen, EGL).

Usage: MUJOCO_GL=egl python views.py MODEL.xml OUTDIR [--pose preview-pose.json] [--size 640]
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image

VIEWS = {  # name: (azimuth, elevation, distance)
    "front": (180, -15, 0.45), "side": (90, -10, 0.45), "back": (0, -15, 0.45),
    "three_quarter": (145, -25, 0.5), "top": (180, -80, 0.45), "head": (165, -12, 0.22),
}

def apply_pose(model, data, pose):
    for name, value in pose.get("joints", pose).items() if isinstance(pose, dict) else []:
        if not isinstance(value, (int, float)):
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            lo, hi = model.jnt_range[jid]
            data.qpos[model.jnt_qposadr[jid]] = np.clip(value, lo, hi) if model.jnt_limited[jid] else value

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model"); p.add_argument("out")
    p.add_argument("--pose"); p.add_argument("--size", type=int, default=640)
    p.add_argument("--views", default=",".join(VIEWS))
    p.add_argument("--lookat", type=float, nargs=3)
    a = p.parse_args()
    model = mujoco.MjModel.from_xml_path(a.model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, a.size)
    model.vis.global_.offheight = max(model.vis.global_.offheight, a.size)
    data = mujoco.MjData(model)
    if a.pose:
        pose = json.loads(Path(a.pose).read_text())
        apply_pose(model, data, pose)
        if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and "base_height_m" in pose:
            data.qpos[2] = pose["base_height_m"]
    mujoco.mj_forward(model, data)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(model, a.size, a.size)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    for name in a.views.split(","):
        az, el, dist = VIEWS[name]
        cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = a.lookat if a.lookat else (data.xpos[bid] if bid >= 0 else model.stat.center)
        r.update_scene(data, cam, opt)
        Image.fromarray(r.render()).save(out / f"{name}.png")
    print("wrote", sorted(p.name for p in out.glob("*.png")))

if __name__ == "__main__":
    main()

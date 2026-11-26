"""Render the recorded run on the skinned robot in the classroom map, shot by shot.

    MUJOCO_GL=egl python render/render_shots.py --sequence motion/sequence.npz \\
        --skin output/king_crab_professor.skin --map output/king_crab_classroom.map \\
        --out render/out/raw_main.mp4 [--start 0 --end 30 --fps 30 --scale 1.0]

The motion is the simulation's, replayed kinematically: every frame sets the skinned robot's
root and 22 joints, and each paper's free joint, to the recorded state nearest that time (the
run is recorded at the controller's 200 Hz). Train has no .skin loader, so this is how the
skin appears on the motion; the map world is the same package both sides loaded.

Writes the video (H.264, high quality, no overlays) and `<out>.json` with, per frame, the
screen position of the crown and the corners of every paper -- the overlay anchors for post.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import compose  # noqa: E402
import shots  # noqa: E402

DESIGN = Path(os.environ.get("DESIGN", "../jumper-design")).resolve()  # a jumper-design checkout
PAPER_HALF = (0.06, 0.045, 0.0011)
# Crown, in base_link's frame: CAD (0, -52, 73) mm of the upper shell through the profile's
# CAD->upper_shell transform, plus the upper shell's offset in base_link, raised to its band top.
CROWN_BASE = np.array([0.0377, 0.0, 0.088])
CROWN_TOP = np.array([0.0377, 0.0, 0.105])
HEAD_FRONT = np.array([0.095, 0.0, 0.055])


class Replay:
    def __init__(self, model, seq):
        self.m, self.s = model, seq
        self.d = mujoco.MjData(model)
        self.t = seq["t"]
        self.root = compose.free_joint_of(model, "base_link")
        self.joint_adr = []
        for k, name in enumerate(seq["joint_names"]):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name).split("/")[-1])
            if j < 0:
                raise KeyError(f"skin robot has no joint {name}")
            self.joint_adr.append((k, int(model.jnt_qposadr[j])))
        self.bodies = []
        for k, name in enumerate(seq["free_bodies"]):
            name = str(name)
            if name.endswith("base_link"):
                continue
            adr = compose.free_joint_of(model, name)
            if adr is None:
                raise KeyError(f"map has no free body {name}")
            self.bodies.append((k, adr, name))

    def index(self, t):
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))

    def set(self, t):
        i = self.index(t)
        q = self.d.qpos
        q[self.root:self.root + 3] = self.s["root_pos"][i]
        q[self.root + 3:self.root + 7] = self.s["root_quat"][i]
        for k, adr in self.joint_adr:
            q[adr] = self.s["qpos"][i, k]
        for k, adr, _ in self.bodies:
            q[adr:adr + 3] = self.s["body_pos"][i, k]
            q[adr + 3:adr + 7] = self.s["body_quat"][i, k]
        mujoco.mj_forward(self.m, self.d)
        return i

    def body(self, suffix):
        for b in range(self.m.nbody):
            n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if n == suffix or n.endswith(suffix):
                return b
        raise KeyError(suffix)

    def world(self, body, local):
        return self.d.xpos[body] + self.d.xmat[body].reshape(3, 3) @ np.asarray(local)


class Lens:
    """The free camera MuJoCo renders with, rebuilt from the same numbers (MjvCamera + fovy)."""

    def __init__(self, spec, W, H):
        az, el = math.radians(spec["azimuth"]), math.radians(spec["elevation"])
        self.fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        self.pos = np.asarray(spec["lookat"], float) - spec["distance"] * self.fwd
        self.right = np.cross(self.fwd, [0, 0, 1.0]); self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.fwd)
        self.f = (H / 2) / math.tan(math.radians(spec.get("fovy", 40.0)) / 2)
        self.W, self.H = W, H


def project(lens, W, H, p):
    """World point -> pixel (u, v) and depth."""
    d = np.asarray(p, float) - lens.pos
    z = float(d @ lens.fwd)
    if z <= 1e-6:
        return None
    u = W / 2 + lens.f * float(d @ lens.right) / z
    v = H / 2 - lens.f * float(d @ lens.up) / z
    return [round(u, 2), round(v, 2), round(z, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--skin", type=Path, required=True)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--scale", type=float, default=1.0, help="0.5 for a quick half-size check")
    ap.add_argument("--frames", type=str, default=None, help="comma list of times: write PNGs instead")
    a = ap.parse_args()

    W, H = int(1080 * a.scale) // 2 * 2, int(1920 * a.scale) // 2 * 2
    model, prefix = compose.load(a.skin, a.map, DESIGN, width=W, height=H)
    seq = dict(np.load(a.sequence))
    rep = Replay(model, seq)
    base = rep.body("base_link")
    papers = {n.split("_", 1)[-1] if n.startswith(prefix) else n: rep.body(n) for _, _, n in rep.bodies}
    renderer = mujoco.Renderer(model, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    cam = mujoco.MjvCamera()

    def frame(t):
        i = rep.set(t)
        spec = shots.camera(t, rep, base)
        cam.lookat[:] = spec["lookat"]; cam.distance = spec["distance"]
        cam.azimuth = spec["azimuth"]; cam.elevation = spec["elevation"]
        model.vis.global_.fovy = spec.get("fovy", 40.0)
        renderer.update_scene(rep.d, cam, opt)
        img = renderer.render()
        gl = Lens(spec, W, H)
        meta = {"t": round(t, 4), "i": i, "mode": str(seq["mode"][i]), "shot": spec["name"],
                "crown_top": project(gl, W, H, rep.world(base, CROWN_TOP)),
                "crown": project(gl, W, H, rep.world(base, CROWN_BASE)),
                "head": project(gl, W, H, rep.world(base, HEAD_FRONT)), "papers": {}}
        for name, b in papers.items():
            corners = [rep.world(b, (sx * PAPER_HALF[0], sy * PAPER_HALF[1], PAPER_HALF[2]))
                       for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            normal = rep.d.xmat[b].reshape(3, 3)[:, 2]
            to_cam = np.asarray(gl.pos) - rep.d.xpos[b]
            meta["papers"][name] = {"corners": [project(gl, W, H, c) for c in corners],
                                    "center": project(gl, W, H, rep.world(b, (0, 0, PAPER_HALF[2]))),
                                    "faces_camera": bool(normal @ to_cam > 0)}
        return img, meta

    if a.frames:
        from PIL import Image
        a.out.mkdir(parents=True, exist_ok=True)
        for t in (float(x) for x in a.frames.split(",")):
            img, meta = frame(t)
            from PIL import ImageDraw
            im = Image.fromarray(img); dr = ImageDraw.Draw(im)
            for key in ("crown_top", "head"):
                if meta[key]:
                    u, v, _ = meta[key]; dr.ellipse([u - 4, v - 4, u + 4, v + 4], outline=(255, 0, 255), width=2)
            for name, pp in meta["papers"].items():
                pts = [tuple(c[:2]) for c in pp["corners"] if c]
                if len(pts) == 4:
                    dr.polygon(pts, outline=(0, 255, 255) if pp["faces_camera"] else (255, 80, 0))
            im.save(a.out / f"t{t:05.2f}.png")
            print(json.dumps(meta)[:200])
        return

    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = int(round((a.end - a.start) * a.fps))
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "12",
         "-pix_fmt", "yuv420p", str(a.out)], stdin=subprocess.PIPE)
    metas = []
    for k in range(n):
        t = a.start + k / a.fps
        img, meta = frame(t)
        enc.stdin.write(img.tobytes())
        metas.append(meta)
        if k % 30 == 0:
            print(f"\r[render] {t:6.2f}s  {meta['shot']:<10s} frame {k}/{n}", end="", flush=True)
    enc.stdin.close(); enc.wait()
    a.out.with_suffix(".json").write_text(json.dumps({"fps": a.fps, "width": W, "height": H,
                                                       "start": a.start, "frames": metas}))
    print(f"\n[render] wrote {a.out} ({n} frames) and {a.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()

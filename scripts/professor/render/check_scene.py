"""Render a composed skin + scene from its named cameras: python check_scene.py SKIN WORLD OUTDIR"""
import json, os, sys
from pathlib import Path
import mujoco, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import compose

DESIGN = Path(os.environ.get("DESIGN", "../jumper-design")).resolve()  # a jumper-design checkout
skin, world, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
model, prefix = compose.load(skin, world, DESIGN, width=1080, height=1920)
data = mujoco.MjData(model)
pose = json.loads((DESIGN / "robots/jumper/preview-pose.json").read_text())
for n, v in pose["joints"].items():
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    data.qpos[model.jnt_qposadr[j]] = np.clip(v, *model.jnt_range[j])
a = compose.free_joint_of(model, "base_link")
data.qpos[a:a + 3] = [0, 0, pose["base_height_m"]]
for _ in range(300):  # let the papers settle
    mujoco.mj_step(model, data)
    data.qpos[a:a + 3] = [0, 0, pose["base_height_m"]]; data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)
r = mujoco.Renderer(model, 960, 540)
for cam in ("vertical_main", "hook_closeup"):
    r.update_scene(data, camera=prefix + cam)
    Image.fromarray(r.render()).save(out / f"{cam}.png")
free = mujoco.MjvCamera(); free.lookat[:] = [0.15, -0.45, 0.05]; free.distance = 1.6; free.azimuth = 200; free.elevation = -45
r2 = mujoco.Renderer(model, 1280, 720); r2.update_scene(data, free); Image.fromarray(r2.render()).save(out / "overview.png")
print("ok", sorted(p.name for p in out.glob("*.png")))

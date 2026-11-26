"""Render the map's robot-free package thumbnail from the compiled scene (native MuJoCo).

    MUJOCO_GL=egl python scene/render_preview.py <built scene.xml> <out.png>

The papers settle for 0.5 s first. This is a native render, not the be-web/1 Web importer
render the formal jumper-design library requires (see README).
"""
import sys

import mujoco
from PIL import Image

model = mujoco.MjModel.from_xml_path(sys.argv[1])
model.vis.global_.offwidth = 1280
model.vis.global_.offheight = 1280
model.vis.quality.shadowsize = 4096
data = mujoco.MjData(model)
for _ in range(500):
    mujoco.mj_step(model, data)
renderer = mujoco.Renderer(model, 1024, 1024)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.0, -0.45, 0.06]
cam.distance, cam.azimuth, cam.elevation = 1.55, 205, -32
renderer.update_scene(data, cam)
Image.fromarray(renderer.render()).save(sys.argv[2])

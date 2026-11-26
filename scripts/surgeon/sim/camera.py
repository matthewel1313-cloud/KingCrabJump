"""Offscreen rendering for King Crab: a renderer and the three named cameras.

Cameras are free cameras defined by (lookat, distance, azimuth, elevation),
computed per frame from the recorded state, so a shot can track the robot or
the claw. Vertical 9:16 frames are rendered directly (width 1080, height 1920).
"""

from __future__ import annotations

import mujoco
import numpy as np


class Cam:
    def __init__(self, lookat, distance, azimuth, elevation, fovy=None):
        self.lookat = np.asarray(lookat, dtype=float)
        self.distance = float(distance)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)
        self.fovy = fovy

    def lerp(self, other: "Cam", a: float) -> "Cam":
        a = float(np.clip(a, 0, 1))
        s = a * a * (3 - 2 * a)  # smoothstep
        # interpolate azimuth the short way round
        daz = (other.azimuth - self.azimuth + 180) % 360 - 180
        return Cam(self.lookat + s * (other.lookat - self.lookat),
                   self.distance + s * (other.distance - self.distance),
                   self.azimuth + s * daz,
                   self.elevation + s * (other.elevation - self.elevation),
                   self.fovy)


class Renderer:
    def __init__(self, model: mujoco.MjModel, width: int = 1080, height: int = 1920,
                 fovy: float = 40.0):
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        model.vis.global_.fovy = fovy
        model.vis.quality.shadowsize = 4096
        model.vis.map.znear = 0.01
        model.vis.headlight.ambient[:] = [0.20, 0.21, 0.22]
        model.vis.headlight.diffuse[:] = [0.22, 0.23, 0.24]
        model.vis.headlight.specular[:] = [0.05, 0.05, 0.05]
        self.model = model
        self.r = mujoco.Renderer(model, height, width)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        for g in (0, 2):  # world + robot visuals; hide collision meshes (1) and sites
            self.opt.geomgroup[g] = 1
        self.opt.sitegroup[:] = 0
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.default_fovy = fovy

    def render(self, data: mujoco.MjData, cam: Cam) -> np.ndarray:
        self.cam.lookat[:] = cam.lookat
        self.cam.distance = cam.distance
        self.cam.azimuth = cam.azimuth
        self.cam.elevation = cam.elevation
        self.model.vis.global_.fovy = cam.fovy or self.default_fovy
        self.r.update_scene(data, self.cam, self.opt)
        scn = self.r.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        return self.r.render()

"""Compose a .skin robot and a .map world into one MuJoCo model for rendering.

Follows jumper-design's examples/load_skin_map.py: both packages are read with the reference
reader (`shellflow.package_io.read_archive`), the robot spec is the host and the map's world is
attached into it with `MjSpec.attach` under the map's declared prefix -- never XML
concatenation. Run in the jumper-design environment.

Also accepts an unpacked scene directory (scene.xml + assets/) in place of a .map, for
checking a scene before it is packaged.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import mujoco


def _design_src(design: Path):
    import sys
    sys.path.insert(0, str(design / "src"))
    sys.path.insert(0, str(design / "examples"))


def load(skin: Path, world: Path, design: Path, *, width=1080, height=1920, shadow=4096):
    """Return (model, prefix) with the skin's robot and the map's world in one model."""
    _design_src(design)
    from shellflow.map_package import MAP_LIMITS
    from shellflow.package_io import json_object, read_archive
    from load_skin_map import make_spec

    skin_files = read_archive(skin)
    robot_spec, _ = make_spec(skin_files, "assembled/jumper/robot.xml", "assembled/jumper")

    if world.is_dir():
        files = {str(p.relative_to(world)).replace("\\", "/"): p.read_bytes()
                 for p in world.rglob("*") if p.is_file()}
        xml, prefix = "scene.xml", "scn_"
        assets = {k: v for k, v in files.items() if k != xml}
        # mjscene writes file="name.png" with meshdir/texturedir set; give MuJoCo both spellings
        assets.update({PurePosixPath(k).name: v for k, v in files.items() if k.startswith("assets/")})
        map_spec = mujoco.MjSpec.from_string(files[xml].decode("utf-8"), assets=assets)
    else:
        map_files = read_archive(world, limits=MAP_LIMITS)
        manifest = json_object(map_files["scene-package.json"])
        xml = manifest["world"]["file"]
        prefix = manifest["world"]["attach"]["prefix"]
        env = {k: v for k, v in map_files.items() if not k.startswith("robot/")}
        parent = PurePosixPath(xml).parent
        map_spec, _ = make_spec(env, xml, "" if parent == PurePosixPath(".") else str(parent))

    robot_spec.attach(map_spec, prefix=prefix, frame=robot_spec.worldbody.add_frame(name="map_frame"))
    robot_spec.visual.global_.offwidth = width
    robot_spec.visual.global_.offheight = height
    robot_spec.visual.quality.shadowsize = shadow
    robot_spec.visual.map.znear = 0.002
    model = robot_spec.compile()
    return model, prefix


def free_joint_of(model, body_suffix: str) -> int | None:
    """The qpos address of the free joint of the body whose name ends with `body_suffix`."""
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[j]) or ""
        if name == body_suffix or name.endswith(body_suffix):
            return int(model.jnt_qposadr[j])
    return None

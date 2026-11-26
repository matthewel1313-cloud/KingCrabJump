#!/usr/bin/env python3
"""Drive a jumper `.app` with a scripted gamepad in one continuous simulation, and record it.

This is `scripts/play.py --app` with the person replaced by a timeline: the same environment
construction (`tasks.load_env_cfg(play=True)`, `apply_scene`, `step_the_world_at`), the same
deployment controller (`mjrl.app_play.AppPlayer` -- every mode, its ramps and its hooks), and the
pad handed to it through the same `pad=` argument a real gamepad uses. Nothing in the jumper
repository is modified.

Run from the jumper checkout, in its environment:

    python /path/to/run_sequence.py --app king_crab_grader.app --timeline timeline.py \\
        --scene king_crab_classroom.map --seconds 34 --out sequence.npz [--video debug.mp4]

The timeline module defines `pad(t) -> dict` (buttons, lx, ly, rx, ry, lt, rt, hat_x, hat_y; see
controller.xbox.GamepadState for the conventions: Ly positive *down*, hat_y positive *down*) and
may define `attach(t, state) -> str | None` naming a free body to hold kinematically in the
claw (the fallback the README documents) and `PRE_ROLL` seconds to discard before t = 0.

The output `.npz` holds, per control step: time, controller mode, root pose, joint positions by
name, and the pose of every free body in the world -- everything the render stage needs to
replay the run on the skinned robot.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

JUMPER = Path.cwd()
sys.path.insert(0, str(JUMPER / "scripts"))


@dataclass
class PadState:
    """The fields `AppPlayer._person` reads from a gamepad's state."""

    buttons: frozenset = field(default_factory=frozenset)
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    hat_x: int = 0
    hat_y: int = 0


class ScriptedPad:
    """A gamepad whose state is a function of simulated time."""

    connected = True

    def __init__(self, fn, clock):
        self._fn, self._clock = fn, clock

    def state(self) -> PadState:
        d = dict(self._fn(self._clock()))
        d["buttons"] = frozenset(d.get("buttons", ()))
        return PadState(**d)

    def close(self):
        pass


def load_timeline(path: Path):
    spec = importlib.util.spec_from_file_location("timeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", type=Path, required=True)
    ap.add_argument("--timeline", type=Path, required=True)
    ap.add_argument("--scene", default=None, help="a .map (kk-scene-package/2) or a built-in scene")
    ap.add_argument("--seconds", type=float, required=True, help="length after the pre-roll")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video", type=Path, default=None, help="debug MP4 of the stock robot")
    ap.add_argument("--physics-hz", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=7, help="fixes the startup randomisation")
    a = ap.parse_args()

    timeline = load_timeline(a.timeline)
    pre_roll = float(getattr(timeline, "PRE_ROLL", 3.0))

    from mjrl.app_play import AppPlayer, open_app, step_the_world_at
    from _cli import add_scene_args, add_viewer_args, build_parser, parse_with_task_args, resolve_all
    from _cli import apply_scene
    import play as play_script  # for _set_physics_rate, the replay physics-rate rule

    opened = open_app(a.app)
    parser = build_parser("play.py", "", replay=True)
    add_viewer_args(parser, ui_default=False, fps_default=60.0)
    add_scene_args(parser)
    argv = ["--headless", "--backend", "native", "--device", "cpu"]
    if a.scene:
        argv += ["--scene", str(a.scene)]
    sys.argv = ["play.py", *argv]
    args = parse_with_task_args(parser, opened.default_task)
    spec, res, asset = resolve_all(args)

    import numpy as np
    import torch
    from mjrl.backend.select import use_backend

    import tasks

    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    env_cfg = tasks.load_env_cfg(spec.id, asset=asset, play=True, task_args=args.task_args)
    play_script._set_physics_rate(env_cfg, a.physics_hz)
    step_the_world_at(env_cfg, opened.control_hz)
    apply_scene(env_cfg, args)
    env_cfg.scene.num_envs = 1
    # Reproducible: one seed, and the reset puts the robot on the map's spawn (which the
    # importer moves to the origin, heading +X) instead of a random pose within +-0.5 m and any
    # heading. The startup randomisation (friction, COM, encoder bias) is kept, seeded.
    env_cfg.seed = a.seed
    if "reset_base" in env_cfg.events:
        env_cfg.events["reset_base"].params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.01, 0.01), "yaw": (0.0, 0.0)}
    agent_cfg = tasks.load_agent_cfg(spec.id)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
    from mjrl.replay import skip_rewards

    skip_rewards(env)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = wrapped.unwrapped
    dt = unwrapped.step_dt
    tick = {"n": 0}

    def clock() -> float:
        return tick["n"] * dt - pre_roll

    pad = ScriptedPad(timeline.pad, clock)
    player = AppPlayer(opened, wrapped, pad=pad)

    import mujoco

    sim = unwrapped.sim
    model = sim.mj_model
    robot = unwrapped.scene["robot"]
    joint_names = list(robot.joint_names)
    free_bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[j])
                   for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    free_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in free_bodies]
    print(f"[seq] free bodies: {free_bodies}")
    track = [n for n in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody))
             if n and n.split("/")[-1] in ("RF_palm_link", "RF_finger_tip_link", "RF_finger_link",
                                           "RF_palm_pad_f_link", "RF_palm_pad_b_link")]
    track_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in track]
    origin = unwrapped.scene.env_origins[0].cpu().numpy() if hasattr(unwrapped.scene, "env_origins") else np.zeros(3)
    attach_fn = getattr(timeline, "attach", None)
    held: dict[str, np.ndarray] = {}

    def body_pose(i):
        d = sim.data
        return (d.xpos[0, i].cpu().numpy().copy(), d.xquat[0, i].cpu().numpy().copy())

    video = None
    if a.video is not None:
        from mjrl.viewer.video import ReplayVideo

        video = ReplayVideo(env, a.video, fps=30, width=540, height=960, env_index=0,
                            camera_distance=0.9, azimuth=150, elevation=-20)
        video.__enter__()

    rec = {k: [] for k in ("t", "mode", "root_pos", "root_quat", "qpos", "body_pos", "body_quat", "pad", "track")}
    total = int(round((pre_roll + a.seconds) / dt))
    obs = wrapped.get_observations()
    try:
        for _ in range(total):
            with torch.inference_mode():
                obs, _, dones, _ = wrapped.step(player(obs))
            tick["n"] += 1
            t = clock()
            if attach_fn is not None:
                _kinematic_attach(attach_fn, t, sim, model, free_bodies, free_ids, held)
            if video is not None and t >= 0:
                video.update()
            if t < 0:
                continue
            rec["t"].append(t)
            rec["mode"].append(player.fsm.mode)
            rec["root_pos"].append(robot.data.root_link_pos_w[0].cpu().numpy() - origin)
            rec["root_quat"].append(robot.data.root_link_quat_w[0].cpu().numpy())
            rec["qpos"].append(robot.data.joint_pos[0].cpu().numpy())
            poses = [body_pose(i) for i in free_ids]
            rec["body_pos"].append(np.array([p - origin for p, _ in poses]).reshape(-1, 3))
            rec["body_quat"].append(np.array([q for _, q in poses]).reshape(-1, 4))
            rec["pad"].append(json.dumps(timeline.pad(t), default=list))
            rec["track"].append(np.array([body_pose(i)[0] - origin for i in track_ids]).reshape(-1, 3))
            if bool(dones.any()):
                print(f"\n[seq] episode reset at t={t:.2f}s -- the recording after this is a new attempt")
    finally:
        if video is not None:
            video.close()
        player.close()
        opened.close()
        env.close()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, t=np.array(rec["t"]), mode=np.array(rec["mode"]), root_pos=np.array(rec["root_pos"]),
        root_quat=np.array(rec["root_quat"]), qpos=np.array(rec["qpos"]),
        body_pos=np.array(rec["body_pos"]), body_quat=np.array(rec["body_quat"]),
        pad=np.array(rec["pad"]), track=np.array(rec["track"]), track_names=np.array(track), joint_names=np.array(joint_names), free_bodies=np.array(free_bodies),
        control_dt=dt, scene=str(a.scene), app=str(a.app))
    modes = rec["mode"]
    spans, start = [], 0
    for i in range(1, len(modes) + 1):
        if i == len(modes) or modes[i] != modes[start]:
            spans.append(f"{modes[start]} {rec['t'][start]:.2f}-{rec['t'][i - 1]:.2f}s")
            start = i
    print(f"[seq] wrote {a.out} ({len(modes)} steps at {1 / dt:g} Hz); modes: " + "; ".join(spans))


def _kinematic_attach(attach_fn, t, sim, model, free_bodies, free_ids, held):
    """Hold a named free body rigidly in the right claw's palm frame while `attach` says so.

    The fallback for paper physics a claw cannot hold: on the first step a body is named its
    current pose relative to the palm is frozen, and from then on its free joint is written
    from the palm's pose every control step with zero velocity. Released the step it is not
    named, from wherever it is, with zero velocity, so it falls and settles under physics.
    """
    import mujoco
    import numpy as np
    import torch

    name = attach_fn(t)
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/RF_palm_link")
    if palm < 0:
        palm = next(i for i in range(model.nbody)
                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("RF_palm_link"))
    d = sim.data
    for other in [k for k in held if k != name]:
        held.pop(other)
    if name is None:
        return
    match = [k for k, b in enumerate(free_bodies) if b == name or b.endswith(name)]
    if not match:
        return  # nothing by that name in this world (a dry run without the map)
    i = free_ids[match[0]]
    j = next(j for j in range(model.njnt) if model.jnt_bodyid[j] == i)
    qa, va = model.jnt_qposadr[j], model.jnt_dofadr[j]
    ppos, pquat = d.xpos[0, palm].cpu().numpy(), d.xquat[0, palm].cpu().numpy()
    if name not in held:
        bpos, bquat = d.xpos[0, i].cpu().numpy(), d.xquat[0, i].cpu().numpy()
        inv = np.zeros(4); mujoco.mju_negQuat(inv, pquat)
        rel_q = np.zeros(4); mujoco.mju_mulQuat(rel_q, inv, bquat)
        rel_p = np.zeros(3); mujoco.mju_rotVecQuat(rel_p, bpos - ppos, inv)
        held[name] = np.concatenate([rel_p, rel_q])
    rel_p, rel_q = held[name][:3], held[name][3:]
    wp = np.zeros(3); mujoco.mju_rotVecQuat(wp, rel_p, pquat)
    wq = np.zeros(4); mujoco.mju_mulQuat(wq, pquat, rel_q)
    qpos = d.qpos.clone(); qvel = d.qvel.clone()
    qpos[0, qa:qa + 3] = torch.as_tensor(ppos + wp, dtype=qpos.dtype)
    qpos[0, qa + 3:qa + 7] = torch.as_tensor(wq, dtype=qpos.dtype)
    qvel[0, va:va + 6] = 0
    d.qpos[:] = qpos; d.qvel[:] = qvel


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capture a Jumper policy replay as poses, so it can be filmed afterwards.

Runs the Jumper toolkit's own `scripts/play.py` (same env, same scene, same
pretrained checkpoint), but instead of its fixed video camera it records the
robot's full joint state (qpos) at every control step, plus the compiled
appearance model as an .mjb. `render.py` then films that capture with any
camera moves, slow motion and extra props we like, without re-simulating.

With BALL="px,py,pz,vx,vy,vz" set, a real physical volleyball is added to the
simulation (a free body with contact, via the toolkit's own scene-prop
mechanism), launched from that position/velocity at the start of every episode.
The robot and ball interact through MuJoCo contact both ways; nothing is scripted.

Usage (run from anywhere; JUMPER points at a checkout of KingKongRobotics/jumper):

    MUJOCO_GL=egl python tools/capture.py OUT_NAME -- \
        --task jumper.jump --checkpoint tasks/jumper/jump/out/example/model_3100.pt \
        --scene beach --steps 800
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

JUMPER = Path(os.environ.get("JUMPER", "")).resolve()
OUT_DIR = Path(os.environ.get("CAPTURE_DIR", Path(__file__).resolve().parents[1] / "captures"))


#: A volleyball scaled to the crab: 9 cm across, a thin shell (mass scales with
#: surface area: 0.27 kg * (4.5/10.5)^2 ~= 50 g), lively bounce.
BALL_RADIUS = 0.045
BALL_MASS = 0.05


def _volleyball_spec():
    import mujoco

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="volleyball", pos=[0.0, 0.0, 0.0])
    body.add_freejoint()
    body.add_geom(
        name="volleyball", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS, 0.0, 0.0], mass=BALL_MASS, rgba=[0.97, 0.97, 0.97, 1.0],
        condim=6, friction=[0.6, 0.005, 0.01], solref=[0.01, 0.15],  # volleyball-like bounce
        contype=1, conaffinity=1, group=2,
    )
    return spec


def add_volleyball(env_cfg, pos, vel):
    """Add the ball as a scene entity, reset to `pos` with velocity `vel` each episode."""
    from mjlab.entity import EntityCfg
    from mjlab.envs.mdp import events as mdp_events
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    env_cfg.scene.entities["volleyball"] = EntityCfg(
        spec_fn=_volleyball_spec,
        init_state=EntityCfg.InitialStateCfg(pos=tuple(pos)),
    )
    vr = {k: (v, v) for k, v in zip(("x", "y", "z"), vel)}
    env_cfg.events["reset_volleyball"] = EventTermCfg(
        func=mdp_events.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": vr,
                "asset_cfg": SceneEntityCfg("volleyball")},
    )
    print(f"[capture] physical volleyball at {pos} launched with {vel} m/s")


def main() -> None:
    if "--" not in sys.argv or not (JUMPER / "scripts" / "play.py").exists():
        raise SystemExit(__doc__)
    split = sys.argv.index("--")
    name = sys.argv[1]
    play_args = sys.argv[split + 1 :]

    os.chdir(JUMPER)
    sys.path.insert(0, str(JUMPER / "scripts"))
    import mujoco
    import play  # the toolkit's own replay entry point

    captured: dict = {}
    original_loop = play._loop

    ball = os.environ.get("BALL")
    if ball:
        bp = [float(x) for x in ball.split(",")]
        original_apply = play.apply_scene

        def apply_with_ball(env_cfg, args, **kw):
            original_apply(env_cfg, args, **kw)
            add_volleyball(env_cfg, bp[:3], bp[3:6])

        play.apply_scene = apply_with_ball

    class PoseRecorder:
        """Stands in for play's MP4 recorder: same update() hook, records poses."""

        def __init__(self, env):
            self.env = getattr(env, "unwrapped", env)
            self.sim = self.env.sim
            self.qpos = []
            self.update()

        def update(self):
            q = self.sim.data.qpos[0]
            self.qpos.append(q.detach().cpu().numpy().astype(np.float64).reshape(-1).copy())

    def loop(env, policy, viewer, max_steps, *a, **kw):
        rec = PoseRecorder(env)
        # play passes (speed, readout, monitor, video, evaluation) positionally.
        a = list(a)
        if len(a) >= 4:
            a[3] = rec
        else:
            kw["video"] = rec
        steps = original_loop(env, policy, viewer, max_steps, *a, **kw)
        sim = rec.sim
        model = getattr(sim, "render_model", sim.mj_model)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveModel(model, str(OUT_DIR / f"{name}.mjb"), None)
        captured.update(qpos=np.stack(rec.qpos), dt=float(rec.env.step_dt))
        return steps

    play._loop = loop
    sys.argv = ["play.py", "--headless", "--backend", "native", "--device", "cpu", *play_args]
    play.main()
    np.savez_compressed(OUT_DIR / f"{name}.npz", **captured)
    print(f"[capture] {name}: {len(captured['qpos'])} poses at dt={captured['dt']:.4f}s "
          f"-> {OUT_DIR / name}.npz/.mjb")


if __name__ == "__main__":
    main()

"""Build the real jumper.posture env (as `play.py --app` does) and dump its compiled MJCF.

The dumped XML carries the training-time collision bits, contact params, solver
options, actuators and joint params, so the standalone sequencer runs the robot
under the same physics the shipped policies were trained in.
Run from the jumper checkout with its venv active.
"""
import sys
from pathlib import Path
import mujoco

out = Path(sys.argv[1])
from mjrl.backend.resolve import resolve
from mjrl.backend.select import use_backend
import tasks

res = resolve(backend="native", device="cpu", num_envs=1)
use_backend(res)
from mjlab.envs import ManagerBasedRlEnv

cfg = tasks.load_env_cfg("jumper.posture", play=True)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
u = env.unwrapped
m = u.sim.mj_model
print("timestep", m.opt.timestep, "decimation", cfg.decimation, "integrator", m.opt.integrator,
      "cone", m.opt.cone, "iters", m.opt.iterations, "nu", m.nu, "ngeom", m.ngeom)
robot = u.scene["robot"]
for a in robot.actuators:
    print(type(a).__name__, {k: getattr(a, k) for k in ("plateau_torque","corner_speed","decay_speed","cutoff_speed") if hasattr(a,k)} )
spec = u.scene.spec
xml = spec.to_xml()
# MjSpec.to_xml emits an anonymous nested <default/> the XML parser then rejects
# ("empty class name"); it carries no attributes, so it is dropped.
xml = xml.replace("<default>\n    <default/>\n", "<default>\n", 1)
out.write_text(xml)
print("wrote", out, len(xml))

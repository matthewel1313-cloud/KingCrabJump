"""Standalone MuJoCo runtime for the Jumper crab, driven by the shipped app.

What runs here is the same stack `scripts/play.py --app` runs, minus mjlab:

* the robot model is the **compiled training model** of `jumper.posture`
  (dumped by `tools/dump_robot_model.py`): same collision bits, contact params,
  joint armature/damping/frictionloss and actuator limits;
* the controller is the bundle's **compiled `mjrl_fsm` extension**
  (`runtime/mjlab/linux-x86_64/controller.so`) loaded exactly as
  `mjrl/app_play.py::_load_extension` loads it;
* each mode's policy is the bundle's **ONNX** file, run on onnxruntime CPU;
* the servos are the training actuator: PD torque with the controller's own
  per-joint kp/kd, clipped to the measured servo torque-speed curve
  (`tasks/jumper/common/actuator.py::servo_torque_limit`). The thermal derate
  is not modelled (the clip is short).

The person on the pad is replaced by a script: `PadState` is a gamepad frame
(sticks, triggers, buttons, d-pad) handed to `Fsm.set_pad_frame` every tick,
which is what the robot's pad service does on hardware.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

# Servo curve, `assets/jumper/motor/motor_config.yaml` as read by the task.
PLATEAU = 1.7464
CORNER = 30.7352
DECAY = 25.1851
CUTOFF = 63.9838

PHYSICS_DT = 0.001  # jumper.posture's timestep
CONTROL_HZ = 200.0  # the app's fastest mode (locomotion); others infer at their own rate
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * PHYSICS_DT)))


def servo_limit(speed: np.ndarray) -> np.ndarray:
    w = np.abs(speed)
    lim = PLATEAU * np.exp(-np.clip(w - CORNER, 0.0, None) / DECAY)
    return np.where(w >= CUTOFF, 0.0, lim)


@dataclass
class PadState:
    """One gamepad frame, in the robot pad service's units."""

    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    buttons: set = field(default_factory=set)
    dpad_x: int = 0
    dpad_y: int = 0  # up is +1


def load_controller(bundle: Path):
    manifest = json.loads((bundle / "bundle.json").read_text("utf-8"))
    entry = manifest["runtimes"]["mjlab"]["extensions"]["linux-x86_64"]
    path = bundle / entry["file"]
    loader = importlib.machinery.ExtensionFileLoader("mjrl_fsm", str(path))
    spec = importlib.util.spec_from_file_location("mjrl_fsm", path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module, manifest


def open_bundle(app: Path, into: Path) -> Path:
    """An `.app` is a zip of a bundle directory; a directory is used as is."""
    app = Path(app)
    if app.is_dir():
        return app
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(app) as z:
        z.extractall(into)
    return into


class CrabRuntime:
    """The robot inside an arbitrary MuJoCo world, run by the app's controller."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, bundle: Path,
                 prefix: str = "robot/") -> None:
        self.m, self.d = model, data
        self.bundle = Path(bundle)
        fsm_mod, manifest = load_controller(self.bundle)
        contracts = {mode: (self.bundle / e["contract"]).read_text("utf-8")
                     for mode, e in manifest["modes"].items()}
        trajectories = {mode: (self.bundle / e["trajectory"]).read_text("utf-8")
                        for mode, e in manifest["modes"].items() if e.get("trajectory")}
        parsed = {k: json.loads(v) for k, v in contracts.items()}
        self.wire = next(iter(parsed.values()))["wire_joint_order"]
        self.contracts = parsed

        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n) for n in self.wire]
        if min(jids) < 0:
            raise RuntimeError("robot joints missing from the model")
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.vadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n)
                             for n in self.wire])
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base")
        self.root_q = model.jnt_qposadr[root]
        self.root_v = model.jnt_dofadr[root]
        self.base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link")
        lo = [float(model.jnt_range[j][0]) for j in jids]
        hi = [float(model.jnt_range[j][1]) for j in jids]

        gait = next((t.get("params") or {} for c in parsed.values()
                     for t in c["observation"]["terms"] if t["name"] == "gait_phase"), {})
        self.fsm = fsm_mod.Fsm(
            (self.bundle / "controller.toml").read_text("utf-8"),
            contracts,
            {"joint_names": self.wire, "joint_pos_lo": lo, "joint_pos_hi": hi,
             "output_rate_hz": CONTROL_HZ, "gait_period": gait.get("period"),
             "gait_gate_threshold": gait.get("command_threshold")},
            trajectories=trajectories,
            operators=True,
        )
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sessions = {mode: ort.InferenceSession(str(self.bundle / e["models"]["onnx"]), so,
                                                    providers=["CPUExecutionProvider"])
                         for mode, e in manifest["modes"].items()}
        self.tick_n = 0
        self.targets = np.zeros(len(self.wire))
        self.kp = np.zeros(len(self.wire))
        self.kd = np.zeros(len(self.wire))
        self.tau = np.zeros(len(self.wire))
        self.log: list[tuple] = []
        self.inferences = {m: 0 for m in self.sessions}
        # physics hook (welds, kinematic hands) called before every physics substep
        self.pre_substep = None

    # -- state ---------------------------------------------------------------
    def reset_pose(self, xy=(0.0, 0.0), z_ground=0.0, yaw=0.0, mode="locomotion"):
        c = self.contracts[mode]
        d = self.d
        d.qpos[self.root_q:self.root_q + 3] = [xy[0], xy[1], z_ground + c["base_height"] + 0.004]
        d.qpos[self.root_q + 3:self.root_q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i, n in enumerate(self.wire):
            d.qpos[self.qadr[i]] = c["default_joint_pos"][n]
        d.qvel[:] = 0
        self.targets = np.array([c["default_joint_pos"][n] for n in self.wire])
        self.kp[:] = 10.0
        self.kd[:] = 0.5
        mujoco.mj_forward(self.m, d)

    @property
    def now_us(self) -> int:
        return int(round(self.tick_n * 1e6 / CONTROL_HZ))

    @property
    def mode(self) -> str:
        return self.fsm.mode

    def base_pos(self) -> np.ndarray:
        return self.d.xpos[self.base_body].copy()

    def base_yaw(self) -> float:
        q = self.d.qpos[self.root_q + 3:self.root_q + 7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    # -- one control tick (5 ms) -------------------------------------------------
    def step(self, pad: PadState) -> None:
        d = self.d
        now = self.now_us
        self.fsm.set_state(
            [float(x) for x in d.qpos[self.qadr]],
            [float(x) for x in d.qvel[self.vadr]],
            [float(x) for x in self.tau],
            [float(x) for x in d.qpos[self.root_q + 3:self.root_q + 7]],
            # free-joint angular velocity is already in the body frame
            [float(x) for x in d.qvel[self.root_v + 3:self.root_v + 6]],
            now,
        )
        self.fsm.set_pad_frame(
            sorted(pad.buttons),
            {"Lx": pad.lx, "Ly": pad.ly, "Rx": pad.rx, "Ry": pad.ry, "LT": pad.lt, "RT": pad.rt},
            dpad_x=int(pad.dpad_x), dpad_y=int(pad.dpad_y), connected=True, now_us=now,
        )
        mode = self.fsm.tick(now)
        if mode is not None:
            s = self.sessions[mode]
            obs = np.asarray([self.fsm.observation()], dtype=np.float32)
            act = s.run(None, {s.get_inputs()[0].name: obs})[0][0]
            self.fsm.resume([float(v) for v in act])
            self.inferences[mode] += 1
        for entry in self.fsm.take_log():
            self.log.append(entry)
        self.targets = np.asarray(self.fsm.positions())
        self.kp = np.asarray(self.fsm.kp())
        self.kd = np.asarray(self.fsm.kd())
        for _ in range(SUBSTEPS):
            if self.pre_substep is not None:
                self.pre_substep()
            q = d.qpos[self.qadr]
            qd = d.qvel[self.vadr]
            tau = self.kp * (self.targets - q) - self.kd * qd
            lim = servo_limit(qd)
            tau = np.clip(tau, -lim, lim)
            d.ctrl[self.act] = tau
            self.tau = tau
            mujoco.mj_step(self.m, d)
        self.tick_n += 1

"""Shared LoopMimic EMA actor, reference alignment, and 29-joint DDS output."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation, Slerp
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_


def read_metadata(bundle: Path) -> dict:
    return json.loads((bundle / "deployment.json").read_text(encoding="utf-8"))


def rotation(wxyz: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(wxyz, scalar_first=True).as_matrix()


class Motion:
    def __init__(self, path: Path, meta: dict):
        if not path.is_file():
            raise FileNotFoundError(f"{path}. Run bash deploy/setup.sh first.")
        with np.load(path, allow_pickle=False) as data:
            if not np.isclose(data["fps"].item() * meta["control_dt"], 1):
                raise ValueError("Reference fps must match the 50 Hz policy")
            names = data["joint_names"].tolist()
            expected = meta["joint_names"]
            if len(names) != 29 or set(names) != set(expected):
                raise ValueError("Reference joint names must match G1 29-DoF")
            order = [names.index(name) for name in expected]
            self.frames = np.column_stack((
                data["root_pos_w"], data["root_quat_w"],
                data["joint_pos"][:, order], data["joint_vel"][:, order],
            )).astype(np.float32)
        if not len(self.frames) or not np.isfinite(self.frames).all():
            raise ValueError("Reference must have finite frames")
        if not np.allclose(np.linalg.norm(self.frames[:, 3:7], axis=1), 1.0, atol=1e-3):
            raise ValueError("Reference pelvis quaternions must be unit wxyz")
        self.offsets = np.arange(meta["goal_frames"]) * meta["goal_stride"]


@dataclass(frozen=True)
class Observation:
    position: np.ndarray
    orientation_wxyz: np.ndarray
    joint_pos: np.ndarray
    linear_velocity_w: np.ndarray
    angular_velocity_w: np.ndarray
    joint_vel: np.ndarray

    def vector(self) -> np.ndarray:
        return np.concatenate((
            self.position, self.orientation_wxyz, self.joint_pos,
            self.linear_velocity_w, self.angular_velocity_w, self.joint_vel,
        )).astype(np.float32)


class DancePolicy:
    PREPARATION_S = 1.20

    def __init__(self, bundle: Path, motion: Path):
        self.meta = read_metadata(bundle)
        self.motion = Motion(motion, self.meta)
        self.dt = self.meta["control_dt"]
        self.prepare_ticks = round(self.PREPARATION_S / self.dt)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.actor = ort.InferenceSession(str(bundle / "motion.onnx"), sess_options=options,
                                          providers=["CPUExecutionProvider"])
        self.default_q = np.asarray(self.meta["default_q"])
        self.scale = np.asarray(self.meta["position_scale"])
        self.cursor = 0
        self.reference = np.empty((0, 65), dtype=np.float32)
        self.max_root_height_error = 0.0

    def start(self, state: Observation):
        """Align the motion in x/y/yaw and connect the live pose to frame zero."""
        frames = self.motion.frames.copy()
        robot_r = rotation(state.orientation_wxyz)
        first_r = rotation(frames[0, 3:7])
        yaw = np.arctan2(robot_r[1, 0], robot_r[0, 0]) - np.arctan2(first_r[1, 0], first_r[0, 0])
        yaw_r = Rotation.from_rotvec([0.0, 0.0, yaw])
        frames[:, :2] = ((frames[:, :3] - frames[0, :3]) @ yaw_r.as_matrix().T)[:, :2]
        frames[:, :2] += state.position[:2]
        frames[:, 3:7] = (yaw_r * Rotation.from_quat(frames[:, 3:7], scalar_first=True)).as_quat(
            scalar_first=True
        )

        t = np.arange(self.prepare_ticks, dtype=np.float64) / self.prepare_ticks
        blend = (t * t * (3.0 - 2.0 * t))[:, None]
        pos = state.position + blend * (frames[0, :3] - state.position)
        endpoints = np.stack((state.orientation_wxyz, frames[0, 3:7]))
        quat = Slerp([0, 1], Rotation.from_quat(endpoints, scalar_first=True))(blend[:, 0]).as_quat(
            scalar_first=True
        )
        delta_q = frames[0, 7:36] - state.joint_pos
        q_pos = state.joint_pos + blend * delta_q
        q_vel = (6 * t * (1 - t) / self.PREPARATION_S)[:, None] * delta_q
        prep = np.column_stack((pos, quat, q_pos, q_vel)).astype(np.float32)
        self.reference = np.concatenate((prep, frames))
        self.cursor = 0
        self.max_root_height_error = 0.0

    @property
    def done(self):
        return self.cursor == len(self.reference)

    def target(self, state: Observation) -> np.ndarray:
        if self.done:
            raise RuntimeError("No remaining dance reference frames")
        ref = self.reference[self.cursor]
        height_error = abs(float(state.position[2] - ref[2]))
        tilt_error = abs(float(rotation(state.orientation_wxyz)[2, 2] - rotation(ref[3:7])[2, 2]))
        self.max_root_height_error = max(self.max_root_height_error, height_error)
        if (height_error > self.meta["root_height_threshold"] or
                tilt_error > self.meta["root_orientation_threshold"]):
            raise RuntimeError(f"LoopMimic tracking left its limits at reference step {self.cursor}")
        idx = np.minimum(self.cursor + self.motion.offsets, len(self.reference) - 1)
        vector = state.vector()
        if not np.isfinite(vector).all():
            raise RuntimeError("Non-finite LoopMimic observation")
        actions = self.actor.run(["actions"], {"state": vector[None], "reference": self.reference[idx][None]})[0][0]
        target = self.default_q + self.scale * actions
        if not np.isfinite(target).all():
            raise RuntimeError("Non-finite LoopMimic action")
        self.cursor += 1
        return target


class TargetPublisher:
    """Publish candidate targets. The native SONIC process owns rt/lowcmd."""

    def __init__(self, meta):
        self.channel = ChannelPublisher("rt/loopmimic/dance_target", LowCmd_)
        self.channel.Init()
        self.message = unitree_hg_msg_dds__LowCmd_()
        pd = meta["tracking_pd"]
        for index, (kp, kd) in enumerate(zip(pd["kp"], pd["kd"], strict=True)):
            self.message.motor_cmd[index].kp = kp
            self.message.motor_cmd[index].kd = kd

    def send(self, token: int, targets: np.ndarray | None):
        self.message.reserve[0] = int(targets is not None)
        self.message.reserve[1] = token
        if targets is not None:
            for index, target in enumerate(targets):
                self.message.motor_cmd[index].q = float(target)
        self.channel.Write(self.message)

    def close(self):
        self.channel.Close()

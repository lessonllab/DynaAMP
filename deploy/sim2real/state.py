"""Build the actor's world-frame pelvis state from timestamped Unitree DDS samples."""

from dataclasses import dataclass
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, OdoState_
from unitree_sdk2py.utils.crc import CRC

from shared.dance import Observation, rotation


@dataclass(frozen=True)
class Estimate:
    observation: Observation
    low_age_s: float
    odom_age_s: float
    low_tick_ms: int


class StateSource:
    """Use pelvis odometry for xyz and velocity, and LowState for IMU and encoders."""

    def __init__(self, odom_type, odom_topic, odom_z_offset):
        self.lock = threading.Lock()
        self.crc = CRC()
        self.low = self.odom = None
        self.odom_type = odom_type
        self.z_offset = odom_z_offset
        self.low_channel = ChannelSubscriber("rt/lowstate", LowState_)
        self.odom_channel = ChannelSubscriber(odom_topic, OdoState_ if odom_type == "hg" else SportModeState_)
        self.low_channel.Init(self._on_low, 1)
        self.odom_channel.Init(self._on_odom, 1)

    def _on_low(self, msg):
        received = time.monotonic()
        if self.crc.Crc(msg) != msg.crc:
            return
        with self.lock:
            if self.low is None or msg.tick != self.low[1].tick:
                self.low = (received, msg)

    def _on_odom(self, msg):
        received = time.monotonic()
        if self.odom_type == "sport":
            if msg.error_code != 0:
                return
            stamp = (msg.stamp.sec, msg.stamp.nanosec)
        else:
            stamp = msg.tick
        with self.lock:
            if self.odom is None or stamp != self.odom[1]:
                self.odom = (received, stamp, msg)

    def estimate(self) -> Estimate:
        with self.lock:
            low = self.low
            odom = self.odom
        if low is None or odom is None:
            raise RuntimeError("Waiting for CRC-valid LowState and pelvis odometry")
        now = time.monotonic()
        t_low, low_state = low
        t_odom, _, odom_state = odom
        if now - t_low > 0.08 or now - t_odom > 0.12 or abs(t_low - t_odom) > 0.10:
            raise RuntimeError("Robot state or pelvis odometry is stale")
        if low_state.mode_pr != 0:
            raise RuntimeError("LoopMimic requires PR joint coordinates")
        p = np.asarray(odom_state.position, dtype=np.float64).copy()
        p[2] += self.z_offset
        quat = np.asarray(low_state.imu_state.quaternion, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or not 0.95 <= norm <= 1.05:
            raise RuntimeError("Invalid pelvis IMU quaternion")
        quat = quat / norm
        gyro = np.asarray(low_state.imu_state.gyroscope, dtype=np.float64)
        velocity = odom_state.linear_velocity if self.odom_type == "hg" else odom_state.velocity
        motors = low_state.motor_state[:29]
        q = np.asarray([motor.q for motor in motors])
        dq = np.asarray([motor.dq for motor in motors])
        state = Observation(p, quat, q, np.asarray(velocity), rotation(quat) @ gyro, dq)
        if not np.isfinite(state.vector()).all() or not 0.10 < p[2] < 1.30:
            raise RuntimeError("Invalid pelvis estimate. Verify odometry frame and height offset")
        return Estimate(state, now - t_low, now - t_odom, int(low_state.tick))

    def close(self):
        self.low_channel.Close()
        self.odom_channel.Close()

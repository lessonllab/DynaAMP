"""The gamepad-triggered LoopMimic provider for hardware and the DDS loopback demo."""

import argparse
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from shared.dance import DancePolicy, TargetPublisher, rotation
from shared.runtime import DEFAULT_BUNDLE, DEFAULT_MOTION
from sim2real.state import StateSource


class DanceStatus:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = ("idle", 0, 0.0)
        self.channel = ChannelSubscriber("rt/loopmimic/dance_status", String_)
        self.channel.Init(self._on_status, 1)

    def _on_status(self, sample):
        phase, sep, token = sample.data.partition(":")
        if sep and token.isdecimal() and phase in ("idle", "requested", "playing", "returning"):
            with self.lock:
                self.status = (phase, int(token), time.monotonic())

    def snapshot(self):
        with self.lock:
            return self.status

    def close(self):
        self.channel.Close()


def standing(estimate):
    state = estimate.observation
    return (0.63 <= state.position[2] <= 0.94 and rotation(state.orientation_wxyz)[2, 2] > 0.97
            and np.linalg.norm(state.linear_velocity_w) < 0.14
            and np.linalg.norm(state.angular_velocity_w) < 0.50)


def check_state(source, timeout):
    deadline = time.monotonic() + timeout
    samples = 0
    last_error = "No state samples"
    while time.monotonic() < deadline:
        try:
            estimate = source.estimate()
            if not standing(estimate):
                raise RuntimeError("Stand still on a level floor to check the pelvis estimate")
            samples += 1
            if samples >= 15:
                print(json.dumps({
                    "standing": True,
                    "pelvis_z_m": float(estimate.observation.position[2]),
                    "linear_speed_m_s": float(np.linalg.norm(estimate.observation.linear_velocity_w)),
                    "lowstate_age_s": estimate.low_age_s,
                    "odometry_age_s": estimate.odom_age_s,
                }))
                return
        except RuntimeError as exc:
            last_error = str(exc)
            samples = 0
        time.sleep(0.02)
    raise RuntimeError(f"State check failed: {last_error}")


def play(source, bundle, motion, stop, simulation_clock=False):
    actor = DancePolicy(bundle, motion)
    joint_limits = np.asarray(actor.meta["soft_joint_limits"])
    publisher = TargetPublisher(actor.meta)
    status = DanceStatus()
    active = False
    token = 0
    stand_since = None
    step_at_ms = 0
    print(f"LoopMimic ready: {len(actor.motion.frames)} frames, {len(actor.motion.frames)*actor.dt:.2f} s", flush=True)
    try:
        while not stop.is_set():
            now = time.monotonic()
            phase, request, timestamp = status.snapshot()
            if now - timestamp > 0.25:
                phase = "unavailable"
            try:
                estimate = source.estimate()
                stand_since = (stand_since or now) if standing(estimate) else None
            except RuntimeError as exc:
                estimate = None
                stand_since = None
                if active:
                    print(f"Returning to Idle: {exc}", flush=True)
            if active and (phase in ("idle", "returning", "unavailable") or estimate is None or request != token):
                print(f"Ended dance at {actor.cursor}/{len(actor.reference)} frames. SONIC state: {phase}.",
                      flush=True)
                publisher.send(token, None)
                active = False
            if phase == "requested" and request != token:
                token = request
                if estimate is None or stand_since is None or now - stand_since < 0.24:
                    print("Dance refused: stand still with a fresh pelvis estimate, then press A again.", flush=True)
                    publisher.send(token, None)
                else:
                    actor.start(estimate.observation)
                    active = True
                    step_at_ms = estimate.low_tick_ms
                    print(f"Playing full dance, request {token}", flush=True)
            if active and (not simulation_clock or estimate.low_tick_ms >= step_at_ms):
                if actor.done:
                    publisher.send(token, None)
                    active = False
                    print(f"Completed {len(actor.motion.frames)} reference frames", flush=True)
                else:
                    try:
                        targets = actor.target(estimate.observation)
                    except RuntimeError as exc:
                        publisher.send(token, None)
                        active = False
                        print(f"Returning to Idle: {exc}", flush=True)
                    else:
                        publisher.send(token, np.clip(targets, joint_limits[:, 0], joint_limits[:, 1]))
                        step_at_ms += round(1000 * actor.dt)
            period = 0.002 if simulation_clock else actor.dt
            stop.wait(max(0.0, now + period - time.monotonic()))
    finally:
        if active:
            publisher.send(token, None)
        status.close()
        publisher.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dance", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--odom-type", choices=("sport", "hg"), default="sport")
    parser.add_argument("--odom-topic")
    parser.add_argument("--odom-z-offset", type=float, default=0.0)
    parser.add_argument("--check-state", action="store_true")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--loopback-sim", action="store_true")
    args = parser.parse_args()
    if args.loopback_sim != (args.interface == "lo"):
        parser.error("Use --loopback-sim only with --interface lo")
    if not np.isfinite(args.odom_z_offset) or args.timeout <= 0:
        parser.error("Use finite odometry offset and positive timeout")
    topic = args.odom_topic or ("rt/odommodestate" if args.odom_type == "sport" else "rt/odostate")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    ChannelFactoryInitialize(0, args.interface)
    source = StateSource(args.odom_type, topic, args.odom_z_offset)
    try:
        if args.check_state:
            check_state(source, args.timeout)
        else:
            play(source, args.bundle, args.dance, stop, simulation_clock=args.loopback_sim)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"{exc}\n")
    finally:
        source.close()


if __name__ == "__main__":
    main()

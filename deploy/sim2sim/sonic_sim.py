"""Run the native SONIC deploy executable against NVIDIA's MuJoCo simulation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
import tty

import mujoco
import numpy as np

from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from unitree_sdk2py.utils.crc import CRC
from shared.dance import Motion, read_metadata
from shared.runtime import (
    DEFAULT_BUNDLE, DEFAULT_MOTION, G1_SCENE, SONIC_SOURCE, prepare_reference_directory, sonic_command,
)
from sonic_dance import Dance


HERE = Path(__file__).resolve().parent
STOCK_SCENE = SONIC_SOURCE / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
CONTROL_DT = 0.02
PHYSICS_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL,
)


class NativeDeploy:
    def __init__(self, output: Path, dance: bool, port: int, has_dex3: bool, gamepad: bool):
        refs = prepare_reference_directory(output)
        env = os.environ.copy()
        args = sonic_command("lo", refs, "gamepad" if gamepad else "keyboard", port) + [
            "--logs-dir", str(output / "native_logs"), "--enable-csv-logs",
            "--planner-motion-logfile", str(output / "planner_motion.csv"),
        ]
        if not gamepad:
            args.append("--disable-crc-check")
        env.pop("LOOPMIMIC_DANCE", None)
        if dance or gamepad:
            env["LOOPMIMIC_DANCE"] = "gamepad" if gamepad else "sim"
        if has_dex3:
            env.pop("SONIC_NO_DEX3", None)
        else:
            env["SONIC_NO_DEX3"] = "1"
        self.log = (output / "sonic.log").open("wb")
        try:
            self.proc = subprocess.Popen(args, cwd=output, env=env, stdin=subprocess.PIPE,
                                         stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close()
            raise

    def send(self, keys: str):
        if self.proc.poll() is not None:
            raise RuntimeError(f"SONIC exited with status {self.proc.returncode}")
        self.proc.stdin.write(keys.encode("ascii"))
        self.proc.stdin.flush()

    def close(self):
        try:
            if self.proc.poll() is None:
                self.send("o")
                self.proc.wait(timeout=3)
        except (TimeoutError, subprocess.TimeoutExpired, BrokenPipeError):
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        finally:
            self.proc.stdin.close()
            self.log.close()


class DeployLog:
    def __init__(self, path: Path):
        self.stream = path.open("r", errors="replace")
        self.markers = set()

    def update(self):
        for line in self.stream:
            if "motion name is planner_motion" in line:
                self.markers.add("planner")
            if "[LoopMimic] Entering dance" in line:
                self.markers.add("entering")
            if "[LoopMimic] Returning to SONIC Idle" in line:
                self.markers.add("returning")
            if "[LoopMimic] SONIC Idle resumed" in line:
                self.markers.add("resumed")


class SimulatedGamepad:
    """Feed the same 40-byte LowState remote packet that a G1 sends."""

    START, SELECT, F1, A, B, R2 = (1 << bit for bit in (2, 3, 6, 8, 9, 4))

    def __init__(self, bridge):
        self.bridge = bridge
        self.crc = CRC()
        self.release_at = 0.0
        self.axes = (0.0, 0.0, 0.0, 0.0)
        self.write = bridge.low_state_puber.Write
        bridge.low_state_puber.Write = self.publish
        self.set()

    def set(self, button=0, forward=0.0, hold_s=0.10):
        self.axes = (0.0, 0.0, 0.0, forward)
        self.release_at = time.monotonic() + hold_s
        self.button = button

    def publish(self, state):
        button = self.button if time.monotonic() < self.release_at else 0
        lx, rx, ry, ly = self.axes
        state.wireless_remote[:] = struct.pack("<2BH5f16x", 0xFE, 0xEF, button, lx, rx, ry, 0.0, ly)
        state.crc = self.crc.Crc(state)
        return self.write(state)


@contextmanager
def keyboard_input(enabled: bool):
    settings = None
    if enabled and sys.stdin.isatty():
        settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin)
    try:
        yield
    finally:
        if settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def foot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, bool]:
    contact = set()
    floor = model.geom("floor").id
    for point in data.contact:
        if point.efc_address < 0:
            continue
        a, b = point.geom
        if floor in (a, b):
            contact.add(int(model.geom_bodyid[b if a == floor else a]))
    return tuple(model.body(f"{side}_ankle_roll_link").id in contact for side in ("left", "right"))


def sample(env, phase: str) -> dict:
    state = env.mj_data
    w, x, y, z = state.qpos[3:7]
    rot_z = 1.0 - 2.0 * (x * x + y * y)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return {
        "time_s": state.time, "phase": phase, "qpos": state.qpos.copy(), "velocity": state.qvel[:6].copy(),
        "tilt_deg": np.degrees(np.arccos(np.clip(rot_z, -1, 1))), "yaw_rad": yaw,
        "pelvis_height_m": state.qpos[2],
        "contacts": foot_contacts(env.mj_model, state),
    }


def save_report(args, rows, outcome, dance, model, remote_report=None):
    if not rows:
        return
    np.savez_compressed(args.output / "trace.npz", **{
        name: np.array([row[name] for row in rows]) for name in rows[0]
    })
    report = {"outcome": outcome, "physics_s": rows[-1]["time_s"], "scene": str(args.scene),
              "actuated_joints": model.nu,
              "physics_dt_s": model.opt.timestep,
              "physics_integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
              "control_dt_s": CONTROL_DT, "phases": {}}
    if args.dance:
        report["dance_reference"] = str(args.dance.resolve())
        report["loopmimic_actor"] = str((args.bundle / "motion.onnx").resolve())
    if dance is not None:
        report["reference_frames"] = len(dance.motion.frames)
        report["dance_frames_played"] = max(0, dance.cursor - dance.prepare_ticks)
        report["pose_preparation_s"] = dance.PREPARATION_S
        if dance.cursor:
            report["dance_max_root_height_error_m"] = dance.max_root_height_error
    for phase in dict.fromkeys(row["phase"] for row in rows):
        seq = [row for row in rows if row["phase"] == phase]
        xy = np.array([row["qpos"][:2] for row in seq])
        speed = np.linalg.norm(np.asarray([row["velocity"][:2] for row in seq]), axis=1)
        azimuth = np.unwrap([row["yaw_rad"] for row in seq])
        report["phases"][phase] = {
            "start_s": seq[0]["time_s"], "end_s": seq[-1]["time_s"],
            "net_xy_m": float(np.linalg.norm(xy[-1] - xy[0])),
            "peak_xy_m": float(np.linalg.norm(xy - xy[0], axis=1).max()),
            "mean_speed_m_s": float(speed.mean()), "peak_speed_m_s": float(speed.max()),
            "yaw_change_deg": float(np.degrees(azimuth[-1] - azimuth[0])),
            "max_tilt_deg": float(max(row["tilt_deg"] for row in seq)),
            "min_pelvis_height_m": float(min(row["pelvis_height_m"] for row in seq)),
            "two_foot_frames": sum(all(row["contacts"]) for row in seq), "frames": len(seq),
        }
    results = (args.output / "results.json")
    phases = report["phases"]
    if args.gamepad_demo:
        report.update(remote_report or {})
        report["passed"] = (
            outcome == "completed"
            and all(name in phases for name in ("IDLE", "WALK", "STOP"))
            and report.get("reference_frames") is not None
            and report.get("dance_frames_played") == report["reference_frames"]
            and phases["WALK"]["net_xy_m"] > 0.6
            and phases["STOP"]["two_foot_frames"] >= 0.7 * phases["STOP"]["frames"]
            and np.max(np.linalg.norm([r["velocity"][:2] for r in rows[-25:]], axis=1)) < 0.2
        )
    elif args.native_demo or args.dance_demo:
        report["passed"] = (
            outcome == "completed"
            and all(name in phases for name in ("IDLE", "WALK", "RUN", "STOP"))
            and phases["WALK"]["net_xy_m"] >= 0.6
            and phases["RUN"]["mean_speed_m_s"] >= 1.2
            and phases["STOP"]["two_foot_frames"] >= 0.7 * phases["STOP"]["frames"]
            and np.max(np.linalg.norm([r["velocity"][:2] for r in rows[-25:]], axis=1)) < 0.2
            and (not args.dance_demo or report["dance_frames_played"] == report["reference_frames"])
        )
    results.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {results}", flush=True)
    return report


def run(args):
    os.chdir(SONIC_SOURCE)
    cfg = SimLoopConfig(sim_frequency=args.physics_hz, enable_onscreen=args.view).load_wbc_yaml()
    cfg.update(INTERFACE="lo", DOMAIN_ID=0, PRINT_SCENE_INFORMATION=False, ROBOT_SCENE=str(args.scene))
    if args.scene != STOCK_SCENE:
        cfg.update(NUM_HAND_MOTORS=0, NUM_HAND_JOINTS=0)
        limits = cfg["motor_effort_limit_list"]
        cfg["motor_effort_limit_list"] = limits[:22] + limits[29:36]
    existing_threads = set(threading.enumerate())
    simulator = BaseSimulator(cfg, onscreen=args.view)
    viewer_thread = next(
        (thread for thread in threading.enumerate()
         if thread not in existing_threads and thread.name.endswith("(_launch_internal)")),
        None,
    )
    env = simulator.sim_env
    if args.integrator == "implicitfast":
        env.mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    gamepad = args.gamepad_demo
    remote = SimulatedGamepad(env.unitree_bridge) if gamepad else None
    dance = Dance(args.bundle, args.dance, env) if args.dance and not gamepad else None
    native = NativeDeploy(args.output, dance is not None, args.port, env.num_hand_dof > 0, gamepad)
    log = DeployLog(args.output / "sonic.log")
    provider_log = (args.output / "loopmimic.log").open("w") if gamepad else None
    provider = None
    if gamepad:
        provider = subprocess.Popen([
            sys.executable, "-m", "sim2real.controller", "--interface", "lo", "--loopback-sim", "--odom-type", "hg",
            "--bundle", str(args.bundle), "--dance", str(args.dance),
        ], stdout=provider_log, stderr=subprocess.STDOUT)
    # Count failures instead of letting the upstream simulator reset a fall.
    env.check_fall = lambda: None
    flag = mujoco.mjtDisableBit.__members__.get("mjDSBL_AUTORESET")
    if flag is not None:
        env.mj_model.opt.disableflags |= flag
    step_sample = round(CONTROL_DT / env.sim_dt)
    assert np.isclose(step_sample * env.sim_dt, CONTROL_DT)
    phase = "INIT"
    rows = []
    start = time.monotonic()
    outcome = "interrupted"
    i = 0
    init_closing = 0.0
    phase_start = 0.0
    flight_started = False
    try:
        with keyboard_input(args.view):
            while native.proc.poll() is None:
                tick_start = time.monotonic()
                d = env.mj_data
                now = d.time
                elapsed = now - phase_start
                if args.view and not env.viewer.is_running():
                    break

                def enter(name, keys=""):
                    nonlocal phase, phase_start
                    phase, phase_start = name, now
                    if keys:
                        native.send(keys)
                    print(f"{now:7.2f}s  {name}", flush=True)

                if i % step_sample == 0:
                    log.update()
                    if provider and provider.poll() is not None:
                        raise RuntimeError("LoopMimic DDS provider exited. See loopmimic.log")

                if phase == "INIT":
                    if env.unitree_bridge.cmd_received() and init_closing == 0:
                        init_closing = now + 4.0
                    if init_closing and now >= init_closing:
                        if remote:
                            remote.set(remote.START)
                            enter("STARTING_CONTROL")
                        else:
                            enter("PLANNER_START", "]\n")
                    elif time.monotonic() - start > 60:
                        raise RuntimeError("SONIC did not publish the startup motor command")
                elif phase == "STARTING_CONTROL":
                    if elapsed >= 0.4:
                        remote.set(remote.F1)
                        enter("PLANNER_START")
                elif phase == "PLANNER_START":
                    if "planner" in log.markers:
                        if remote:
                            enter("LOWERING")
                        else:
                            env.elastic_band.enable = False
                            flight_started = True
                            enter("IDLE_BEFORE_DANCE" if args.dance else "IDLE")
                        if args.view:
                            print("Use this terminal for SONIC controls. Press Y while standing to dance.", flush=True)
                    elif elapsed > 8:
                        raise RuntimeError("SONIC did not enter native Planner mode")
                elif phase == "LOWERING":
                    # Lower the startup harness until both feet support the robot.
                    # This makes the loopback test independent of DDS startup timing.
                    env.elastic_band.length = -min(elapsed * 0.12, 0.24)
                    if elapsed >= 2.5 and all(foot_contacts(env.mj_model, d)):
                        env.elastic_band.enable = False
                        flight_started = True
                        enter("IDLE_BEFORE_DANCE")
                    elif elapsed > 4:
                        raise RuntimeError("The startup harness did not lower G1 onto both feet")
                elif phase == "IDLE_BEFORE_DANCE" and gamepad and elapsed >= 4.0:
                    remote.set(remote.A)
                    enter("DANCE_REQUEST")
                elif phase == "DANCE_REQUEST":
                    if "entering" in log.markers:
                        enter("DANCE")
                    elif elapsed > 3:
                        raise RuntimeError("Remote A did not start LoopMimic. See sonic.log and loopmimic.log")
                elif phase == "IDLE_BEFORE_DANCE" and args.dance_demo and elapsed >= 4.0:
                    log.markers.discard("resumed")
                    dance.start()
                    enter("DANCE")
                elif phase == "DANCE":
                    if gamepad:
                        if "returning" in log.markers:
                            enter("RETURN_TO_IDLE")
                        elif elapsed > 360:
                            raise RuntimeError("The remote dance did not finish")
                    elif dance.done:
                        dance.stop()
                        enter("RETURN_TO_IDLE")
                    elif i % step_sample == 0:
                        dance.step()
                elif phase == "RETURN_TO_IDLE":
                    if "resumed" in log.markers:
                        enter("IDLE")
                    elif elapsed > 3:
                        raise RuntimeError("SONIC did not take back control after the dance")
                elif gamepad:
                    if phase == "IDLE" and elapsed >= 4.0:
                        remote.set(remote.R2, forward=0.75, hold_s=0.20)
                        enter("WALK")
                    elif phase == "WALK" and elapsed >= 4.0:
                        remote.set(remote.B)
                        enter("STOP")
                    elif phase == "STOP" and elapsed >= 5.0:
                        outcome = "completed"
                        break
                elif args.native_demo or args.dance_demo:
                    if phase == "IDLE" and elapsed >= 4.0:
                        enter("WALK", "1000w")
                    elif phase == "WALK" and elapsed >= 3.0:
                        enter("FAST_WALK", "200w")
                    elif phase == "FAST_WALK" and elapsed >= 2.0:
                        enter("RUN", "3w")
                    elif phase == "RUN" and elapsed >= 3.0:
                        enter("DECELERATE", "200w")
                    elif phase == "DECELERATE" and elapsed >= 2.0:
                        enter("STOP", "r")
                    elif phase == "STOP" and elapsed >= 5.0:
                        outcome = "completed"
                        break
                    if phase in ("WALK", "FAST_WALK", "RUN", "DECELERATE") and i % round(0.10 / env.sim_dt) == 0:
                        native.send("w")
                if args.view and select.select([sys.stdin], [], [], 0)[0]:
                    key = os.read(sys.stdin.fileno(), 1).decode("ascii", errors="ignore")
                    if key in ("o", "O", "\x1b", ""):
                        break
                    if (key.lower() == "y" and dance and phase in ("IDLE_BEFORE_DANCE", "IDLE")
                            and np.linalg.norm(d.qvel[:3]) < 0.13 and sum(foot_contacts(env.mj_model, d)) == 2):
                        log.markers.discard("resumed")
                        native.send("r")
                        dance.start()
                        enter("DANCE")
                    elif key.lower() == "r" and phase == "DANCE":
                        dance.stop()
                        enter("RETURN_TO_IDLE")
                    elif phase in ("IDLE_BEFORE_DANCE", "IDLE"):
                        native.send(key)

                env.sim_step()
                for warning in PHYSICS_WARNINGS:
                    if d.warning[warning].number:
                        raise RuntimeError(f"{warning.name} at {now:.2f}s in {phase}")
                if (not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel))
                        or not np.isclose(d.time, now + env.sim_dt, rtol=0.0, atol=1e-8)):
                    raise RuntimeError(f"Invalid physics state or clock reset at {now:.2f}s in {phase}")
                # Dance.step uses reference-relative limits for poses such as a deep squat.
                if flight_started and phase != "DANCE" and d.qpos[2] < 0.45:
                    raise RuntimeError(f"Lost balance at {now:.2f}s in {phase}")
                if i % step_sample == 0:
                    rows.append(sample(env, phase))
                    if args.view:
                        env.viewer.sync()
                i += 1
                delay = tick_start + env.sim_dt - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            if native.proc.poll() is not None:
                outcome = f"SONIC exited with status {native.proc.returncode}"
                raise RuntimeError(outcome)
    except RuntimeError as error:
        outcome = str(error)
        raise
    finally:
        if remote and native.proc.poll() is None:
            remote.set(remote.SELECT)
            for _ in range(50):
                if native.proc.poll() is not None:
                    break
                remote.publish(env.unitree_bridge.low_state)
                time.sleep(0.002)
        native.close()
        remote_report = None
        if provider:
            if provider.poll() is None:
                provider.terminate()
            provider.wait(timeout=5)
            provider_log.close()
            frames = len(Motion(args.dance, read_metadata(args.bundle)).frames)
            completed = f"Completed {frames} reference frames" in (args.output / "loopmimic.log").read_text()
            remote_report = {"reference_frames": frames, "dance_frames_played": frames if completed else None,
                             "crc_validated": True, "input": "Unitree gamepad packets"}
        log.stream.close()
        simulator.close()
        # The passive viewer has to finish destroying GLFW before interpreter shutdown.
        if viewer_thread is not None:
            viewer_thread.join(timeout=5)
        report = save_report(args, rows, outcome, dance, env.mj_model, remote_report)
        if dance is not None:
            dance.close()
    if report and report.get("passed") is False:
        raise RuntimeError("Sim2Sim did not meet the checks in results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physics-hz", type=int, default=500, choices=(200, 500, 1000))
    parser.add_argument("--integrator", choices=("native", "implicitfast"), default="native")
    parser.add_argument("--scene", type=Path, default=G1_SCENE,
                        help="Unitree G1 29-DoF MJCF without Dex3, or the SONIC stock scene")
    parser.add_argument("--port", type=int, default=5587)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dance", type=Path, help="50 Hz LoopMimic dance reference NPZ")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--native-demo", action="store_true")
    mode.add_argument("--dance-demo", action="store_true")
    mode.add_argument("--gamepad-demo", action="store_true", help="Validate the Sim2Real controller over loopback DDS")
    mode.add_argument("--view", action="store_true")
    args = parser.parse_args()
    if args.native_demo and args.dance:
        parser.error("Use --dance-demo or --view with a dance reference")
    args.output = args.output.resolve()
    args.scene = args.scene.resolve()
    args.bundle = args.bundle.resolve()
    if not args.native_demo:
        args.dance = (args.dance or DEFAULT_MOTION).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock_dir = HERE / "sonic_validation"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / ".dds.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another SONIC Sim2Sim is already using the local DDS simulator")
        run(args)


if __name__ == "__main__":
    main()

"""Launch the SONIC Unitree controller and the LoopMimic dance provider."""

import argparse
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from shared.runtime import DEFAULT_BUNDLE, DEFAULT_MOTION, prepare_reference_directory, sonic_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True, help="Ethernet interface connected to the G1")
    parser.add_argument("--odom-type", choices=("sport", "hg"), default="sport")
    parser.add_argument("--odom-topic")
    parser.add_argument("--odom-z-offset", type=float, default=0)
    parser.add_argument("--check-state", action="store_true", help="Read DDS without starting a motor publisher")
    parser.add_argument("--dance", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--port", type=int, default=5587)
    parser.add_argument("--output", type=Path, default=Path("loopmimic_real_run"))
    args = parser.parse_args()
    if args.interface == "lo" or not Path("/sys/class/net", args.interface).is_dir():
        parser.error("Use the robot's Ethernet interface. Use Sim2Sim for loopback.")

    controller = [
        sys.executable, str(Path(__file__).with_name("controller.py")), "--interface", args.interface,
        "--bundle", str(args.bundle.resolve()), "--dance", str(args.dance.resolve()),
        "--odom-type", args.odom_type, "--odom-z-offset", str(args.odom_z_offset),
    ]
    if args.odom_topic:
        controller += ["--odom-topic", args.odom_topic]
    # Check the state source before the native process releases the built-in motion service.
    if subprocess.run([*controller, "--check-state"], check=False).returncode != 0:
        return 1
    if args.check_state:
        return 0

    from shared.dance import Motion, read_metadata
    meta = read_metadata(args.bundle)
    Motion(args.dance, meta)
    with (args.bundle / "motion.onnx").open("rb") as actor:
        if hashlib.file_digest(actor, "sha256").hexdigest() != meta["motion_onnx_sha256"]:
            parser.error("The policy ONNX does not match the deployment configuration")
    output = args.output.resolve()
    native = sonic_command(args.interface, prepare_reference_directory(output), "gamepad", args.port)
    env = dict(os.environ, SONIC_NO_DEX3="1", LOOPMIMIC_DANCE="gamepad")
    stop = False

    def mark_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, mark_stop)
    signal.signal(signal.SIGTERM, mark_stop)
    dance_proc = subprocess.Popen(controller, start_new_session=True)
    sonic = None
    try:
        sonic = subprocess.Popen(native, cwd=output, env=env, start_new_session=True)
        reported = False
        while sonic.poll() is None and not stop:
            if dance_proc.poll() is not None and not reported:
                print("Dance provider stopped. SONIC remains available. Press Select to exit.", flush=True)
                reported = True
            time.sleep(0.10)
        return sonic.returncode or 0
    finally:
        try:
            if sonic is not None and sonic.poll() is None:
                sonic.terminate()
                sonic.wait(timeout=10)
        finally:
            if dance_proc.poll() is None:
                dance_proc.terminate()
            dance_proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())

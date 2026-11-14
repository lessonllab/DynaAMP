"""Download and convert the complete default LAFAN1 G1 reference in place."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from shared.runtime import DEFAULT_BUNDLE


CSV_URL = (
    "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/"
    "resolve/main/g1/dance1_subject2.csv"
)
CSV_SHA256 = "3c2d37391ab08af18224f7a63ea55f1cd83480bdec779c320acb4d0b3062633e"


def convert(source: Path, output: Path):
    if hashlib.sha256(source.read_bytes()).hexdigest() != CSV_SHA256:
        raise ValueError(f"Unexpected LAFAN1 source SHA-256: {source}")
    meta = json.loads((DEFAULT_BUNDLE / "deployment.json").read_text())
    q = np.loadtxt(source, delimiter=",", dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 36 or not np.isfinite(q).all():
        raise ValueError("Expected position, xyzw quaternion and 29 G1 joint angles per row")
    source_time = np.arange(len(q)) / 30
    new_time = np.arange((len(q) - 1) * 50 // 30 + 1) / 50
    lower = (new_time * 30).astype(int)
    upper = np.minimum(lower + 1, len(q) - 1)
    alpha = (new_time * 30 - lower).astype(np.float32)[:, None]
    samples = q[lower] * (1 - alpha) + q[upper] * alpha
    rot = Slerp(source_time, Rotation.from_quat(q[:, 3:7]))(new_time)
    # Keep quaternion signs continuous, as in the input.
    quat = rot.as_quat(scalar_first=True)
    for i in range(1, len(quat)):
        if quat[i - 1] @ quat[i] < 0:
            quat[i] *= -1
    omega = (rot[2:] * rot[:-2].inv()).as_rotvec() / 0.04
    omega = np.concatenate((omega[:1], omega, omega[-1:])).astype(np.float32)
    pos, joints = samples[:, :3], samples[:, 7:]
    if len(pos) != 6574:
        raise ValueError(f"Unexpected playback length: {len(pos)}")
    with output.with_suffix(".npz.part").open("wb") as tmp:
        np.savez_compressed(
            tmp, fps=np.asarray([50]), joint_names=np.asarray(meta["joint_names"]),
            root_pos_w=pos, root_quat_w=quat.astype(np.float32),
            root_lin_vel_w=np.gradient(pos, 0.02, axis=0), root_ang_vel_w=omega,
            joint_pos=joints, joint_vel=np.gradient(joints, 0.02, axis=0),
        )
    output.with_suffix(".npz.part").replace(output)
    print(f"Complete LAFAN1 dance reference: {output} ({len(pos)} frames, 131.48 s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, help="Use a previously downloaded upstream CSV")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = args.csv or args.output.with_suffix(".csv")
    if args.csv is None and not source.is_file():
        part = source.with_suffix(".csv.part")
        with urlopen(CSV_URL, timeout=90) as response, part.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if hashlib.sha256(part.read_bytes()).hexdigest() != CSV_SHA256:
            raise ValueError("Downloaded LAFAN1 CSV does not match its expected SHA-256")
        part.replace(source)
    convert(source, args.output)


if __name__ == "__main__":
    main()

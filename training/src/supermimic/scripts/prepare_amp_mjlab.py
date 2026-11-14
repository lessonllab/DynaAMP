"""Lossless format adapter and FK validation for AMP_mjlab's 50 Hz WalkandRun."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import tyro
from mjlab.scripts.csv_to_npz import JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401


def main(
    source: Path,
    output: Path = Path(__file__).resolve().parents[3] / "data/amp_mjlab_walkrun_50hz",
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(source.glob("*.npz")):
        with np.load(path) as src:
            assert float(src["fps"].reshape(-1)[0]) == 50.0, path
            assert src["joint_pos"].shape[1] == len(JOINT_NAMES), path
            arrays = {"fps": src["fps"], "joint_names": np.asarray(JOINT_NAMES),
                      "joint_pos": src["joint_pos"], "joint_vel": src["joint_vel"]}
            for dst, key in [("root_pos_w", "body_pos_w"), ("root_quat_w", "body_quat_w"),
                             ("root_lin_vel_w", "body_lin_vel_w"), ("root_ang_vel_w", "body_ang_vel_w")]:
                arrays[dst] = src[key][:, 0]
            assert all(np.isfinite(v).all() for k, v in arrays.items() if k != "joint_names")
            assert np.max(np.abs(np.linalg.norm(arrays["root_quat_w"], axis=-1) - 1)) < 1e-4
            dest = output / path.name
            np.savez(dest, **arrays)
            records.append({"name": path.stem, "frames": len(arrays["joint_pos"]),
                            "source": str(path.resolve()), "output": str(dest.resolve()),
                            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "output_sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
    if not records:
        raise ValueError(f"No source motions at {source}")
    cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=True)
    cfg.scene.num_envs = 1
    cfg.commands["motion"].motion_file = str(output / "*.npz")
    env = ManagerBasedRlEnv(cfg, device="cpu")
    cmd = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    assert robot.body_names[0] == "pelvis"
    assert list(robot.joint_names) == list(JOINT_NAMES)
    body_ids = [robot.body_names.index(n) for n in cmd.cfg.body_names]
    offset = 0
    for record in records:
        frames = torch.arange(offset, offset + record["frames"])
        predicted = cmd.motion.body_state(frames)
        with np.load(record["source"]) as src:
            errors = {}
            for key, pred in zip(["body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"], predicted, strict=True):
                truth = torch.from_numpy(src[key][:, body_ids])
                delta = (pred - truth).abs()
                if key == "body_quat_w":
                    delta = torch.minimum(delta.amax(-1), (pred + truth).abs().amax(-1))
                errors[key] = float(delta.max())
            record["fk_max_abs_error"] = errors
            # Geometry confirms joint/body order without trusting absent metadata.
            assert errors["body_pos_w"] < 5e-4, record
            assert errors["body_quat_w"] < 5e-4, record
        offset += record["frames"]
    env.close()
    result = {"source_repo": "https://github.com/ccrpRepo/AMP_mjlab",
              "fps": 50, "clips": len(records), "frames": offset, "seconds": offset / 50,
              "conversion": "No resampling: retain all joint values; body index 0 supplies root pose/twist.",
              "joint_names": list(JOINT_NAMES), "body_names": list(robot.body_names),
              "fk_validated_bodies": list(cmd.cfg.body_names), "motions": records}
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["clips", "frames", "seconds"]}))


if __name__ == "__main__":
    tyro.cli(main)

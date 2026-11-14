"""Export a LoopMimic G1 EMA checkpoint into the shared deployment format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import onnx
import torch
from mjlab.scene import Scene
from mjlab.utils.lab_api.math import quat_apply_inverse
from torch import nn

from prepare_data import sha256
from supermimic.representations import (
    GOAL_FRAME_STRIDE,
    NUM_GOAL_FRAMES,
    assemble_anchor_window,
    create_representation,
)
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor
from supermimic.tasks.env_cfgs import gencoord_env_cfg
from supermimic.utils.gencoord_helpers import compute_gencoord_state


class MotionActor(nn.Module):
    """Motion encoder, shared decoder, and the exact GenCoord preprocessing."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.encoder = actor.encoders["motion"]
        self.decoder = actor.decoder
        self.obs_normalizer = actor.obs_normalizer
        self.goal_normalizer = actor.goal_normalizers["motion"]
        self.root_normalizer = actor.root_normalizer

    def forward(self, state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # State: root pose + q + world root velocities + dq.
        # Reference: [root position, root quaternion, q, dq] per future frame.
        root_pos, root_quat = state[:, :3], state[:, 3:7]
        _, orientation = assemble_anchor_window(
            root_pos, root_quat, reference[:, :, :3], reference[:, :, 3:7]
        )
        goal = torch.cat((reference[:, :, 7:].flatten(1), orientation), dim=1)
        latent = self.encoder(self.goal_normalizer(goal)).tanh()
        history = self.obs_normalizer(compute_gencoord_state(state))
        error = quat_apply_inverse(root_quat, reference[:, 0, :3] - root_pos)
        return self.decoder(torch.cat((latent, history, self.root_normalizer(error)), dim=1))


def g1_metadata() -> tuple[dict, int]:
    cfg = gencoord_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.scene.sensors = ()
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    joints = [model.joint(f"robot/{name}") for name in scene["robot"].joint_names]
    names = [joint.name.split("/")[-1] for joint in joints]
    if len(joints) != 29:
        raise ValueError("Expected a 29-DoF Unitree G1 scene")
    actuator_by_joint = {int(row[0]): i for i, row in enumerate(model.actuator_trnid)}
    motors = np.array([actuator_by_joint[joint.id] for joint in joints])
    q_addresses = [int(joint.qposadr[0]) for joint in joints]
    default_q = model.key("init_state").qpos[q_addresses].copy()
    limits = model.jnt_range[[joint.id for joint in joints]]
    half_range = (limits[:, 1] - limits[:, 0]) * 0.5
    center = limits.mean(axis=1)
    ratio = cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor
    soft_limits = np.stack((center - ratio * half_range, center + ratio * half_range), axis=1)
    scale_exprs = cfg.actions["joint_pos"].scale
    scale = np.array([
        next(value for pattern, value in scale_exprs.items() if re.fullmatch(pattern, name))
        for name in names
    ])
    kp = model.actuator_gainprm[motors, 0].copy()
    kd = -model.actuator_biasprm[motors, 2].copy()
    effort = model.actuator_forcerange[motors, 1].copy()
    np.testing.assert_allclose(scale, 0.25 * effort / kp)

    return {
        "joint_names": names,
        "default_q": default_q.tolist(),
        "soft_joint_limits": soft_limits.tolist(),
        "position_scale": scale.tolist(),
        "tracking_pd": {"kp": kp.tolist(), "kd": kd.tolist(), "effort": effort.tolist()},
        "root_height_threshold": cfg.terminations["anchor_pos"].params["threshold"],
        "root_orientation_threshold": cfg.terminations["anchor_ori"].params["threshold"],
        "control_dt": cfg.sim.mujoco.timestep * cfg.decimation,
    }, cfg.decimation


def export(checkpoint: Path, output: Path) -> None:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = load_runner_cfg(saved)
    if (cfg.representation, cfg.actor_obs, cfg.history_length,
            cfg.root_position_feedback) != ("gencoord", "rep_state", 1, True):
        raise ValueError("Expected the GenCoord state actor with root position feedback")
    if cfg.slot_mix != {"motion": 1.0, "velocity": 0.0}:
        raise ValueError("Expected a motion-only checkpoint")
    if "actor_ema" not in saved:
        raise ValueError("The checkpoint must contain EMA actor weights")

    meta, decimation = g1_metadata()
    default_q = torch.tensor(meta["default_q"], dtype=torch.float32)
    num_actions = default_q.numel()
    reference_dim = 7 + 2 * num_actions
    state_dim = reference_dim + 6
    representation = create_representation(
        "gencoord", state_dim, num_actions, meta["control_dt"], decimation,
        default_q, "rep_state",
    )
    spec = SimpleNamespace(
        representation=representation, num_actions=num_actions,
        num_goals=NUM_GOAL_FRAMES * (reference_dim + 2),
    )
    actor = build_actor(cfg, spec, "cpu").eval()
    actor.load_state_dict(saved["actor_ema"])

    state = torch.zeros(1, state_dim)
    reference = torch.zeros(1, NUM_GOAL_FRAMES, reference_dim)
    state[:, 3] = reference[:, :, 3] = 1.0
    output.mkdir(parents=True, exist_ok=True)
    model = output / "motion.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            MotionActor(actor).eval(), (state, reference), str(model),
            opset_version=18, dynamo=False, input_names=["state", "reference"],
            output_names=["actions"],
        )
    onnx.checker.check_model(str(model))
    meta.update(
        checkpoint_sha256=sha256(checkpoint),
        policy_updates=saved["policy_updates"],
        actor_weights="EMA",
        goal_frames=NUM_GOAL_FRAMES,
        goal_stride=GOAL_FRAME_STRIDE,
        motion_onnx_sha256=sha256(model),
    )
    (output / "deployment.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Exported {meta['policy_updates']} updates EMA to {model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.output)

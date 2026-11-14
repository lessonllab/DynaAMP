"""Measure actual task/style parameter gradients without changing a checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from tensordict import TensorDict

import supermimic.tasks  # noqa: F401
from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.learners.amp import DifferentiableMotionPrior
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
)
from supermimic.models import build_mlp
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.gradient_audit import audit_policy_objective
from supermimic.scripts.train_utils import setup_device


def main(
  checkpoint: Path,
  paired_data: Path,
  output: Path,
  diagnostic_batch_size: int | None = None,
  seed: int = 17,
  device: str = "cuda:0",
) -> None:
  setup_device(device)
  torch.manual_seed(seed)
  saved = torch.load(checkpoint, map_location=device, weights_only=False)
  pair = torch.load(paired_data, map_location=device, weights_only=False)
  cfg = load_runner_cfg(saved)
  batch_size = (
    cfg.policy.batch_size if diagnostic_batch_size is None else diagnostic_batch_size
  )
  if batch_size <= 0:
    raise ValueError("diagnostic_batch_size must be positive")
  if cfg.history_length != 1 or cfg.actor_obs != "rep_state" or not cfg.amp.enabled:
    raise ValueError("This audit requires the current rep_state/history=1 AMP model")
  env_cfg = load_env_cfg(saved["task_id"], play=True)
  # Only instantiate the robot to bind official kinematics/reward parameters.
  # No simulation rollout occurs. A diagnostic size override never changes cfg
  # or the saved training batch; by default we inspect all 4096 starts.
  env_cfg.scene.num_envs = 1
  env_cfg.sim.nconmax = max(256, env_cfg.sim.nconmax or 0)
  env_cfg.sim.njmax = max(1024, env_cfg.sim.njmax or 0)
  env_cfg.commands["motion"].motion_file = saved["motion_file"]
  env_cfg.commands["motion"].debug_vis = False
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=device))
  try:
    wm = build_world_model(cfg, env, device)
    wm.load_bundle(saved["world_model"])
    actor = build_actor(cfg, wm, device)
    actor.load_state_dict(saved["actor"])
    actor.activation_checkpointing = cfg.policy_activation_checkpointing
    wm.dynamics.activation_checkpointing = cfg.policy_activation_checkpointing
    wm_env = MultiTaskWorldModelEnv(
      wm.dynamics,
      wm.representation,
      env,
      batch_size,
      None,  # Explicit starts below: sampling from a replay buffer is unnecessary.
      actor.task_specs,
      cfg.history_length,
      cfg.termination_softness,
      device,
    )
    prior = DifferentiableMotionPrior.from_motion(
      wm_env.motion, cfg.amp, env.unwrapped.command_manager.get_term("motion").cfg.body_names
    )
    prior.load_state_dict(saved["amp"])
    prior.activation_checkpointing = cfg.policy_activation_checkpointing

    ids = torch.randint(len(pair["initial"]), (batch_size,), device=device)
    # State after the first real branch step has a known previous action.
    # This preserves the official action-rate penalty instead of inventing a_(-1).
    times = pair["motion_time"][ids]
    ends = wm_env.motion.motion_ends
    times = torch.minimum(
      times + 1, ends[torch.searchsorted(ends, times, right=True)] - 1
    )
    start = TensorDict(
      {
        "state": pair["real_base"][ids, 1],
        "prev_action": pair["base_actions"][ids, 0],
        "sim_ctx": pair["sim_ctx"][ids],
        "motion_time": times,
      },
      batch_size=[batch_size],
    )
    if isinstance(cfg.policy, ActorCriticPolicyLearnerCfg):
      critic_dim = wm_env.reset(start)["critic"].shape[-1]
      critic = build_mlp(critic_dim, cfg.critic_hidden_dims, 1).to(device)
      critic.load_state_dict(saved["critic"])
      learner = ActorCriticPolicyLearner(actor, critic, wm_env, cfg.policy, device)
    else:
      learner = BpttPolicyLearner(actor, wm_env, cfg.policy, device)
    learner.motion_prior = prior
    learner._rollout_actor = actor  # Eager diagnostic, outside the captured update.
    # Some representation helpers are individually torch.compiled. Their donated
    # backward buffers cannot be reused by separate task/style autograd.grad calls.
    # Force only this diagnostic to eager mode, retaining the training fast path.
    with torch.compiler.set_stance("force_eager"):
      result = audit_policy_objective(learner, start)
    result.update(
      checkpoint=str(checkpoint.resolve()),
      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      training_step=saved["step"],
      paired_data=str(paired_data.resolve()),
      available_real_starts=len(pair["initial"]),
      sampled_unique_starts=len(torch.unique(ids)),
      seed=seed,
      mlp_bf16=cfg.mlp_bf16,
      mlp_bf16_active=cfg.mlp_bf16 and torch.device(device).type == "cuda",
      activation_checkpointing=cfg.policy_activation_checkpointing,
      training_batch_size=cfg.policy.batch_size,
      device=device,
      saved_batch_preserved=result["batch_size"] == cfg.policy.batch_size,
      saved_horizon_preserved=result["horizon"] == cfg.policy.horizon,
      sampling="with replacement from recorded real branch states at t+1",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
      raise SystemExit(1)
  finally:
    env.close()


if __name__ == "__main__":
  tyro.cli(main)

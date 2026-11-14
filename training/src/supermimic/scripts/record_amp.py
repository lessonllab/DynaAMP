"""Evaluate and film an actual MuJoCo rollout from a local training checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.viewer.viewer_config import ViewerConfig
from PIL import Image, ImageDraw, ImageFont

import supermimic.tasks  # noqa: F401
from supermimic.learners.amp import DifferentiableMotionPrior
from supermimic.representations import extract_actor_goal
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.gradient_audit import audit_recorded_states
from supermimic.scripts.train_utils import setup_device
from supermimic.tasks.task_spec import build_actor_obs_td


def main(
  checkpoint: Path,
  output: Path,
  seconds: float = 30.0,
  start_frame: int = 0,
  num_envs: int = 1,
  render: bool = True,
  gradient_audit: bool = True,
  device: str = "cuda:0",
) -> None:
  setup_device(device)
  saved = torch.load(checkpoint, map_location=device, weights_only=False)
  runner_cfg = load_runner_cfg(saved)
  env_cfg = load_env_cfg(saved["task_id"], play=True)
  env_cfg.scene.num_envs = num_envs
  # Training amortizes contact capacity across many worlds. A single filmed
  # world needs enough candidate-pair capacity by itself, including near falls.
  env_cfg.sim.nconmax = max(256, env_cfg.sim.nconmax or 0)
  env_cfg.sim.njmax = max(1024, env_cfg.sim.njmax or 0)
  env_cfg.auto_reset = False
  env_cfg.episode_length_s = seconds + 2
  motion_cfg = env_cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  motion_cfg.motion_file = saved["motion_file"]
  motion_cfg.sampling_mode = "start"
  motion_cfg.debug_vis = False
  env_cfg.viewer = ViewerConfig(
    width=960,
    height=720,
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    entity_name="robot",
    distance=3.5,
    elevation=-15,
    azimuth=135,
    enable_shadows=False,
    enable_reflections=False,
  )
  base = ManagerBasedRlEnv(
    env_cfg, device=device, render_mode="rgb_array" if render else None
  )
  env = RslRlVecEnvWrapper(base)
  wm = build_world_model(runner_cfg, env, device)
  wm.load_bundle(saved["world_model"])
  actor = build_actor(runner_cfg, wm, device)
  actor.load_state_dict(saved["actor"])
  actor.eval()
  env.reset()
  command = base.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)
  command.reset_to_frame(torch.arange(num_envs, device=device), start_frame)
  base.scene.write_data_to_sim()
  base.sim.forward()
  base.sim.sense()
  command.update_relative_body_poses()
  base.observation_manager.compute(update_history=True)
  obs = env.get_observations()
  ctx = wm.sim_context.read(base)
  history = None
  output.parent.mkdir(parents=True, exist_ok=True)
  font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
  font = (
    ImageFont.truetype(str(font_path), 17)
    if font_path.exists()
    else ImageFont.load_default()
  )
  writer = (
    imageio.get_writer(str(output), fps=50, codec="libx264", quality=8)
    if render
    else None
  )
  reward_sum = torch.zeros(num_envs, device=device)
  first_failure = torch.full((num_envs,), -1, device=device, dtype=torch.long)
  root_errors, joint_errors = [], []
  steps = min(
    round(seconds / base.step_dt), command.motion.time_step_total - start_frame - 1
  )
  actual_steps = 0
  audit_states, audit_times = [], []
  try:
    with torch.inference_mode():
      for step in range(steps):
        if gradient_audit and step % 50 == 0 and len(audit_states) < 16:
          audit_states.append(obs["full_state"][:1].clone())
          audit_times.append(command.time_steps[:1].clone())
        rep = wm.representation.represent(obs["full_state"])
        sensor = wm.representation.compute_actor_sensor(
          rep,
          ctx,
          sim_actor_obs=obs["actor"] if runner_cfg.actor_obs == "proprio" else None,
        )
        history = (
          sensor[:, None].repeat(1, runner_cfg.history_length, 1)
          if history is None
          else torch.cat([history[:, 1:], sensor[:, None]], 1)
        )
        actor_obs = build_actor_obs_td(
          {"motion": extract_actor_goal(obs["goal"])}, history
        )
        obs, reward, done, _ = env.step(actor(actor_obs))
        failed = base.termination_manager.terminated
        first_failure[(first_failure < 0) & failed] = step
        reward_sum += reward
        root_errors.append(command.metrics["error_anchor_pos"].mean().item())
        joint_errors.append(command.metrics["error_joint_pos"].mean().item())
        actual_steps = step + 1
        if writer is not None:
          pixels = base.render()
          assert pixels is not None
          frame = Image.fromarray(pixels)
          draw = ImageDraw.Draw(frame)
          draw.rectangle((0, 0, 960, 72), fill=(20, 24, 30))
          draw.text(
            (18, 12),
            "Official SuperMimic + differentiable AMP | MuJoCo physics",
            fill="white",
            font=font,
          )
          draw.text(
            (18, 38),
            f"step {saved['step']} | t={actual_steps * base.step_dt:.2f}s | reset=0 | termination={bool(failed[0])}",
            fill="white",
            font=font,
          )
          writer.append_data(np.asarray(frame))
        if done.any():
          # End the take at termination: never hide a fall behind an automatic reset.
          break
    if gradient_audit and runner_cfg.amp.enabled:
      if runner_cfg.history_length != 1 or runner_cfg.actor_obs != "rep_state":
        raise ValueError(
          "Recorded gradient audit currently uses official rep_state/history=1"
        )
      prior = DifferentiableMotionPrior.from_motion(command.motion, runner_cfg.amp, command.cfg.body_names)
      prior.load_state_dict(saved["amp"])
      prior.activation_checkpointing = runner_cfg.policy_activation_checkpointing
      actor.activation_checkpointing = runner_cfg.policy_activation_checkpointing
      wm.dynamics.activation_checkpointing = runner_cfg.policy_activation_checkpointing
      audit = audit_recorded_states(
        actor,
        wm,
        prior,
        command.motion,
        audit_states,
        audit_times,
        ctx[:1],
        steps=runner_cfg.policy.horizon,
      )
      output.with_suffix(".gradient.json").write_text(json.dumps(audit, indent=2))
      print(f"[AMP] Full-horizon recorded-state gradient audit: {audit}")
  finally:
    if writer is not None:
      writer.close()
    env.close()
  result = {
    "checkpoint": str(checkpoint.resolve()),
    "training_step": saved["step"],
    "requested_seconds": seconds,
    "actual_seconds": actual_steps * base.step_dt,
    "start_frame": start_frame,
    "auto_reset": False,
    "physics_execution": True,
    "nconmax": env_cfg.sim.nconmax,
    "njmax": env_cfg.sim.njmax,
    "first_failure_steps": first_failure.cpu().tolist(),
    "reward_sum": reward_sum.cpu().tolist(),
    "mean_anchor_position_error_m": sum(root_errors) / max(1, len(root_errors)),
    "mean_joint_error_l2_rad": sum(joint_errors) / max(1, len(joint_errors)),
  }
  output.with_suffix(".json").write_text(json.dumps(result, indent=2))
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  tyro.cli(main)

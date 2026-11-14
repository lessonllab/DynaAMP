"""Matched-task, real-transition-budget training using pinned AMP_mjlab PPO.

Both arms start from scratch. This adapts upstream AMP+PPO to SuperMimic's
tracking task; it is not AMP_mjlab's native velocity-command/recovery benchmark.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.representations import create_representation, extract_actor_goal
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.sim_context import SimContext
from supermimic.utils.gencoord_helpers import compute_gencoord_state
from supermimic.vendor.amp_mjlab.algorithms import AMPPPO
from supermimic.vendor.amp_mjlab.modules import ActorCritic, Discriminator, EmpiricalNormalization
from supermimic.vendor.amp_mjlab.utils import Normalizer

TASK = "SuperMimic-GenCoord-G1"
DATA = Path(__file__).resolve().parents[3] / "data/amp_mjlab_walkrun_50hz"
PPO_SETTINGS = dict(num_learning_epochs=5, num_mini_batches=4, clip_param=0.2,
                    gamma=0.99, lam=0.95, value_loss_coef=1.0, entropy_coef=0.005,
                    learning_rate=1e-3, max_grad_norm=1.0,
                    use_clipped_value_loss=True, schedule="adaptive", desired_kl=0.01)


def json_value(value):
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def common_env_cfg(num_envs: int, seed: int, play: bool = False, data: Path = DATA):
    cfg = load_env_cfg(TASK, play=play)
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.commands["motion"].motion_file = str(data / "*.npz")
    cfg.commands["motion"].sampling_mode = "uniform"
    return cfg


class SharedObservations:
    """Same state, goals and privileged simulator context as the BPTT arm."""
    def __init__(self, env):
        self.env = env
        obs = env.get_observations()
        base = env.unwrapped
        self.context = SimContext.from_env(base)
        self.rep = create_representation(
            "gencoord", obs["full_state"].shape[-1], env.num_actions,
            base.step_dt, base.cfg.decimation,
            base.scene["robot"].data.default_joint_pos[0], actor_obs_mode="rep_state",
        ).to(env.device)
        self.rep.bind(base, self.context)

    def __call__(self, obs):
        state = self.rep.represent(obs["full_state"])
        actor = torch.cat([state, extract_actor_goal(obs["goal"])], -1)
        critic = torch.cat([state, obs["goal"], self.context.read(self.env.unwrapped)], -1)
        return actor, critic, state


class ExpertPairs:
    """Uniform valid adjacent frames, identically scoped to our AMP expert data."""
    def __init__(self, motion, encoder=None):
        self.frames = compute_gencoord_state(torch.cat([
            motion.root_pos_w, motion.root_quat_w, motion.joint_pos,
            motion.root_lin_vel_w, motion.root_ang_vel_w, motion.joint_vel,
        ], -1))
        idx = torch.arange(len(self.frames) - 1, device=self.frames.device)
        self.starts = idx[~torch.isin(idx, motion.motion_ends - 1)]
        if encoder is not None:
            self.frames = encoder.expert_frames(self.frames)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            idx = self.starts[torch.randint(len(self.starts), (mini_batch_size,), device=self.frames.device)]
            yield self.frames[idx], self.frames[idx + 1]


def train_ppo(env, output: Path, budget: int, checkpoints: tuple[int, ...], seed: int,
              body195: bool = False, resume: dict | None = None):
    features = SharedObservations(env)
    actor_obs, critic_obs, amp_obs = features(env.get_observations())
    encoder = None
    command = env.unwrapped.command_manager.get_term("motion")
    if body195:
        from supermimic.learners.body_amp_features import BodyAMPFeatures
        encoder = BodyAMPFeatures(command.motion, command.cfg.body_names)
        amp_obs = encoder(amp_obs)
    amp_dim = amp_obs.shape[-1]
    actor_dim, critic_dim = actor_obs.shape[-1], critic_obs.shape[-1]
    policy = ActorCritic(actor_dim, critic_dim, env.num_actions,
                        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                        activation="elu", init_noise_std=1.0).to(env.device)
    obs_norm = EmpiricalNormalization([actor_dim], until=1e8).to(env.device)
    critic_norm = EmpiricalNormalization([critic_dim], until=1e8).to(env.device)
    discriminator = Discriminator(amp_dim * 2, 0.1, [1024, 512, 256], env.device, task_reward_lerp=0.75)
    amp_norm = Normalizer(amp_dim)
    expert = ExpertPairs(command.motion, encoder)
    algorithm = AMPPPO(policy, discriminator, expert, amp_norm,
                       amp_replay_buffer_size=100000,
                       min_std=torch.full((env.num_actions,), 0.05, device=env.device),
                       device=env.device, **PPO_SETTINGS)
    horizon = 24
    transitions_per_iteration = env.num_envs * horizon
    if budget % transitions_per_iteration:
        raise ValueError("PPO budget must contain whole 24-step rollouts")
    algorithm.init_storage("rl", env.num_envs, horizon, [actor_dim], [critic_dim], [env.num_actions])
    start_iteration = 0
    prior_elapsed = 0.0
    total = 0
    episode_length = torch.zeros(env.num_envs, device=env.device)
    episode_return = torch.zeros_like(episode_length)
    cumulative_failures = 0
    skipped = 0
    if resume is not None:
        if resume["ppo_settings"] != PPO_SETTINGS or resume["amp_dim"] != amp_dim:
            raise ValueError("PPO settings or AMP features differ from the checkpoint")
        start_iteration = resume["iteration"]
        total = resume["real_sim_transitions"]
        if total != start_iteration * transitions_per_iteration or total >= budget:
            raise ValueError("PPO resume iteration and transition counts do not match the target")
        policy.load_state_dict(resume["policy"])
        discriminator.load_state_dict(resume["discriminator"])
        algorithm.optimizer.load_state_dict(resume["optimizer"])
        algorithm.learning_rate = resume["learning_rate"]
        obs_norm.load_state_dict(resume["obs_normalizer"])
        critic_norm.load_state_dict(resume["critic_normalizer"])
        vars(amp_norm).update(resume["amp_normalizer"])
        prior_elapsed = resume["elapsed_s"]
        skipped = resume["skipped_non_finite_batches_total"]
        cumulative_failures = resume.get("cumulative_failures", 0)
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state_all(resume["cuda_rng"])
        print(f"[INFO] Restored PPO at update {start_iteration}; rebuilding rollout and AMP replay", flush=True)
    start = time.monotonic()
    with torch.no_grad():
        actor_obs, critic_obs = obs_norm(actor_obs), critic_norm(critic_obs)

    def save(iteration):
        state = {"method": "amp_mjlab_ppo_matched_tracking", "task_id": TASK,
                 "motion_file": command.cfg.motion_file, "seed": seed, "iteration": iteration,
                 "env_variant": "branch_body195" if body195 else "gencoord68", "amp_dim": amp_dim,
                 "real_sim_transitions": total, "num_envs": env.num_envs,
                 "policy": policy.state_dict(), "discriminator": discriminator.state_dict(),
                 "optimizer": algorithm.optimizer.state_dict(), "learning_rate": algorithm.learning_rate,
                 "obs_normalizer": obs_norm.state_dict(), "critic_normalizer": critic_norm.state_dict(),
                 "amp_normalizer": vars(amp_norm), "actor_dim": actor_dim, "critic_dim": critic_dim,
                 "ppo_settings": PPO_SETTINGS, "elapsed_s": prior_elapsed + time.monotonic() - start,
                 "cumulative_failures": cumulative_failures,
                 "skipped_non_finite_batches_total": skipped,
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        torch.save(state, output / "training_state.tmp")
        (output / "training_state.tmp").replace(output / "training_state.pt")
        if total in checkpoints:
            import shutil
            shutil.copyfile(output / "training_state.pt", output / f"budget_{total:09d}.pt")
        (output / "progress.json").write_text(json.dumps({k: state[k] for k in [
            "method", "iteration", "real_sim_transitions", "elapsed_s", "skipped_non_finite_batches_total"]}, indent=2))

    for iteration in range(start_iteration + 1, budget // transitions_per_iteration + 1):
        completed_lengths, completed_returns, task_rewards = [], [], []
        with torch.no_grad():
            for _ in range(horizon):
                actions = algorithm.act(actor_obs, critic_obs, amp_obs)
                obs, task_reward, done, info = env.step(actions)
                raw_actor, raw_critic, next_amp = features(obs)
                if encoder is not None:
                    next_amp = encoder(next_amp)
                actor_obs, critic_obs = obs_norm(raw_actor), critic_norm(raw_critic)
                # Preserve AMP_mjlab runner's terminal-pair convention (self-pair),
                # never pair an old episode with an unrelated reset frame.
                terminal_amp = torch.where(done.bool()[:, None], amp_obs, next_amp)
                reward, _ = discriminator.predict_amp_reward(amp_obs, terminal_amp, task_reward, amp_norm)
                algorithm.process_env_step(reward, done, info, terminal_amp)
                amp_obs = next_amp
                total += env.num_envs
                episode_length += 1
                episode_return += task_reward
                task_rewards.append(task_reward.mean())
                if done.any():
                    mask = done.bool()
                    completed_lengths.extend(episode_length[mask].cpu().tolist())
                    completed_returns.extend(episode_return[mask].cpu().tolist())
                    episode_length[mask] = 0
                    episode_return[mask] = 0
                cumulative_failures += int(env.unwrapped.termination_manager.terminated.sum())
            algorithm.compute_returns(critic_obs)
        metrics = algorithm.update()
        skipped += int(metrics["skipped_non_finite_batches"])
        if not all(np.isfinite(x) for x in metrics.values()):
            raise FloatingPointError(metrics)
        row = {"iteration": iteration, "real_sim_transitions": total,
               "elapsed_s": prior_elapsed + time.monotonic() - start, **metrics,
               "mean_task_reward_per_step": float(torch.stack(task_rewards).mean()),
               "mean_completed_episode_steps": float(np.mean(completed_lengths)) if completed_lengths else None,
               "mean_completed_episode_task_return": float(np.mean(completed_returns)) if completed_returns else None,
               "cumulative_failures": cumulative_failures,
               "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30}
        with (output / "training_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if total in checkpoints or total == budget or iteration % 100 == 0:
            save(iteration)
    env.close()


def main(method: Literal["ppo", "bptt"], output: Path, budget: int = 24_576_000,
         num_envs: int = 2048, seed: int = 0,
         checkpoints: tuple[int, ...] = (4_915_200, 12_288_000, 24_576_000),
         device: str = "cuda:0"):
    if (output / "training_metrics.jsonl").exists():
        raise FileExistsError(f"Refusing to mix a fresh run into {output}")
    output.mkdir(parents=True, exist_ok=True)
    setup_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = common_env_cfg(num_envs, seed)
    env_json = json.dumps(asdict(cfg), sort_keys=True, default=json_value)
    data_manifest = DATA / "manifest.json"
    checkpoints = tuple(x for x in checkpoints if x <= budget)
    contract = {"method": method, "seed": seed, "num_envs": num_envs,
                "budget": budget, "checkpoints": checkpoints, "from_scratch": True,
                "task_id": TASK, "physics_dt": cfg.sim.mujoco.timestep,
                "control_decimation": cfg.decimation, "data_manifest": str(data_manifest),
                "data_manifest_sha256": hashlib.sha256(data_manifest.read_bytes()).hexdigest(),
                "env_cfg_sha256": hashlib.sha256(env_json.encode()).hexdigest(),
                "actor_observation": "GenCoord68 + official motion actor goal; history=1",
                "critic_observation": "GenCoord68 + official full motion goal + sim context",
                "amp_observation": "GenCoord68 consecutive pair, 136 inputs",
                "baseline_adaptations": ["Native velocity/recovery task replaced by shared reference tracking",
                    "Native 195-D body AMP features replaced by shared 68-D GenCoord",
                    "Expert pair sampling uniform over valid adjacent frames, as in BPTT arm",
                    "Current shared mjlab/MuJoCo runtime; unchanged upstream PPO and discriminator mathematics"],
                "interaction_unit": "One real environment transition at 50 Hz; all five feedback branches included",
                "evaluation_interactions": "Reported separately, never used for training"}
    (output / "contract.json").write_text(json.dumps(contract, indent=2))
    (output / "env_config.json").write_text(env_json)
    env = create_env(cfg, num_envs, device)
    if method == "ppo":
        train_ppo(env, output, budget, checkpoints, seed)
    else:
        runner_cfg = SuperMimicRunnerCfg()
        runner_cfg.max_iterations = budget // num_envs
        runner_cfg.max_sim_transitions = budget
        runner_cfg.budget_checkpoints = checkpoints
        runner_cfg.amp.enabled = True
        runner_cfg.amp_feedback.enabled = True
        runner_cfg.policy_activation_checkpointing = True
        runner_cfg.mlp_bf16 = True
        runner_cfg.sampling_mode = "uniform"
        runner_cfg.run_name = f"matched_walkrun_bptt_seed{seed}"

        class MatchedRunner(SuperMimicRunner):
            def _create_log_dir(self, *extra_parts):
                return output

        runner = MatchedRunner(env, runner_cfg, TASK, device=device)
        runner.run()


if __name__ == "__main__":
    tyro.cli(main)

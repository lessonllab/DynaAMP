"""Same-task comparison with JonnyDing's actual `supermimic` branch.

The vendored optimizer, model, state increment, AMP and critic update are intact.
Only task observations/reward/termination and physics collection are adapted to
the shared SuperMimic tracking task. This is not the upstream native benchmark.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg

from supermimic.learners.body_amp_features import AMP_BODIES, BodyAMPFeatures, encode_bodies
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.compare_amp import DATA, TASK, SharedObservations, common_env_cfg, json_value
from supermimic.scripts.train_utils import setup_device
from supermimic.utils.gencoord_helpers import compute_gencoord_state
from supermimic.vendor.jonny_wmrl.adapters.mjlab import read_physics
from supermimic.vendor.jonny_wmrl.core.batching import accumulate_imagination, learning_rate, world_model_update
from supermimic.vendor.jonny_wmrl.core.checkpoint import atomic_save, restore_rng, rng_state
from supermimic.vendor.jonny_wmrl.core.imagination import ValueModel, freeze_parameters, update_critic
from supermimic.vendor.jonny_wmrl.core.normalization import TensorAMPNormalizer
from supermimic.vendor.jonny_wmrl.core.policy import create_policy
from supermimic.vendor.jonny_wmrl.core.replay import SequenceReplay
from supermimic.vendor.jonny_wmrl.core.state import pack, unpack
from supermimic.vendor.jonny_wmrl.core.world_model import DYNAMICS_KEYS, WorldModel, rollout_loss
from supermimic.vendor.jonny_wmrl.mdp.amp import DifferentiableAMP, DiscriminatorUpdates, ExpertPairs
from supermimic.vendor.jonny_wmrl.mdp.kinematics import RobotKinematics

UPSTREAM_SHA = "596dfe37e0f75769b635386cd589960cce7faaa2"
START_KEYS = ["state", "context", "motion_time", "previous_action"]


def branch_env_cfg(num_envs, seed, play=False, data=DATA):
    cfg = common_env_cfg(num_envs, seed, play, data=data)
    # Sensor-only addition for the upstream model's supervised contact head.
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True),)
    return cfg


def to_official(state):
    return torch.cat((state[..., :7], state[..., 13:42], state[..., 7:13], state[..., 42:]), -1)


class CaptureTrackingEnv(ManagerBasedRlEnv):
    """Capture all targets at reward time, before automatic reset/command advance."""
    def step(self, action):
        original = self.reward_manager.compute

        def capture(*args, **kwargs):
            reward = original(*args, **kwargs)
            self.transition_next_state = pack(read_physics(self))
            self.transition_feet = (self.scene["feet_ground_contact"].data.found > 0).reshape(self.num_envs, 2).clone()
            collision = self.scene["self_collision"].data
            if collision.force_history is not None:
                self.transition_collision = (collision.force_history.norm(dim=-1) > 10).any(dim=1).sum(-1).float()
            else:
                self.transition_collision = collision.found.reshape(self.num_envs, -1).sum(-1).float().clone()
            return reward

        self.reward_manager.compute = capture
        try:
            return super().step(action)
        finally:
            self.reward_manager.compute = original


class BranchFeatures:
    def __init__(self, command):
        self.encoder = BodyAMPFeatures(command.motion, command.cfg.body_names)

    def __call__(self, state):
        return self.encoder(compute_gencoord_state(to_official(state)))


class TrackingCollection:
    def __init__(self, env, shared, policy, norm, replay, train_ids):
        self.env, self.shared, self.policy, self.norm = env, shared, policy, norm
        self.replay, self.train_ids = replay, train_ids
        self.world_model = None
        self.episodes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.total_steps = self.failures = 0
        self.action_noise = 0.1
        self.last_reward = 0.0

    @torch.no_grad()
    def collect(self, count):
        base = self.env.unwrapped
        self.policy.eval(); self.norm.eval()
        for _ in range(count):
            obs = self.env.get_observations()
            actor = self.shared(obs)[0]
            self.norm.update(actor[self.train_ids])
            action = self.policy.act_inference(self.norm(actor))
            action = action + self.action_noise * torch.randn_like(action)
            state = pack(read_physics(base))
            context = self.shared.context.read(base).clone()
            previous = base.action_manager.action.clone()
            frame = base.command_manager.get_term("motion").time_steps.clone()
            _, reward, _, _ = self.env.step(action)
            terminated = base.termination_manager.terminated.clone()
            truncated = base.termination_manager.time_outs.clone()
            ns = base.transition_next_state
            if self.world_model is not None:
                ids = self.train_ids
                self.world_model.update_stats(state[ids], ns[ids], action[ids], context[ids])
            self.replay.add_values(dict(
                state=state, next_state=ns, action=action, context=context,
                terminated=terminated, truncated=truncated, episode_id=self.episodes.clone(),
                dynamics_valid=torch.ones_like(terminated), reward=reward.clone(),
                motion_time=frame, previous_action=previous,
                **{"labels/feet_in_contact": base.transition_feet,
                   "labels/self_collision_raw": base.transition_collision}))
            self.episodes += (terminated | truncated).long()
            self.total_steps += 1
            self.failures += int(terminated.sum())
            self.last_reward = float(reward.mean())


class TrackingImagination:
    """Native FastP4 objective with shared tracking task heads (32 connected steps)."""
    def __init__(self, wm, policy, norm, critic, amp, features, shared, command):
        self.wm, self.actor, self.norm, self.critic = wm, policy.actor, norm, critic
        self.amp, self.features, self.rep, self.motion = amp, features, shared.rep, command.motion
        self.horizon, self.gamma = 32, 0.99
        self.wm.mlp_dtype = self.critic.mlp_dtype = torch.bfloat16
        self.amp.normalizer = TensorAMPNormalizer(amp.normalizer, amp.device)
        self.anchor = command.motion_anchor_body_index
        cfg = shared.env.unwrapped.cfg
        self.z_threshold = cfg.terminations["anchor_pos"].params["threshold"]
        self.tilt_threshold = cfg.terminations["anchor_ori"].params["threshold"]
        # Constant reference lookup; all predicted-state FK stays differentiable.
        with torch.no_grad():
            chunks = [self.motion.body_state(ids) for ids in
                      torch.arange(self.motion.time_step_total, device=amp.device).split(8192)]
            self.reference = tuple(torch.cat(parts, 0) for parts in zip(*chunks))
        self.step = torch.compile(self._step, fullgraph=True)
        self.reward = torch.compile(self._reward, fullgraph=True)
        self.values = torch.compile(self._value_inputs, fullgraph=True)
        self.wm_loss = torch.compile(self._wm_loss, fullgraph=True)
        self.audit = False

    def _value_inputs(self, state, frame, context):
        full = to_official(state)
        rep = compute_gencoord_state(full)
        goal, actor_goal = self.rep.compute_wm_goal(self.motion, frame, state[:, :7])
        return torch.cat((rep, actor_goal), -1), torch.cat((rep, goal, context), -1)

    def _step(self, state, frame, context):
        obs, critic_input = self._value_inputs(state, frame, context)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = self.actor(self.norm(obs)).float()
        ns, _ = self.wm(state, action, context)
        ends = self.motion.motion_ends
        nxt = frame + 1
        idx = torch.searchsorted(ends, nxt).clamp(max=ends.numel() - 1)
        nxt = torch.where(ends[idx] == nxt, frame, nxt)
        return ns, action, nxt, critic_input

    def _reward(self, states, actions, previous, frames):
        horizon, batch = actions.shape[:2]
        flat = lambda x: x.flatten(0, 1)
        following = flat(states[1:])
        rep = compute_gencoord_state(to_official(following))
        references = tuple(x[flat(frames)] for x in self.reference)
        task = self.rep.compute_wm_reward_from_reference(
            rep, following[:, :7], self.rep.joint_state(rep), flat(actions), flat(previous), *references)
        features = self.features(flat(states)).reshape(horizon + 1, batch, 195)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rewards, scores = self.amp.reward(flat(features[:-1]), flat(features[1:]), task)
        pos, quat = references[:2]
        height = (pos[:, self.anchor, 2] - following[:, 2]).abs()
        tilt = (2 * (quat[:, self.anchor, 1:3].square().sum(-1) - following[:, 4:6].square().sum(-1))).abs()
        termination = torch.sigmoid((torch.maximum(height / self.z_threshold, tilt / self.tilt_threshold) - 1) / .05)
        return rewards.reshape(horizon, batch), scores.reshape(horizon, batch), (1 - termination).reshape(horizon, batch)

    def _wm_loss(self, window):
        return rollout_loss(self.wm, window)

    def __call__(self, starts):
        state, context, frame, previous = (starts[k][:, 0] for k in START_KEYS)
        states, actions, previous_actions, frames, critic_inputs = [state], [], [], [], []
        first_action = None
        for _ in range(self.horizon):
            if self.audit and first_action is None:
                # AOTAutograd exposes fused (action, next_state) as sibling
                # outputs. Asking d(next_state)/d(returned_action) then misses
                # the internal edge. Expose the identical first-step action as
                # an actual WM input for this intermediate-gradient audit.
                obs, critic_input = self.values(state, frame, context)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    action = self.actor(self.norm(obs)).float()
                ns, _ = self.wm(state, action, context)
                ends = self.motion.motion_ends
                nxt = frame + 1
                idx = torch.searchsorted(ends, nxt).clamp(max=ends.numel() - 1)
                nxt = torch.where(ends[idx] == nxt, frame, nxt)
            else:
                ns, action, nxt, critic_input = self.step(state, frame, context)
            if first_action is None:
                first_action = action
            states.append(ns); actions.append(action); previous_actions.append(previous)
            frames.append(frame); critic_inputs.append(critic_input)
            state, previous, frame = ns, action, nxt
        critic_inputs.append(self.values(state, frame, context)[1])
        states, actions, previous_actions, frames, x = map(torch.stack, (states, actions, previous_actions, frames, critic_inputs))
        rewards, scores, cont = self.reward(states, actions, previous_actions, frames)
        alive = torch.cumprod(cont, 0)
        weights = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
        discounts = self.gamma ** torch.arange(self.horizon, device=state.device, dtype=state.dtype)[:, None]
        total = (discounts * weights * rewards).sum(0)
        bootstrap = self.gamma ** self.horizon * alive[-1] * self.critic(x[-1])
        result = dict(objective=(total + bootstrap).mean(), task_return=total.mean(),
                      critic_inputs=x, rewards=rewards, continuations=cont,
                      imagined_actions=actions.detach(), recovery_mode_fraction=state.new_zeros(()),
                      style_score=scores.mean())
        if self.audit:
            # Native generator style reward, excluding both task and bootstrap.
            result["final_style"] = (.1 * (1 - .25 * (scores[-1] - 1).square()).clamp_min(0)).mean()
            result["first_action"] = first_action
        return result


def verify_features(env, shared, features):
    base = env.unwrapped
    state = pack(read_physics(base))
    kin = RobotKinematics(base)
    names = list(base.scene["robot"].body_names)
    bodies, anchor = [names.index(n) for n in AMP_BODIES], names.index("torso_link")
    with torch.no_grad():
        native = encode_bodies(*kin(unpack(state)), bodies, anchor)
        common = features(state)
        error = float((native - common).abs().max())
        if not torch.allclose(native, common, atol=1e-4, rtol=1e-4):
            raise AssertionError(f"Body feature mismatch: {error}")
    return {"upstream_vs_shared_body195_max_abs_error": error, "passed": True}


def train_branch(env, output, budget, checkpoints, seed, smoke=False, *,
                 max_policy_updates=None, resume=None):
    shared = SharedObservations(env)
    obs_actor, obs_critic, _ = shared(env.get_observations())
    policy, norm = create_policy(env.device, obs_actor.shape[-1], obs_critic.shape[-1], env.num_actions)
    command = env.unwrapped.command_manager.get_term("motion")
    features = BranchFeatures(command)
    feature_audit = verify_features(env, shared, features)
    (output / "feature_audit.json").write_text(json.dumps(feature_audit, indent=2))
    with torch.no_grad():
        expert = features.encoder.expert_frames(compute_gencoord_state(torch.cat((
            command.motion.root_pos_w, command.motion.root_quat_w, command.motion.joint_pos,
            command.motion.root_lin_vel_w, command.motion.root_ang_vel_w, command.motion.joint_vel), -1)))
    ends = command.motion.motion_ends.tolist()
    starts = [0] + ends[:-1]
    experts = ExpertPairs([expert[a:b] for a, b in zip(starts, ends)], command.motion.motion_names)
    replay = SequenceReplay(256, env.num_envs, env.device)
    ids = torch.randperm(env.num_envs, generator=torch.Generator().manual_seed(seed)).tolist()
    train_ids, holdout = ids[:env.num_envs * 3 // 4], ids[env.num_envs * 3 // 4:]
    if resume is not None:
        train_ids, holdout = resume["train_ids"], resume["holdout_ids"]
        policy.load_state_dict(resume["model_state_dict"])
        norm.load_state_dict(resume["obs_norm_state_dict"])
    collector = TrackingCollection(env, shared, policy, norm, replay, train_ids)
    start_time = time.monotonic()
    if resume is None:
        collector.collect(64)
    wm = WorldModel(env.num_actions, shared.context.output_dim, hidden=(2048,) * 3, statistics_mode="running").to(env.device)
    if resume is None:
        fit = replay.sample(8192, 1, train_ids, ["state", "next_state", "action", "context"])
        wm.fit_stats(fit["state"], fit["next_state"], fit["action"], fit["context"])
    else:
        wm.load_state_dict(resume["world_model"])
    collector.world_model = wm
    critic = ValueModel(obs_critic.shape[-1], normalize=True).to(env.device)
    amp = DifferentiableAMP(resume, env.device)
    engine = TrackingImagination(wm, policy, norm, critic, amp, features, shared, command)
    discriminator = DiscriminatorUpdates(amp, features, experts, lr=1e-5)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=2e-4)
    opt_actor = torch.optim.Adam(policy.actor.parameters(), lr=2e-4, betas=(.9, .95))
    opt_v = torch.optim.Adam(critic.parameters(), lr=2e-4)
    update = actor_updates = 0
    accumulator = prior_elapsed = 0.0
    if resume is not None:
        update, actor_updates = resume["update"], resume["actor_updates"]
        prior_elapsed = resume["elapsed_s"]
        accumulator = resume.get("collection_accumulator", (update * 2.5) % 1)
        collector.total_steps = resume["real_sim_transitions"] // env.num_envs
        collector.failures = resume.get("cumulative_failures", 0)
        critic.load_state_dict(resume["p4_critic"])
        opt_wm.load_state_dict(resume["wm_optimizer"])
        opt_actor.load_state_dict(resume["actor_optimizer"])
        opt_v.load_state_dict(resume["critic_optimizer"])
        discriminator.optimizer.load_state_dict(resume["discriminator_optimizer"])
        discriminator.updates = update
        experts.cursor = resume["expert_cursor"]
        restore_rng(resume["rng"])
        print(f"[INFO] Restored branch at {actor_updates} policy updates; collecting 64 new replay steps", flush=True)
        collector.collect(64)
    with torch.no_grad():
        actual = pack(read_physics(env.unwrapped))
        actor_expected, critic_expected, _ = shared(env.get_observations())
        actor_rebuilt, critic_rebuilt = engine._value_inputs(
            actual, command.time_steps, shared.context.read(env.unwrapped))
        row = (replay.ptr - 1) % replay.capacity
        ended = replay.storage["terminated"][row] | replay.storage["truncated"][row]
        next_error = (replay.storage["next_state"][row] - actual).abs().amax(-1)
        collection_audit = dict(
            actor_observation_max_abs_error=float((actor_expected - actor_rebuilt).abs().max()),
            critic_observation_max_abs_error=float((critic_expected - critic_rebuilt).abs().max()),
            nonterminal_next_state_max_abs_error=float(next_error[~ended].max()),
            terminal_rows_last_step=int(ended.sum()),
            terminal_targets_differ_from_reset_state=int((next_error[ended] > 1e-5).sum()))
        collection_audit["passed"] = bool(
            torch.allclose(actor_expected, actor_rebuilt, atol=1e-4, rtol=1e-4)
            and torch.allclose(critic_expected, critic_rebuilt, atol=1e-4, rtol=1e-4)
            and next_error[~ended].max() < 1e-5
            and (next_error[ended] > 1e-5).all())
        (output / "collection_audit.json").write_text(json.dumps(collection_audit, indent=2))
        if not collection_audit["passed"]:
            raise AssertionError(collection_audit)
    # Full effective native batches; BF16 is an upstream-supported option.
    config = dict(horizon=32, actor_batch=1024, wm_batch=2048, wm_horizon=8, wm_batches=10,
                  actor_microbatch=1024, wm_microbatch=2048, precision="bf16", utd=.4,
                  wm_pretrain_updates=500, critic_epochs=5, critic_minibatches=4,
                  actor_hidden=[512,256,128], wm_hidden=[2048,2048,2048], seed=seed)
    (output / "training_config.json").write_text(json.dumps(config, indent=2))
    validation = replay.sample_prefix(2048, 8, holdout, DYNAMICS_KEYS)
    with torch.no_grad():
        initial_holdout = float(engine.wm_loss(validation)[0])
    pretrain = 2 if smoke else 500
    final_steps = 74 if smoke else budget // env.num_envs if budget is not None else None
    markers = sorted({x // env.num_envs for x in checkpoints if budget is None or x <= budget})
    if final_steps is not None:
        markers = sorted(set(markers) | {final_steps})
    pending = []

    def save(update):
        total = collector.total_steps * env.num_envs
        state = dict(method="jonny_wmrl_supermimic_matched_body195", branch_sha=UPSTREAM_SHA,
            env_variant="branch_body195", task_id=TASK, motion_file=command.cfg.motion_file, seed=seed,
            update=update, actor_updates=actor_updates, real_sim_transitions=total, num_envs=env.num_envs,
            model_state_dict=policy.state_dict(), obs_norm_state_dict=norm.state_dict(),
            actor_dim=obs_actor.shape[-1], critic_dim=obs_critic.shape[-1], world_model=wm.state_dict(),
            architecture=wm.architecture(), p4_critic=critic.state_dict(), discriminator_state_dict=amp.discriminator.state_dict(),
            amp_normalizer=amp.normalizer.native(), amp_normalization="raw", expert_cursor=experts.cursor,
            wm_optimizer=opt_wm.state_dict(), actor_optimizer=opt_actor.state_dict(), critic_optimizer=opt_v.state_dict(),
            discriminator_optimizer=discriminator.optimizer.state_dict(), rng=rng_state(),
            train_ids=train_ids, holdout_ids=holdout, config=config,
            elapsed_s=prior_elapsed + time.monotonic() - start_time,
            collection_accumulator=accumulator, cumulative_failures=collector.failures,
            replay_included=False, resume_note="Evaluation checkpoint; exact training continuation would require replay and simulator state")
        atomic_save(state, output / "training_state.pt")
        if collector.total_steps in markers:
            import shutil
            shutil.copyfile(output / "training_state.pt", output / f"budget_{total:09d}.pt")

    while (final_steps is None or collector.total_steps < final_steps) and (
        max_policy_updates is None or actor_updates < max_policy_updates
    ):
        update += 1
        accumulator += 2.5
        count = int(accumulator); accumulator -= count
        next_marker = min((s for s in markers if s > collector.total_steps), default=None)
        if next_marker is not None:
            count = min(count, next_marker - collector.total_steps)
        torch.set_float32_matmul_precision("highest")
        collector.collect(count)
        torch.set_float32_matmul_precision("high")
        if update > pretrain:
            for opt in (opt_wm, opt_actor, opt_v):
                for group in opt.param_groups:
                    group["lr"] = learning_rate(2e-4, update - 1)
        losses = []
        for _ in range(10):
            window = replay.sample_prefix(2048, 8, train_ids, DYNAMICS_KEYS)
            losses.append(world_model_update(wm, opt_wm, window, engine.wm_loss, 2048))
        metrics = discriminator.update(replay, train_ids, 2048 if update <= pretrain else 1024, return_tensors=True)
        metrics["wm_loss"] = torch.stack(losses).mean()
        if update > pretrain:
            opt_wm.zero_grad(set_to_none=True); opt_v.zero_grad(set_to_none=True)
            samples = replay.sample(1024, 1, train_ids, START_KEYS)
            with freeze_parameters(wm, critic, amp.discriminator):
                if actor_updates == 0:
                    engine.audit = True
                    audit = engine(samples)
                    gradient = torch.autograd.grad(audit["final_style"], audit["first_action"], retain_graph=True)[0]
                    params = list(policy.actor.parameters())
                    pg = torch.autograd.grad(audit["final_style"], params, allow_unused=True)
                    report = {"horizon": 32, "batch": 1024, "final_style_to_first_action_norm": float(gradient.norm()),
                              "style_to_policy_norm": float(torch.stack([g.square().sum() for g in pg if g is not None]).sum().sqrt()),
                              "world_model_parameter_grads_absent": all(p.grad is None for p in wm.parameters()),
                              "discriminator_parameter_grads_absent": all(p.grad is None for p in amp.discriminator.parameters()),
                              "style_value": float(audit["final_style"].detach())}
                    report["audit_note"] = "Identical first step expanded to expose the intermediate action edge across AOTAutograd's multi-output boundary"
                    report["passed"] = bool(torch.isfinite(gradient).all() and gradient.norm() > 0 and report["style_to_policy_norm"] > 0 and report["world_model_parameter_grads_absent"] and report["discriminator_parameter_grads_absent"])
                    (output / "gradient_audit.json").write_text(json.dumps(report, indent=2))
                    if not report["passed"]:
                        raise AssertionError(report)
                    engine.audit = False
                    del audit, gradient, pg
                opt_actor.zero_grad(set_to_none=True)
                rollout = accumulate_imagination(engine, samples, policy.actor, 1024)
                grad = torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.)
                torch.nn.utils.clip_grad_value_(policy.actor.parameters(), .1)
                opt_actor.step()
                actor_updates += 1
            value, diagnostics = update_critic(critic, opt_v, rollout, 5, 4, return_tensor=True, return_metrics=True)
            metrics.update(diagnostics, actor_gradient_norm=grad, critic_loss=value, imagined_return=rollout["task_return"], style_score=rollout["style_score"])
        pending.append({k: float(v.detach()) for k, v in metrics.items()})
        at_marker = collector.total_steps in markers
        at_update_target = actor_updates == max_policy_updates
        if update == 1 or update % 25 == 0 or update == pretrain or at_marker or at_update_target:
            with torch.no_grad():
                heldout = float(engine.wm_loss(validation)[0])
            keys = set.intersection(*(set(x) for x in pending))
            row = dict(update=update, actor_updates=actor_updates, real_sim_transitions=collector.total_steps*env.num_envs,
                       elapsed_s=prior_elapsed + time.monotonic() - start_time, wm_initial_holdout=initial_holdout, wm_holdout=heldout,
                       cumulative_failures=collector.failures, mean_task_reward_per_step=collector.last_reward,
                       peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                       **{k: float(np.mean([x[k] for x in pending])) for k in keys})
            if not all(np.isfinite(x) for x in row.values()):
                raise FloatingPointError(row)
            with (output / "training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            (output / "progress.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
            pending.clear()
        if at_marker or at_update_target or update % 500 == 0:
            save(update)
    env.close()


def main(method: Literal["branch", "ours", "ppo"], output: Path, budget: int = 24_576_000,
         num_envs: int = 2048, seed: int = 0, smoke: bool = False,
         checkpoints: tuple[int, ...] = (4_915_200, 12_288_000, 24_576_000),
         device: str = "cuda:0", data: Path = DATA,
         max_policy_updates: int | None = None, resume_checkpoint: Path | None = None,
         update_checkpoints: tuple[int, ...] = (),
         evaluate_checkpoint: Annotated[Callable[[int, Path], None] | None, tyro.conf.Suppress] = None):
    if (output / "training_metrics.jsonl").exists() or (output / "contract.json").exists():
        raise FileExistsError(f"Refusing to mix a fresh run into {output}")
    if budget % num_envs or any(x % num_envs for x in checkpoints):
        raise ValueError("Interaction budgets must contain whole vector steps")
    if max_policy_updates is not None and (max_policy_updates <= 0 or smoke):
        raise ValueError("An update target must be positive and cannot be used in smoke mode")
    if resume_checkpoint is not None and max_policy_updates is None:
        raise ValueError("Continuation requires a cumulative policy update target")
    if update_checkpoints and (method != "ours" or max_policy_updates is None):
        raise ValueError("Update checkpoints require the AMP feedback arm and an update target")
    if max_policy_updates is not None:
        # Transition checkpoints belong to the separate equal-interaction run.
        checkpoints = ()
        budget = max_policy_updates * num_envs * 24 if method == "ppo" else None
    output.mkdir(parents=True, exist_ok=True)
    setup_device(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data = data.resolve()
    cfg = branch_env_cfg(num_envs, seed, data=data)
    env_json = json.dumps(asdict(cfg), sort_keys=True, default=json_value)
    env_hash = hashlib.sha256(env_json.encode()).hexdigest()
    data_hash = hashlib.sha256((data/"manifest.json").read_bytes()).hexdigest()
    resume = None
    start_updates = 0
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.resolve()
        source_contract = json.loads((resume_checkpoint.parent / "contract.json").read_text())
        expected = dict(method=method, seed=seed, num_envs=num_envs,
                        env_cfg_sha256=env_hash, data_manifest_sha256=data_hash)
        if any(source_contract[key] != value for key, value in expected.items()):
            raise ValueError(f"Continuation configuration differs from {resume_checkpoint.parent}")
        resume = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if method == "ours":
            start_updates = int(resume["amp"]["updates"])
        else:
            start_updates = resume["actor_updates" if method == "branch" else "iteration"]
        if start_updates >= max_policy_updates:
            raise ValueError("Target must exceed the saved policy update count")
    contract = dict(method=method, upstream_branch="supermimic", upstream_sha=UPSTREAM_SHA,
        seed=seed, num_envs=num_envs, budget=budget, checkpoints=checkpoints,
        from_scratch=resume is None, smoke=smoke, target_policy_updates=max_policy_updates,
        update_checkpoints=update_checkpoints,
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None,
        start_policy_updates=start_updates,
        start_real_sim_transitions=resume["real_sim_transitions"] if resume is not None else 0,
        env_variant="branch_body195", task_id=TASK,
        data_directory=str(data), data_manifest_sha256=data_hash,
        env_cfg_sha256=env_hash,
        actor_observation="GenCoord68 + official 10-frame actor goal; current state, history=1",
        critic_observation="GenCoord68 + full reference goal + shared simulator context",
        amp_observation="Upstream 13-body torso-relative pose and body-local velocities; 195D per frame",
        branch_adaptations=["Native velocity/recovery task replaced with shared tracking observations, rewards and termination",
            "Common current mjlab/MuJoCo runtime and simulator randomization context",
            "Equivalent differentiable FK implementation audited against upstream RobotKinematics",
            "Replay stores frame indices instead of redundant full reference observations",
            f"{num_envs} real environments; native effective batches/networks and 32-step BPTT retained; supported BF16 mode"],
        interaction_unit="One real environment transition; includes warmup, holdout and all feedback branches",
        comparison_scope="Matched full systems; native discriminator objectives, normalizers, model architecture and update rates retained separately")
    if method == "ppo":
        from supermimic.scripts.compare_amp import PPO_SETTINGS
        contract.update(optimizer="AMP_mjlab AMP+PPO", rollout_horizon=24, ppo_settings=PPO_SETTINGS,
                        baseline_adaptations=["Shared reference tracking task and body195 features", "Native AMP_mjlab PPO, LSGAN reward, normalizers and terminal-pair convention"])
    (output/"contract.json").write_text(json.dumps(contract, indent=2))
    (output/"env_config.json").write_text(env_json)
    base = (CaptureTrackingEnv if method == "branch" else ManagerBasedRlEnv)(cfg=cfg, device=device)
    env = RslRlVecEnvWrapper(base)
    if method == "branch":
        train_branch(env, output, budget, checkpoints, seed, smoke,
                     max_policy_updates=max_policy_updates, resume=resume)
    elif method == "ppo":
        from supermimic.scripts.compare_amp import train_ppo
        train_ppo(env, output, budget, checkpoints, seed, body195=True, resume=resume)
    else:
        if resume is None:
            runner_cfg = SuperMimicRunnerCfg()
            runner_cfg.amp.enabled = True
            runner_cfg.amp.feature_mode = "body195"
            runner_cfg.amp_feedback.enabled = True
            runner_cfg.policy_activation_checkpointing = True
            runner_cfg.mlp_bf16 = True
            runner_cfg.sampling_mode = "uniform"
        else:
            runner_cfg = load_runner_cfg(resume)
            runner_cfg.resume_checkpoint = str(resume_checkpoint)
        runner_cfg.max_iterations = budget // num_envs if budget is not None else SuperMimicRunnerCfg().max_iterations
        runner_cfg.max_sim_transitions = budget
        runner_cfg.max_amp_updates = max_policy_updates
        runner_cfg.budget_checkpoints = tuple(x for x in checkpoints if budget is None or x <= budget)
        runner_cfg.run_name = f"matched_body195_bptt_seed{seed}"

        class MatchedRunner(SuperMimicRunner):
            def _create_log_dir(self, *extra_parts):
                return output

        runner = MatchedRunner(env, runner_cfg, TASK, device=device)
        runner.run(update_checkpoints=update_checkpoints, evaluate_checkpoint=evaluate_checkpoint)


if __name__ == "__main__":
    tyro.cli(main)

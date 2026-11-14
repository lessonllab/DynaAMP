"""Fixed-seed full-clip evaluation and actual-physics videos for both methods."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer.viewer_config import ViewerConfig
from PIL import Image, ImageDraw, ImageFont

from supermimic.representations import extract_actor_goal, root_position_error
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.compare_amp import DATA, ExpertPairs, SharedObservations, common_env_cfg
from supermimic.scripts.train_utils import setup_device
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.vendor.amp_mjlab.modules import ActorCritic, EmpiricalNormalization


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def initial_snapshot(base, features, obs, output):
    snapshot = {"full_state": obs["full_state"].detach().cpu().clone(),
                "qpos": base.sim.data.qpos.detach().cpu().clone(),
                "qvel": base.sim.data.qvel.detach().cpu().clone(),
                "sim_context": features.context.read(base).detach().cpu().clone()}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, output.with_suffix(".initial.pt"))
    return tensor_hash(torch.cat([snapshot["qpos"], snapshot["qvel"]], -1))


def load_policy(saved, env):
    features = SharedObservations(env)
    if saved.get("method", "").startswith("jonny_wmrl"):
        from supermimic.vendor.jonny_wmrl.core.policy import create_policy
        policy, norm = create_policy(env.device, saved["actor_dim"], saved["critic_dim"], env.num_actions)
        policy.load_state_dict(saved["model_state_dict"])
        norm.load_state_dict(saved["obs_norm_state_dict"])
        policy.eval(); norm.eval()

        def act(obs):
            return policy.act_inference(norm(features(obs)[0]))
        return act, features, "supermimic branch: world model + AMP + BPTT"
    if saved.get("method", "").startswith("amp_mjlab"):
        policy = ActorCritic(saved["actor_dim"], saved["critic_dim"], env.num_actions,
                             actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                             activation="elu").to(env.device)
        policy.load_state_dict(saved["policy"])
        norm = EmpiricalNormalization([saved["actor_dim"]]).to(env.device)
        norm.load_state_dict(saved["obs_normalizer"])
        policy.eval()
        norm.eval()

        def act(obs):
            return policy.act_inference(norm(features(obs)[0]))
        return act, features, "AMP_mjlab AMP+PPO"
    cfg = load_runner_cfg(saved)
    wm = build_world_model(cfg, env, env.device)
    actor = build_actor(cfg, wm, env.device)
    actor.load_state_dict(saved.get("actor_ema", saved["actor"]))
    actor.eval()

    def act(obs):
        state = wm.representation.represent(obs["full_state"])
        command = env.unwrapped.command_manager.get_term("motion")
        aux = wm.representation.extract_aux_state(obs["full_state"])
        return actor(build_actor_obs_td({"motion": extract_actor_goal(obs["goal"])}, state[:, None],
                     root_error=root_position_error(command.motion, command.time_steps, aux)))
    name = "LoopMimic v2 (EMA)" if cfg.root_position_feedback else "SuperMimic + differentiable AMP + feedback"
    return act, features, name


def evaluation_env_cfg(saved, num_envs, seed, data: Path | None = None):
    if data is None:
        data = Path(saved.get("motion_file", str(DATA / "*.npz"))).parent
    body195 = saved.get("env_variant") == "branch_body195" or saved.get("runner_cfg", {}).get("amp", {}).get("feature_mode") in ("body195", "grounded199")
    if body195:
        from supermimic.scripts.compare_wmrl_branch import branch_env_cfg
        return branch_env_cfg(num_envs, seed, play=True, data=data)
    return common_env_cfg(num_envs, seed, play=True, data=data)


def reset_clips(base, command, frames, seed):
    base.reset(seed=seed)
    for frame in frames.unique().tolist():
        ids = torch.where(frames == frame)[0]
        command.reset_to_frame(ids, int(frame))
    base.scene.write_data_to_sim()
    base.sim.forward()
    base.sim.sense()
    command.update_relative_body_poses()
    # Manual frame writes do not invalidate ObservationManager's cached reset
    # observation. Refresh once so the first action sees the actual test frame.
    return base.observation_manager.compute(update_history=True)


def mmd_style(pairs, expert, seed):
    """Biased RBF MMD^2 in shared expert-standardized features; no learned D."""
    generator = torch.Generator(device=pairs.device).manual_seed(seed)
    starts = expert.starts[torch.randperm(len(expert.starts), generator=generator, device=pairs.device)[:2048]]
    ref = torch.cat([expert.frames[starts], expert.frames[starts + 1]], -1)
    ids = torch.randperm(len(pairs), generator=generator, device=pairs.device)[:2048]
    pred = pairs[ids]
    std = ref.std(0).clamp_min(0.1)
    ref, pred = ref / std, pred / std
    rr = torch.cdist(ref, ref).square()
    bandwidth = rr[rr > 0].median().clamp_min(1e-6)
    pp, pr = torch.cdist(pred, pred).square(), torch.cdist(pred, ref).square()
    value = torch.exp(-pp / (2 * bandwidth)).mean() + torch.exp(-rr / (2 * bandwidth)).mean() - 2 * torch.exp(-pr / (2 * bandwidth)).mean()
    return {"rbf_mmd2_biased": float(value), "bandwidth_squared": float(bandwidth),
            "policy_pairs": len(pred), "expert_pairs": len(ref),
            "scope": "All valid pre-termination pairs; shared features and expert-only fixed bandwidth."}


def evaluate(checkpoint: Path, output: Path, replicas: int = 4, seed: int = 20261113,
             device: str = "cuda:0", data: Path | None = None):
    setup_device(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if data is None:
        data = Path(saved.get("motion_file", str(DATA / "*.npz"))).parent
    data = data.resolve()
    count = len(list(data.glob("*.npz"))) * replicas
    if count <= 0:
        raise ValueError(f"No evaluation episodes. Check the reference directory and replica count: {data}")
    cfg = evaluation_env_cfg(saved, count, seed, data)
    cfg.auto_reset = False
    cfg.sim.nconmax = max(256, cfg.sim.nconmax or 0)
    cfg.sim.njmax = max(1024, cfg.sim.njmax or 0)
    base = ManagerBasedRlEnv(cfg, device=device)
    env = RslRlVecEnvWrapper(base)
    act, features, method = load_policy(saved, env)
    cmd = base.command_manager.get_term("motion")
    ends = cmd.motion.motion_ends
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    initial = starts.repeat_interleave(replicas)
    lengths = (ends - starts - 1).repeat_interleave(replicas)
    obs = reset_clips(base, cmd, initial, seed)
    initial_hash = tensor_hash(obs["full_state"])
    context_hash = tensor_hash(features.context.read(base))
    physics_hash = initial_snapshot(base, features, obs, output)
    active = torch.ones(count, dtype=torch.bool, device=device)
    steps = torch.zeros(count, dtype=torch.long, device=device)
    failed = torch.zeros_like(active)
    errors = {k: torch.zeros(count, device=device) for k in ["root_pos_m", "joint_rmse_rad", "body_pos_m", "task_return"]}
    slip_sum, slip_count = torch.zeros(count, device=device), torch.zeros(count, device=device)
    contact_sum, contact_count = torch.zeros_like(slip_sum), torch.zeros_like(slip_count)
    has_foot_contact = any(s.name == "feet_ground_contact" for s in (cfg.scene.sensors or ()))
    feet = [base.scene["robot"].body_names.index(f"{side}_ankle_roll_link") for side in ["left", "right"]]
    pairs = []
    physics_steps = 0
    with torch.inference_mode():
        while active.any():
            before = features.rep.represent(obs["full_state"])
            obs, reward, done, _ = env.step(act(obs))
            physics_steps += 1
            after = features.rep.represent(obs["full_state"])
            mask = active.clone()
            # Full LAFAN1 evaluation has ~1.76M transitions. Keep raw metric
            # pairs on CPU; only the fixed MMD sample needs GPU FK.
            pairs.append(torch.cat([before[mask], after[mask]], -1).cpu())
            steps[mask] += 1
            target = initial + steps
            robot = base.scene["robot"]
            pos_error = (robot.data.root_link_pos_w - base.scene.env_origins - cmd.motion.root_pos_w[target]).norm(dim=-1)
            joint_error = (robot.data.joint_pos - cmd.motion.joint_pos[target]).square().mean(-1).sqrt()
            errors["root_pos_m"][mask] += pos_error[mask]
            errors["joint_rmse_rad"][mask] += joint_error[mask]
            errors["body_pos_m"][mask] += cmd.metrics["error_body_pos"][mask]
            errors["task_return"][mask] += reward[mask]
            foot_height = robot.data.body_link_pos_w[:, feet, 2] - base.scene.env_origins[:, 2, None]
            foot_speed = robot.data.body_link_lin_vel_w[:, feet, :2].norm(dim=-1)
            near_ground = (foot_height < 0.08) & mask[:, None]
            slip_sum += (foot_speed * near_ground).sum(-1)
            slip_count += near_ground.sum(-1)
            if has_foot_contact:
                contact = (base.scene["feet_ground_contact"].data.found > 0).reshape(count, 2) & mask[:, None]
                contact_sum += (foot_speed * contact).sum(-1)
                contact_count += contact.sum(-1)
            failed |= mask & base.termination_manager.terminated
            active &= ~failed & (steps < lengths) & ~done.bool()
            # Finished/fallen rows never re-enter measurement. Reset only to keep
            # the batched simulator legal while other independent clips finish.
            if done.any() and active.any():
                base.reset(env_ids=torch.where(done.bool())[0])
                obs = env.get_observations()
            if physics_steps % 500 == 0:
                print(json.dumps({"evaluation_step": physics_steps, "active": int(active.sum()),
                    "failed": int(failed.sum()), "completed": int(((steps >= lengths) & ~failed).sum())}), flush=True)
    results = []
    for i in range(count):
        results.append({"clip": cmd.motion.motion_names[i // replicas], "replica": i % replicas,
                        "completed": bool(steps[i] >= lengths[i] and not failed[i]),
                        "failed": bool(failed[i]), "steps": int(steps[i]), "target_steps": int(lengths[i]),
                        "duration_fraction": float(steps[i] / lengths[i]),
                        **{k: float(v[i] / (1 if k == "task_return" else steps[i].clamp_min(1))) for k, v in errors.items()},
                        "near_ground_foot_speed_mps": float(slip_sum[i] / slip_count[i].clamp_min(1))})
        if has_foot_contact:
            results[-1]["contact_foot_speed_mps"] = float(contact_sum[i] / contact_count[i].clamp_min(1))
            results[-1]["foot_contact_samples"] = int(contact_count[i])
    summary = {"completion_rate": float(np.mean([r["completed"] for r in results])),
               "mean_duration_fraction": float(np.mean([r["duration_fraction"] for r in results])),
               **{k: float(np.mean([r[k] for r in results])) for k in [*errors, "near_ground_foot_speed_mps"]}}
    if has_foot_contact:
        summary["contact_foot_speed_mps"] = float(contact_sum.sum() / contact_count.sum().clamp_min(1))
        summary["foot_contact_fraction"] = float(contact_count.sum() / (2 * steps.sum()).clamp_min(1))
    all_pairs = torch.cat(pairs)
    del pairs
    sample_ids = torch.randperm(len(all_pairs), generator=torch.Generator().manual_seed(seed))[:2048]
    metric_pairs = all_pairs[sample_ids].to(device)
    measured_pairs = len(all_pairs)
    del all_pairs
    result = {"checkpoint": str(checkpoint.resolve()), "method": method,
              "motion_file": str(data / "*.npz"), "valid_measured_transitions": measured_pairs,
              "training_real_sim_transitions": saved["real_sim_transitions"],
              "evaluation_physics_transitions": physics_steps * count,
              "seed": seed, "replicas_per_clip": replicas, "clips": len(starts),
              "initial_state_sha256": initial_hash, "sim_context_sha256": context_hash,
              "initial_physics_state_sha256": physics_hash,
              "summary": summary, "style_distribution": mmd_style(metric_pairs, ExpertPairs(cmd.motion), seed),
              "metric_notes": "Exact clip starts, deterministic actions, original termination criteria. Errors averaged per episode until first termination. Near-ground speed uses ankle height < 0.08 m, a slip proxy, not a contact measurement.",
              "episodes": results}
    if has_foot_contact:
        from types import SimpleNamespace
        from supermimic.learners.body_amp_features import BodyAMPFeatures
        encoder = BodyAMPFeatures(cmd.motion, cmd.cfg.body_names)
        experts = ExpertPairs(cmd.motion)
        with torch.no_grad():
            body_pairs = encoder(metric_pairs.reshape(-1, 2, 68)).flatten(1)
            body_experts = SimpleNamespace(frames=encoder.expert_frames(experts.frames), starts=experts.starts)
            result["style_distribution_body195"] = mmd_style(body_pairs, body_experts, seed)
        result["contact_metric_note"] = "Mean ankle horizontal speed during sensor-confirmed ground contact, weighted by contact samples; reward-time contact sensing can lag geometry by one physics substep."
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(summary), flush=True)
    env.close()


def video(checkpoint: Path, output: Path, clip: str, seed: int = 20261113, device: str = "cuda:0",
          seconds: float | None = None, data: Path | None = None):
    setup_device(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = evaluation_env_cfg(saved, 1, seed, data)
    cfg.auto_reset = False
    cfg.sim.nconmax = max(256, cfg.sim.nconmax or 0)
    cfg.sim.njmax = max(1024, cfg.sim.njmax or 0)
    cfg.commands["motion"].debug_vis = False
    cfg.viewer = ViewerConfig(width=960, height=720, origin_type=ViewerConfig.OriginType.ASSET_ROOT,
                             entity_name="robot", distance=3.5, elevation=-15, azimuth=135,
                             enable_shadows=False, enable_reflections=False)
    base = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(base)
    act, features, method = load_policy(saved, env)
    cmd = base.command_manager.get_term("motion")
    index = cmd.motion.motion_names.index(clip)
    start = 0 if index == 0 else int(cmd.motion.motion_ends[index - 1])
    length = int(cmd.motion.motion_ends[index]) - start - 1
    clip_length = length
    if seconds is not None:
        length = min(length, int(seconds / base.step_dt))
    obs = reset_clips(base, cmd, torch.tensor([start], device=device), seed)
    initial_hash = tensor_hash(obs["full_state"])
    context_hash = tensor_hash(features.context.read(base))
    physics_hash = initial_snapshot(base, features, obs, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    failure_step = None
    final_pixels = None
    with imageio.get_writer(str(output), fps=50, codec="libx264", quality=8) as writer, torch.inference_mode():
        for step in range(length):
            if failure_step is None:
                obs, _, done, _ = env.step(act(obs))
                final_pixels = base.render()
                if bool(done[0]):
                    failure_step = step + 1
            frame = Image.fromarray(final_pixels.copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 960, 104), fill=(20, 24, 30))
            draw.text((14, 10), f"{method} | MuJoCo physics", font=font, fill="white")
            draw.text((14, 40), f"{clip} | train transitions: {saved['real_sim_transitions']:,}", font=font, fill="white")
            status = f"t={(step + 1) / 50:.2f}s | reset=0"
            if failure_step is not None:
                status += f" | TERMINATED at {failure_step / 50:.2f}s (last frame held)"
            draw.text((14, 70), status, font=font, fill="white" if failure_step is None else "#ff7777")
            writer.append_data(np.asarray(frame))
    output.with_suffix(".json").write_text(json.dumps({"checkpoint": str(checkpoint), "clip": clip,
        "training_real_sim_transitions": saved["real_sim_transitions"], "target_steps": length,
        "full_clip_steps": clip_length, "scope": "full_clip" if length == clip_length else "prefix",
        "initial_state_sha256": initial_hash, "sim_context_sha256": context_hash,
        "initial_physics_state_sha256": physics_hash,
        "failure_step": failure_step, "completed": failure_step is None,
        "actual_physics_steps": failure_step or length, "seed": seed}, indent=2))
    env.close()


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"evaluate": evaluate, "video": video})

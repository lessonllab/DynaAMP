"""Validate the full temporal discriminator gradient on actual recorded states."""

import torch

from supermimic.tasks.task_spec import build_actor_obs_td


def audit_policy_objective(learner, start):
  """Inspect the exact training objective on an isolated, eager learner.

  This deliberately runs outside CUDA graph capture and takes no optimizer
  steps. Norms/cosines are raw parameter gradients before clipping or Muon/Adam
  preconditioning. The diagnostic must not run concurrently with WM/D updates.
  """
  if learner.motion_prior is None:
    raise ValueError("The loss-gradient diagnostic requires an AMP discriminator")
  learner.actor.train().requires_grad_(True)
  for module in (learner.wm_env.dynamics, learner.wm_env.representation):
    module.eval().requires_grad_(False)
    for parameter in module.parameters():
      parameter.grad = None
  learner.motion_prior.freeze()
  objective = learner._imagine_batch(start)
  params = [p for p in learner.actor_params if p.requires_grad]
  task = torch.autograd.grad(
    objective.task_loss, params, retain_graph=True, allow_unused=True
  )
  style = torch.autograd.grad(
    objective.style_loss, params, retain_graph=True, allow_unused=True
  )
  zero = objective.task_loss.new_zeros(())
  task_sq, style_sq, dot = zero, zero, zero
  finite = True
  for gt, gs in zip(task, style, strict=True):
    if gt is not None:
      task_sq = task_sq + gt.float().square().sum()
      finite = finite and bool(torch.isfinite(gt).all())
    if gs is not None:
      style_sq = style_sq + gs.float().square().sum()
      finite = finite and bool(torch.isfinite(gs).all())
    if gt is not None and gs is not None:
      dot = dot + (gt.float() * gs.float()).sum()
  task_norm, style_norm = task_sq.sqrt(), style_sq.sqrt()
  weight = objective.style_weight.detach()
  weighted_style_norm = weight.abs() * style_norm
  ratio = weighted_style_norm / task_norm.clamp_min(1e-12)
  cosine = dot / (task_norm * style_norm).clamp_min(1e-12)
  finite = finite and bool(
    torch.isfinite(torch.stack([task_norm, style_norm, ratio, cosine])).all()
  )
  # A final-pair-only loss cannot reach a_0 through an earlier local style loss.
  # This therefore detects a detach anywhere along the full temporal path.
  first_gradient = torch.autograd.grad(
    objective.final_style_cost,
    objective.first_action,
    allow_unused=True,
  )[0]
  first_norm = first_gradient.norm() if first_gradient is not None else zero
  connected = (
    first_gradient is not None
    and bool(torch.isfinite(first_gradient).all())
    and float(first_norm) > 0
  )
  return {
    "passed": finite and connected and float(style_norm) > 0,
    "horizon": learner.cfg.horizon,
    "batch_size": start.shape[0],
    "task_loss": float(objective.task_loss.detach()),
    "style_loss": float(objective.style_loss.detach()),
    "weighted_style_loss": float(objective.weighted_style_loss.detach()),
    "bootstrap_loss": float(objective.bootstrap_loss.detach()),
    "auxiliary_loss": float(objective.auxiliary_loss.detach()),
    "total_loss": float(objective.total_loss.detach()),
    "amp_weight": float(weight),
    "task_gradient_norm": float(task_norm),
    "unweighted_style_gradient_norm": float(style_norm),
    "weighted_style_gradient_norm": float(weighted_style_norm),
    "weighted_style_to_task_gradient_ratio": float(ratio),
    "task_style_gradient_cosine": float(cosine),
    "final_pair_to_first_action_gradient_norm": float(first_norm),
    "world_model_frozen": not any(
      p.requires_grad for p in learner.wm_env.dynamics.parameters()
    ),
    "discriminator_frozen": not any(
      p.requires_grad for p in learner.motion_prior.parameters()
    ),
    "world_model_parameter_grads_empty": all(
      p.grad is None for p in learner.wm_env.dynamics.parameters()
    ),
    "discriminator_parameter_grads_empty": all(
      p.grad is None for p in learner.motion_prior.parameters()
    ),
    "optimizer_steps": 0,
    "gradient_scope": "tracking and style only; bootstrap/auxiliary gradients excluded",
    "interpretation": "Raw gradient diagnostic, not a bound on the Muon/Adam update",
  }


def audit_recorded_states(actor, wm, prior, motion, states, times, sim_ctx, steps=32):
  """Only the final pair's style loss; no intermediate loss or task reward.

  Recording uses inference_mode. Clone its snapshots here, outside that mode,
  so autograd may save normal tensors. Keep weights fixed throughout the audit.
  """
  full = torch.cat([s.clone() for s in states], 0)
  times = torch.cat([t.clone() for t in times], 0)
  ctx = sim_ctx.detach().clone().expand(len(full), -1)
  rep = wm.representation.represent(full).detach()
  aux = wm.representation.extract_aux_state(full).detach()
  wm.dynamics.eval().requires_grad_(False)
  wm.representation.eval().requires_grad_(False)
  prior.freeze()
  actor.eval().requires_grad_(True)
  first_action = None
  previous = rep
  for _ in range(steps):
    sensor = wm.representation.compute_actor_sensor(rep, ctx, aux_state=aux)
    goals = {
      spec.name: spec.extract_training_goal(motion, times, aux)
      for spec, (lo, hi) in zip(
        actor.task_specs, actor.slot_ranges(len(full)), strict=True
      )
      if lo < hi
    }
    actions = actor(build_actor_obs_td(goals, sensor[:, None]))
    if first_action is None:
      first_action = actions
    previous = rep
    aux, rep = wm.dynamics(rep, actions, aux, ctx)
    clip = torch.searchsorted(motion.motion_ends, times, right=True)
    times = torch.minimum(times + 1, motion.motion_ends[clip] - 1)
  assert first_action is not None
  unweighted = prior.style_cost(previous, rep).mean()
  loss = prior.weight * unweighted
  parameters = list(actor.parameters())
  gradients = torch.autograd.grad(loss, [*parameters, first_action], allow_unused=True)
  actor_gradients = [g for g in gradients[:-1] if g is not None]
  first_gradient = gradients[-1]
  actor_norm = torch.stack([g.square().sum() for g in actor_gradients]).sum().sqrt()
  action_norm = (
    first_gradient.norm() if first_gradient is not None else loss.new_zeros(())
  )
  finite = all(bool(torch.isfinite(g).all()) for g in actor_gradients)
  finite = (
    finite and first_gradient is not None and bool(torch.isfinite(first_gradient).all())
  )
  finite = (
    finite and bool(torch.isfinite(actor_norm)) and bool(torch.isfinite(action_norm))
  )
  return {
    "passed": finite and float(actor_norm) > 0 and float(action_norm) > 0,
    "steps": steps,
    "batch_size": len(full),
    "task_reward_used": False,
    "only_final_pair_loss": True,
    "amp_weight": float(prior.weight),
    "amp_updates": int(prior.updates),
    "unweighted_style_cost": float(unweighted.detach()),
    "actor_gradient_norm": float(actor_norm),
    "first_action_gradient_norm": float(action_norm),
    "world_model_frozen": not any(p.requires_grad for p in wm.dynamics.parameters()),
    "discriminator_frozen": not any(p.requires_grad for p in prior.parameters()),
    "mlp_bf16": wm.dynamics.mlp_bf16,
  }

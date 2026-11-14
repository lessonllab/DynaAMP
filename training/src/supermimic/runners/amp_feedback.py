"""Use AMP improvement disagreement to acquire and replay real WM transitions.

The policy objective is unchanged. AMP generates bounded action proposals and
compares their *relative improvement* in learned and real dynamics. Fresh physical
windows train the WM through its original supervised loss, never a loss that asks
the WM itself to fool D. D and its normalizer are fixed throughout each paired
comparison and before/after WM check.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.representations import root_position_error
from supermimic.utils.sim_branch import PhysicsBranch

if TYPE_CHECKING:
  from supermimic.runners.supermimic_runner import SuperMimicRunner


@dataclass
class AmpFeedbackCfg:
  enabled: bool = False
  interval: int = 250
  min_amp_updates: int = 100
  action_delta_rms: float = 0.03
  action_delta_max: float = 0.1
  local_noise: float = 0.03
  selected_fraction: float = 0.25
  priority_fraction: float = 0.25
  priority_cap: float = 20.0
  capacity: int = 16384
  holdout_stride: int = 8
  repeat_tolerance: float = 2e-4
  policy_start_fraction: float = 0.25
  discriminator_fraction: float = 0.25
  priority_decay: float = 0.99


def improvement_disagreement(
  wm_base: torch.Tensor,
  wm_adjusted: torch.Tensor,
  real_base: torch.Tensor,
  real_adjusted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Scores are larger-is-better; no absolute 'bad style' acquisition term."""
  predicted = wm_adjusted - wm_base
  actual = real_adjusted - real_base
  return (predicted - actual).abs(), predicted, actual


class WindowReplay:
  """Bounded priority replay of indivisible, real, reset-free WM windows."""

  def __init__(self, capacity: int):
    self.capacity = capacity
    self.size = 0
    self.cursor = 0
    self.data: TensorDict | None = None
    self.priority: torch.Tensor | None = None

  def insert(self, windows: TensorDict, priority: torch.Tensor) -> None:
    if len(windows) == 0:
      return
    if len(windows) > self.capacity:
      windows, priority = windows[-self.capacity :], priority[-self.capacity :]
    if not torch.isfinite(priority).all() or (priority <= 0).any():
      raise ValueError("Priorities must be positive and finite")
    if self.data is None:
      self.data = TensorDict(
        {
          key: value.new_zeros((self.capacity, *value.shape[1:]))
          for key, value in windows.items()
        },
        batch_size=[self.capacity, windows.shape[1]],
      )
      self.priority = priority.new_ones(self.capacity)
    assert self.priority is not None
    idx = (
      torch.arange(len(windows), device=priority.device) + self.cursor
    ) % self.capacity
    self.data[idx] = windows.detach()
    self.priority[idx] = priority.detach()
    self.cursor = (self.cursor + len(windows)) % self.capacity
    self.size = min(self.capacity, self.size + len(windows))

  def sample(self, batch_size: int) -> TensorDict:
    if self.data is None or self.priority is None or self.size == 0:
      raise RuntimeError("Cannot sample empty feedback replay")
    idx = torch.multinomial(self.priority[: self.size], batch_size, replacement=True)
    return self.data[idx]


class AmpFeedbackLoop:
  def __init__(self, runner: SuperMimicRunner, cfg: AmpFeedbackCfg):
    if runner.motion_prior is None:
      raise ValueError("AMP feedback requires the differentiable motion prior")
    if runner.cfg.actor_obs != "rep_state" or runner.cfg.history_length != 1:
      raise ValueError(
        "First feedback implementation uses official rep_state/history=1"
      )
    if runner.cfg.slot_mix != {"motion": 1.0, "velocity": 0.0}:
      raise ValueError(
        "Feedback currently supports the reference-conditioned motion task"
      )
    if not 0 < cfg.priority_fraction < 1 or not 0 < cfg.selected_fraction <= 1:
      raise ValueError("Keep both uniform WM data and priority feedback data")
    if cfg.holdout_stride < 2 or cfg.interval < 1 or cfg.capacity < 1:
      raise ValueError("Invalid feedback replay/holdout settings")
    self.runner, self.cfg = runner, cfg
    self.prior = runner.motion_prior
    self.branch = PhysicsBranch(runner.env)
    self.replay = WindowReplay(cfg.capacity)
    self.paired_replay = WindowReplay(cfg.capacity // 2)
    self.rounds = 0
    self.main_transitions = 0
    self._last_step = runner._start_step
    self._pending: dict | None = None
    self._round_record: dict = {}
    self._artifact: dict = {}
    self.priority_windows_used = 0
    self.fresh_windows = 0

  def due(self, step: int) -> bool:
    return (
      step - self._last_step >= self.cfg.interval
      and int(self.prior.updates) >= self.cfg.min_amp_updates
    )

  def _model_rollout(self, full_state, actions, context):
    r = self.runner._representation
    rep, aux = r.represent(full_state), r.extract_aux_state(full_state)
    states = [rep]
    for action in actions.unbind(1):
      aux, rep = self.runner._dynamics(rep, action, aux, context)
      states.append(rep)
    return torch.stack(states, dim=1)

  def _score(self, rep_states):
    # Same transform, discriminator, normalization and temporal reduction for
    # real and imagined branches; larger score means lower existing AMP cost.
    return -self.prior.style_cost(rep_states[:, :-1], rep_states[:, 1:]).mean(1)

  @torch.no_grad()
  def _baseline_actions(self, initial, context, times, horizon):
    runner, r = self.runner, self.runner._representation
    rep, aux = r.represent(initial), r.extract_aux_state(initial)
    times = times.clone()
    actions = []
    ends = runner.motion.motion_ends
    clip = torch.searchsorted(ends, times, right=True).clamp(max=len(ends) - 1)
    last = ends[clip] - 1
    for _ in range(horizon):
      goal = runner._active_task_specs[0].extract_training_goal(
        runner.motion, times, aux
      )
      obs = build_actor_obs_td({"motion": goal}, rep[:, None],
                              root_error=root_position_error(runner.motion, times, aux))
      action = runner.policy_learner.actor(obs)
      actions.append(action)
      aux, rep = runner._dynamics(rep, action, aux, context)
      times = torch.minimum(times + 1, last)
    return torch.stack(actions, dim=1)

  def _window(self, states, actions, context, times=None, previous=None):
    length = states.shape[1]
    # Official WM learner consumes L states and only the first L-1 actions.
    padded_actions = torch.cat([actions, torch.zeros_like(actions[:, :1])], 1)
    fields = {
        "state": states,
        "action": padded_actions,
        "sim_ctx": context[:, None].expand(-1, length, -1),
        "transition_mask": torch.ones(states.shape[:2], device=states.device),
        "feedback_pair_mask": torch.zeros(states.shape[:2], device=states.device),
    }
    if times is not None:
      ends = self.runner.motion.motion_ends
      clip = torch.searchsorted(ends, times, right=True).clamp(max=len(ends)-1)
      fields["motion_time"] = torch.minimum(times[:, None] + torch.arange(length, device=states.device), ends[clip, None]-1)
      fields["prev_action"] = torch.cat((previous[:, None], actions), dim=1)
    return TensorDict(fields, batch_size=list(states.shape[:2]))

  def collect(self, step: int, log_dir: Path) -> dict[str, torch.Tensor]:
    runner, cfg = self.runner, self.cfg
    runner._dynamics.eval().requires_grad_(False)
    runner.policy_learner.actor.eval()
    self.prior.freeze()
    context = runner._sim_context.read(runner.env.unwrapped).detach().clone()
    self.branch.capture()
    initial = self.branch.full_state().detach().clone()
    times = runner.motion_cmd.time_steps.detach().clone()
    previous = runner.action_manager.action.detach().clone()
    # Acquisition windows match the official 8-state WM horizon. Policy BPTT
    # remains 32; this is an additional data collection operation only.
    actions = self._baseline_actions(initial, context, times, runner.cfg.wm.horizon - 1)
    proposal = actions.detach().requires_grad_(True)
    predicted = self._model_rollout(initial, proposal, context)
    gradient = torch.autograd.grad(self._score(predicted).sum(), proposal)[0]
    if not torch.isfinite(gradient).all():
      raise RuntimeError("Nonfinite AMP action proposal gradient")
    rms = gradient.square().mean((1, 2), keepdim=True).sqrt().clamp(min=1e-8)
    delta = (
      (cfg.action_delta_rms * gradient / rms)
      .clamp(-cfg.action_delta_max, cfg.action_delta_max)
      .detach()
    )
    adjusted = actions + delta
    del predicted, gradient, proposal

    with torch.no_grad():
      wm_base = self._model_rollout(initial, actions, context)
      wm_adjusted = self._model_rollout(initial, adjusted, context)
      base_score = self._score(wm_base)
      # A large normalized step can overshoot a useful local AMP direction.
      # Backtrack per state using only the model; this costs no sim interactions.
      for _ in range(3):
        worse = self._score(wm_adjusted) < base_score
        if not worse.any():
          break
        delta = torch.where(worse[:, None, None], delta * 0.5, delta)
        adjusted = actions + delta
        wm_adjusted = self._model_rollout(initial, adjusted, context)
      real_base = self.branch.rollout(actions)
      real_adjusted = self.branch.rollout(adjusted)
      repeated_base = self.branch.rollout(actions)
      repeated_adjusted = self.branch.rollout(adjusted)
      base_repeat_error = (repeated_base - real_base).abs()
      adjusted_repeat_error = (repeated_adjusted - real_adjusted).abs()
      repeat_error = float(
        torch.maximum(base_repeat_error.max(), adjusted_repeat_error.max())
      )
      per_env_repeat = torch.maximum(
        (base_repeat_error / (1 + real_base.abs())).flatten(1).amax(1),
        (adjusted_repeat_error / (1 + real_adjusted.abs())).flatten(1).amax(1),
      )
      reliable = per_env_repeat <= cfg.repeat_tolerance
      if float(reliable.float().mean()) < 0.8:
        raise RuntimeError(
          "Fewer than 80% of physical pairs pass repeatability; inspect the simulator"
        )
      r = runner._representation
      real_base_rep, real_adjusted_rep = (
        r.represent(real_base),
        r.represent(real_adjusted),
      )
      real_base_score, real_adjusted_score = (
        self._score(real_base_rep),
        self._score(real_adjusted_rep),
      )
      score_noise = (
        self._score(r.represent(repeated_base)) - real_base_score
      ).abs() + (
        self._score(r.represent(repeated_adjusted)) - real_adjusted_score
      ).abs()
      gap, predicted_gain, actual_gain = improvement_disagreement(
        self._score(wm_base),
        self._score(wm_adjusted),
        real_base_score,
        real_adjusted_score,
      )
      if not torch.isfinite(gap).all():
        raise RuntimeError("Nonfinite AMP discrepancy")
      count = len(initial)
      # A few difficult contact states can amplify GPU reduction noise under the
      # unchanged official solver budget. Do not label that as a model error.
      signal = (gap - 5 * score_noise - 1e-5).clamp(min=0)
      eligible = reliable.nonzero().squeeze(-1)
      heldout = torch.zeros(count, dtype=torch.bool, device=initial.device)
      heldout[
        eligible[
          torch.randperm(len(eligible), device=initial.device)[
            : max(1, len(eligible) // cfg.holdout_stride)
          ]
        ]
      ] = True
      train_ids = (reliable & ~heldout).nonzero().squeeze(-1)
      selected_count = max(1, int(len(train_ids) * cfg.selected_fraction))
      selected = train_ids[signal[train_ids].topk(selected_count).indices]
      selected_mask = torch.zeros_like(heldout)
      selected_mask[selected] = True

      # Revisit exactly those snapshots with fresh local action perturbations.
      # Warp advances all E worlds; non-selected rows are counted as interactions
      # and retained as general WM windows, not hidden from the sample budget.
      extra_actions = torch.where(selected_mask[:, None, None], adjusted, actions)
      extra_actions = extra_actions + (
        torch.randn_like(actions) * cfg.local_noise * selected_mask[:, None, None]
      )
      real_extra = self.branch.rollout(extra_actions)
      priority = 1 + (signal / signal[eligible].median().clamp(min=1e-5)).clamp(
        max=cfg.priority_cap - 1
      )
      for states, controls in [
        (real_base, actions),
        (real_adjusted, adjusted),
        (real_extra, extra_actions),
      ]:
        windows = self._window(states, controls, context, times, previous)
        self.replay.insert(windows[train_ids], priority[train_ids])
        self.fresh_windows += len(train_ids)

      # Preserve the two interventions as a pair, not independent priority rows.
      # Training recomputes their real score difference with the CURRENT frozen D.
      paired = self._window(real_base, actions, context)
      paired["adjusted_state"] = real_adjusted
      paired["adjusted_action"] = torch.cat((adjusted, torch.zeros_like(adjusted[:, :1])), 1)
      self.paired_replay.insert(paired[train_ids], priority[train_ids])
      for replay in (self.replay, self.paired_replay):
        replay.priority[:replay.size].lerp_(torch.ones_like(replay.priority[:replay.size]), 1-cfg.priority_decay)

      restore_error = self.branch.restoration_error()
      state_error = float((self.branch.full_state() - initial).abs().max())
      context_error = float(
        (runner._sim_context.read(runner.env.unwrapped) - context).abs().max()
      )
      if max(restore_error, state_error, context_error) > cfg.repeat_tolerance:
        raise RuntimeError("Feedback collection changed the live training state")
      if not torch.equal(times, runner.motion_cmd.time_steps):
        raise RuntimeError("Feedback collection advanced the reference clock")

      ids = heldout.nonzero().squeeze(-1)
      self._pending = {
        "initial": initial[ids],
        "context": context[ids],
        "base_actions": actions[ids],
        "adjusted_actions": adjusted[ids],
        "real_base_rep": real_base_rep[ids],
        "real_adjusted_rep": real_adjusted_rep[ids],
        "actual_gain": actual_gain[ids],
        "prior_version": int(self.prior.updates),
        "before_gap": float(gap[ids].mean()),
      }
      floor = self._state_error_scale(wm_base)
      before_error = 0.5 * (
        ((wm_base[ids, 1:] - real_base_rep[ids, 1:]) / floor).square().mean()
        + ((wm_adjusted[ids, 1:] - real_adjusted_rep[ids, 1:]) / floor).square().mean()
      )
      self.rounds += 1
      self._last_step = step
      self._round_record = {
        "round": self.rounds,
        "step": step,
        "amp_version": int(self.prior.updates),
        "physics_repeat_max_error": repeat_error,
        "physics_repeat_scaled_max_error": float(per_env_repeat.max()),
        "accepted_repeat_scaled_max_error": float(per_env_repeat[eligible].max()),
        "physics_repeat_score_max_error": float(score_noise.max()),
        "accepted_mean_score_noise": float(score_noise[eligible].mean()),
        "repeat_rejected_environments": int((~reliable).sum()),
        "restoration_max_error": restore_error,
        "initial_state_max_error": state_error,
        "context_max_error": context_error,
        "environments": count,
        "selected_environments": len(selected),
        "heldout_environments": len(ids),
        "paired_transitions": count * actions.shape[1] * 2,
        "repeat_check_transitions": count * actions.shape[1] * 2,
        "supplement_transitions": count * actions.shape[1],
        "selected_supplement_transitions": len(selected) * actions.shape[1],
        "mean_gain_disagreement": float(gap.mean()),
        "accepted_mean_gain_disagreement": float(gap[eligible].mean()),
        "selected_mean_disagreement": float(gap[selected].mean()),
        "mean_predicted_gain": float(predicted_gain.mean()),
        "mean_actual_gain": float(actual_gain.mean()),
        "false_improvement_fraction": float(
          ((predicted_gain > 1e-5) & (actual_gain < 0)).float().mean()
        ),
        "action_delta_rms": float(delta.square().mean().sqrt()),
        "heldout_gap_before": self._pending["before_gap"],
        "heldout_normalized_state_mse_before": float(before_error),
        "priority_windows_stored": self.replay.size,
      }
      # Keep a concrete matched-action record. Holdout rows never enter WM replay.
      self._artifact = {
        "config": asdict(cfg),
        "record": self._round_record,
        "initial": initial[ids],
        "sim_ctx": context[ids],
        "motion_time": times[ids],
        "base_actions": actions[ids],
        "adjusted_actions": adjusted[ids],
        "real_base": real_base[ids],
        "real_adjusted": real_adjusted[ids],
        "predicted_base_rep": wm_base[ids],
        "predicted_adjusted_rep": wm_adjusted[ids],
        "holdout_env_ids": ids,
        "selected_train_env_ids": selected,
        "prior_cfg": asdict(self.prior.cfg),
        "feature_encoder": (
          self.prior.feature_encoder.metadata()
          if self.prior.feature_encoder is not None else None
        ),
        "prior_state": {
          key: value.detach().clone() for key, value in self.prior.state_dict().items()
        },
        "real_base_score": real_base_score[ids],
        "real_adjusted_score": real_adjusted_score[ids],
      }
    return self.metrics()

  def mix_batch(self, uniform: TensorDict) -> TensorDict:
    uniform = uniform.select("state", "action", "sim_ctx", "transition_mask")
    uniform["feedback_pair_mask"] = torch.zeros(uniform.shape, device=uniform["state"].device)
    if not self.paired_replay.size:
      return uniform
    total = int(len(uniform) * self.cfg.priority_fraction)
    count = total // 4
    self.priority_windows_used += total
    pair = self.paired_replay.sample(count)
    keys = ("state", "action", "sim_ctx", "transition_mask", "feedback_pair_mask")
    base = pair.select(*keys).clone()
    adjusted = base.clone()
    adjusted["state"] = pair["adjusted_state"]
    adjusted["action"] = pair["adjusted_action"]
    base["feedback_pair_mask"].fill_(1)
    adjusted["feedback_pair_mask"].fill_(1)
    # Half of feedback slots are matched interventions; the remaining half also
    # includes fresh local perturbations, which must still supervise the WM.
    fresh = self.replay.sample(total-2*count).select(*keys).clone()
    fresh["feedback_pair_mask"].zero_()
    return torch.cat((base, adjusted, fresh, uniform[total:]), dim=0)

  def mix_amp_pairs(self, uniform: torch.Tensor) -> torch.Tensor:
    if not self.replay.size:
      return uniform
    count = int(len(uniform) * self.cfg.discriminator_fraction)
    windows = self.replay.sample(count)
    start = torch.randint(windows.shape[1]-1, (count,), device=uniform.device)
    rows = torch.arange(count, device=uniform.device)
    pairs = torch.stack((windows["state"][rows, start], windows["state"][rows, start+1]), dim=1)
    return torch.cat((uniform[count:], pairs), dim=0)

  def mix_policy_starts(self, uniform: TensorDict) -> TensorDict:
    if not self.replay.size:
      return uniform
    count = int(len(uniform) * self.cfg.policy_start_fraction)
    if count == 0:
      return uniform
    windows = self.replay.sample(count)
    times = windows["motion_time"][:, 1]
    state = windows["state"][:, 1]
    motion = self.runner.motion
    height = (state[:, 2] - motion.root_pos_w[times, 2]).abs()
    gravity = 2 * state[:, 4:6].square().sum(-1)-1
    reference_gravity = 2 * motion.root_quat_w[times, 1:3].square().sum(-1)-1
    valid = (height < self.runner.wm_env._anchor_pos_threshold) & ((gravity-reference_gravity).abs() < self.runner.wm_env._anchor_ori_threshold)
    idx = valid.long()  # first real successor if alive, otherwise the captured start
    selected = windows[torch.arange(count, device=state.device), idx]
    result = uniform.clone()
    for key in ("state", "prev_action", "motion_time", "sim_ctx"):
      result[key][:count] = selected[key]
    return result

  @torch.no_grad()
  def after_world_update(self, log_dir: Path) -> dict[str, torch.Tensor]:
    if self._pending is None:
      return self.metrics()
    p = self._pending
    if int(self.prior.updates) != p["prior_version"]:
      raise RuntimeError("D changed during the before/after world-model comparison")
    self.runner._dynamics.eval().requires_grad_(False)
    base = self._model_rollout(p["initial"], p["base_actions"], p["context"])
    adjusted = self._model_rollout(p["initial"], p["adjusted_actions"], p["context"])
    after_gap = (
      (self._score(adjusted) - self._score(base) - p["actual_gain"]).abs().mean()
    )
    floor = self._state_error_scale(base)
    state_error = 0.5 * (
      ((base[:, 1:] - p["real_base_rep"][:, 1:]) / floor).square().mean()
      + ((adjusted[:, 1:] - p["real_adjusted_rep"][:, 1:]) / floor).square().mean()
    )
    self._round_record.update(
      {
        "heldout_gap_after": float(after_gap),
        "heldout_gap_change": float(after_gap) - p["before_gap"],
        "heldout_normalized_state_mse_after": float(state_error),
        **self.budget(),
      }
    )
    self._artifact["updated_base_rep"] = base
    self._artifact["updated_adjusted_rep"] = adjusted
    with (log_dir / "amp_feedback_rounds.jsonl").open("a") as out:
      out.write(json.dumps(self._round_record) + "\n")
    torch.save(self._artifact, log_dir / "amp_feedback_last_pair.pt")
    print("[AMP feedback] " + json.dumps(self._round_record), flush=True)
    self._pending = None
    self._artifact = {}
    return self.metrics()

  def budget(self) -> dict[str, int]:
    return {
      "main_transitions": self.main_transitions,
      "feedback_transitions": self.branch.simulated_transitions,
      "total_transitions": self.main_transitions + self.branch.simulated_transitions,
      "priority_windows_used": self.priority_windows_used,
      "fresh_windows": self.fresh_windows,
      "rounds": self.rounds,
    }

  @staticmethod
  def _state_error_scale(states: torch.Tensor) -> torch.Tensor:
    """GenCoord diagnostic units, independent of the discriminator features.

    Retain the original 68-D state diagnostic when AMP uses body features.
    This quantity is only logged; acquisition uses paired discriminator gains.
    """
    dim = states.shape[-1]
    scale = states.new_full((dim,), 0.1)
    scale[4:10] = 0.5
    scale[10 + (dim - 10) // 2:] = 1.0
    return scale

  def metrics(self) -> dict[str, torch.Tensor]:
    values = {**self._round_record, **self.budget()}
    return {
      f"Feedback/{key}": torch.tensor(value, device=self.runner.device)
      for key, value in values.items()
      if isinstance(value, (int, float))
    }

  def state_dict(self) -> dict:
    # Like the official simulator replay, priority replay is recollected on resume.
    return {"budget": self.budget(), "last_step": self._last_step}

  def load_state_dict(self, state: dict) -> None:
    budget = state["budget"]
    self.main_transitions = budget["main_transitions"]
    self.branch.simulated_transitions = budget["feedback_transitions"]
    self.priority_windows_used = budget["priority_windows_used"]
    self.fresh_windows = budget["fresh_windows"]
    self.rounds = budget["rounds"]
    self._last_step = state["last_step"]

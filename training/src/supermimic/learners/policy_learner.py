"""Policy learners using a differentiable world model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.learners.amp import DifferentiableMotionPrior
from supermimic.learners.learner_utils import (
  CapturedUpdate,
  _optimize_step,
  build_optimizers,
  graph_state,
)


@dataclass
class BpttPolicyLearnerCfg:
  lr: float = 2e-4
  num_batches: int = 1
  batch_size: int = 4096
  noise: float = 0.1
  horizon: int = 32
  gamma: float = 1.0
  survival_masking: bool = True
  beta2: float = 0.95  # faster Adam EMA recovery after gradient spikes
  grad_clip_value: float = 0.1  # per-parameter gradient element clipping
  detach_style_survival: bool = True


@dataclass
class ActorCriticPolicyLearnerCfg(BpttPolicyLearnerCfg):
  gamma: float = 0.99
  lam: float = 0.95


@dataclass
class ImaginedPolicyObjective:
  """One connected rollout, shared by training and offline gradient diagnostics.

  ``task_loss`` contains the original tracking return. The critic bootstrap is
  kept separate because the critic learns the combined task/style return.
  Only ``critic_rollout`` is detached; every loss keeps its path to the actor.
  """

  task_loss: torch.Tensor
  style_loss: torch.Tensor
  style_weight: torch.Tensor
  bootstrap_loss: torch.Tensor
  auxiliary_loss: torch.Tensor
  style_cost: torch.Tensor
  critic_rollout: list[TensorDict]
  last_obs: TensorDict
  first_action: torch.Tensor
  final_style_cost: torch.Tensor

  @property
  def weighted_style_loss(self) -> torch.Tensor:
    return self.style_weight * self.style_loss

  @property
  def total_loss(self) -> torch.Tensor:
    return (
      self.task_loss
      + self.weighted_style_loss
      + self.bootstrap_loss
      + self.auxiliary_loss
    )


def gae_returns(
  values: torch.Tensor,
  last_value: torch.Tensor,
  rewards: torch.Tensor,
  dones: torch.Tensor,
  gamma: float,
  lam: float,
) -> torch.Tensor:
  """GAE returns ``A_t + V(s_t)`` of a time-major rollout. A soft done ``p_t``
  discounts the next state by ``(1 - p_t) * gamma``."""
  returns = torch.empty_like(values)
  advantage = 0.0
  next_value = last_value
  for t in reversed(range(values.shape[0])):
    not_done = 1.0 - dones[t]
    delta = rewards[t] + not_done * gamma * next_value - values[t]
    advantage = delta + not_done * gamma * lam * advantage
    returns[t] = advantage + values[t]
    next_value = values[t]
  return returns


class BpttPolicyLearner:
  """Policy Learner using direct Backpropagation Through Time (BPTT)."""

  def __init__(
    self,
    policy: nn.Module,
    wm_env: WorldModelEnv,
    cfg: BpttPolicyLearnerCfg,
    device: str = "cpu",
  ):
    self.actor = policy
    self.wm_env = wm_env
    self.cfg = cfg
    self.motion_prior: DifferentiableMotionPrior | None = None
    self.actor_params = list(self.actor.parameters())
    self.actor_optimizers = build_optimizers(
      [self.actor], self.cfg.lr, betas=(0.9, self.cfg.beta2), device=device
    )
    self._rollout_actor = (
      torch.compile(self.actor) if torch.device(device).type == "cuda" else self.actor
    )
    self._captured_update = CapturedUpdate(self._update_from, self._graph_state)

  def update(self) -> dict[str, torch.Tensor]:
    self.actor.train()
    for module in (self.wm_env.dynamics, self.wm_env.representation):
      module.eval()
      module.requires_grad_(False)
    if self.motion_prior is not None:
      self.motion_prior.freeze()
    starts = [self.wm_env.sample_initial_states() for _ in range(self.cfg.num_batches)]
    return self._captured_update(torch.stack(starts))

  def _update_from(self, starts: TensorDict) -> dict[str, torch.Tensor]:
    totals: dict[str, torch.Tensor] = {}
    for start in starts.unbind(0):
      for k, v in self._train_batch(start).items():
        totals[k] = totals[k] + v if k in totals else v
    n = self.cfg.num_batches
    reward = totals.pop("return") / self.cfg.horizon / n / self.wm_env.representation.dt
    return {"Policy/reward": reward} | {f"Policy/{k}": v / n for k, v in totals.items()}

  def _train_batch(self, start: TensorDict) -> dict[str, torch.Tensor]:
    """One actor step on explicitly separated, end-to-end differentiable losses."""
    objective = self._imagine_batch(start)
    loss = objective.total_loss
    grad_norm = _optimize_step(
      self.actor_optimizers, loss, self.actor_params, 1.0, self.cfg.grad_clip_value
    )
    return {
      "return": (
        -objective.task_loss - objective.weighted_style_loss - objective.bootstrap_loss
      ).detach(),
      "grad_norm": grad_norm,
      "task_loss": objective.task_loss.detach(),
      "style_loss": objective.style_loss.detach(),
      "weighted_style_loss": objective.weighted_style_loss.detach(),
      "bootstrap_loss": objective.bootstrap_loss.detach(),
      "auxiliary_loss": objective.auxiliary_loss.detach(),
      "total_loss": loss.detach(),
      **(
        {"style_cost": objective.style_cost.detach()}
        if self.motion_prior is not None
        else {}
      ),
      **self._fit_critic(objective.critic_rollout, objective.last_obs),
      **getattr(self.actor, "auxiliary_stats", lambda: {})(),
    }

  def _imagine_batch(self, start: TensorDict) -> ImaginedPolicyObjective:
    """Build the official BPTT objective without stepping any optimizer.

    Frozen WM/D parameters must still be evaluated in grad mode. Do not detach
    actions, imagined states, or normalised AMP inputs. The optional detached
    style survival mask changes only the weighting derivative, not the path
    from the discriminator through predicted states and actions.
    ``reset`` detaches only the replay start; the following 32 steps stay joined.
    """
    cfg = self.cfg
    obs = self.wm_env.reset(start)
    zero = obs["critic"].new_zeros(())
    task_return, style_return, aux_loss, survival = zero, zero, zero, 1.0
    rollout: list[TensorDict] = []
    style_cost_sum, style_cost = zero, zero
    weight = self.motion_prior.weight if self.motion_prior is not None else zero
    dt = self.wm_env.representation.dt
    first_action = None
    for t in range(cfg.horizon):
      prior_state = self.wm_env.rep_state if self.motion_prior is not None else None
      actions = self._rollout_actor(obs)
      if first_action is None:
        first_action = actions
      next_obs, task_reward, term_prob, _ = self.wm_env.step(actions)
      discount = cfg.gamma**t * survival
      task_return = task_return + discount * task_reward
      combined_reward = task_reward
      if self.motion_prior is not None:
        assert prior_state is not None
        style_cost = self.motion_prior.style_cost(prior_state, self.wm_env.rep_state)
        style_cost_sum = style_cost_sum + style_cost.mean()
        # A positive BCE cost must not offer a direct shortcut by terminating
        # the imagined trajectory. Task survival gradients remain unchanged.
        style_discount = discount.detach() if torch.is_tensor(discount) and cfg.detach_style_survival else discount
        style_return = style_return + style_discount * dt * style_cost
        combined_reward = task_reward - weight * dt * style_cost
      # Only the critic's target copy is detached. Neither actor loss uses it.
      step = {"critic": obs["critic"], "reward": combined_reward, "done": term_prob}
      rollout.append(TensorDict(step, batch_size=task_reward.shape).detach())
      aux_loss = aux_loss + getattr(self.actor, "auxiliary_loss", lambda: 0.0)()
      if cfg.survival_masking:
        survival = survival * (1 - term_prob)
      obs = next_obs
    assert first_action is not None, "BPTT horizon must be positive"
    bootstrap = self._bootstrap(obs, survival, term_prob)
    return ImaginedPolicyObjective(
      task_loss=-task_return.mean(),
      style_loss=style_return.mean(),
      style_weight=weight,
      bootstrap_loss=-(zero + bootstrap).mean(),
      auxiliary_loss=aux_loss.mean(),
      style_cost=style_cost_sum / cfg.horizon,
      critic_rollout=rollout,
      last_obs=obs,
      first_action=first_action,
      final_style_cost=style_cost.mean(),
    )

  def _bootstrap(
    self, obs: TensorDict, survival: torch.Tensor, term_prob: torch.Tensor
  ) -> torch.Tensor | float:
    """Value of the rollout's last state, discounted into its return."""
    del obs, survival, term_prob
    return 0.0

  def _fit_critic(
    self, rollout: list[TensorDict], last_obs: TensorDict
  ) -> dict[str, torch.Tensor]:
    """Fit a critic to the rollout after the actor step."""
    del rollout, last_obs
    return {}

  def _graph_state(self) -> list[torch.Tensor]:
    """What the captured update reads that Python can rebind."""
    wm = self.wm_env
    modules = [self.actor, wm.dynamics, wm.representation]
    if self.motion_prior is not None:
      modules.append(self.motion_prior)
    return graph_state(modules, self.actor_optimizers)


class ActorCriticPolicyLearner(BpttPolicyLearner):
  """Actor-critic variant that adds a critic bootstrap and GAE critic update."""

  def __init__(
    self,
    policy: nn.Module,
    critic: nn.Module,
    wm_env: WorldModelEnv,
    cfg: ActorCriticPolicyLearnerCfg,
    device: str = "cpu",
  ):
    super().__init__(policy, wm_env, cfg, device)
    self.critic = critic
    self.critic_optimizers = build_optimizers([critic], self.cfg.lr, device=device)
    self._v_max = self.wm_env.max_step_reward / (1 - self.cfg.gamma)

  def _bootstrap(
    self, obs: TensorDict, survival: torch.Tensor, term_prob: torch.Tensor
  ) -> torch.Tensor:
    self.critic.requires_grad_(False)
    value = self.critic(obs["critic"]).clamp(max=self._v_max).squeeze(-1)
    alive = survival if self.cfg.survival_masking else 1.0 - term_prob
    return self.cfg.gamma**self.cfg.horizon * value * alive

  def _fit_critic(
    self, rollout: list[TensorDict], last_obs: TensorDict
  ) -> dict[str, torch.Tensor]:
    steps = torch.stack(rollout)
    with torch.no_grad():
      values = self.critic(steps["critic"])
      last_value = self.critic(last_obs["critic"])
    returns = gae_returns(
      values,
      last_value,
      steps["reward"].unsqueeze(-1),
      steps["done"].unsqueeze(-1),
      self.cfg.gamma,
      self.cfg.lam,
    ).clamp(max=self._v_max)
    obs, returns = steps["critic"].flatten(0, 1), returns.flatten(0, 1)
    perm = torch.randperm(len(returns) // 4 * 4, device=returns.device)
    self.critic.requires_grad_(True)
    losses = []
    for _ in range(5):
      for idx in perm.view(4, -1):
        loss = (returns[idx] - self.critic(obs[idx])).pow(2).mean()
        _optimize_step(
          self.critic_optimizers, loss, self.critic.parameters(), max_norm=1.0
        )
        losses.append(loss.detach())
    return {"value_function": torch.stack(losses).mean()}

  def _graph_state(self) -> list[torch.Tensor]:
    return super()._graph_state() + graph_state([self.critic], self.critic_optimizers)

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.amp import AmpCfg, DifferentiableMotionPrior
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
)
from supermimic.models import build_mlp, run_mlp
from supermimic.scripts.gradient_audit import audit_policy_objective


def make_prior():
  torch.manual_seed(12)
  frames = torch.randn(80, 14)
  return DifferentiableMotionPrior(
    frames, torch.tensor([40, 80]), AmpCfg(hidden_dims=(32, 16))
  )


def test_expert_pairs_never_cross_clips():
  prior = make_prior()
  assert 39 not in prior.pair_starts.tolist()
  assert 79 not in prior.pair_starts.tolist()
  assert len(prior.pair_starts) == 78


def test_style_gradient_passes_through_frozen_dynamics_to_earlier_action():
  prior = make_prior()
  actor = nn.Linear(14, 3)
  dynamics = nn.Linear(17, 14).requires_grad_(False)
  state = torch.randn(8, 14)
  first_action = actor(state)
  next_state = dynamics(torch.cat([state, first_action], -1))
  final_state = dynamics(torch.cat([next_state, actor(next_state)], -1))
  loss = prior.style_cost(next_state, final_state).mean()
  grads = torch.autograd.grad(loss, [actor.weight, first_action])
  assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
  assert all(p.grad is None and not p.requires_grad for p in prior.parameters())
  assert all(p.grad is None for p in dynamics.parameters())


def test_normalization_is_differentiable_and_checkpointed():
  prior = make_prior()
  states = torch.randn(8, 14, requires_grad=True)
  loss = prior.style_cost(states, states + 0.1).mean()
  assert torch.autograd.grad(loss, states)[0].norm() > 0
  restored = make_prior()
  restored.load_state_dict(prior.state_dict())
  assert torch.equal(
    prior.style_cost(states, states), restored.style_cost(states, states)
  )


def test_activation_recomputation_preserves_outputs_and_full_gradients():
  torch.manual_seed(8)
  mlp = build_mlp(14, (32, 16), 5)
  inputs = torch.randn(8, 14, requires_grad=True)
  expected = run_mlp(mlp, inputs, False)
  grad_expected = torch.autograd.grad(
    expected.square().mean(), [inputs, *mlp.parameters()]
  )
  actual = run_mlp(mlp, inputs, True)
  grad_actual = torch.autograd.grad(actual.square().mean(), [inputs, *mlp.parameters()])
  torch.testing.assert_close(actual, expected)
  for actual_grad, expected_grad in zip(grad_actual, grad_expected, strict=True):
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 verification")
def test_bf16_mlp_retains_finite_aligned_pathwise_gradients():
  torch.manual_seed(8)
  mlp = build_mlp(14, (128, 64), 5).cuda()
  inputs = torch.randn(32, 14, device="cuda", requires_grad=True)
  exact = run_mlp(mlp, inputs, True, False)
  exact_grads = torch.autograd.grad(exact.square().mean(), [inputs, *mlp.parameters()])
  fast = run_mlp(mlp, inputs, True, True)
  fast_grads = torch.autograd.grad(fast.square().mean(), [inputs, *mlp.parameters()])
  assert fast.dtype == torch.float32
  exact_vec = torch.cat([g.flatten() for g in exact_grads])
  fast_vec = torch.cat([g.flatten() for g in fast_grads])
  assert torch.isfinite(fast_vec).all()
  assert torch.nn.functional.cosine_similarity(exact_vec, fast_vec, dim=0) > 0.99
  assert (fast_vec - exact_vec).norm() / exact_vec.norm() < 0.05


class _AuditActor(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(14, 3)

  def forward(self, obs):
    return self.linear(obs["critic"])

  def auxiliary_loss(self):
    return 0.01 * self.linear.weight.square().mean()


class _AuditEnv:
  """State-dependent reward/termination, with optional deliberate graph damage."""

  max_step_reward = 1.0

  def __init__(self, break_graph=False):
    self.dynamics = nn.Linear(17, 14).requires_grad_(False)
    self.representation = nn.Module()
    self.representation.dt = 0.02
    self.break_graph = break_graph
    with torch.no_grad():
      self.dynamics.weight.zero_()
      self.dynamics.weight[:, :14].copy_(0.95 * torch.eye(14))
      self.dynamics.weight[:, 14:].fill_(0.015)
      self.dynamics.bias.fill_(0.001)

  def reset(self, batch):
    self.rep_state = batch["state"].detach()
    self.steps = []
    return TensorDict({"critic": self.rep_state}, batch_size=[len(self.rep_state)])

  def step(self, action):
    previous = self.rep_state
    self.rep_state = self.dynamics(torch.cat([previous, action], -1))
    if self.break_graph:
      self.rep_state = self.rep_state.detach().requires_grad_(True)
    reward = self.representation.dt * torch.exp(-self.rep_state.square().mean(-1))
    done = torch.sigmoid(self.rep_state[:, 0] - 4)
    self.steps.append((previous, self.rep_state, reward, done))
    obs = TensorDict({"critic": self.rep_state}, batch_size=[len(self.rep_state)])
    return obs, reward, done, {}


def _make_audit_learner(break_graph=False):
  prior = make_prior()
  prior.weight.fill_(0.1)
  env = _AuditEnv(break_graph)
  learner = ActorCriticPolicyLearner(
    _AuditActor(),
    nn.Linear(14, 1),
    env,
    ActorCriticPolicyLearnerCfg(horizon=32),
  )
  learner.motion_prior = prior
  start = TensorDict({"state": torch.randn(8, 14)}, batch_size=[8])
  return learner, start


def test_explicit_losses_preserve_original_return_and_parameter_gradients():
  learner, start = _make_audit_learner()
  learner.cfg.detach_style_survival = False  # Legacy objective equivalence mode.
  objective = learner._imagine_batch(start)
  # Independently reconstruct the previous, per-step combined-return objective.
  ret, survival = 0.0, 1.0
  prior = learner.motion_prior
  for t, (previous, state, task_reward, done) in enumerate(learner.wm_env.steps):
    reward = task_reward - prior.weight * 0.02 * prior.style_cost(previous, state)
    ret = ret + learner.cfg.gamma**t * survival * reward
    survival = survival * (1 - done)
  bootstrap = learner._bootstrap(objective.last_obs, survival, done)
  original = (
    -ret - bootstrap + learner.cfg.horizon * learner.actor.auxiliary_loss()
  ).mean()
  torch.testing.assert_close(objective.total_loss, original)
  actual = torch.autograd.grad(
    objective.total_loss, learner.actor_params, retain_graph=True
  )
  expected = torch.autograd.grad(original, learner.actor_params)
  for actual_grad, expected_grad in zip(actual, expected, strict=True):
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-7, rtol=2e-5)
  assert not objective.critic_rollout[0]["reward"].requires_grad
  assert objective.style_loss.requires_grad


@pytest.mark.parametrize("break_graph", [False, True])
def test_loss_audit_detects_full_horizon_graph_break_without_updating_weights(
  break_graph,
):
  learner, start = _make_audit_learner(break_graph)
  before = {key: value.clone() for key, value in learner.actor.state_dict().items()}
  audit = audit_policy_objective(learner, start)
  assert audit["passed"] is not break_graph
  assert audit["horizon"] == 32
  assert audit["optimizer_steps"] == 0
  assert audit["world_model_parameter_grads_empty"]
  assert audit["discriminator_parameter_grads_empty"]
  if not break_graph:
    assert audit["final_pair_to_first_action_gradient_norm"] > 0
    assert -1.00001 <= audit["task_style_gradient_cosine"] <= 1.00001
    assert audit["weighted_style_to_task_gradient_ratio"] > 0
  for key, value in learner.actor.state_dict().items():
    torch.testing.assert_close(value, before[key], atol=0, rtol=0)

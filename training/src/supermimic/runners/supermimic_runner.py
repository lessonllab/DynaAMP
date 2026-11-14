from __future__ import annotations

import json
import copy
import math
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import torch
import torch.nn as nn
import tyro
import wandb
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from supermimic.envs.motion_tracking_env import MotionTrackingEnv
from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.learners.amp import AmpCfg, DifferentiableMotionPrior
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics, MultiEncoderAE, build_mlp
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.representations import StateRepresentation, create_representation, root_position_error
from supermimic.runners.amp_feedback import AmpFeedbackCfg, AmpFeedbackLoop
from supermimic.runners.exporter import export_actor_as_onnx
from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
from supermimic.sim_context import SimContext
from supermimic.tasks.task_spec import (
  TasksCfg,
  build_actor_obs_td,
  resolve_task_specs,
)
from supermimic.utils.helpers import (
  extract_action_scale_offset,
  unpack_obs_dims,
)
from supermimic.utils.logger import Logger
from supermimic.utils.wandb_artifacts import download_model_dir, log_model_artifact


@dataclass
class BaseRunnerCfg:
  """Shared configuration fields for all runners."""

  wandb_project: str = "supermimic"
  run_name: str | None = None
  wandb_run_path: str | None = None
  headless: bool = True
  buffer_size: int = 1000
  representation: str = "gencoord"  # "gencoord" or "maxcoord"
  actor_obs: str = "rep_state"  # "rep_state" or "proprio"
  log_interval: int = 10
  save_interval: int = 1000
  history_length: int = 1
  termination_softness: float = 0.05
  dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  mlp_bf16: bool = False
  """BlinkMimic-style BF16 actor/dynamics MLPs; FP32 state and losses."""
  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)


BpttCfg = Annotated[BpttPolicyLearnerCfg, tyro.conf.subcommand(name="bptt")]
ActorCriticCfg = Annotated[
  ActorCriticPolicyLearnerCfg, tyro.conf.subcommand(name="actor-critic")
]


@dataclass
class SuperMimicRunnerCfg(BaseRunnerCfg):
  root_position_feedback: bool = False
  policy_ema_decay: float = 0.995
  failure_focus_fraction: float = 0.2
  failure_lookback_seconds: float = 3.0
  max_iterations: int = 1000000
  max_sim_transitions: int | None = None
  """Optional real-transition cap, including every feedback physics branch."""
  max_amp_updates: int | None = None
  """Stop after this many joint world-model, AMP and policy updates."""
  budget_checkpoints: tuple[int, ...] = ()
  amp: AmpCfg = field(default_factory=AmpCfg)
  amp_feedback: AmpFeedbackCfg = field(default_factory=AmpFeedbackCfg)
  resume_checkpoint: str | None = None
  policy_activation_checkpointing: bool = False
  """Recompute MLP activations during BPTT; preserve full batch and horizon."""
  replay_ratio: int = 16
  latent_dim: int = 32
  actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden_dims: tuple[int, ...] = (256, 128)
  latent_bc_weight: float = 1e-1
  """Weight on the latent BC loss pulling non-anchor encoders toward the
  anchor encoder's latent on the same state. 0 disables."""
  tasks: TasksCfg = field(default_factory=TasksCfg)
  """Per-task hyperparameters."""
  slot_mix: dict[str, float] = field(
    default_factory=lambda: {"motion": 1.0, "velocity": 0.0}
  )
  """Fractional slot allocation per task. Must sum to 1.0."""
  policy: BpttCfg | ActorCriticCfg = field(default_factory=ActorCriticCfg)

  sampling_mode: Literal["uniform", "adaptive", "start", "return"] = "return"
  """RSI reset distribution: 'return' (concentrate resets on the bins where the
  policy earns low return; see ReturnSampler) or one of mjlab's native modes.
  train.py resolves this into the motion command's cfg before the env is built
  ('return' drives mjlab's 'external' mode)."""
  return_sampler: ReturnSamplerCfg = field(default_factory=ReturnSamplerCfg)


@dataclass
class WorldModel:
  """The world model plus the env dimensions that sized it."""

  representation: StateRepresentation
  dynamics: Dynamics
  sim_context: SimContext
  num_goals: int
  num_actions: int

  def bundle(self) -> dict:
    """The ``world_model.pt`` contents."""
    return {
      "dynamics": self.dynamics.state_dict(),
      "sim_context": self.sim_context.state_dict(),
    }

  def load_bundle(self, bundle: dict) -> None:
    """Load a ``world_model.pt`` bundle."""
    self.dynamics.load_state_dict(bundle["dynamics"])
    self.sim_context.load_state_dict(bundle["sim_context"])


def build_world_model(cfg: BaseRunnerCfg, env, device: str) -> WorldModel:
  """Construct the representation and dynamics for ``env`` sized by ``cfg``.

  Resets ``env`` to read its observation dimensions.
  """
  if cfg.history_length > 1 and cfg.actor_obs != "proprio":
    raise RuntimeError(
      "history_length > 1 requires actor_obs='proprio' (--runner.actor-obs proprio)"
    )

  obs, _ = env.reset()
  num_full_state, num_goals = unpack_obs_dims(obs)
  num_actions = env.num_actions

  print(f"[INFO] State Dim: {num_full_state}, Goal Dim: {num_goals}")

  sim_context = SimContext.from_env(env.unwrapped)
  sim_ctx_dim = sim_context.output_dim
  print(
    f"[INFO] Sim Context: {sim_ctx_dim} dim ({[s.name for s in sim_context.specs]})"
  )

  default_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0].to(device)
  representation = create_representation(
    cfg.representation,
    num_full_state,
    num_actions,
    env.unwrapped.step_dt,
    env.unwrapped.cfg.decimation,
    default_joint_pos,
    actor_obs_mode=cfg.actor_obs,
  ).to(device)

  representation.bind(env.unwrapped, sim_context)

  print(
    f"[INFO] Representation: {cfg.representation} "
    f"(full={representation.full_state_dim}, rep={representation.rep_state_dim}, "
    f"pred={representation.prediction_dim})"
  )

  action_scale, action_offset = extract_action_scale_offset(env)
  dynamics = Dynamics(
    representation,
    num_actions,
    sim_ctx_dim,
    action_scale,
    action_offset,
    hidden_dims=cfg.dynamics_hidden_dims,
  ).to(device)
  dynamics.mlp_bf16 = cfg.mlp_bf16

  return WorldModel(
    representation=representation,
    dynamics=dynamics,
    sim_context=sim_context,
    num_goals=num_goals,
    num_actions=num_actions,
  )


def build_actor(
  cfg: SuperMimicRunnerCfg, wm: WorldModel, device: str
) -> MultiEncoderAE:
  """Actor sized by ``cfg``; the resolved task specs live on ``actor.task_specs``."""
  representation = wm.representation
  obs_dim = (
    representation.rep_state_dim
    if cfg.actor_obs == "rep_state"
    else representation.proprio_obs_dim(wm.num_actions)
  )
  actor = MultiEncoderAE(
    obs_dim=obs_dim,
    action_dim=wm.num_actions,
    task_specs=resolve_task_specs(cfg.tasks, representation, wm.num_goals, device),
    slot_mix=cfg.slot_mix,
    latent_dim=cfg.latent_dim,
    hidden_dims=cfg.actor_hidden_dims,
    latent_bc_weight=cfg.latent_bc_weight,
    history_length=cfg.history_length,
    root_position_feedback=cfg.root_position_feedback,
  ).to(device)
  actor.mlp_bf16 = cfg.mlp_bf16
  return actor


def load_wandb_checkpoint(wandb_run_path: str, wm: WorldModel, device: str) -> None:
  """Warm-start ``wm`` from another W&B run's world-model bundle."""
  print(f"[INFO] Loading World Model from W&B: {wandb_run_path}")
  run = wandb.Api().run(wandb_run_path)
  wm.load_bundle(
    torch.load(download_model_dir(run) / "world_model.pt", map_location=device)
  )


class Ratio:
  def __init__(self, ratio: float) -> None:
    self._ratio = ratio
    self._prev: int | None = None

  def __call__(self, step: int) -> int:
    step = int(step)
    if self._ratio == 0:
      return 0
    if self._ratio < 0:
      return 1
    if self._prev is None:
      self._prev = step
      return 1
    repeats = int((step - self._prev) * self._ratio)
    self._prev += repeats / self._ratio
    return repeats


class BaseRunner[CfgT: BaseRunnerCfg]:
  """Base class with shared initialization for all runners.

  Handles: env dimension extraction, representation creation, world model
  setup, replay buffer, action manager, and logger. Subclasses implement
  ``_init_policy()`` for their specific policy setup.
  """

  cfg: CfgT

  def __init__(
    self,
    env,
    cfg: CfgT,
    task_id: str,
    registry_name: str | None = None,
    device: str = "cuda:0",
  ):
    self.env = env
    self.cfg = cfg
    self.task_id = task_id
    self.registry_name = registry_name
    self.device = device

    self._init_common()
    self._init_policy()

    self.logger = Logger(
      run_name=cfg.run_name,
      num_envs=env.num_envs,
      max_episode_length=env.max_episode_length,
      device=device,
    )

  def _init_common(self) -> None:
    wm = build_world_model(self.cfg, self.env, self.device)
    self._wm = wm
    self._representation = wm.representation
    self._dynamics = wm.dynamics
    self._sim_context = wm.sim_context
    self._num_goals = wm.num_goals
    self._num_actions = wm.num_actions

    if self.cfg.wandb_run_path is not None:
      load_wandb_checkpoint(self.cfg.wandb_run_path, wm, self.device)

    self.action_manager = self.env.unwrapped.action_manager
    self.motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    self.motion = self.motion_cmd.motion

    self.replay_buffer = ReplayBuffer(
      schema=self._buffer_schema(),
      buffer_size=self.cfg.buffer_size,
      num_envs=self.env.num_envs,
      device=self.device,
    )
    self.world_model_learner = WorldModelLearner(
      self._dynamics, self._representation, self.cfg.wm, device=self.device
    )

    self._pending_norm_updates: list[tuple] = []
    self._sim_history: torch.Tensor | None = None

  def _seed_sim_history(self) -> None:
    """Seed the real-sim obs-history ring with the noised current sensor."""
    if self.cfg.history_length == 1:
      return
    seed = self._representation.add_proprio_noise(
      self.env.get_observations()["actor"].to(self.device)
    )
    self._sim_history = seed.unsqueeze(1).repeat(1, self.cfg.history_length, 1)

  def _collection_history(self, sensor: torch.Tensor) -> torch.Tensor:
    """The ``(B, history_length, sensor_dim)`` window for a real-sim actor step.
    history_length > 1 needs ``_seed_sim_history()`` first."""
    if self.cfg.history_length == 1:
      return sensor.unsqueeze(1)
    assert self._sim_history is not None
    return self._sim_history

  def _advance_sim_history(self, next_actor: torch.Tensor, dones: torch.Tensor) -> None:
    """Append the noised next sensor; rows whose episode just reset repeat that
    single realization, matching the warm-start padding."""
    if self.cfg.history_length == 1:
      return
    assert self._sim_history is not None
    next_sensor = self._representation.add_proprio_noise(next_actor)
    shifted = torch.cat([self._sim_history[:, 1:], next_sensor.unsqueeze(1)], dim=1)
    reset = dones.bool().view(-1, 1, 1)
    self._sim_history = torch.where(reset, next_sensor.unsqueeze(1), shifted)

  def _policy_input_dims(self) -> tuple[int, int]:
    """Compute actor and critic input dimensions from the representation."""
    actor_dim = self._representation.actor_obs_dim(
      self._num_actions, self._num_goals, self.cfg.history_length
    )
    critic_dim = (
      self._representation.rep_state_dim
      + self._num_goals
      + self._sim_context.output_dim
    )
    return actor_dim, critic_dim

  def _build_mlp(
    self,
    name: str,
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    **kwargs: object,
  ) -> nn.Module:
    """Build an MLPModel with input obs normalization."""
    obs = TensorDict(
      {name: torch.zeros(1, input_dim, device=self.device)},
      batch_size=[1],
      device=self.device,
    )
    return MLPModel(
      obs=obs,
      obs_groups={name: [name]},
      obs_set=name,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      obs_normalization=True,
      **kwargs,
    ).to(self.device)

  def _create_wm_env(self, batch_size: int) -> MotionTrackingEnv:
    """Create a :class:`MotionTrackingEnv` bound to this runner's world model and buffer."""
    return MotionTrackingEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      batch_size=batch_size,
      replay_buffer=self.replay_buffer,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
      device=self.device,
    )

  @property
  def _stores_actor_obs(self) -> bool:
    """The buffer carries the real sensor only for its sole reader, the
    history warm-start (``WorldModelEnv.sample_initial_states``)."""
    return (
      self._representation.actor_obs_mode == "proprio" and self.cfg.history_length > 1
    )

  def _buffer_schema(self) -> dict[str, FieldSpec]:
    """Override to customize the replay buffer schema (e.g. extra fields)."""
    schema = {
      "state": FieldSpec(self._representation.full_state_dim),
      "next_state": FieldSpec(self._representation.full_state_dim),
      "action": FieldSpec(self._num_actions),
      "prev_action": FieldSpec(self._num_actions),
      "motion_time": FieldSpec(dtype=torch.long),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    }
    if self._stores_actor_obs:
      schema["actor_obs"] = FieldSpec(
        self._representation.proprio_obs_dim(self._num_actions)
      )
    schema["sim_ctx"] = FieldSpec(self._sim_context.output_dim)
    return schema

  def _init_policy(self) -> None:
    """Subclasses implement this to create their policy learner."""
    raise NotImplementedError

  def _step_and_store(
    self,
    actions: torch.Tensor,
    curr_states: torch.Tensor,
    actor_td: TensorDict | None = None,
    extra_fields: dict[str, torch.Tensor] | None = None,
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Step the env and write the transition to the replay buffer.

    When given, ``actor_td`` is queued for the actor-normalizer update.
    """
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    motion_times = motion_cmd.time_steps.clone()

    raw_actions = actions.detach().clone()
    prev_action = self.action_manager.action.clone()

    next_obs, rewards, dones, extras = self.env.step(actions)

    truncated = extras.get(
      "time_outs",
      torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
    )
    terminated = dones.bool() & ~truncated

    fields: dict[str, torch.Tensor] = {
      "state": curr_states,
      "next_state": self.env.unwrapped.transition_next_state,
      "action": raw_actions,
      "prev_action": prev_action,
      "motion_time": motion_times,
      "terminated": terminated,
      "truncated": truncated,
    }
    if extra_fields:
      fields.update(extra_fields)
    self.replay_buffer.insert(TensorDict(fields, batch_size=[self.env.num_envs]))

    if actor_td is not None:
      self._pending_norm_updates.append(
        (
          curr_states.clone(),
          raw_actions.clone(),
          actor_td.detach().clone(),
        )
      )
    return next_obs, rewards, dones, extras

  def _flush_normalization(self) -> None:
    """Apply queued normalizer updates. Call outside ``inference_mode()``."""
    if not self._pending_norm_updates:
      return
    states, actions, actor_inputs = zip(*self._pending_norm_updates, strict=True)
    self.world_model_learner.update_normalization(torch.cat(states), torch.cat(actions))
    self._update_actor_normalization(actor_inputs)
    self._pending_norm_updates.clear()

  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    """Subclasses override to update the actor obs normalizer."""
    raise NotImplementedError

  def _create_log_dir(self, *extra_parts: str) -> Path:
    """Create and return a timestamped log directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("logs", self.task_id, *extra_parts, timestamp)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")
    return log_dir

  def _init_wandb(self, log_dir: Path, name_suffix: str = "") -> None:
    """Initialize the W&B run and link the registry artifacts."""
    wandb_config = {
      "env_cfg": asdict(self.env.cfg),
      "runner_cfg": asdict(self.cfg),
    }

    default_name = f"{self.task_id}{name_suffix}-{log_dir.name}"
    wandb.init(
      project=self.cfg.wandb_project,
      name=self.cfg.run_name or default_name,
      config=wandb_config,
    )

    if self.registry_name:
      api = wandb.Api()
      for name in self.registry_name.split(","):
        name = name.strip()
        if ":" not in name:
          name = name + ":latest"
        print(f"[INFO] Linking registry artifact: {name}")
        artifact = api.artifact(name)
        wandb.run.use_artifact(artifact)

  def _cleanup(self) -> None:
    """Close environment and finish W&B run."""
    env_close_fn = getattr(self.env, "close", None)
    if env_close_fn:
      env_close_fn()
    wandb.finish()


class SuperMimicRunner(BaseRunner[SuperMimicRunnerCfg]):
  """Trains WM + multi-encoder actor end-to-end via BPTT/AC.

  Slots are contiguous blocks of the env batch; each block routes through its
  task's encoder. Slots share motion-tracking rewards and optionally a
  differentiable AMP objective; feedback collection augments supervised WM data.
  """

  def _init_policy(self) -> None:
    actor = build_actor(self.cfg, self._wm, self.device)
    actor.activation_checkpointing = self.cfg.policy_activation_checkpointing
    self._dynamics.activation_checkpointing = self.cfg.policy_activation_checkpointing
    self._active_task_specs = [
      ts for ts in actor.task_specs if self.cfg.slot_mix.get(ts.name, 0.0) > 0.0
    ]

    print("[INFO] Sim slot ranges:")
    for ts, (lo, hi) in zip(
      actor.task_specs, actor.slot_ranges(self.env.num_envs), strict=True
    ):
      print(f"  {ts.name}: [{lo}, {hi}) — goal_dim={ts.goal_dim}")
    _, critic_input_dim = self._policy_input_dims()
    critic = build_mlp(
      critic_input_dim,
      self.cfg.critic_hidden_dims,
      output_dim=1,
    ).to(self.device)

    policy_cfg = self.cfg.policy

    self.wm_env = MultiTaskWorldModelEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      batch_size=policy_cfg.batch_size,
      replay_buffer=self.replay_buffer,
      task_specs=self._active_task_specs,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
      device=self.device,
    )
    if torch.cuda.is_available():
      self.wm_env.step = torch.compile(self.wm_env.step)
      self.wm_env.reset = torch.compile(self.wm_env.reset)

    self._configure_reset_sampling()

    if isinstance(policy_cfg, ActorCriticPolicyLearnerCfg):
      self.policy_learner = ActorCriticPolicyLearner(
        actor, critic, self.wm_env, policy_cfg, device=self.device
      )
    else:
      self.policy_learner = BpttPolicyLearner(
        actor, self.wm_env, policy_cfg, device=self.device
      )
    self.motion_prior = None
    self.actor_ema = copy.deepcopy(actor).eval().requires_grad_(False)
    self.amp_feedback: AmpFeedbackLoop | None = None
    self._resume_feedback_state: dict | None = None
    self._amp_audited = False
    self._start_step = 0
    self._completed_policy_updates = 0
    if self.cfg.amp.enabled:
      if self.cfg.representation != "gencoord":
        raise ValueError("Differentiable AMP currently requires GenCoord-G1")
      self.motion_prior = DifferentiableMotionPrior.from_motion(
        self.motion, self.cfg.amp, self.motion_cmd.cfg.body_names
      )
      self.motion_prior.activation_checkpointing = (
        self.cfg.policy_activation_checkpointing
      )
      self.policy_learner.motion_prior = self.motion_prior
      self.world_model_learner.motion_prior = self.motion_prior
    if self.cfg.resume_checkpoint:
      self._load_training_state(self.cfg.resume_checkpoint)
    if self.cfg.amp_feedback.enabled:
      self.amp_feedback = AmpFeedbackLoop(self, self.cfg.amp_feedback)
      self.world_model_learner.feedback_pair_count = int(self.cfg.wm.batch_size * self.cfg.amp_feedback.priority_fraction) // 4
      self.wm_env.feedback_start_sampler = self.amp_feedback.mix_policy_starts
      if self._resume_feedback_state is not None:
        self.amp_feedback.load_state_dict(self._resume_feedback_state)

  def _save_training_state(self, log_dir: Path, step: int, *, snapshot: Path | None = None) -> None:
    """Atomic model/optimizer checkpoint; replay is recollected when resuming."""
    prior = self.motion_prior
    learner = self.policy_learner
    state = {
      "step": step,
      "policy_updates": self._completed_policy_updates,
      "task_id": self.task_id,
      "runner_cfg": asdict(self.cfg),
      "motion_file": self.motion_cmd.cfg.motion_file,
      "world_model": self._wm.bundle(),
      "actor": learner.actor.state_dict(),
      "actor_ema": self.actor_ema.state_dict(),
      "evaluation_policy": "ema",
      "world_optimizers": [o.state_dict() for o in self.world_model_learner.optimizers],
      "actor_optimizers": [o.state_dict() for o in learner.actor_optimizers],
      "torch_rng": torch.get_rng_state(),
      "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
      "resume_semantics": "restore models and optimizers; recollect simulator replay",
      "real_sim_transitions": self._real_transition_count(step),
      "reset_sampler": self._sampler.state_dict() if self._sampler is not None else None,
      "elapsed_training_s": time.time() - self._run_start_time if hasattr(self, "_run_start_time") else None,
    }
    if isinstance(learner, ActorCriticPolicyLearner):
      state["critic"] = learner.critic.state_dict()
      state["critic_optimizers"] = [o.state_dict() for o in learner.critic_optimizers]
    if prior is not None:
      state["amp"] = prior.state_dict()
      state["amp_optimizer"] = prior.optimizer.state_dict()
    if self.amp_feedback is not None:
      state["amp_feedback"] = self.amp_feedback.state_dict()
      state["interaction_budget_scope"] = (
        "since feedback was first enabled; ancestor pretraining excluded"
      )
    if snapshot is not None and snapshot.exists():
      raise FileExistsError(snapshot)
    tmp = log_dir / "training_state.tmp"
    torch.save(state, tmp)
    latest = log_dir / "training_state.pt"
    tmp.replace(latest)
    if snapshot is not None:
      # Future saves replace the inode, so the dated hard link keeps its weights.
      snapshot.hardlink_to(latest)
    (log_dir / "progress.json").write_text(
      json.dumps(
        {
          "step": step,
          "amp_updates": int(prior.updates) if prior is not None else 0,
          "policy_updates": self._completed_policy_updates,
          "amp_weight": float(prior.weight) if prior is not None else 0.0,
          "num_envs": self.env.num_envs,
          "policy_batch_size": self.cfg.policy.batch_size,
          "horizon": self.cfg.policy.horizon,
          "real_sim_transitions": self._real_transition_count(step),
          "elapsed_training_s": state["elapsed_training_s"],
          "feedback": self.amp_feedback.budget() if self.amp_feedback else None,
        },
        indent=2,
      )
    )

  def _load_training_state(self, path: str) -> None:
    state = torch.load(path, map_location=self.device, weights_only=False)
    self._wm.load_bundle(state["world_model"])
    learner = self.policy_learner
    learner.actor.load_state_dict(state["actor"])
    self.actor_ema.load_state_dict(state.get("actor_ema", state["actor"]))
    if self._sampler is not None and state.get("reset_sampler") is not None:
      self._sampler.load_state_dict(state["reset_sampler"])
    for opts, key in [
      (self.world_model_learner.optimizers, "world_optimizers"),
      (learner.actor_optimizers, "actor_optimizers"),
    ]:
      for optimizer, saved in zip(opts, state[key], strict=True):
        optimizer.load_state_dict(saved)
    if isinstance(learner, ActorCriticPolicyLearner):
      learner.critic.load_state_dict(state["critic"])
      for optimizer, saved in zip(
        learner.critic_optimizers, state["critic_optimizers"], strict=True
      ):
        optimizer.load_state_dict(saved)
    if self.motion_prior is not None:
      self.motion_prior.load_state_dict(state["amp"])
      self.motion_prior.optimizer.load_state_dict(state["amp_optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
      torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    self._start_step = state["step"]
    self._completed_policy_updates = int(state.get("policy_updates", state.get("amp", {}).get("updates", 0)))
    self._resume_feedback_state = state.get("amp_feedback")
    print(f"[INFO] Restored training state at {self._start_step}; recollecting replay")

  def _audit_amp_gradient(self, log_dir: Path) -> None:
    """Actual actor/dynamics/discriminator graph, without a task reward."""
    assert self.motion_prior is not None
    if self.cfg.history_length != 1:
      raise ValueError(
        "Initial gradient audit currently expects official history_length=1"
      )
    td = self.replay_buffer.sample_causal(1, 16)[0]
    rep = self._representation.represent(td["state"][:, 0]).detach()
    aux = self._representation.extract_aux_state(td["state"][:, 0]).detach()
    ctx, times = td["sim_ctx"][:, 0], td["motion_time"][:, 0]
    self._dynamics.eval().requires_grad_(False)
    self.motion_prior.freeze()
    first_action = None
    previous = rep
    for _ in range(self.cfg.policy.horizon):
      obs = self._build_sim_obs_td(rep, aux, rep, times, ctx)
      actions = self.policy_learner.actor(obs)
      if first_action is None:
        first_action = actions
      previous = rep
      aux, rep = self._dynamics(rep, actions, aux, ctx)
      clip = torch.searchsorted(self.motion.motion_ends, times, right=True)
      times = torch.minimum(times + 1, self.motion.motion_ends[clip] - 1)
    cost = self.motion_prior.style_cost(previous, rep).mean()
    params = list(self.policy_learner.actor.parameters())
    assert first_action is not None
    grads = torch.autograd.grad(cost, [*params, first_action], allow_unused=True)
    valid = [g for g in grads[:-1] if g is not None]
    actor_norm = torch.stack([g.square().sum() for g in valid]).sum().sqrt()
    action_norm = grads[-1].norm()
    finite = all(bool(torch.isfinite(g).all()) for g in [*valid, grads[-1]])
    if not finite or float(actor_norm) == 0 or float(action_norm) == 0:
      raise RuntimeError("AMP pathwise gradient audit failed")
    result = {
      "passed": True,
      "steps": self.cfg.policy.horizon,
      "only_final_pair_loss": True,
      "task_reward_used": False,
      "unweighted_style_cost": float(cost.detach()),
      "actor_gradient_norm": float(actor_norm),
      "first_action_gradient_norm": float(action_norm),
      "world_model_frozen": not any(
        p.requires_grad for p in self._dynamics.parameters()
      ),
      "discriminator_frozen": not any(
        p.requires_grad for p in self.motion_prior.parameters()
      ),
    }
    (log_dir / "amp_gradient_audit.json").write_text(json.dumps(result, indent=2))
    print(f"[AMP] Real-model gradient audit: {result}")
    self._amp_audited = True

  def _build_sim_obs_td(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    sim_actor_obs: torch.Tensor,
    motion_times: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> TensorDict:
    proprio = self.cfg.actor_obs == "proprio"
    sensor = self._representation.compute_actor_sensor(
      rep_state,
      sim_ctx,
      aux_state=aux_state,
      sim_actor_obs=sim_actor_obs if proprio else None,
      noisy=proprio,
    )
    goals = {
      ts.name: ts.extract_training_goal(self.motion, motion_times, aux_state)
      for ts in self._active_task_specs
    }
    return build_actor_obs_td(goals, self._collection_history(sensor),
                             root_error=root_position_error(self.motion, motion_times, aux_state))

  def _configure_reset_sampling(self) -> None:
    self._sampler: ReturnSampler | None = None
    env_mode = self.motion_cmd.cfg.sampling_mode
    expected = (
      "external" if self.cfg.sampling_mode == "return" else self.cfg.sampling_mode
    )
    if env_mode != expected:
      raise RuntimeError(
        f"runner sampling_mode='{self.cfg.sampling_mode}' needs the motion command "
        f"in '{expected}' mode but the env was built with '{env_mode}'; resolve it "
        "into the env cfg before env creation ('external' also needs an mjlab with "
        "MotionCommand external-mode support)"
      )
    if self.cfg.sampling_mode != "return":
      return
    self._sampler = ReturnSampler(
      cfg=self.cfg.return_sampler,
      bin_count=self.motion_cmd.bin_count,
      frame_count=self.motion.time_step_total,
      num_envs=self.env.num_envs,
      max_step_reward=self.wm_env.max_step_reward,
      device=self.device,
    )
    self._sampler.configure_failure_focus(
      self.motion.motion_ends, self.env.unwrapped.step_dt,
      self.cfg.failure_focus_fraction, self.cfg.failure_lookback_seconds,
    )

  def step(self) -> int:
    with torch.inference_mode():
      obs = self.env.get_observations()
      curr_states = obs["full_state"].to(self.device)
      sim_actor_obs = obs["actor"].to(self.device)

      # Pre-step frames; env.step() advances time_steps in place.
      motion_times = self.motion_cmd.time_steps.to(self.device).clone()

      rep_state = self._representation.represent(curr_states)
      aux_state = self._representation.extract_aux_state(curr_states)
      sim_ctx = self._sim_context.read(self.env.unwrapped)
      actor_td = self._build_sim_obs_td(
        rep_state, aux_state, sim_actor_obs, motion_times, sim_ctx
      )

      actions = self.policy_learner.actor(actor_td)
      actions = actions + torch.randn_like(actions) * self.policy_learner.cfg.noise

      extra_fields = {"sim_ctx": sim_ctx}
      if self._stores_actor_obs:
        extra_fields["actor_obs"] = sim_actor_obs

      next_obs, rewards, dones, extras = self._step_and_store(
        actions,
        curr_states,
        actor_td=actor_td,
        extra_fields=extra_fields,
      )
      self._advance_sim_history(next_obs["actor"].to(self.device), dones)
      terminated = self.env.unwrapped.termination_manager.terminated
      self.logger.process_env_step(rewards, terminated, extras)

      if self._sampler is not None:
        # A failure on a step that also times out counts as the time-out.
        manager = self.env.unwrapped.termination_manager
        truncated = manager.time_outs.to(self.device, torch.bool)
        clip_end = manager.get_term("motion_end").to(self.device, torch.bool)
        weights = self._sampler.record(
          motion_times,
          rewards,
          terminated.to(self.device, torch.bool) & ~truncated,
          clip_end,
          truncated & ~clip_end,
        )
        if weights is not None:
          self.motion_cmd.set_sampling_distribution(weights)

    self._flush_normalization()
    return self.env.num_envs

  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    self.policy_learner.actor.train()
    for actor_td in actor_inputs:
      self.policy_learner.actor.update_normalization(actor_td)

  def _world_model_batches(self):
    for _ in range(self.cfg.wm.num_batches):
      batch = self.replay_buffer.sample_world_windows(self.cfg.wm.horizon, self.cfg.wm.batch_size)
      yield (
        self.amp_feedback.mix_batch(batch) if self.amp_feedback is not None else batch
      )

  def _real_transition_count(self, step: int) -> int:
    if self.amp_feedback is not None:
      return self.amp_feedback.budget()["total_transitions"]
    return step * self.env.num_envs

  def _next_budget_boundary(self, count: int) -> int | None:
    boundaries = [x for x in self.cfg.budget_checkpoints if x > count]
    if self.cfg.max_sim_transitions is not None:
      boundaries.append(self.cfg.max_sim_transitions)
    return min(boundaries) if boundaries else None

  def run(
    self,
    *,
    update_checkpoints: tuple[int, ...] = (),
    evaluate_checkpoint: Callable[[int, Path], bool | None] | None = None,
  ) -> None:
    update_limit = self.cfg.max_amp_updates
    if update_limit is not None and (update_limit <= 0 or self.motion_prior is None):
      raise ValueError("A positive max_amp_updates requires AMP training")
    for count in (*self.cfg.budget_checkpoints, self.cfg.max_sim_transitions):
      if count is not None and (count <= 0 or count % self.env.num_envs):
        raise ValueError("Real-transition budgets must be positive multiples of num_envs")
    log_dir = self._create_log_dir()
    (log_dir / "run_config.json").write_text(
      json.dumps(
        {
          "task_id": self.task_id,
          "runner_cfg": asdict(self.cfg),
          "motion_file": self.motion_cmd.cfg.motion_file,
          "num_envs": self.env.num_envs,
        },
        indent=2,
      )
    )
    self._init_wandb(log_dir)

    self.env.reset()
    self._seed_sim_history()

    print(f"[INFO] Starting training loop for {self.cfg.max_iterations} iterations...")

    should_train = Ratio(
      self.env.num_envs
      / self.cfg.wm.num_batches
      / self.cfg.wm.batch_size
      / self.cfg.wm.horizon
      * self.cfg.replay_ratio
    )

    num_collected_steps = 0
    start_time = time.time()
    wm_learn_time = 0.0
    policy_learn_time = 0.0
    loss_dict: dict[str, torch.Tensor] = {}
    log_count = 0
    uploader = ThreadPoolExecutor(max_workers=1)
    upload: Future | None = None
    final_step = self._start_step
    training_start = time.time()
    self._run_start_time = training_start
    amp_updates = int(self.motion_prior.updates) if self.motion_prior is not None else 0
    if update_checkpoints and (
      update_limit is None
      or any(not amp_updates < n <= update_limit for n in update_checkpoints)
      or len(set(update_checkpoints)) != len(update_checkpoints)
    ):
      raise ValueError("Update checkpoints must be unique and between the resume and target updates")
    update_milestones = set(update_checkpoints)
    updates_complete = update_limit is not None and amp_updates >= update_limit

    for step in range(self._start_step, self.cfg.max_iterations):
      count = self._real_transition_count(step)
      if updates_complete or (
        self.cfg.max_sim_transitions is not None and count >= self.cfg.max_sim_transitions
      ):
        break
      num_collected_steps += self.step()
      final_step = step + 1
      if self.amp_feedback is not None:
        self.amp_feedback.main_transitions += self.env.num_envs

      evaluation_s = 0.0
      at_update_checkpoint = False
      stop_after_evaluation = False
      for _ in range(should_train(step)):
        if self.replay_buffer.num_transitions < self.cfg.wm.horizon:
          continue

        feedback = self.amp_feedback
        boundary = self._next_budget_boundary(self._real_transition_count(final_step))
        feedback_fits = (
          boundary is None
          or self._real_transition_count(final_step)
          + 5 * self.env.num_envs * (self.cfg.wm.horizon - 1) <= boundary
        )
        # Do not step over a checkpoint reached by the main physics step.
        at_milestone = self._real_transition_count(final_step) in self.cfg.budget_checkpoints
        if feedback is not None and feedback.due(step + 1) and feedback_fits and not at_milestone:
          loss_dict.update(feedback.collect(step + 1, log_dir))

        data = self._world_model_batches()

        start_wm_time = time.time()
        wm_logs = self.world_model_learner.update(data)
        if feedback is not None:
          loss_dict.update(feedback.after_world_update(log_dir))
        wm_learn_time += time.time() - start_wm_time

        start_policy_time = time.time()
        if self.motion_prior is not None:
          loss_dict.update(self.motion_prior.update(self.replay_buffer, self.amp_feedback))
          amp_updates += 1
          if not self._amp_audited:
            self._audit_amp_gradient(log_dir)
        policy_logs = self.policy_learner.update()
        self._completed_policy_updates += self.cfg.policy.num_batches
        self._update_actor_ema()
        policy_learn_time += time.time() - start_policy_time

        loss_dict.update(wm_logs)
        loss_dict.update(policy_logs)
        if amp_updates in update_milestones:
          at_update_checkpoint = True
          snapshot = log_dir / f"updates_{amp_updates:06d}.pt"
          self._save_training_state(log_dir, final_step, snapshot=snapshot)
          if evaluate_checkpoint is not None:
            before_evaluation = time.time()
            stop_after_evaluation = evaluate_checkpoint(amp_updates, snapshot) is True
            wait_s = time.time() - before_evaluation
            evaluation_s += wait_s
            training_start += wait_s
            self._run_start_time = training_start
            start_time += wait_s
        updates_complete = stop_after_evaluation or (update_limit is not None and amp_updates >= update_limit)
        if updates_complete:
          break

      if step % self.cfg.log_interval == 0 or updates_complete or at_update_checkpoint:
        numeric_logs = {k: float(v) for k, v in loss_dict.items()}
        nonfinite = {k: v for k, v in numeric_logs.items() if not math.isfinite(v)}
        if nonfinite:
          raise FloatingPointError(f"Nonfinite training metrics at update {amp_updates}: {nonfinite}")
        with (log_dir / "training_metrics.jsonl").open("a") as stream:
          stream.write(json.dumps({
            "step": final_step,
            "amp_updates": amp_updates,
            "real_sim_transitions": self._real_transition_count(final_step),
            "elapsed_s": time.time() - training_start,
            **({"evaluation_s": evaluation_s} if at_update_checkpoint else {}),
            **numeric_logs,
          }) + "\n")
        if torch.cuda.is_available():
          loss_dict.update(
            {
              "Memory/allocated_GiB": torch.tensor(
                torch.cuda.memory_allocated() / 2**30, device=self.device
              ),
              "Memory/peak_allocated_GiB": torch.tensor(
                torch.cuda.max_memory_allocated() / 2**30, device=self.device
              ),
            }
          )
        elapsed_time = time.time() - start_time
        total_train_time = wm_learn_time + policy_learn_time
        collect_time = elapsed_time - total_train_time

        self.logger.log(
          it=log_count,
          total_it=self.cfg.max_iterations // self.cfg.log_interval,
          collect_time=collect_time,
          wm_learn_time=wm_learn_time,
          policy_learn_time=policy_learn_time,
          loss_dict=loss_dict,
          num_collected_steps=num_collected_steps,
        )

        log_count += 1
        start_time = time.time()
        wm_learn_time = 0
        policy_learn_time = 0
        num_collected_steps = 0

      if step % self.cfg.save_interval == 0:
        self._save_training_state(log_dir, step + 1)
        assert wandb.run is not None  # _init_wandb ran before the loop
        # The previous version must be logged before its files are overwritten.
        if upload is not None:
          upload.result()
        world_model_path = str(log_dir / "world_model.pt")
        actor_path = str(log_dir / "actor.pt")
        torch.save(self._wm.bundle(), world_model_path)
        torch.save(self.policy_learner.actor.state_dict(), actor_path)
        checkpoint_files = [world_model_path, actor_path]
        critic = getattr(self.policy_learner, "critic", None)
        if critic is not None:
          critic_path = str(log_dir / "critic.pt")
          torch.save(critic.state_dict(), critic_path)
          checkpoint_files.append(critic_path)
        # Skip ONNX export on huge datasets.
        if (
          self._representation.actor_obs_mode == "proprio"
          and self.motion.time_step_total <= 15_000
        ):
          checkpoint_files += export_actor_as_onnx(
            actor=self.policy_learner.actor,
            env=self.env.unwrapped,
            path=str(log_dir),
            run_name=wandb.run.name,
            term_widths=self._representation.proprio_term_widths(self._num_actions),
          )
        upload = uploader.submit(log_model_artifact, checkpoint_files)

      count = self._real_transition_count(final_step)
      if count in self.cfg.budget_checkpoints:
        import shutil

        self._save_training_state(log_dir, final_step)
        shutil.copyfile(log_dir / "training_state.pt", log_dir / f"budget_{count:09d}.pt")

    self._save_training_state(log_dir, final_step)
    uploader.shutdown()
    self._cleanup()

  @torch.no_grad()
  def _update_actor_ema(self):
    live = self.policy_learner.actor
    torch._foreach_lerp_(list(self.actor_ema.parameters()), list(live.parameters()), 1-self.cfg.policy_ema_decay)
    # Normalization describes the current input coordinates. Copy statistics
    # exactly instead of mixing them with delayed counts/variances.
    for dest, src in zip(self.actor_ema.buffers(), live.buffers(), strict=True):
      dest.copy_(src)

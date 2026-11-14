"""Motion prior whose input derivatives remain connected to imagined dynamics.

Two-frame discrimination, expert gradient regularization and replay sampling are
informed by ccrpRepo/AMP_mjlab. Policy learning is exclusively SuperMimic BPTT;
this module intentionally has no PPO reward-prediction/no_grad interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from supermimic.models import run_mlp
from supermimic.utils.gencoord_helpers import compute_gencoord_state

if TYPE_CHECKING:
  from mjlab.tasks.tracking.mdp.commands import MotionLoader

  from supermimic.replay_buffer import ReplayBuffer


@dataclass
class AmpCfg:
  enabled: bool = False
  feature_mode: str = "gencoord68"
  hidden_dims: tuple[int, ...] = (1024, 512, 256)
  batch_size: int = 2048
  lr: float = 1e-4
  grad_penalty: float = 10.0
  weight_decay: float = 1e-4
  weight: float = 0.1
  warmup_updates: int = 100
  ramp_updates: int = 400
  norm_rate: float = 0.01


class DifferentiableMotionPrior(nn.Module):
  expert_frames: torch.Tensor
  pair_starts: torch.Tensor
  variance_floor: torch.Tensor
  mean: torch.Tensor
  variance: torch.Tensor
  weight: torch.Tensor
  updates: torch.Tensor

  def __init__(
    self, expert_frames: torch.Tensor, motion_ends: torch.Tensor, cfg: AmpCfg,
    feature_encoder=None,
  ):
    super().__init__()
    self.cfg = cfg
    self.feature_encoder = feature_encoder
    self.activation_checkpointing = False
    dim = expert_frames.shape[-1]
    self.register_buffer("expert_frames", expert_frames.detach(), persistent=False)
    starts = torch.arange(len(expert_frames) - 1, device=expert_frames.device)
    starts = starts[~torch.isin(starts, motion_ends - 1)]
    if starts.numel() == 0:
      raise ValueError("AMP needs at least one consecutive in-clip frame pair")
    self.register_buffer("pair_starts", starts, persistent=False)
    floor = expert_frames.new_full((dim,), 0.1)
    if cfg.feature_mode in ("body195", "grounded199"):
      floor[39:117] = 0.5
      floor[117:] = 1.0
      if cfg.feature_mode == "grounded199":
        floor[195:] = 0.1
    else:
      floor[4:10] = 0.5
      floor[10 + (dim - 10) // 2 :] = 1.0
    self.register_buffer("variance_floor", floor.square())
    self.register_buffer("mean", expert_frames.mean(0))
    self.register_buffer("variance", expert_frames.var(0, unbiased=False))
    self.register_buffer("weight", expert_frames.new_zeros(()))
    self.register_buffer(
      "updates", torch.zeros((), dtype=torch.long, device=expert_frames.device)
    )
    layers: list[nn.Module] = []
    width = dim * 2
    for hidden in cfg.hidden_dims:
      layers += [nn.Linear(width, hidden), nn.ReLU()]
      width = hidden
    layers.append(nn.Linear(width, 1))
    self.net = nn.Sequential(*layers).to(expert_frames.device)
    self.optimizer = torch.optim.AdamW(
      self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    self.freeze()

  @classmethod
  def from_motion(cls, motion: MotionLoader, cfg: AmpCfg, body_names=None):
    full = torch.cat(
      [
        motion.root_pos_w,
        motion.root_quat_w,
        motion.joint_pos,
        motion.root_lin_vel_w,
        motion.root_ang_vel_w,
        motion.joint_vel,
      ],
      dim=-1,
    )
    frames = compute_gencoord_state(full)
    encoder = None
    if cfg.feature_mode in ("body195", "grounded199"):
      from supermimic.learners.body_amp_features import BodyAMPFeatures, GroundedBodyAMPFeatures
      if body_names is None:
        raise ValueError("body195 needs the motion FK's ordered body names")
      encoder_type = GroundedBodyAMPFeatures if cfg.feature_mode == "grounded199" else BodyAMPFeatures
      encoder = encoder_type(motion, body_names)
      frames = encoder.expert_frames(frames)
    elif cfg.feature_mode != "gencoord68":
      raise ValueError(f"Unknown AMP feature mode: {cfg.feature_mode}")
    return cls(frames, motion.motion_ends, cfg, encoder)

  def encode(self, frames: torch.Tensor) -> torch.Tensor:
    if self.feature_encoder is None:
      return frames
    if self.activation_checkpointing and frames.requires_grad:
      from torch.utils.checkpoint import checkpoint
      # Complete recomputation: an early-stop exception cannot cross the
      # TorchScript quaternion helpers when a diagnostic forces eager mode.
      return checkpoint(self.feature_encoder, frames, use_reentrant=False,
                        preserve_rng_state=False, early_stop=False)
    return self.feature_encoder(frames)

  def freeze(self) -> None:
    self.net.eval().requires_grad_(False)
    for p in self.net.parameters():
      p.grad = None

  def normalize(self, frames: torch.Tensor) -> torch.Tensor:
    return (frames - self.mean) * torch.rsqrt(
      torch.maximum(self.variance, self.variance_floor)
    )

  def logits(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return run_mlp(
      self.net,
      torch.cat([self.normalize(self.encode(first)), self.normalize(self.encode(second))], -1),
      self.activation_checkpointing,
    ).squeeze(-1)

  def style_cost(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    # Frozen weights do not detach inputs. Smooth non-saturating generator loss.
    return F.softplus(-self.logits(first, second))

  def sample_expert(self, batch_size: int) -> torch.Tensor:
    idx = self.pair_starts[
      torch.randint(
        self.pair_starts.numel(), (batch_size,), device=self.pair_starts.device
      )
    ]
    return torch.stack([self.expert_frames[idx], self.expert_frames[idx + 1]], 1)

  def update(self, replay: ReplayBuffer, feedback=None) -> dict[str, torch.Tensor]:
    # Official replay rejects windows that cross terminated/truncated transitions.
    real = replay.sample_pairs(self.cfg.batch_size).detach()
    if feedback is not None:
      real = feedback.mix_amp_pairs(real)
    policy = self.encode(compute_gencoord_state(real))
    expert = self.sample_expert(self.cfg.batch_size)
    with torch.no_grad():
      frames = torch.cat([expert.flatten(0, 1), policy.flatten(0, 1)])
      mean = frames.mean(0)
      variance = frames.var(0, unbiased=False)
      rate = self.cfg.norm_rate
      self.variance.mul_(1 - rate).add_(variance, alpha=rate).add_(
        (mean - self.mean).square(), alpha=rate * (1 - rate)
      )
      self.mean.lerp_(mean, rate)
    self.net.train().requires_grad_(True)
    positive = self.normalize(expert).flatten(1).detach().requires_grad_(True)
    negative = self.normalize(policy).flatten(1).detach()
    expert_logits = self.net(positive).squeeze(-1)
    policy_logits = self.net(negative).squeeze(-1)
    classification = 0.5 * (
      F.softplus(-expert_logits).mean() + F.softplus(policy_logits).mean()
    )
    grad = torch.autograd.grad(expert_logits.sum(), positive, create_graph=True)[0]
    penalty = grad.square().sum(-1).mean()
    loss = classification + self.cfg.grad_penalty * penalty
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
    self.optimizer.step()
    with torch.no_grad():
      self.updates.add_(1)
      ramp = (
        (self.updates - self.cfg.warmup_updates) / max(1, self.cfg.ramp_updates)
      ).clamp(0, 1)
      self.weight.copy_(ramp * self.cfg.weight)
    self.freeze()
    return {
      "AMP/discriminator_loss": loss.detach(),
      "AMP/gradient_penalty": penalty.detach(),
      "AMP/expert_accuracy": (expert_logits > 0).float().mean().detach(),
      "AMP/policy_accuracy": (policy_logits < 0).float().mean().detach(),
      "AMP/expert_logit": expert_logits.mean().detach(),
      "AMP/policy_logit": policy_logits.mean().detach(),
      "AMP/discriminator_grad_norm": grad_norm.detach(),
      "AMP/weight": self.weight.clone(),
    }

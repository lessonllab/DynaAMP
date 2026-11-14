"""Reversible physical branches for the official, undelayed G1 environment.

Branches run the identical control/physics path but do not advance task managers,
reference clocks, sensor histories, resets or episode metrics. This intentionally
supports only builtin position actuators without delay. All integration state,
including solver warm starts, and action/target buffers are restored in place.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.actuator.builtin_actuator import BuiltinPositionActuator


class PhysicsBranch:
  STATE_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "act",
    "history",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
    "qacc",
    "act_dot",
    "userdata",
    "sensordata",
    "tree_asleep",
  )

  def __init__(self, env: Any):
    if getattr(env, "clip_actions", None) is not None:
      raise ValueError("Branch action tensors must match unclipped official controls")
    self.env = env.unwrapped
    for entity in self.env.scene.entities.values():
      for actuator in entity.actuators:
        if not isinstance(actuator, BuiltinPositionActuator) or actuator.has_delay:
          raise ValueError("AMP branches require undelayed builtin position actuators")
    self.simulated_transitions = 0
    self._state: dict[str, torch.Tensor] = {}
    self._live: dict[str, torch.Tensor] = {}
    self._buffers: list[tuple[Any, str, torch.Tensor, torch.Tensor]] = []

  def full_state(self) -> torch.Tensor:
    data = self.env.scene["robot"].data
    return torch.cat(
      [
        data.root_link_pos_w - self.env.scene.env_origins,
        data.root_link_quat_w,
        data.joint_pos,
        data.root_link_lin_vel_w,
        data.root_link_ang_vel_w,
        data.joint_vel,
      ],
      dim=-1,
    )

  @torch.no_grad()
  def capture(self) -> None:
    # Include contact/constraint caches and derived state, not just qpos/qvel.
    # Re-running forward() during restoration can change contact ordering and
    # solver caches before the next supposedly identical branch.
    def arrays(value, prefix=""):
      for field in dataclasses.fields(value):
        item = getattr(value, field.name)
        name = prefix + field.name
        if isinstance(item, wp.array) and item.size:
          yield name, wp.to_torch(item)
        elif dataclasses.is_dataclass(item):
          yield from arrays(item, name + ".")

    self._live = dict(arrays(self.env.sim.wp_data))
    self._state = {name: value.clone() for name, value in self._live.items()}
    owners = [self.env.action_manager]
    owners += [
      self.env.action_manager.get_term(name)
      for name in self.env.action_manager.active_terms
    ]
    for entity in self.env.scene.entities.values():
      owners.extend([entity, entity.data])
    self._buffers = [
      (owner, name, value, value.clone())
      for owner in owners
      for name, value in vars(owner).items()
      if isinstance(value, torch.Tensor)
    ]

  @torch.inference_mode()
  def restore(self) -> None:
    if not self._state:
      raise RuntimeError("Capture a physics snapshot before branching")
    for owner, name, original, saved in self._buffers:
      original.copy_(saved)
      setattr(owner, name, original)
    for name, saved in self._state.items():
      self._live[name].copy_(saved)

  @torch.no_grad()
  def rollout(self, actions: torch.Tensor) -> torch.Tensor:
    """Execute a fixed (E, H, A) action sequence, returning E x (H+1) states.

    Count every advanced MuJoCo world, including discarded/held-out data. No
    early resets are allowed to replace terminal states in a training window.
    """
    if actions.shape[0] != self.env.num_envs:
      raise ValueError("Physics branches must include every advanced world")
    self.restore()
    states = [self.full_state().clone()]
    try:
      for action in actions.unbind(1):
        self.env.action_manager.process_action(action)
        for _ in range(self.env.cfg.decimation):
          self.env.action_manager.apply_action()
          self.env.scene.write_data_to_sim()
          self.env.sim.step()
        self.simulated_transitions += self.env.num_envs
        self.env.sim.forward()
        capacity_mask = int(mjwarp.OverflowType.ALL) & ~int(
          mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS
        )
        if (self.env.sim.data.overflow[:] & capacity_mask).any():
          raise RuntimeError("Physics capacity overflow in a feedback branch")
        states.append(self.full_state().clone())
      result = torch.stack(states, dim=1)
      if not torch.isfinite(result).all():
        raise RuntimeError("Nonfinite physical branch; do not insert into replay")
      return result
    finally:
      self.restore()

  @torch.no_grad()
  def restoration_error(self) -> float:
    errors = [
      (self._live[key].float() - saved.float()).abs().max()
      for key, saved in self._state.items()
      if key in self.STATE_FIELDS and saved.numel()
    ]
    errors += [
      (getattr(owner, name).float() - saved.float()).abs().max()
      for owner, name, _, saved in self._buffers
      if saved.numel()
    ]
    return float(torch.stack(errors).max())

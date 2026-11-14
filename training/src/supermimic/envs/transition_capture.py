"""Capture real transition targets at reward time, before mjlab auto reset."""

from mjlab.envs import ManagerBasedRlEnv
import torch


class TransitionCaptureEnv(ManagerBasedRlEnv):
  @torch.no_grad()
  def physics_state(self):
    robot = self.scene["robot"]
    idx = robot.indexing
    q, v = self.sim.data.qpos, self.sim.data.qvel
    from mjlab.utils.lab_api.math import quat_apply
    pose = q[:, idx.free_joint_q_adr.long()]
    vel = v[:, idx.free_joint_v_adr.long()]
    return torch.cat((
      pose[:, :3] - self.scene.env_origins, pose[:, 3:7],
      q[:, idx.joint_q_adr.long()], vel[:, :3],
      quat_apply(pose[:, 3:7], vel[:, 3:6]), v[:, idx.joint_v_adr.long()],
    ), dim=-1).clone()

  def step(self, action):
    compute = self.reward_manager.compute
    def capture(*args, **kwargs):
      reward = compute(*args, **kwargs)
      self.transition_next_state = self.physics_state()
      return reward
    self.reward_manager.compute = capture
    try:
      return super().step(action)
    finally:
      self.reward_manager.compute = compute

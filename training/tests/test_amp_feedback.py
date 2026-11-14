import os
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from supermimic.runners.amp_feedback import WindowReplay, improvement_disagreement


def test_acquisition_measures_action_effect_not_absolute_style_error():
  # A constant score bias is not a wrong action gradient. In contrast the
  # second proposal is predicted to improve but degrades actual motion.
  gap, predicted, actual = improvement_disagreement(
    torch.tensor([-2.0, -2.0]),
    torch.tensor([-1.0, -1.0]),
    torch.tensor([-8.0, -2.0]),
    torch.tensor([-7.0, -3.0]),
  )
  torch.testing.assert_close(gap, torch.tensor([0.0, 2.0]))
  assert predicted[1] > 0 and actual[1] < 0


def test_window_replay_preserves_real_sequences_through_wrap_and_priority():
  torch.manual_seed(41)
  replay = WindowReplay(capacity=5)

  def windows(ids):
    trajectory = ids[:, None, None] * 100 + torch.arange(8)[None, :, None]
    return TensorDict(
      {"state": trajectory.float(), "action": trajectory.float()}, [len(ids), 8]
    )

  replay.insert(windows(torch.tensor([0, 1, 2, 3])), torch.ones(4))
  replay.insert(windows(torch.tensor([4, 5, 6])), torch.tensor([1.0, 1.0, 1000.0]))
  sampled = replay.sample(2000)
  ids = (sampled["state"][:, 0, 0] / 100).long()
  assert set(ids.tolist()) <= {2, 3, 4, 5, 6}
  assert (ids == 6).float().mean() > 0.97
  torch.testing.assert_close(
    sampled["state"][:, 1:] - sampled["state"][:, :-1], torch.ones(2000, 7, 1)
  )
  assert sampled.shape == (2000, 8)


@pytest.mark.skipif(
  os.environ.get("SUPERAMP_PHYSICS_TEST") != "1", reason="explicit real MuJoCo test"
)
def test_physics_branch_matches_official_step_and_restores_live_state():
  from mjlab.tasks.registry import load_env_cfg

  import supermimic.tasks  # noqa: F401
  from supermimic.scripts.train_utils import create_env, setup_device
  from supermimic.utils.sim_branch import PhysicsBranch

  setup_device("cuda:0")
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  motion_dir = Path(os.environ.get(
    "LOOPMIMIC_MOTION_DIR", Path(__file__).resolve().parents[1] / "data/lafan1_g1_50hz"
  ))
  cfg.commands["motion"].motion_file = str(motion_dir / "run1_subject5.npz")
  cfg.commands["motion"].sampling_mode = "start"
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 1024
  cfg.auto_reset = False
  env = create_env(cfg, 16, "cuda:0")
  try:
    env.reset()
    base = env.unwrapped
    # Real collection creates action-term tensors inside inference_mode.
    with torch.inference_mode():
      env.step(torch.zeros(16, env.num_actions, device="cuda:0"))
    branch = PhysicsBranch(env)
    branch.capture()
    initial = branch.full_state().clone()
    reference_time = base.command_manager.get_term("motion").time_steps.clone()
    episode_lengths = base.episode_length_buf.clone()
    actions = torch.zeros(16, 3, env.num_actions, device="cuda:0")
    # Use position targets at the current reference pose for a stable test.
    term = base.action_manager.get_term("joint_pos")
    actions[:] = ((base.scene["robot"].data.joint_pos - term._offset) / term._scale)[
      :, None
    ]
    actions[:, :, 0] += 0.02
    first = branch.rollout(actions)
    second = branch.rollout(actions)
    # MuJoCo Warp's GPU contact reductions are not bitwise deterministic.
    torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(branch.full_state(), initial, atol=2e-5, rtol=0)
    assert branch.restoration_error() == 0
    assert torch.equal(
      reference_time, base.command_manager.get_term("motion").time_steps
    )
    assert torch.equal(episode_lengths, base.episode_length_buf)
    assert branch.simulated_transitions == 16 * 3 * 2
    obs, _, _, _ = env.step(actions[:, 0])
    torch.testing.assert_close(obs["full_state"], first[:, 1], atol=2e-5, rtol=2e-5)
  finally:
    env.close()

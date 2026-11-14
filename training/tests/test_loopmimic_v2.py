"""Behavioral checks for the new observation and paired-real-data contracts."""
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from supermimic.models import MultiEncoderAE
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.representations import root_position_error
from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.runners.amp_feedback import AmpFeedbackLoop, AmpFeedbackCfg, WindowReplay
from tests.conftest import make_actor_specs, make_actor_obs


def test_root_feedback_preserves_initial_actor_but_can_learn_correction():
    kwargs = dict(obs_dim=68, action_dim=29, task_specs=make_actor_specs(False),
                  slot_mix={"motion": 1.0}, latent_dim=32, hidden_dims=(32, 16),
                  latent_bc_weight=0.1, history_length=1)
    torch.manual_seed(10)
    old = MultiEncoderAE(**kwargs)
    torch.manual_seed(10)
    new = MultiEncoderAE(**kwargs, root_position_feedback=True)
    obs = make_actor_obs(4)
    obs["root_error"] = torch.tensor([[0.25, 0., 0.]]).expand(4, -1).clone()
    torch.testing.assert_close(new(obs), old(obs), atol=0, rtol=0)
    new(obs).square().sum().backward()
    assert new.decoder[0].weight.grad[:, -3:].norm() > 0
    with torch.no_grad():
        new.decoder[0].weight[:, -3:] -= new.decoder[0].weight.grad[:, -3:]
    shifted = obs.clone()
    shifted["root_error"].zero_()
    assert not torch.equal(new(obs), new(shifted))


def test_root_feedback_is_robot_frame_and_has_pose_gradient():
    motion = SimpleNamespace(root_pos_w=torch.tensor([[1., 2., 3.]]))
    q = 2**-0.5
    aux = torch.tensor([[0., 0., 0., q, 0., 0., q]], requires_grad=True)
    result = root_position_error(motion, torch.tensor([0]), aux)
    torch.testing.assert_close(result, torch.tensor([[2., -1., 3.]]))
    result.sum().backward()
    assert aux.grad[:, :3].norm() > 0


def test_replay_preserves_terminal_target_and_masks_resets():
    schema = {"state": FieldSpec(1), "next_state": FieldSpec(1), "action": FieldSpec(1),
              "sim_ctx": FieldSpec(1), "terminated": FieldSpec(dtype=torch.bool),
              "truncated": FieldSpec(dtype=torch.bool)}
    buf = ReplayBuffer(schema, 4, 1)
    for before, after, done in [(0., 1., False), (1., 9., True), (100., 101., False)]:
        buf.insert(TensorDict({"state": torch.tensor([[before]]), "next_state": torch.tensor([[after]]),
                   "action": torch.ones(1, 1), "sim_ctx": torch.zeros(1, 1),
                   "terminated": torch.tensor([done]), "truncated": torch.tensor([False])}, [1]))
    sample = buf.sample_world_windows(4, 8)
    torch.testing.assert_close(sample["state"][0, :, 0], torch.tensor([0., 1., 9., 0.]))
    torch.testing.assert_close(sample["transition_mask"][0], torch.tensor([1., 1., 0., 0.]))
    pairs = buf.sample_pairs(100)
    terminal = pairs[:, 0, 0] == 1
    assert terminal.any() and (pairs[terminal, 1, 0] == 9).all()


def test_gain_supervision_matches_real_response_without_training_discriminator():
    class Prior(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        def style_cost(self, first, second):
            return -second[..., 0] * self.scale
    learner = WorldModelLearner.__new__(WorldModelLearner)
    learner.feedback_pair_count = 1
    learner.motion_prior = Prior()
    real = torch.tensor([[[0.], [1.]], [[0.], [-1.]]])
    predicted = torch.tensor([[[1.]], [[2.]]], requires_grad=True)
    mask = torch.ones(2, 2)
    loss = learner._gain_consistency(predicted, real, mask)
    gradient = torch.autograd.grad(loss, predicted)[0]
    assert gradient[1, 0, 0] > 0  # Real adjustment made style worse: correct the optimistic WM.
    assert gradient[0, 0, 0] < 0
    assert learner.motion_prior.scale.grad is None
    better = learner._gain_consistency(predicted-0.1*gradient, real, mask)
    assert better < loss
    assert learner._gain_consistency(predicted, real, mask*0) == 0


def test_failure_focus_never_spills_into_preceding_clip_and_keeps_exploration():
    sampler = ReturnSampler(ReturnSamplerCfg(horizon=1), 20, 100, 1, 1.)
    sampler.configure_failure_focus(torch.tensor([50, 100]), .02, .2, 3.)
    weights = sampler.record(torch.tensor([51]), torch.ones(1), torch.tensor([True]),
                             torch.tensor([False]), torch.tensor([False]))
    assert sampler.failure_count[:10].sum() == 0
    assert sampler.failure_count[10] > 0
    assert (weights >= .8 * .5 / 20).all()
    torch.testing.assert_close(weights.sum(), torch.tensor(1.))
    state = sampler.state_dict()
    restored = ReturnSampler(sampler.cfg, 20, 100, 1, 1.)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.failure_count, sampler.failure_count)


def test_feedback_mask_follows_tensor_device_without_tensordict_device_annotation():
    loop = AmpFeedbackLoop.__new__(AmpFeedbackLoop)
    loop.paired_replay = WindowReplay(4)
    batch = TensorDict({k: torch.zeros(2, 8, 1, device='meta') for k in ('state', 'action', 'sim_ctx')}, [2, 8])
    batch['transition_mask'] = torch.ones(2, 8, device='meta')
    assert batch.device is None
    assert loop.mix_batch(batch)['feedback_pair_mask'].device.type == 'meta'


def test_wm_feedback_batch_keeps_matched_interventions_and_local_data():
    loop = AmpFeedbackLoop.__new__(AmpFeedbackLoop)
    loop.cfg = AmpFeedbackCfg()
    loop.paired_replay, loop.replay = WindowReplay(4), WindowReplay(4)
    loop.priority_windows_used = 0
    def window(value):
        return TensorDict({'state': torch.full((4,8,1), value), 'action': torch.zeros(4,8,1),
            'sim_ctx': torch.zeros(4,8,1), 'transition_mask': torch.ones(4,8),
            'feedback_pair_mask': torch.zeros(4,8)}, [4,8])
    paired = window(1.)
    paired['adjusted_state'] = torch.full((4,8,1), 2.)
    paired['adjusted_action'] = torch.zeros(4,8,1)
    loop.paired_replay.insert(paired, torch.ones(4))
    loop.replay.insert(window(3.), torch.ones(4))
    ordinary = window(0.).repeat(8, 1)
    batch = loop.mix_batch(ordinary)
    assert len(batch) == 32
    assert (batch['state'][:2] == 1).all() and (batch['state'][2:4] == 2).all()
    assert (batch['state'][4:8] == 3).all() and (batch['state'][8:] == 0).all()
    assert (batch['feedback_pair_mask'][:4] == 1).all()
    assert (batch['feedback_pair_mask'][4:] == 0).all()

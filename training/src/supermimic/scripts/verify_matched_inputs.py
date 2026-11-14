"""Check numerical observation/expert equivalence, including clip boundaries."""
import json
from pathlib import Path

import torch
import tyro
from mjlab.tasks.registry import load_env_cfg

from supermimic.learners.amp import AmpCfg, DifferentiableMotionPrior
from supermimic.representations import extract_actor_goal
from supermimic.scripts.compare_amp import DATA, ExpertPairs, SharedObservations
from supermimic.scripts.train_utils import create_env


def main(data: Path = DATA, output: Path = Path("training/runs/input_equivalence.json")):
    cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=True)
    cfg.seed = 7
    cfg.commands["motion"].motion_file = str(data / "*.npz")
    env = create_env(cfg, 17, "cpu")
    features = SharedObservations(env)
    base = env.unwrapped
    cmd = base.command_manager.get_term("motion")
    ends = cmd.motion.motion_ends
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    errors = []
    for frames in [starts, ends - 2, (starts + ends) // 2]:
        for i, frame in enumerate(frames.tolist()):
            cmd.reset_to_frame(torch.tensor([i]), frame)
        base.scene.write_data_to_sim()
        base.sim.forward()
        base.sim.sense()
        base.observation_manager.compute(update_history=True)
        obs = env.get_observations()
        expected_goal, expected_actor_goal = features.rep.compute_wm_goal(
            cmd.motion, cmd.time_steps, features.rep.extract_aux_state(obs["full_state"]))
        errors.append(float((expected_goal - obs["goal"]).abs().max()))
        torch.testing.assert_close(expected_goal, obs["goal"], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(expected_actor_goal, extract_actor_goal(obs["goal"]), atol=2e-5, rtol=2e-5)
    original = DifferentiableMotionPrior.from_motion(cmd.motion, AmpCfg())
    adapted = ExpertPairs(cmd.motion)
    assert torch.equal(original.expert_frames, adapted.frames)
    assert torch.equal(original.pair_starts, adapted.starts)
    assert not torch.isin(adapted.starts, ends - 1).any()
    result = {"passed": True, "tested_clips": 17, "tested_frame_sets": ["start", "penultimate", "midpoint"],
              "goal_max_abs_errors": errors, "amp_expert_features_bitwise_equal": True,
              "valid_pair_starts_bitwise_equal": True, "expert_frames": len(adapted.frames),
              "valid_expert_pairs": len(adapted.starts), "device": "cpu"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    tyro.cli(main)

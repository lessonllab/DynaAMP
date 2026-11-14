"""Recompute paired AMP evidence and summarize a completed feedback run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
import tyro

from supermimic.learners.amp import AmpCfg, DifferentiableMotionPrior
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg
from supermimic.scripts.train_utils import setup_device
from supermimic.utils.gencoord_helpers import compute_gencoord_state


def main(
  log_dir: Path,
  output: Path,
  ancestor: Path | None = None,
  earlier_log_dir: Path | None = None,
  earlier_log_dirs: tuple[Path, ...] = (),
  video_json: Path | None = None,
  device: str = "cuda:0",
) -> None:
  setup_device(device)
  saved = torch.load(
    log_dir / "training_state.pt", map_location="cpu", weights_only=False
  )
  old_cfg = (
    torch.load(ancestor, map_location="cpu", weights_only=False)["runner_cfg"]
    if ancestor is not None
    else asdict(SuperMimicRunnerCfg())
  )
  records = []
  folders = dict.fromkeys(
    folder.resolve()
    for folder in [earlier_log_dir, *earlier_log_dirs, log_dir]
    if folder is not None
  )
  for folder in folders:
    records += [
      json.loads(line)
      for line in (folder / "amp_feedback_rounds.jsonl").read_text().splitlines()
    ]
  pair = torch.load(
    log_dir / "amp_feedback_last_pair.pt", map_location=device, weights_only=False
  )
  dim = pair["prior_state"]["mean"].shape[-1]
  feature_encoder = None
  if pair.get("feature_encoder") is not None:
    from supermimic.learners.body_amp_features import BodyAMPFeatures
    feature_encoder = BodyAMPFeatures.from_metadata(pair["feature_encoder"], device)
  prior = DifferentiableMotionPrior(
    torch.zeros(2, dim, device=device),
    torch.tensor([2], device=device),
    AmpCfg(**pair["prior_cfg"]),
    feature_encoder,
  )
  prior.load_state_dict(pair["prior_state"])
  prior.freeze()
  with torch.no_grad():

    def score(states):
      return -prior.style_cost(states[:, :-1], states[:, 1:]).mean(1)

    real_base = score(compute_gencoord_state(pair["real_base"]))
    real_adjusted = score(compute_gencoord_state(pair["real_adjusted"]))
    actual_gain = real_adjusted - real_base
    predicted_gain = score(pair["predicted_adjusted_rep"]) - score(
      pair["predicted_base_rep"]
    )
    updated_gain = score(pair["updated_adjusted_rep"]) - score(pair["updated_base_rep"])
    before = float((predicted_gain - actual_gain).abs().mean())
    after = float((updated_gain - actual_gain).abs().mean())

  cfg = saved["runner_cfg"]
  # CUDA training uses TF32; CPU IEEE FP32 has slightly different absolute scores.
  score_atol = 1e-4 if device == "cpu" else 1e-5
  preserved = {
    key: cfg[key] == old_cfg[key]
    for key in [
      "actor_hidden_dims",
      "critic_hidden_dims",
      "dynamics_hidden_dims",
      "latent_dim",
      "policy",
      "wm",
      "slot_mix",
    ]
  }
  checks = {
    "official_architecture_batches_bptt_preserved": all(preserved.values()),
    "paired_same_initial_state": torch.equal(
      pair["real_base"][:, 0], pair["real_adjusted"][:, 0]
    )
    and torch.equal(pair["initial"], pair["real_base"][:, 0]),
    "nonzero_action_adjustment": bool(
      (pair["base_actions"] != pair["adjusted_actions"]).any()
    ),
    "same_discriminator_scores_reproduced": torch.allclose(
      real_base, pair["real_base_score"], atol=score_atol, rtol=1e-5
    )
    and torch.allclose(
      real_adjusted, pair["real_adjusted_score"], atol=score_atol, rtol=1e-5
    ),
    "before_after_gap_reproduced": abs(before - pair["record"]["heldout_gap_before"])
    < 1e-5
    and abs(after - pair["record"]["heldout_gap_after"]) < 1e-5,
    "live_state_and_context_restored": all(
      r["restoration_max_error"]
      == r["initial_state_max_error"]
      == r["context_max_error"]
      == 0
      for r in records
    ),
    "selected_data_excludes_holdout": not bool(
      torch.isin(pair["holdout_env_ids"], pair["selected_train_env_ids"]).any()
    ),
    "priority_data_used": saved["amp_feedback"]["budget"]["priority_windows_used"] > 0,
    "feedback_round_count_reconciles": len(records)
    == saved["amp_feedback"]["budget"]["rounds"],
    "feedback_budget_reconciles": sum(
      r["paired_transitions"]
      + r["repeat_check_transitions"]
      + r["supplement_transitions"]
      for r in records
    )
    == saved["amp_feedback"]["budget"]["feedback_transitions"],
  }
  if video_json is not None:
    video = json.loads(video_json.read_text())
    gradient = json.loads(video_json.with_suffix(".gradient.json").read_text())
    checks["physical_video_completed"] = video["actual_seconds"] == video[
      "requested_seconds"
    ] and all(x < 0 for x in video["first_failure_steps"])
    checks["full_bptt_gradient_connected"] = (
      gradient["passed"] and gradient["steps"] == 32
    )
  else:
    video, gradient = None, None

  def mean(key):
    return sum(r[key] for r in records) / len(records)

  result = {
    "passed": all(checks.values()),
    "checks": checks,
    "checkpoint": str(log_dir / "training_state.pt"),
    "training_step": saved["step"],
    "checkpoint_sha256": hashlib.sha256(
      (log_dir / "training_state.pt").read_bytes()
    ).hexdigest(),
    "configuration_preserved": preserved,
    "configuration_reference": str(ancestor) if ancestor is not None else "official runner defaults; fresh training",
    "score_recompute_device": device,
    "absolute_score_tolerance": score_atol,
    "feedback_rounds": len(records),
    "feedback_log_dirs": [str(folder) for folder in folders],
    "budget": saved["amp_feedback"]["budget"],
    "budget_scope": saved["interaction_budget_scope"],
    "heldout_mean_gap_before": mean("heldout_gap_before"),
    "heldout_mean_gap_after": mean("heldout_gap_after"),
    "rounds_with_lower_heldout_gap": sum(
      r["heldout_gap_after"] < r["heldout_gap_before"] for r in records
    ),
    "heldout_mean_state_mse_before": mean("heldout_normalized_state_mse_before"),
    "heldout_mean_state_mse_after": mean("heldout_normalized_state_mse_after"),
    "mean_repeat_rejected_environments": mean("repeat_rejected_environments"),
    "latest_pair_recomputed_gap": {"before": before, "after": after},
    "video": video,
    "gradient_audit": gradient,
    "interpretation": "Closed-loop implementation validation; no sample-efficiency or causal motion-quality superiority claim. Holdouts exclude this round's branch replay, not all historical training experience.",
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2))
  print(json.dumps(result, indent=2))
  if not result["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  tyro.cli(main)

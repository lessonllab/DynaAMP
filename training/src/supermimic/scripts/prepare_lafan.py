"""Use the official mjlab converter, saving local NPZs without an upload."""

import hashlib
import json
from pathlib import Path

import numpy as np
import tyro
from mjlab.scripts.csv_to_npz import MotionLoader


def main(input_dir: Path, output_dir: Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  records = []
  for source in sorted(input_dir.glob("*.csv")):
    motion = MotionLoader(str(source), input_fps=30, output_fps=50, device="cpu")
    arrays = motion.arrays()
    if not all(np.isfinite(v).all() for k, v in arrays.items() if k != "joint_names"):
      raise ValueError(f"Non-finite reference data: {source}")
    target = output_dir / f"{source.stem}.npz"
    if target.exists():
      with np.load(target) as existing:
        if set(existing.files) != set(arrays) or any(
          not np.array_equal(existing[k], v) for k, v in arrays.items()
        ):
          raise ValueError(f"Existing converted data differs: {target}")
    else:
      np.savez(target, **arrays)
    records.append(
      {
        "source": str(source.resolve()),
        "output": str(target.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "input_fps": 30,
        "output_fps": 50,
        "frames": len(arrays["joint_pos"]),
        "joint_names": arrays["joint_names"].tolist(),
      }
    )
    print(f"[DATA] {target}: {len(arrays['joint_pos'])} frames at 50 Hz")
  if not records:
    raise ValueError(f"No CSV files in {input_dir}")
  (output_dir / "manifest.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
  tyro.cli(main)

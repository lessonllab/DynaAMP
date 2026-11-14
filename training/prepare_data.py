"""Fetch the exact 40 G1 clips and convert them with pinned MjLab at 50 Hz."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

import numpy as np
from mjlab.scripts.csv_to_npz import MotionLoader

ROOT = Path(__file__).resolve().parent
SOURCE = (
    "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/"
    "resolve/ce1572906efe6157840e8474d5a0d7aa87481e74/g1"
)
CATALOG = ROOT / "config/lafan1_g1.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(info, directory: Path) -> Path:
    path = directory / info["csv"]
    if path.is_file() and path.stat().st_size == info["size"] and sha256(path) == info["sha256"]:
        return path
    part = path.with_suffix(".csv.part")
    try:
        with urlopen(f"{SOURCE}/{path.name}", timeout=120) as response, part.open("wb") as file:
            shutil.copyfileobj(response, file)
        if part.stat().st_size != info["size"] or sha256(part) != info["sha256"]:
            raise ValueError(f"Source CSV differs from the release manifest: {path.name}")
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/lafan1_g1_50hz")
    parser.add_argument("--csv-dir", type=Path, default=ROOT / "data/lafan1_g1_csv")
    args = parser.parse_args()
    data, csv_dir = args.data.resolve(), args.csv_dir.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(CATALOG.read_text())
    with ThreadPoolExecutor(max_workers=4) as workers:
        sources = list(workers.map(lambda info: fetch(info, csv_dir), catalog))

    manifest = []
    for info, source in zip(catalog, sources, strict=True):
        arrays = MotionLoader(str(source), input_fps=30, output_fps=50, device="cpu").arrays()
        frames = len(arrays["joint_pos"])
        if frames != info["frames"] or not all(
            np.isfinite(value).all() for key, value in arrays.items() if key != "joint_names"
        ):
            raise ValueError(f"Unexpected converted motion: {source.name}")
        target = data / f"{source.stem}.npz"
        tmp = target.with_suffix(".npz.part")
        with tmp.open("wb") as f:
            np.savez(f, **arrays)
        tmp.replace(target)
        manifest.append({
            "source": source.name, "output": target.name,
            "source_sha256": info["sha256"], "output_sha256": sha256(target),
            "input_fps": 30, "output_fps": 50, "frames": frames,
            "joint_names": arrays["joint_names"].tolist(),
        })

    path = data / "manifest.json"
    part = path.with_suffix(".json.part")
    part.write_text(json.dumps(manifest, indent=2) + "\n")
    part.replace(path)
    print(f"Prepared {len(manifest)} motions in {data}")


if __name__ == "__main__":
    main()

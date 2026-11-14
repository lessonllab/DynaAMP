"""Paths shared by the two deployment entry points."""

from pathlib import Path
import os


DEPLOY = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("LOOPMIMIC_RUNTIME", DEPLOY / ".runtime")).expanduser().resolve()
SONIC_SOURCE = Path(os.environ.get("SONIC_SOURCE", RUNTIME / "sonic")).resolve()
SONIC_WEIGHTS = Path(os.environ.get("SONIC_WEIGHTS", RUNTIME / "sonic_weights")).resolve()
SONIC_DEPLOY = SONIC_SOURCE / "gear_sonic_deploy"
SONIC_BIN = SONIC_DEPLOY / "target/release/g1_deploy_onnx_ref"
SONIC_REFERENCE = SONIC_DEPLOY / "reference/example/macarena_001__A545"
DEFAULT_BUNDLE = DEPLOY / "shared/models/loopmimic_014000"
DEFAULT_MOTION = RUNTIME / "motions/dance1_subject2.npz"
G1_SCENE = RUNTIME / "unitree_mujoco/unitree_robots/g1/scene_29dof.xml"


def sonic_command(interface: str, references: Path, input_type: str, port: int) -> list[str]:
    planner = SONIC_WEIGHTS / "planner_sonic_V2.onnx"
    decoder = SONIC_WEIGHTS / "model_decoder.onnx"
    encoder = SONIC_WEIGHTS / "model_encoder.onnx"
    obs = SONIC_WEIGHTS / "observation_config.yaml"
    for path in (SONIC_BIN, planner, decoder, encoder, obs, SONIC_REFERENCE / "joint_pos.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"{path}. Run bash deploy/setup.sh first.")
    return [
        str(SONIC_BIN), interface, str(decoder), str(references), "--encoder-file", str(encoder),
        "--obs-config", str(obs), "--planner-file", str(planner), "--input-type", input_type,
        "--zmq-out-port", str(port),
    ]


def prepare_reference_directory(output: Path) -> Path:
    folder = output / "references"
    folder.mkdir(exist_ok=True, parents=True)
    example = folder / "macarena"
    if not example.is_symlink() and not example.exists():
        example.symlink_to(SONIC_REFERENCE, target_is_directory=True)
    return folder

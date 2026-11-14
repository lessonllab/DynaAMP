# Sourced by the Sim2Sim and Sim2Real launchers.
loopmimic_deploy="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LOOPMIMIC_RUNTIME="${LOOPMIMIC_RUNTIME:-$loopmimic_deploy/.runtime}"
export SONIC_SOURCE="${SONIC_SOURCE:-$LOOPMIMIC_RUNTIME/sonic}"
loopmimic_python="${SONIC_PYTHON:-$LOOPMIMIC_RUNTIME/venv/bin/python}"
if [[ ! -x "$loopmimic_python" ]]; then
  echo "Run bash deploy/setup.sh first, or set SONIC_PYTHON." >&2
  exit 1
fi
export PYTHONPATH="$loopmimic_deploy:$SONIC_SOURCE:$SONIC_SOURCE/external_dependencies/unitree_sdk2_python${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${TensorRT_ROOT:+$TensorRT_ROOT/lib:}${onnxruntime_ROOT:+$onnxruntime_ROOT/lib:}$LOOPMIMIC_RUNTIME/onnxruntime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

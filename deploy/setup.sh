#!/usr/bin/env bash
set -euo pipefail

deploy_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime="${LOOPMIMIC_RUNTIME:-$deploy_dir/.runtime}"
mkdir -p "$runtime"
runtime="$(cd "$runtime" && pwd)"
sonic="$runtime/sonic"
weights="$runtime/sonic_weights"
sonic_repo=https://github.com/NVlabs/GR00T-WholeBodyControl.git
sonic_commit=1e1ec4f5f27e6c3e03048f693bd7c60cce04afda
mujoco_repo=https://github.com/jie0110/hiking-in-the-wild-sim2sim.git
mujoco_commit=557d07479d76a6a077b142636fba9d1de58c7e06
hf=https://huggingface.co/nvidia/GEAR-SONIC/resolve/6733128a3d8a523b1418b06bca3cdf61c8b0987f
sonic_media="https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/$sonic_commit"
sdk=gear_sonic_deploy/thirdparty/unitree_sdk2
sdk_python=external_dependencies/unitree_sdk2_python
macarena=gear_sonic_deploy/reference/example/macarena_001__A545
architecture="$(uname -m)"

case "$architecture" in
  x86_64) crc_lib=crc_amd64.so; ort_arch=x64 ;;
  aarch64) crc_lib=crc_aarch64.so; ort_arch=aarch64 ;;
  *) echo "SONIC supports Linux x86_64 and aarch64." >&2; exit 1 ;;
esac

for tool in git curl cmake sha256sum; do
  command -v "$tool" >/dev/null || { echo "Install $tool first." >&2; exit 1; }
done

checkout_paths() {
  local directory="$1" repo="$2" commit="$3"
  shift 3
  if [[ -d "$directory/.git" ]]; then
    local head
    head="$(git -C "$directory" rev-parse --verify HEAD 2>/dev/null || true)"
    if [[ -n "$head" && "$head" != "$commit" ]]; then
      echo "$directory is not at the required commit $commit." >&2
      return 1
    fi
    if [[ "$head" == "$commit" ]]; then return; fi
  else
    mkdir -p "$directory"
    git -C "$directory" init --quiet
  fi
  if ! git -C "$directory" remote get-url origin >/dev/null 2>&1; then
    git -C "$directory" remote add origin "$repo"
  fi
  git -C "$directory" sparse-checkout set --no-cone "$@"
  git -C "$directory" fetch --quiet --depth=1 origin "$commit"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$directory" checkout --quiet --detach FETCH_HEAD
}

download() {
  local url="$1" dest="$2" checksum="${3:-}"
  if [[ -f "$dest" && -n "$checksum" ]] && printf '%s  %s\n' "$checksum" "$dest" | sha256sum --status --check; then
    return
  fi
  mkdir -p "$(dirname "$dest")"
  curl --fail --silent --show-error --location --retry 3 --connect-timeout 20     "$url" --output "$dest.part"
  if [[ -n "$checksum" ]]; then
    printf '%s  %s\n' "$checksum" "$dest.part" | sha256sum --status --check
  fi
  mv "$dest.part" "$dest"
}

materialize_lfs() {
  local path="$1" checksum
  checksum="$(git -C "$sonic" show "$sonic_commit:$path" | awk '/^oid sha256:/{sub("sha256:", "", $2); print $2}')"
  [[ "${#checksum}" -eq 64 ]] || { echo "Missing LFS hash for $path." >&2; return 1; }
  download "$sonic_media/$path" "$sonic/$path" "$checksum"
}

checkout_paths "$sonic" "$sonic_repo" "$sonic_commit"   '/.gitattributes' '/LICENSE' '/gear_sonic/' '/gear_sonic_deploy/CMakeLists.txt'   '/gear_sonic_deploy/cmake/' '/gear_sonic_deploy/src/' '/gear_sonic_deploy/scripts/'   "/$sdk/" "/$sdk_python/" "/$macarena/"
for file in "$sdk/lib/$architecture/libunitree_sdk2.a"   "$sdk/thirdparty/lib/$architecture/libddsc.so" "$sdk/thirdparty/lib/$architecture/libddscxx.so"   "$sdk_python/unitree_sdk2py/utils/lib/$crc_lib"   "$macarena/"{body_ang_vel,body_lin_vel,body_pos,body_quat,joint_pos,joint_vel}.csv; do
  materialize_lfs "$file"
done

checkout_paths "$runtime/mujoco_source" "$mujoco_repo" "$mujoco_commit" '/unitree_mujoco/unitree_robots/g1/'
ln -sfn "$runtime/mujoco_source/unitree_mujoco" "$runtime/unitree_mujoco"

download "$hf/model_encoder.onnx" "$weights/model_encoder.onnx" 013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3
download "$hf/model_decoder.onnx" "$weights/model_decoder.onnx" c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed
download "$hf/planner_sonic.onnx" "$weights/planner_sonic_V2.onnx" 39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea
download "$hf/observation_config.yaml" "$weights/observation_config.yaml" "$(sed -n '1p' "$deploy_dir/shared/observation_config.sha256")"
if [[ ! -f "$weights/LICENSE" ]]; then download "$hf/LICENSE" "$weights/LICENSE"; fi

venv="$runtime/venv"
if [[ ! -x "$venv/bin/python" ]]; then
  "${LOOPMIMIC_PYTHON:-python3}" -m venv "$venv"
fi
if [[ ! -f "$venv/loopmimic-requirements.sha256" ]] ||
   ! cmp -s "$venv/loopmimic-requirements.sha256" "$deploy_dir/requirements.txt"; then
  if [[ -z "${CYCLONEDDS_HOME:-}" && -f /usr/include/dds/dds.h ]]; then
    export CYCLONEDDS_HOME=/usr
  fi
  "$venv/bin/python" -m pip install -r "$deploy_dir/requirements.txt"
  cp "$deploy_dir/requirements.txt" "$venv/loopmimic-requirements.sha256"
fi
PYTHONPATH="$deploy_dir" "$venv/bin/python" -m shared.prepare_motion --output "$runtime/motions/dance1_subject2.npz"

if [[ -z "${onnxruntime_ROOT:-}" ]]; then
  ort_name="onnxruntime-linux-$ort_arch-1.22.0"
  if [[ ! -f "$runtime/onnxruntime/lib/libonnxruntime.so.1" ]]; then
    download "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/$ort_name.tgz" "$runtime/$ort_name.tgz"
    mkdir -p "$runtime/onnxruntime"
    tar -xzf "$runtime/$ort_name.tgz" -C "$runtime/onnxruntime" --strip-components=1
  fi
  export onnxruntime_ROOT="$runtime/onnxruntime"
fi

# The upstream Findonnxruntime module accepts an explicit runtime root.
source_dir="$sonic/gear_sonic_deploy"
patch="$deploy_dir/shared/sonic_loopmimic.patch"
if git -C "$sonic" apply --check "$patch"; then
  git -C "$sonic" apply "$patch"
else
  git -C "$sonic" apply --reverse --check "$patch"
fi
install -m0644 "$deploy_dir/shared/loopmimic_dance_handoff.hpp"   "$source_dir/src/g1/g1_deploy_onnx_ref/include/loopmimic_dance_handoff.hpp"

export HAS_ROS2=0
cmake -S "$source_dir" -B "$runtime/build" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$runtime/build" --target g1_deploy_onnx_ref -j "${LOOPMIMIC_BUILD_JOBS:-2}"
echo "Ready. Run bash deploy/sim2sim/sonic.sh --dance-demo --output /tmp/loopmimic-sim."

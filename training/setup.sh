#!/usr/bin/env bash
set -euo pipefail

training="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mjlab="$training/../mjlab_research"
revision=dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd
remote=https://github.com/HybridRobotics/mjlab_research.git

command -v uv >/dev/null || { echo "Install uv before setting up the training environment." >&2; exit 1; }
if [[ ! -e "$mjlab" ]]; then
  git clone --filter=blob:none --no-checkout "$remote" "$mjlab"
  git -C "$mjlab" checkout --quiet --detach "$revision"
elif [[ "$(git -C "$mjlab" rev-parse HEAD)" != "$revision" ]] ||
     ! git -C "$mjlab" diff --quiet HEAD; then
  echo "Use a clean mjlab_research checkout at $revision: $mjlab" >&2
  exit 1
fi

uv sync --directory "$training" --locked --python 3.13 "$@"

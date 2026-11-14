#!/usr/bin/env bash
set -euo pipefail
sonic_real_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$sonic_real_dir/../shared/env.sh"
exec "$loopmimic_python" "$sonic_real_dir/run.py" "$@"

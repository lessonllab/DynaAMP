#!/usr/bin/env bash
set -euo pipefail
sonic_sim_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$sonic_sim_dir/../shared/env.sh"
exec "$loopmimic_python" "$sonic_sim_dir/sonic_sim.py" "$@"

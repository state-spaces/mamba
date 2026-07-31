#!/usr/bin/env bash
# Launch the Helion autotune harness detached via nohup, logging to its own file.
# Any arguments are forwarded to the harness, e.g.:
#
#   ./benchmarks/helion/run_autotune.sh --precompile fork
#   LOG_DIR=/data/logs ./benchmarks/helion/run_autotune.sh --sequence-packing
#
# Env: PYTHON (interpreter, default python), LOG_DIR (default <repo>/logs).
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log_dir=${LOG_DIR:-"$repo_root/logs"}
mkdir -p "$log_dir"
log_file="$log_dir/autotune_mamba_mixer_$(date +%Y%m%d_%H%M%S).log"

cd "$repo_root"
# -u so the log is not block-buffered while the run is still in progress.
nohup "${PYTHON:-python}" -u benchmarks/helion/autotune_mamba_mixer.py "$@" \
    >"$log_file" 2>&1 &

echo "pid $! logging to $log_file"
echo "follow with: tail -f $log_file"

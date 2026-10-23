#!/bin/bash
# Multi-process acceptance runner for the NPU test suite.
#
# Lesson from validation (E9): a single in-process PASS is NOT evidence --
# the state-passing pipelining race only shows up at process level. Any test
# that exercises the SSD path (test_ssd_npu.py, test_ssd_bwd_npu.py) must be
# run across multiple fresh processes.
#
# Usage:  ./run_npu_tests.sh [N]        # N processes per SSD test, default 8
set -u
N=${1:-8}
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE/../..:${PYTHONPATH:-}"   # append, never overwrite (CANN paths live in PYTHONPATH)
cd "$HERE"

echo "== env fingerprint =="
python3 -c "import torch, torch_npu, triton; print(torch.__version__, torch_npu.__version__, triton.__version__)"

echo "== smoke =="
python3 example1_add.py
python3 -m pytest example2_broadcast.py -q

echo "== op level =="
for t in test_ssu_npu test_layernorm_npu test_swiglu_npu test_layernorm_gated_bwd_npu \
         test_layer_norm_plain_npu test_chunk_cumsum_npu test_bmm_chunk_npu \
         test_chunk_state_bwd_npu test_chunk_scan_bwd_npu; do
  echo "--- $t"
  python3 "$t.py" || echo "!! $t FAILED (exit $?)"
done

echo "== SSD combined: $N fresh processes each =="
for i in $(seq 1 "$N"); do
  echo "--- test_ssd_npu run$i"
  python3 test_ssd_npu.py || echo "!! test_ssd_npu run$i FAILED"
done
echo "--- test_ssd_bwd_npu"
python3 test_ssd_bwd_npu.py || echo "!! test_ssd_bwd_npu FAILED"

echo "== module level =="
python3 test_mamba2_module.py

echo "== diagnostic (expected UB-overflow error message) =="
python3 test_coredim.py ub || true
echo "== diagnostic (expected pass, sum = 0.0) =="
python3 test_coredim.py coredim

#!/bin/bash
# Check whether a patched bishengir can be handed to TileLang as-is.
#
#   bash verify_patched_bishengir.sh [BIN_DIR]
#
# BIN_DIR is the directory that contains bin/ (for example the unpacked
# bishengir-patched-1.2.0). If omitted, and this script sits inside the package
# (same directory has bin/), that directory is used.
#
# Four checks:
#   1. executable + self-reported version
#   2. a real TileLang kernel compiles (rc=0 and an .o is produced)
#      -- this is the key one: the stock CANN 1.2.0 gives 5/5 SIGSEGV on this IR
#   3. a minimal reproduction IR no longer crashes (rc != 139)
#   4. the effective TILELANG_NPU_COMPILER_PATH / OPT_PATH are printed
#
# Returns 0 if everything passes, 1 otherwise, 2 on a usage error.

set -u

HERE=$(cd "$(dirname "$0")" && pwd)

# The argument is the directory that contains bin/. If omitted: when this script
# lives inside the package (a bin/ sits next to it), use its own directory.
if [ -n "${1:-}" ]; then
  BIN_DIR="$1"
elif [ -x "$HERE/bin/tl-bishengir-compile" ]; then
  BIN_DIR="$HERE"
else
  echo "Usage: bash verify_patched_bishengir.sh <directory containing bin/>"
  echo "       e.g. bash verify_patched_bishengir.sh /opt/bishengir-patched-1.2.0"
  exit 2
fi

REAL_IR="$HERE/real_cube_kernel.npuir"
MIN_IR="$HERE/mmadl1_min.npuir"

FLAGS=(
  --disable-hivm-auto-inject-sync=true
  --enable-auto-multi-buffer=false
  --enable-triton-kernel-compile=true
  --enable-hivm-compile=true
  --disable-hivm-tensor-compile=true
)

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

pass=0
fail=0
ok()   { echo "  [ OK ] $*"; pass=$((pass + 1)); }
bad()  { echo "  [FAIL] $*"; fail=$((fail + 1)); }
note() { echo "  [note] $*"; }

echo "== 0. Locating the compiler =="
CC=""
if [ -x "$BIN_DIR/bin/tl-bishengir-compile" ]; then
  CC="$BIN_DIR/bin/tl-bishengir-compile"
  ok "wrapper: $CC"
elif [ -x "$BIN_DIR/bin/bishengir-compile" ]; then
  CC="$BIN_DIR/bin/bishengir-compile"
  note "only the bare binary was found, without the tl-bishengir-compile wrapper."
  note "The bare binary may fail at the hivmc step because it rejects --enable-lir-compile*."
else
  bad "neither tl-bishengir-compile nor bishengir-compile under $BIN_DIR/bin"
  echo
  echo "Conclusion: FAIL (wrong path — point at the directory that contains bin/)"
  exit 1
fi

echo
echo "== 1. Self-reported version =="
ver=$("$CC" --version 2>&1 | head -1)
if [ -n "$ver" ]; then
  ok "$ver"
else
  bad "no --version output"
fi

echo
echo "== 2. Real kernel compile (key check) =="
if [ ! -f "$REAL_IR" ]; then
  bad "sample missing: $REAL_IR"
else
  "$CC" "$REAL_IR" "${FLAGS[@]}" -o "$TMP/real.o" >"$TMP/real.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] && [ -s "$TMP/real.o" ]; then
    ok "rc=0, produced $(stat -c '%s' "$TMP/real.o" 2>/dev/null || echo '?') B"
  else
    bad "rc=$rc (stock CANN 1.2.0 gives 5/5 SIGSEGV on this IR)"
    tail -3 "$TMP/real.log" | sed 's/^/         /'
  fi
fi

echo
echo "== 3. Minimal reproduction IR no longer crashes =="
if [ ! -f "$MIN_IR" ]; then
  bad "sample missing: $MIN_IR"
else
  "$CC" "$MIN_IR" "${FLAGS[@]}" -o "$TMP/min.o" >"$TMP/min.log" 2>&1
  rc=$?
  if [ "$rc" -eq 139 ] || [ "$rc" -eq 134 ]; then
    bad "rc=$rc — still crashing (139=SIGSEGV, 134=SIGABRT)"
  else
    ok "rc=$rc (not 139/134, so no out-of-bounds crash; this IR is incomplete, rc=1 is expected)"
  fi
fi

echo
echo "== 4. Current environment variables =="
note "TILELANG_NPU_COMPILER_PATH=${TILELANG_NPU_COMPILER_PATH:-<unset, falls back to the stock CANN compiler>}"
note "TILELANG_NPU_OPT_PATH=${TILELANG_NPU_OPT_PATH:-<unset; bishengir-opt stays on the CANN copy>}"
note "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}"
if [ -n "${TILELANG_NPU_COMPILER_PATH:-}" ] && [ "$TILELANG_NPU_COMPILER_PATH" != "$CC" ]; then
  note "Note: the variable points at a different copy than the one just verified ($TILELANG_NPU_COMPILER_PATH)"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "Conclusion: PASS ($pass checks OK). You can export TILELANG_NPU_COMPILER_PATH=$CC"
  exit 0
else
  echo "Conclusion: FAIL ($fail FAIL, $pass OK)"
  exit 1
fi

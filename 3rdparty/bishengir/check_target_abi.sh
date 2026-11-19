#!/bin/bash
# Preflight check for the target machine: 30 seconds to tell whether this prebuilt
# bishengir can be used as-is.
#
#   bash check_target_abi.sh [BIN_DIR]
#
# BIN_DIR is optional. If given, a real compile smoke test is run as well;
# without it, only the runtime environment is inspected.
# Run this on the TARGET machine; the tarball does not need to be there yet.
#
# Why this matters: the binary was relinked on Ubuntu 22.04 (glibc 2.35), so its
# ABI baseline is higher than the stock CANN build. An older glibc fails with
# "version `GLIBC_2.34' not found".

set -u

NEED_GLIBC=2.34
NEED_GLIBCXX=3.4.30
NEED_ARCH=aarch64

pass=0
fail=0
ok()   { echo "  [ OK ] $*"; pass=$((pass + 1)); }
bad()  { echo "  [FAIL] $*"; fail=$((fail + 1)); }
note() { echo "  [note] $*"; }

ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

echo "== 1. CPU architecture =="
arch=$(uname -m)
if [ "$arch" = "$NEED_ARCH" ]; then
  ok "$arch"
else
  bad "$arch (need $NEED_ARCH; cross-architecture means another machine or QEMU emulation, not recommended)"
fi

echo
echo "== 2. glibc version (hard requirement: 2.34; cannot be worked around with bundled libs) =="
glibc=$(ldd --version 2>&1 | head -1 | awk '{print $NF}')
if ge "$glibc" "$NEED_GLIBC"; then
  ok "$glibc >= $NEED_GLIBC"
else
  bad "$glibc < $NEED_GLIBC — you can only relink on this machine, or move to a host/image with a newer glibc"
fi

echo
echo "== 3. libstdc++ GLIBCXX level =="
# Prefer the copy bundled in the package ($ORIGIN/../lib), fall back to the system one.
cand=""
if [ -n "${1:-}" ] && [ -f "$1/lib/libstdc++.so.6" ]; then
  cand="$1/lib/libstdc++.so.6"
  note "using the bundled copy: $cand"
else
  cand=$(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6 /{print $NF; exit}')
  [ -n "$cand" ] && note "using the system copy: $cand"
fi
if [ -z "$cand" ] || [ ! -f "$cand" ]; then
  bad "libstdc++.so.6 not found"
else
  if strings "$cand" 2>/dev/null | grep -qx "GLIBCXX_$NEED_GLIBCXX"; then
    ok "provides up to GLIBCXX_$NEED_GLIBCXX"
  else
    top=$(strings "$cand" 2>/dev/null | grep -oE 'GLIBCXX_3\.4\.[0-9]+' | sort -V | tail -1)
    bad "highest is $top, need GLIBCXX_$NEED_GLIBCXX (dropping a libstdc++ into the package's lib/ works around this)"
  fi
fi

echo
echo "== 4. CANN =="
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -d "$ASCEND_HOME_PATH" ]; then
  ok "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
else
  for d in /usr/local/Ascend/cann-9.1.0 /usr/local/Ascend/ascend-toolkit/latest; do
    if [ -d "$d" ]; then
      note "ASCEND_HOME_PATH is unset, but $d exists (the wrapper falls back to ascend-toolkit/latest)"
      break
    fi
  done
  [ -d /usr/local/Ascend ] || bad "/usr/local/Ascend not found — is CANN installed?"
fi
hivmc=$(command -v hivmc 2>/dev/null || find "${ASCEND_HOME_PATH:-/usr/local/Ascend}" -maxdepth 4 -name hivmc -type f 2>/dev/null | head -1)
if [ -n "$hivmc" ]; then
  ok "hivmc: $hivmc"
else
  note "hivmc not found (needed to compile device code; usually visible after sourcing set_env.sh)"
fi

echo
echo "== 5. Smoke test (optional) =="
if [ -n "${1:-}" ]; then
  CC="$1/bin/tl-bishengir-compile"
  [ -x "$CC" ] || CC="$1/bin/bishengir-compile"
  if [ -x "$CC" ]; then
    if v=$("$CC" --version 2>&1 | head -1); then
      ok "executes: $v"
    else
      bad "failed to execute (most likely an ABI mismatch — see the glibc item above)"
    fi
  else
    bad "no executable compiler under $1/bin"
  fi
else
  note "BIN_DIR not given, skipping (once you have the package, run verify_patched_bishengir.sh for the full acceptance)"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "Conclusion: target machine is usable ($pass checks OK)"
  exit 0
else
  echo "Conclusion: target machine does not meet the requirements ($fail FAIL) — see the remedy noted under each item"
  exit 1
fi

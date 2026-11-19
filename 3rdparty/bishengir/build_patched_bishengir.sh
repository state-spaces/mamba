#!/bin/bash
# Build a patched bishengir from source on a target machine.
#
#   bash build_patched_bishengir.sh --src /path/to/AscendNPU-IR [options]
#
# When you need this: the target machine has glibc < 2.34 (the prebuilt package
# fails with `version 'GLIBC_2.34' not found`), or its CANN is not 9.1.0.
# Building here matches the local toolchain and ABI, and the result is guaranteed
# only for this "distro + CANN version" combination -- so it is built per ABI
# type, not per machine.
#
# Options:
#   --src DIR       AscendNPU-IR source tree root (required; must contain build-tools/build.sh)
#   --out DIR       output directory, default ./bishengir-patched
#   --jobs N        compile parallelism, default nproc
#   --jobs-link N   link parallelism, default 4 (linking is memory hungry, keep it low)
#   --dry-run       only run preflight + apply the patch, do not build
#   --skip-patches  skip the third-party apply_patches.sh (already applied)
#
# Prerequisites:
#   - the source tree is v1.2.0 (git clone --branch v1.2.0 with submodules, or an equivalent source bundle)
#   - cmake / ninja / gcc / g++ / lld / python3 are available
#   - CANN is installed (the compiler you build calls CANN's hivmc at runtime)
#
# Measured cost (256-core machine, -j16, load ~36): source + submodules 2.3 GB,
# build directory 1.7 GB, ~35 minutes of pure compiling. On a shared machine do
# not open --jobs all the way.

set -u

HERE=$(cd "$(dirname "$0")" && pwd)
PATCH="$HERE/bishengir-v1.2.0-buildable.patch"
VERIFY="$HERE/verify_patched_bishengir.sh"

SRC=""
OUT="$PWD/bishengir-patched"
JOBS=$(nproc 2>/dev/null || echo 8)
JOBS_LINK=4
DRY=0
SKIP_PATCHES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --src) SRC="${2:-}"; shift 2 ;;
    --src=*) SRC="${1#--src=}"; shift ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --out=*) OUT="${1#--out=}"; shift ;;
    --jobs) JOBS="${2:-}"; shift 2 ;;
    --jobs=*) JOBS="${1#--jobs=}"; shift ;;
    --jobs-link) JOBS_LINK="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --skip-patches) SKIP_PATCHES=1; shift ;;
    -h|--help) sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

step() { echo; echo "== $* =="; }
die()  { echo "  [FAIL] $*" >&2; exit 1; }

[ -n "$SRC" ] || { echo "--src <source tree> is required" >&2; exit 2; }
[ -d "$SRC" ] || die "source tree does not exist: $SRC"
[ -f "$SRC/build-tools/build.sh" ] || die "$SRC/build-tools/build.sh is missing (incomplete AscendNPU-IR tree?)"
[ -f "$PATCH" ] || die "patch not found: $PATCH"
[ -f "$VERIFY" ] || die "verify script not found: $VERIFY"

step "1. Preflight"
missing=0
for t in cmake ninja gcc g++ python3; do
  if command -v "$t" >/dev/null 2>&1; then
    echo "  [ OK ] $t: $(command -v $t)"
  else
    echo "  [FAIL] missing $t"; missing=1
  fi
done
if command -v ld.lld >/dev/null 2>&1 || command -v lld >/dev/null 2>&1; then
  echo "  [ OK ] lld: $(command -v ld.lld 2>/dev/null || command -v lld)"
else
  echo "  [FAIL] missing lld (apt install lld / dnf install lld)"; missing=1
fi
[ "$missing" -eq 0 ] || die "preflight dependencies are incomplete"

avail_gb=$(df -BG --output=avail "$(dirname "$OUT")" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -lt 8 ]; then
  echo "  [warn] only ${avail_gb}G left on the output partition; the build dir is ~1.7G plus ~2.3G of source"
fi
echo "  [info] parallelism: compile $JOBS, link $JOBS_LINK; logical cores $(nproc)"
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  echo "  [warn] ASCEND_HOME_PATH is unset; the compiler you build needs it to find hivmc at runtime"
fi

step "2. Source tree version and state"
if [ -d "$SRC/.git" ]; then
  echo "  HEAD: $(git -C "$SRC" log -1 --oneline 2>/dev/null)"
  echo "  tag : $(git -C "$SRC" describe --tags --always 2>/dev/null)"
  dirty=$(git -C "$SRC" status --porcelain -- bishengir 2>/dev/null | wc -l)
  [ "$dirty" -eq 0 ] || die "bishengir/ already has $dirty modified file(s); check whether the patch is already applied"
else
  echo "  [info] not a git tree (source bundle), skipping the version check"
fi

step "3. Applying the patch (mmadL1 guard + missing explicit instantiation)"
if patch -p1 -d "$SRC" --dry-run --silent < "$PATCH" >/dev/null 2>&1; then
  patch -p1 -d "$SRC" < "$PATCH" >/dev/null || die "failed to apply the patch"
  echo "  [ OK ] both changes applied"
  echo "    guard: $(grep -c 'getNumResults() == 0' "$SRC/bishengir/lib/Conversion/HFusionToHIVM/HFusionToHIVM.cpp") hit(s)"
  echo "    instantiation: $(grep -c 'collapseAndReplace<mlir::tensor::CollapseShapeOp>' "$SRC/bishengir/lib/Dialect/Tensor/Transforms/PropagateReshape/Utils.cpp") hit(s)"
else
  if patch -p1 -d "$SRC" --dry-run -R --silent < "$PATCH" >/dev/null 2>&1; then
    echo "  [info] the patch looks already applied, continuing"
  else
    die "the patch applies neither forward nor in reverse — wrong source version (v1.2.0 required)"
  fi
fi

if [ "$DRY" -eq 1 ]; then
  echo
  echo "dry-run finished (--dry-run, nothing was compiled)"
  exit 0
fi

step "4. third-party patches"
if [ "$SKIP_PATCHES" -eq 1 ]; then
  echo "  skipped (--skip-patches)"
elif [ -d "$SRC/.git" ]; then
  ( cd "$SRC" && bash ./build-tools/apply_patches.sh ) || die "apply_patches.sh failed"
  echo "  [ OK ] third-party patches applied (llvm-project / torch-mlir)"
else
  echo "  [warn] not a git tree, apply_patches.sh needs git; assuming the bundle is already patched"
fi

step "5. Building (~1-2 hours)"
mkdir -p "$OUT"
CC=$(command -v gcc); CXX=$(command -v g++)
LINKER=$(command -v ld.lld 2>/dev/null || command -v lld)
echo "  CC=$CC  CXX=$CXX  LINKER=$LINKER"
( cd "$SRC" && \
  nice -n 10 bash ./build-tools/build.sh \
    -o "$OUT/build" \
    --c-compiler="$CC" --cxx-compiler="$CXX" \
    --add-cmake-options="-DCMAKE_LINKER=$LINKER -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_RTTI=ON" \
    --bishengir-publish=off \
    --jobs="$JOBS" ) 2>&1 | tee "$OUT/build.log"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || die "build.sh failed (rc=$rc), see the tail of $OUT/build.log"

step "6. Assembling the distributable directory"
INSTALL=""
for c in "$OUT/build/install" "$SRC/build/install"; do
  [ -x "$c/bin/bishengir-compile" ] && INSTALL="$c" && break
done
[ -n "$INSTALL" ] || die "no install tree found (bin/bishengir-compile)"
echo "  install: $INSTALL"
mkdir -p "$OUT/pkg/bin" "$OUT/pkg/lib" "$OUT/pkg/shimdir"
cp "$INSTALL/bin/bishengir-compile" "$OUT/pkg/bin/"

# meta_op*.bc / host.bc are not build products: CANN ships them as device-side
# bitcode. Without them a real kernel dies at "External hivmc run fails".
ASCEND_ROOT="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
bc_src=""
for c in "$ASCEND_ROOT/tools/bishengir/lib" "$INSTALL/lib"; do
  if ls "$c"/*.bc >/dev/null 2>&1; then bc_src="$c"; break; fi
done
[ -n "$bc_src" ] || die "no .bc files (meta_op*/host.bc) found. They come from CANN: $ASCEND_ROOT/tools/bishengir/lib"
cp "$bc_src"/*.bc "$OUT/pkg/lib/"
echo "  [ OK ] .bc from $bc_src ($(ls "$OUT/pkg/lib"/*.bc | wc -l) file(s))"
cp "$INSTALL"/lib/*.a "$OUT/pkg/lib/" 2>/dev/null || true
# Bundle a libstdc++, pinning the GLIBCXX dependency to the machine that built it
cp -L "$(g++ -print-file-name=libstdc++.so.6)" "$OUT/pkg/lib/" 2>/dev/null || true

cat > "$OUT/pkg/bin/tl-bishengir-compile" <<'EOF'
#!/bin/bash
# TileLang -> patched bishengir-compile (guard for the HFusionToHIVM OOB read).
# Self-contained: works from env -i, from cron, from any shell.
set -o pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
: "${ASCEND_HOME_PATH:=/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_HOME_PATH
export PATH="$ROOT/shimdir:$ASCEND_HOME_PATH/bin:$PATH"
exec "$ROOT/bin/bishengir-compile" "$@"
EOF

cat > "$OUT/pkg/shimdir/hivmc" <<'EOF'
#!/bin/bash
# Forward to CANN's hivmc, dropping the two flags it rejects when invoked this way.
set -o pipefail
: "${ASCEND_HOME_PATH:=/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_HOME_PATH
args=()
for a in "$@"; do
  case "$a" in --enable-lir-compile*|--enable-cpu-trace-intrinsic*) ;; *) args+=("$a") ;; esac
done
exec "$ASCEND_HOME_PATH/tools/bishengir/bin/hivmc" "${args[@]}"
EOF

chmod +x "$OUT/pkg/bin/tl-bishengir-compile" "$OUT/pkg/shimdir/hivmc"
echo "  [ OK ] $OUT/pkg assembled"

step "7. Acceptance"
bash "$VERIFY" "$OUT/pkg"

cat <<EOF

Done. To use it with TileLang:

  export TILELANG_NPU_COMPILER_PATH=$OUT/pkg/bin/tl-bishengir-compile

Package it for distribution (machines of the same ABI type can share it):

  tar -czf bishengir-patched-\$(ldd --version | head -1 | awk '{print \$NF}').tar.gz -C "$(dirname "$OUT")" "$(basename "$OUT")/pkg"
EOF

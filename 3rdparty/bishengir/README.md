# bishengir-compile (CANN 9.1.0 · A2 / 910B)

To run the Mamba-3 operators in `mamba_ssm/ops/tilelang_ascend/` on Ascend A2, compilation fails
with **`Return code: -11` (SIGSEGV)** — and it fails *intermittently*: the same kernel may compile
on one attempt and crash on the next. The `bishengir-compile` shipped with CANN 9.1.0 has this
defect; this directory provides a fixed build.

The fix does not change code generation: for a given kernel the patched compiler emits a
byte-identical object file, so there is no cache to invalidate.

| Approach | Requirements | Time |
|---|---|---|
| **A. Use the prebuilt package** | aarch64 + glibc ≥ 2.34 + CANN 9.1.0 | 5 minutes |
| **B. Build from source on your machine** | anything else (older glibc, different CANN) | ~35 minutes |

Neither approach modifies `/usr/local/Ascend/**`. TileLang is pointed at the patched compiler with
a single environment variable.

| File | Purpose |
|---|---|
| `bishengir-patched-1.2.0.tar.gz` | the patched compiler (45.8 MB, 187 MB unpacked) |
| `check_target_abi.sh` | run this first: can this machine use the prebuilt package? |
| `verify_patched_bishengir.sh` | acceptance check, after installing or building |
| `build_patched_bishengir.sh` | build a patched copy on this machine |
| `bishengir-v1.2.0-buildable.patch` | the source changes the build needs (applied automatically) |

All scripts locate their siblings through `dirname $0`, so the directory can be moved as a whole.

---

## 1. Pick an approach

```bash
bash ./check_target_abi.sh
```

| Result | Go to |
|---|---|
| all 4 checks pass (aarch64 / glibc ≥ 2.34 / CANN 9.1.0 / required libs) | [A](#2-a-use-the-prebuilt-package) |
| any ✗ | [B](#4-b-build-from-source-on-your-machine) |

---

## 2. A. Use the prebuilt package

```bash
sha256sum bishengir-patched-1.2.0.tar.gz
# must be e169b46c1687d0458fd4c4943f17c9fcc612033638d2ce9c16ec79f53ecf3a7e

tar -xzf bishengir-patched-1.2.0.tar.gz -C /opt/

export TILELANG_NPU_COMPILER_PATH=/opt/bishengir-patched-1.2.0/bin/tl-bishengir-compile
```

- Paths inside the package are self-locating (the wrapper derives its root with `readlink -f $0`),
  so **you can unpack it anywhere**.
- The package was relinked against glibc 2.35, which makes **glibc ≥ 2.34** a hard requirement
  (the stock CANN binaries only need 2.17). If you cannot meet it, use B.
- `libstdc++` is bundled in the package's `lib/`; the system copy is not used.
- The variable is named `TILELANG_*` because the compiler is invoked by the tilelang wheel.

---

## 3. Verify, then compile your operators

```bash
bash ./verify_patched_bishengir.sh /opt/bishengir-patched-1.2.0
```

All four checks must pass: the compiler reports its version; a real cube kernel compiles to an
`.o` with `rc=0`; it does not crash on a minimal IR; and the effective value of
`TILELANG_NPU_COMPILER_PATH` is printed.

**Reference point:** point the stock CANN `bishengir-compile` at the same kernel and you get
`rc=139`. If your check passes *that* case too, you are still using the stock compiler — recheck
what the variable points to.

After that, call the operators as usual; the environment variable takes care of the compiler:

```python
from mamba_ssm.ops.tilelang_ascend.mamba3 import mamba3_mimo_fwd
```

Put the `export` in your environment setup script, not just in the current shell.

---

## 4. B. Build from source on your machine

### 4.1 Get the source

```bash
git clone --depth 1 --branch v1.2.0 --recurse-submodules --shallow-submodules \
    https://gitcode.com/Ascend/AscendNPU-IR.git
```

- The branch must be **`v1.2.0`**: it matches the bishengir 1.2.0 shipped with CANN 9.1.0. The
  patch will not apply to other tags.
- Three submodules are pulled as well: `third-party/llvm-project`, `third-party/torch-mlir`,
  `third-party/shmem`. **This is the step that usually blocks** — the first two live on GitHub;
  switch to a mirror or use offline bundles if the network cannot reach them.
  Keep `--depth 1 --shallow-submodules` (the full history is 20 GB).
- About 2.3 GB of source, plus roughly 1.7 GB for the build directory.
- Dependencies: `cmake`, `ninja`, `g++`, `lld`, `python3`, `git`.

### 4.2 Build

```bash
bash ./build_patched_bishengir.sh \
    --src /path/to/AscendNPU-IR \
    --out /opt/bishengir-patched \
    --jobs 32
```

The script runs, in order: preflight checks → version and dirty-tree checks → apply the patch →
`apply_patches.sh` → `build.sh` → assemble `pkg/` (binaries + `.bc` + wrapper + `libstdc++`) →
run the acceptance check. `--dry-run` stops after patching, so you can see what it would change.

The result is used exactly like [A](#2-a-use-the-prebuilt-package).

### 4.3 Two things that will block you (both measured)

**① `v1.2.0` does not build as-is.** Linking fails with:

```
ld.lld: error: undefined symbol:
  void mlir::tensor::reshape_utils::collapseAndReplace<mlir::tensor::CollapseShapeOp>(...)
```

The template is defined in `Utils.cpp` and the header does not provide implicit instantiation,
while the source only instantiates the `ExpandShapeOp` form — but `PropagateExpandUp.cpp` uses the
`CollapseShapeOp` one. One explicit instantiation fixes it (the patch in this directory includes
it); with that, the build reaches `[4157/4157]` in one go.

**② `meta_op*.bc` / `host.bc` are not build products** — they must be copied from CANN into the
package's `lib/`:

```
error: External hivmc run fails, returning module before running external compiler
```

These 14 device-side bitcode files ship with CANN and are tied to the `hivmc` version. The script
copies them from the target machine's CANN automatically; if you do it by hand, do not skip them.

---

## 5. Three rules you must follow

| # | Rule | Symptom when violated |
|---|---|---|
| 1 | `TILELANG_NPU_COMPILER_PATH` must be a **file**, not a directory | `FileNotFoundError: ... is not a file` |
| 2 | It must point at the `tl-bishengir-compile` **wrapper**, not the bare `bishengir-compile` | `hivmc: unknown argument '--enable-lir-compile=...'` |
| 3 | Do **not** set `TILELANG_NPU_OPT_PATH` (let `bishengir-opt` stay on CANN's copy) | version mismatch, wrong output |

Rollback: `unset TILELANG_NPU_COMPILER_PATH` and you are back on the stock compiler immediately.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| still `Return code: -11` | the variable did not take effect, or points elsewhere | `echo $TILELANG_NPU_COMPILER_PATH`; use an absolute path |
| `FileNotFoundError ... is not a file` | a directory was passed | point at `.../bin/tl-bishengir-compile` |
| `hivmc: unknown argument '--enable-lir-compile=...'` | pointing at the bare binary | use the wrapper |
| wrong numerical results | `TILELANG_NPU_OPT_PATH` is set | unset it |
| `ld.lld: undefined symbol ... collapseAndReplace` | the source lacks the second hunk | use `bishengir-v1.2.0-buildable.patch` from this directory |
| `External hivmc run fails` | `.bc` files missing from the package | copy CANN's 14 `.bc` files into `lib/` |
| `verify` exits with 2 | no directory argument | pass the directory that contains `bin/` |

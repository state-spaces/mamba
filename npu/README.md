# npu/ — selective_scan（Mamba v1 S6）AscendC 前向/反向算子

## TL;DR (English)

AscendC (Ascend 910B3 / DAV_2201) re-implementation of Mamba v1's S6 `selective_scan`
forward **and** backward kernels, packaged as a standalone PyTorch extension wheel
(`cann_bench`), registered as `torch.ops.cann_bench.selective_scan_fwd` /
`torch.ops.cann_bench.selective_scan_bwd`. A drop-in shim redirects
`mamba_ssm`'s `selective_scan_cuda` to these NPU ops so Mamba-1 trains/runs on NPU.

```bash
cd npu && bash build.sh && pip install dist/cann_bench-*.whl
```
```python
import torch_npu                       # explicit NPU backend
import npu_shim_selective_scan         # import BEFORE mamba_ssm (triton polyfill)
npu_shim_selective_scan.install_npu_shim()
from mamba_ssm import Mamba            # then use Mamba-1 as usual on NPU
```

See the quickstart example, verified results and platform caveats below (Chinese).

---

## 1. 算子简介

本目录将 Mamba v1（S6 选择性扫描）的**前向 `selective_scan_fwd`** 与**反向
`selective_scan_bwd`** 两个算子从 CUDA 迁移为 AscendC 实现（bisheng 编译、
kernel 直调范式，`<<<>>>` 降级 eager-launch 链路），经 torch.library 注册为
PyTorch 自定义算子，打包为独立 wheel（包名沿用 `cann_bench`）：

```text
torch.ops.cann_bench.selective_scan_fwd(Tensor u, Tensor delta, Tensor A, Tensor B, Tensor C,
        Tensor? D=None, Tensor? z=None, Tensor? delta_bias=None, bool delta_softplus=False)
    -> (Tensor out, Tensor x, Tensor out_z)

torch.ops.cann_bench.selective_scan_bwd(Tensor u, Tensor delta, Tensor A, Tensor B, Tensor C, Tensor dout,
        Tensor? D=None, Tensor? z=None, Tensor? delta_bias=None, Tensor? x=None,
        Tensor? out=None, Tensor? dz=None,
        bool delta_softplus=False, bool recompute_out_z=False)
    -> (Tensor du, Tensor ddelta, Tensor dA, Tensor dB, Tensor dC,
        Tensor dD, Tensor ddelta_bias, Tensor dz, Tensor out_z)
```

- **接口语义**与 mamba-ssm CUDA pybind（`selective_scan_cuda.fwd/bwd`）逐位对齐；
  细节差异：schema 返回类型静态，fwd 在 `z=None` 时 `out_z` 返回 shape `{0}` 空
  tensor（CUDA pybind 为动态 2/3 元返回，语义等价）；bwd 恒返 9 元组（缺省槽位为
  空 tensor），由 shim 按 CUDA 语义裁剪。
- **x 布局契约**：fwd 第二个输出 `x` 为 fp32 复合前缀，shape `(b, dim, n_chunks,
  dstate)`，chunk 宽度固定 2048（时间维每 2048 点存一个前缀快照，(a,h) 交错复合）。
  反向时上游须把**同一 fwd 产出的 x 原样透传**给 bwd（bwd 不读 x 内容、仅作契约
  载体，验证明：注入 NaN 垃圾 x 后 9 输出与正常调用 bit 级一致），**严禁伪造**。
- 入参约束（plugin 校验兜底）：`dstate ∈ [1,256]`；u/delta `(b, d, l)`、A
  `(d, dstate)`、静态 B/C `(b, dstate, l)`、变量 B/C `(b, ng, dstate, l)`
  （ng 整除 d）、D/delta_bias `(d,)`、z `(b, d, l)`；全部要求连续（shim 已兜底
  `.contiguous()`）。
- 支撑 dtype：fp32 / fp16 / bf16（归约类输出 dA/dB(静态)/dC(静态)/dD/ddelta_bias
  恒 fp32）。

## 2. 软硬件要求

| 项目 | 要求 | 备注 |
|------|------|------|
| NPU | Ascend 910B3（SoC `Ascend910B3`，NpuArch 2201 / DAV_2201） | 全部验证在该平台完成；`--soc` 亦支持 `ascend910_93`/`ascend950` 编译分支，但**未经验证** |
| CANN | 9.1.0（bisheng 编译器 + ascendc 头文件/库） | `ASCEND_HOME_PATH` 已配置或安装于 `/usr/local/Ascend/cann-*` |
| torch / torch_npu | torch 2.10.0（CPU 版即可）+ torch_npu 2.10.0.post4 | 验证环境 Python 3.12.13 |
| mamba-ssm | **2.3.2.post1**（已验证版本） | 集成验证与源仓 clone 均为该版本；shim patch 点（见 §4）已在本仓源码逐一核对存在 |
| 其它 | cmake ≥ 3.16、python3（含 setuptools/wheel）、g++（AArch64） | kernel 用 bisheng 编译，plugin/extension 用 g++ |

## 3. 构建与安装

```bash
cd npu/
bash build.sh                  # 自动探测 SoC；也可 bash build.sh --soc=ascend910b
pip install dist/cann_bench-*.whl --no-deps
```

- `build.sh` 在 `npu/` 目录内自包含：cmake 驱动 bisheng 编译 2 个 kernel +
  g++ 编译 plugin/extension，产物合并为 `_C.abi3.so`，再经 `setup.py` 打成
  `dist/cann_bench-1.0.0-cp38-abi3-linux_aarch64.whl`。加 `--install` 可构建后自动安装。
- **注册名**：`torch.ops.cann_bench.selective_scan_fwd` / `torch.ops.cann_bench.selective_scan_bwd`
  （`import cann_bench` 即触发注册；wheel 提供 `cann_bench.selective_scan_fwd/bwd` 两个便捷封装函数）。
- 构建输出目录：`npu/build/`、`npu/dist/` 与 `cann_bench/_C.abi3.so` 均为本地产物，
  **建议加入 .gitignore**（mamba 仓根 .gitignore 已覆盖 `build/` 与 `**.so`，
  请再补一行 `npu/dist/`）。
- 以上命令已在干净副本（`cp -r npu /tmp/npu_stage && cd /tmp/npu_stage && bash build.sh`）
  从零构建、安装并跑通 §5 示例验证。

## 4. 接入 mamba_ssm（shim）

`npu_shim_selective_scan.py` 是 mamba-ssm 的 NPU 接入 shim：运行时把
`mamba_ssm.ops.selective_scan_interface.selective_scan_cuda` 模块属性替换为适配
对象，其 `fwd/bwd` 直映射 `torch.ops.cann_bench.selective_scan_fwd/bwd`（参数序、
返回元数严格对齐 CUDA pybind 动态行为，内部统一 `.contiguous()` 兜底）。因
mamba_simple 等处在 import 期按名绑定，**shim 须在 `import mamba_ssm` 之后、模型
构建之前挂载**；同时本模块 import 时会先注入 triton 兼容 polyfill（triton 3.2.0
无 `set_allocator`，不补则 `import mamba_ssm` 直接失败），故**必须在 mamba_ssm
之前 import 本模块**。

```python
import os
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import torch_npu                       # 显式引入 NPU 后端
import npu_shim_selective_scan         # ① 先于 mamba_ssm（triton polyfill）
import mamba_ssm.ops.selective_scan_interface as ssi
from mamba_ssm import Mamba            # ② 再 import mamba_ssm
npu_shim_selective_scan.install_npu_shim()   # ③ 挂载 shim（selective_scan_cuda → NPU 算子）

model = Mamba(d_model=..., d_state=16, d_conv=4, expand=2).to("npu")  # 之后照常使用
```

两个可选开关（按需调用）：

- `npu_shim_selective_scan.disable_triton_selective_state_update()`：triton_ascend
  编译不可用时，令 `Mamba.step()`（decode 逐 token 生成）回退纯 torch 参考步进。
- `npu_shim_selective_scan.enable_triton_cuda_ctx_compat()`：mamba_ssm 各 triton
  kernel 内部以 `with torch.cuda.device(...)` 包裹启动，CPU-torch + torch_npu 环境
  会直接抛错；本调用将其 patch 为 `torch.npu.device` 等价上下文（triton 路径在 NPU
  可用时使用）。

诊断开关（环境变量，问题定位用，默认全关）：

| 环境变量 | 作用 |
|----------|------|
| `NPU_SHIM_TRACE=1` | 打印每次 fwd/bwd 的调用序号与各输入 shape/dtype |
| `NPU_SHIM_SYNC=1` | 每次 fwd/bwd 后强制 device 同步（异步 aivec 异常即时定位） |
| `NPU_SHIM_DUMP=<dir>` | 每次 fwd/bwd 的输入 CPU 副本落盘 `<dir>/{fwd,bwd}{n}.pt` |

> 兼容性说明：已验证 mamba-ssm **2.3.2.post1**——patch 点
> `mamba_ssm/ops/selective_scan_interface.py` 的模块级 `selective_scan_cuda` 名
> （`SelectiveScanFn.forward/backward` 调用期按全局名解析）与
> `mamba_ssm/modules/mamba_simple.py` 的 `selective_state_update` 均在该版本源码
> 中核对存在。其余版本未验证；若 `selective_scan_cuda.bwd` 的调用参数序发生变化
> （当前为 `bwd(u, delta, A, B, C, D, z, delta_bias, dout, x, out, dz,
> delta_softplus, recompute_out_z)`），shim 需同步适配。

## 5. 快速示例与完整用法

本节先给出内置最小示例（5.1），再原样收录上游 README 的 Usage（Mamba-1 相关）与
Examples 生成命令（5.2–5.4）并附 NPU 适配要点。文中「引①/④/⑦」分别指 §1（算子
简介）/ §4（接入 mamba_ssm）/ §7（已知限制与平台注意事项）。

### 5.1 快速示例（quickstart_npu.py）

```bash
# 在 mamba 仓根目录执行（保证 import mamba_ssm 可用）
python npu/examples/quickstart_npu.py [--device 0]
```

`npu/examples/quickstart_npu.py`（约 100 行，固定种子、可选卡）：小 shape
(b=2, d=16, l=64, dstate=16, fp32, 含 D/z/delta_bias/delta_softplus 全通路) →
挂 shim → `selective_scan_fn` 前向（触发 NPU fwd）→ autograd 反向（触发 NPU
bwd）→ 与 CPU fp64 参考逐项对比 9 个输出/梯度的 max_err → 打印 PASS/FAIL。

本仓实测（2026-10-20，干净副本构建产物）：

```text
shape(b,d,l;ds)=(2,16,64;16)  dtype=fp32  device=npu:0
  out          max_err=2.861e-06   du      max_err=3.815e-06   ddelta  max_err=1.907e-06
  dA           max_err=2.384e-07   dB      max_err=1.907e-06   dC      max_err=1.907e-06
  dD           max_err=1.431e-06   dz      max_err=3.815e-06   ddelta_bias max_err=3.815e-06
shim 调用计数: fwd=1, bwd=1        RESULT: PASS
```

配套的 mamba_ssm 集成检查（`Mamba(d_model=16, d_state=16, d_conv=4, expand=2)`
模块前向 + 反向，挂 shim）同环境 PASS：输出有限、9/9 参数梯度均有限、
shim 计数 fwd=1 bwd=1。

### 5.2 Selective SSM（上游 Usage）

**上游原文**（摘自上游 README @ e9594ce · Usage → Selective SSM，原样收录）：

> Mamba is based on a selective SSM layer, which is the focus of the paper (Section 3; Algorithm 2).
>
> Source: [ops/selective_scan_interface.py](mamba_ssm/ops/selective_scan_interface.py).

该小节指向的 `mamba_ssm/ops/selective_scan_interface.py` 中 `selective_scan_fn` 即
selective SSM 层的训练入口，在本包内经 shim 直达 NPU 算子。

> **NPU 使用要点**
>
> - **设备替换**：上游面向 CUDA，NPU 上将张量/模块的 `.to("cuda")` 改为 `.to("npu")`。
> - **先挂 shim**：调用 `selective_scan_fn` 前须按 §4「接入 mamba_ssm（shim）」的挂载
>   代码块完成挂载（import 顺序：`torch_npu` → 本 shim 模块 → `mamba_ssm` →
>   `install_npu_shim()`），此处不重复全文（引④，§4）。
> - **dtype**：支持 fp32 / fp16 / bf16（引①，§1；归约类输出
>   dA/dB(静态)/dC(静态)/dD/ddelta_bias 恒 fp32）。
> - **x 布局契约与约束**：fwd 第二输出 `x` 为 fp32 复合前缀（shape
>   `(b, dim, n_chunks, dstate)`、chunk 宽 2048，反向须把同一 fwd 产出的 x 原样
>   透传、严禁伪造），各入参 shape/连续性要求一并见引①（§1 算子简介）。

### 5.3 Mamba Block（上游 Usage）

**上游原文**（摘自上游 README @ e9594ce · Usage → Mamba Block，原样收录）：

> The main module of this repository is the Mamba architecture block wrapping the selective SSM.
>
> Source: [modules/mamba_simple.py](mamba_ssm/modules/mamba_simple.py).
>
> Usage:

``` python
import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
```

> **NPU 使用要点**
>
> - **设备替换与 shim**：同 5.2——`.to("cuda")` → `.to("npu")`，模型构建前先挂
>   shim（引④，§4）。
> - **fast path 不可用、自动走 ref 路径**：`mamba_simple.Mamba.forward` 的 fast
>   path（`mamba_inner_fn` 融合前向）要求 `causal_conv1d_fn is not None`；本包
>   csrc 仅含 `selective_scan` fwd/bwd 两算子、未迁移 causal_conv1d（见 §8 文件
>   清单），该条件恒不成立，故 NPU 包下 fast path 不可用，前向自动走 ref 路径
>   （`in_proj` → torch 原生 conv1d → `selective_scan_fn`（经 shim 触发 NPU fwd）→
>   `out_proj`）。§5.1 集成检查实测的正是该路径（shim 计数 fwd=1、bwd=1）。
> - **dtype**：同 5.2（fp32 / fp16 / bf16，引①，§1）。

### 5.4 生成示例（上游 Examples，Mamba-1 相关）

**上游原文**（摘自上游 README @ e9594ce · Inference → Examples，命令原样收录；
脚本为 `benchmarks/benchmark_generation_mamba_simple.py`）：

> To test generation latency (e.g. batch size = 1) with different sampling strategies:

``` sh
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-2.8b" --prompt "My cat wrote all this CUDA code for a new language model and" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
python benchmarks/benchmark_generation_mamba_simple.py --model-name "EleutherAI/pythia-2.8b" --prompt "My cat wrote all this CUDA code for a new language model and" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-2.8b" --prompt "My cat wrote all this CUDA code for a new language model and" --minp 0.05 --topk 0 --temperature 0.7 --repetition-penalty 1.2
```

> To test generation throughput with random prompts (e.g. large batch size):

``` sh
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-2.8b" --batch 64
python benchmarks/benchmark_generation_mamba_simple.py --model-name "EleutherAI/pythia-2.8b" --batch 64
```

> With Mamba-2, you just need to change the model name:

``` sh
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba2-2.7b" --prompt "My cat wrote all this CUDA code for a new language model and" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
```

**NPU 适配要点**（上游原命令在 NPU 上跑通所需的最小适配，按条对照）：

> - **① 脚本设备硬编码 → 直接使用随包适配版脚本（无需手工适配）**：上游脚本面向
>   CUDA 编写——`device = "cuda"`（L31），`from_pretrained` 的 `device=device` /
>   `device_map={"": device}`（L38/L41，经该变量），随机 `input_ids`/`attn_mask`
>   字面 `device="cuda"`（L47/48）；另有 `torch.cuda.synchronize()`、decode 计时
>   `torch.cuda.Event`、`cg=True`（CUDA Graph）等 CUDA 专有 API。本包随附上游脚本
>   的 NPU 适配版 **`npu/examples/benchmark_generation_npu.py`**（CLI 参数与上游
>   完全一致、模型加载逻辑未改；内置 shim 挂载 + triton polyfill，适配点：device=npu、
>   `torch.npu.synchronize()`/`torch.npu.Event`、`cg=False`、fused_add_norm 关闭 +
>   RMSNorm 替换为 torch 原生层），将上文命令中的脚本路径替换即可，mamba-2.8b
>   吞吐示例：
>
>   ``` sh
>   ASCEND_LAUNCH_BLOCKING=1 HF_ENDPOINT=https://hf-mirror.com \
>       python npu/examples/benchmark_generation_npu.py --model-name "state-spaces/mamba-2.8b" --batch 32
>   ```
>
>   mamba-2.8b topp 采样延迟示例（上游 cmd1 命令的随包脚本版，对应下方实测表 cmd1'）：
>
>   ``` sh
>   ASCEND_LAUNCH_BLOCKING=1 HF_ENDPOINT=https://hf-mirror.com \
>       python npu/examples/benchmark_generation_npu.py --model-name "state-spaces/mamba-2.8b" \
>       --prompt "My cat wrote all this CUDA code for a new language model and" \
>       --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
>   ```
>
>   环境变量（`ASCEND_LAUNCH_BLOCKING=1` 必加、`HF_ENDPOINT` 下载镜像、
>   `ASCEND_RT_VISIBLE_DEVICES` 选卡）与 batch 提示（共享小卡 64 会 OOM，用 32/16）
>   见该脚本文件头注释；脚本开头会打印 `mamba_ssm` 实际解析路径与 NPU 算子注册
>   状态（透明化核对，引⑦）。适配过程 diff 留档见
>   《mamba仓NPU跑测问题与解决方案.md》§6.4（过程记录来源，非使用必需）。
> - **② mamba-2.8b 真实域须加 `ASCEND_LAUNCH_BLOCKING=1`**：dav-2201 平台 MTE 缺陷
>   在默认异步提交 / OpExecCache 重放路径下概率性复发 aivec 507035（引⑦，§7 第 1
>   条），mamba 系跑测命令统一追加该环境变量强制逐 kernel 同步提交；小模型
>   （130m 级）默认路径实测稳定。
> - **③ 下载走镜像**：直连 huggingface.co 超时的环境统一
>   `HF_ENDPOINT=https://hf-mirror.com`（下载与 transformers 运行时均需要；
>   mamba-2.8b 权重 ~22 GB、pythia-2.8b ~11 GB）。
> - **④ batch 64 资源不足（与余量相关，已复测澄清）**：~7 GB 共享卡余量下 mamba-2.8b
>   `--batch 64` 加载/构造期即 OOM，实测降 `--batch 32` 可跑（`--batch 16` 亦可）；
>   pythia-2.8b `--batch 64` 报 207001（attention workspace 超余量），实测 `--batch 8`
>   可跑。2026-10-20 复测（卡余量 ~62 GB）：mamba-2.8b `--batch 64` 随包脚本实测可跑
>   （46892 ms，EXIT=0）——OOM 与否取决于共享卡当时余量，非算子缺陷；共享环境建议
>   先 `npu-smi info` 看余量选卡（`ASCEND_RT_VISIBLE_DEVICES`）再定 batch。
> - **「With Mamba-2」范围标注**：Mamba-2（如 mamba2-2.7b）用 SSD 算法，prefill 走
>   triton `mamba_chunk_scan_combined`、decode 走 triton `selective_state_update`，
>   **全程不经过本包 selective_scan 算子**；本包适用边界为 Mamba-1 系（引⑦，§7
>   范围边界条）。

**预期输出与判定**（随包脚本 `--batch 32` 干净登录 shell 复跑样例，2026-10-20；
开启 `NPU_SHIM_TRACE=1` 时还会逐次打印 fwd/bwd 调用行，此处不摘，需要时可用该
开关自行核对，引④）：

```text
[npu-bench] mamba_ssm resolved: /home/z30066236/mymamba/mamba/mamba_ssm/__init__.py
[npu-bench] mamba_ssm version: 2.3.2.post1
[npu-bench] torch.ops.cann_bench.selective_scan_fwd registered: True
Loading model state-spaces/mamba-2.8b
Number of parameters: 2768345600
Prompt length: 100, generation length: 100
state-spaces/mamba-2.8b prompt processing + decoding time: 31005ms
```

> 判定要点：
>
> - 退出码 0，生成文本连贯（无乱码/复读；topp 采样运行可直读生成文本）。
> - `registered: False` 时优先检查 cann_bench wheel 是否已安装（§3 构建与安装）。
> - `resolved` 显示仓内路径属预期：CANN `set_env.sh` 导出的 PYTHONPATH 尾随空条目
>   使 Python 将空条目解释为 cwd，从仓根运行时本地源码遮蔽 site-packages，两处
>   内容一致、行为等价（引⑦，§7 第 7 条；问题记录 #25）。
> - prefill 每层走本包 NPU 算子（b32 为 64 层 × 4 轮 = 256 次 fwd），加
>   `NPU_SHIM_TRACE=1` 可逐次核对调用序号与各输入 shape/dtype（引④，§4 诊断开关）。

**NPU 实测参考**（端到端跑通参照，**非性能基线**）：

| 命令 | 模型 | 关键数字 | 说明 |
|------|------|----------|------|
| cmd1（topp 延迟） | mamba-2.8b | **15044 ms**（prompt 14 + gen 100，b1） | prefill 走本包 NPU fwd（64 次/run，全部 PASS），decode 走 triton `selective_state_update` 主导数字；含 `ASCEND_LAUNCH_BLOCKING=1` 同步开销 |
| cmd1'（随包脚本） | mamba-2.8b | **14718 ms**（prompt 14 + gen 100，b1；命令见上方 §5.4 ① topp 采样延迟示例块） | 随包适配版脚本验证（2026-10-20，同 cmd1 口径非性能基线）；生成文本连贯（无乱码/复读）；EXIT=0、256 次 fwd 全 PASS |
| cmd2（topp 对照） | pythia-2.8b | 4936 ms | 纯 transformers，与 SSM 算子无关；与 cmd1 不可直接作性能对比 |
| cmd3（minp 变体） | mamba-2.8b | 15823 ms | 同 cmd1 口径 |
| cmd4（吞吐） | mamba-2.8b | `--batch 32` **31043 ms**（≈103 tok/s） | batch 64 OOM 降 batch 绕行；`--batch 16` 23068 ms 亦 PASS |
| cmd4'（随包脚本） | mamba-2.8b | `npu/examples/benchmark_generation_npu.py --batch 32` **30353 ms**；`--batch 64` **46892 ms** 亦 PASS（余量 ~62 GB） | 随包适配版脚本验证（2026-10-20，与 cmd4 同口径非性能基线）；EXIT=0、prefill 64 次/run（4 轮共 256 次 fwd 全 PASS）、无 error/aivec |

> 表中 decode 由 triton `selective_state_update` 主导、且统一含
> `ASCEND_LAUNCH_BLOCKING=1` 拖慢，**不代表 `selective_scan_fwd` 算子性能**（算子级
> 权威性能见 §6），仅证明 README 原命令在 NPU 端到端跑通且生成文本连贯。来源：
> 《mamba仓NPU跑测问题与解决方案.md》第六章；cmd4' 行来源同文档问题 25。

## 6. 验证结论汇总

> 以下为算子开发期的权威验收结论（QA 独立验收，模式 B 自建 golden；来源报告名见
> 各行标注）。性能为 **30 轮中位数、warmup 5、固定输入**，采集平台 Ascend 910B3 +
> CANN 9.1.0 + torch_npu 2.10.0.post4；验收口径为下限口径（功能正确 + 无明显实现
> 缺陷），量化指标仅作基线留档。

### 前向 selective_scan_fwd

| 项目 | 结论 | 来源 |
|------|------|------|
| 功能用例 | **113/113 PASS**（L0 18 + L1 36 + L2 26 + 白盒 33；计数三处一致） | 《CP3-功能验收报告.md》（fwd，v1.0 2026-10-10） |
| 精度 | 权威口径（逐元素混合容差 AND matched_ratio≥0.99，k 按 seqlen 分档）逐输出达标；最差实测：out fp16 长序列 L1-T-02 (8,2048,8192) mr=0.99994、L1-T-08 (l=16384) mr=0.99947；x 输出 l=32768 max_err=8.392e-05；QA 独立探针 10 条抽样复算全过 | 同上 |
| 白盒覆盖 | 声明分支点 90 个，非防御 86/86 全覆盖，防御类 4 个附不可达依据 | 同上 |
| 性能（30 轮中位数） | T-02 (8,2048,8192;1,16) fp16 **119.75 ms**（1121 Mel/s）；T-08 (l=16384) 239.40 ms；T-03 bf16 60.36 ms；T-05 静态 BC fp32 54.69 ms；T-06 bf16 29.68 ms；跨 chunk 线性（T-08/T-02≈2.0） | 《CP4-性能验收报告.md》（fwd，2026-10-10；2026-10-21 性能迭代后基线） |
| 瓶颈归因 | 标量 pipe 主导（64.7~65.5%、逐核 99.6% 饱和），为逐状态扫描范式的结构固有成本；无未流水/重复搬运/冗余重算缺陷；UB 搬运已双缓冲 | 同上 |

### 反向 selective_scan_bwd

| 项目 | 结论 | 来源 |
|------|------|------|
| 功能用例 | **135/135 PASS**（L0 20 + L1 37 + L2 43 + 白盒 35；QA 侧独立聚合核对） | 《CP3-验收报告.md》（bwd，v1.0 2026-10-15） |
| 精度 | 双基准 golden（fp64 autograd 主判 + 手写逆推 fp64 互检）+ 权威口径，逐输出 mr 最差 0.99902 ≥ 0.99（dB@L1-BW-P-23）；QA 独立探针 18 条（9 元组契约 + SVW 豁免门控无越权 + 尾块 pad 中性 + x 契约不读）复验 | 同上 |
| 白盒覆盖 | WB 35/35（tile 边界/多核分支/plugin 校验增量/A 域三档） | 同上 |
| 性能（30 轮中位数） | T-02 (4,2048,8192;1,16) fp16 **153.05 ms**；T-08 (l=16384) 305.57 ms；T-03 bf16 77.47 ms（逐轮同步口径）；T-05 fp32 139.88 ms；T-06 bf16 37.92 ms；bwd/fwd 每 pair-chunk 2.68~2.72 @S=16/32（需求锚带 2~3× 内） | 《CP4-验收报告.md》（bwd，2026-10-16；2026-10-21 性能迭代后基线） |
| 瓶颈归因 | 标量 pipe 62.4~65.3%（落预判带 60~70%）；原子加跨核归约 PUNCTURE-PASS 无乱序发散 | 同上 |

> 开发期 state.json 曾记录 bwd 的 `pending_retest`（BW-12 逆扫描 seed 根因 E/F 修复
> 后待回归）——**已闭环**：隔离矩阵 l=2048/2049/3072 全绿、BW-12 复测 PASS（bwd
> 《7.1-开发报告.md》修复清单 #6），CP3/CP4 报告终稿均 PASS，CP5 报告确认「无悬置
> pending」。当前未闭环项见 §7 限制节（均为非算子缺陷的流程/观察项）。

> **2026-10-21 性能迭代**：标量维依赖链直线化，fwd/bwd 五用例 −34.8~−46.4%，标量
> 占比 64~65%→19.3%/13.8~17.7%。上表性能数字已同步更新为迭代后新基线，迭代前
> 基线保留于《CP4-性能验收报告.md》原文（历史留痕）。

## 7. 已知限制与平台注意事项

1. **dav-2201 平台 MTE 缺陷与 `ASCEND_LAUNCH_BLOCKING=1`（重要）**：910B3 存在
   平台级 MTE 地址组合缺陷——kernel 以任何引擎访问 `delta_bias` 的 GM 原地址可能
   触发确定性 aivec error 507035（地址组合依赖，与数据值/指令流无关，mssanitizer
   证实无真实越界）。算子内已按方案 V 绕行（plugin 侧 V 引擎生成 bias 安全副本
   `bias_safe = delta_bias * 1.0f`，且为 launch 期首个分配），全部回归通过。
   **残余风险**：大模型真实域（如 mamba-2.8b，64 层 × d_inner=5120）在默认异步
   批量提交 / OpExecCache 重放路径下，绕行前提（分配序列）可能被破坏而概率性复发
   （小模型 130m 级默认路径实测稳定）。规避：mamba 系跑测命令统一追加
   `ASCEND_LAUNCH_BLOCKING=1`（强制逐 kernel 同步提交）；代价是每算子 host 同步，
   该口径下性能数字不可作基线。若 torch/allocator/CANN 升级变更分配或提交行为，
   建议先用小模型 + 大模型各回归一轮。该平台缺陷本体建议按昇腾流程上报
   （来源：《mamba仓NPU跑测问题与解决方案.md》问题 6/13）。
2. **禁改常量 TILE_T=1024**：dav-2201 平台 vgather count>1024 间歇性数据损坏
   （正确性级）。fwd/bwd kernel 的 `TILE_T=1024`（及任何使单次 Gather count 超过
   1024 的改动）**禁止回大或调整**（来源：问题记录 R6 / fwd《CP4-性能验收报告.md》
   已证伪方向表）。
3. **事件池预算（HANG-01 教训）**：dav-2201 上 V_MTE2 事件池上限
   `QUE_MAX_EVENT=8`，fwd 已收敛为 4 条 VECIN 队列（z 走 TBuf+手工事件），bwd 为
   4 条 VECIN×深度 2=8 恰满。**后续修改禁止再增加 TQue 队列**，否则向量核确定性
   挂死（fwd P-24 曾 551s 挂死，已修复为 1.54s PASS）。
4. **bwd 未闭环遗留项（如实声明，均非算子精度缺陷）**：
   - L-1：mock 校准证据链的执行合规缺口（gate 字段未刷新、12 条 over_1_ratio>0
     条目的逐元素归因落盘缺失、reduce 类 max_util 0.317 超 0.25 门禁线）——归属
     测试方案 2.1 的制度执行项；主判 135/135 全绿、超容差元素占比 ≤0.08%（距 1%
     配额线 >12× 余量），实测已证伪配额溢出风险（bwd《CP3-验收报告.md》）。
   - L-2：fp16/bf16 归约类输出的环境瞬态敏感性观察项（QA11/QA12 单次运行 FAIL
     不可复现，3 次独立复跑 + 9 条同构对照全绿，判归环境瞬态；建议后续回归保持
     观察）。
   - L-Q1：性能定性检查③（跨核归约串行化）与⑤（核间负载均衡）缺直接量化条目
     （采集框架无 per-core 维度），以间接证据链定性通过（bwd《CP4-验收报告.md》）。
5. **范围边界**：本包仅覆盖 Mamba-1 的 `selective_scan` fwd/bwd（real 变体）。
   complex 变体、`selective_state_update`（decode triton kernel）、Mamba-2 的 SSD
   路径与 causal_conv1d 均未迁移（后者缺失时 mamba_ssm 自动回退 ref conv 路径）。
   shim 未提供的 `selective_scan_cuda` 属性访问会抛出明确错误。
6. **共享环境资源**：大 shape 用例（b×d×l×dstate×4B >1GiB）在共享卡 HBM 余量紧张
   时可能 OOM，建议独占或选空闲卡（`ASCEND_RT_VISIBLE_DEVICES`）。
7. **禁止在本环境 `pip install` 本仓根的 mamba_ssm 源码（裸 import 全断风险）**：
   本仓 clone 的上游源码（setup.py 版本号仍为 2.3.2.post1）含 Mamba-3 代码路径——
   `mamba_ssm/__init__.py` L6 无条件 `from ...mamba3 import Mamba3`，其 triton
   kernel 模块（`mamba3_siso_fwd.py` L457）在 **import 期**调用 `triton.set_allocator`；
   本环境 triton 3.2.0（triton_ascend 系）无该 API（triton ≥3.3 才引入），任何未加
   polyfill 的 `import mamba_ssm` 都会 `AttributeError` 直接失败（官方 v2.3.2.post1
   tag 同样含该调用，已核实）。**已验证基线为 site-packages 安装的 mamba-ssm
   2.3.2.post1**（本仓集成验证、§5.1 quickstart 与 §5.4 Example 均基于此，shim
   模块 import 时先注入 triton 兼容 polyfill，按 §4 顺序接入即可正常使用）。
   `npu/examples/benchmark_generation_npu.py` 开头会打印 `mamba_ssm` 实际解析
   路径供核对：CANN `set_env.sh` 导出的 PYTHONPATH 以冒号结尾（尾随空条目），
   Python 将空条目解释为 cwd，故从仓根运行时本地源码会遮蔽 site-packages——
   两者内容一致、行为等价，但若本地源码被改动（如 pip 重装过其它版本）则不再
   等价，务必以打印结果为准。详见问题记录 #25。

## 8. 文件清单

```text
npu/
├── README.md                                  # 本文件
├── build.sh                                   # 构建入口（SoC 探测 + cmake/bisheng + wheel 打包）
├── setup.py                                   # wheel 打包（cann_bench 包，cp38-abi3 标签）
├── CMakeLists.txt                             # 顶层 cmake 工程（bisheng kernel + g++ plugin + 链接）
├── cmake/                                     # cmake 模块（ascend/func/python/torch/torch_npu 发现）
│   ├── ascend.cmake
│   ├── func.cmake                             # register_direct_launch_op 算子自注册宏
│   ├── python.cmake
│   ├── torch.cmake
│   └── torch_npu.cmake
├── cann_bench/
│   └── __init__.py                            # 包入口：import 即注册；fwd/bwd 便捷封装
├── csrc/
│   ├── extension.cpp                          # Python 扩展入口（触发 TORCH_LIBRARY 静态注册）
│   └── ops/
│       ├── CMakeLists.txt                     # 算子子目录聚合
│       ├── selective_scan/                    # 前向算子
│       │   ├── CMakeLists.txt
│       │   ├── op_kernel/
│       │   │   ├── selective_scan_kernel.cpp  # fwd kernel + tiling（Hillis-Steele 扫描）
│       │   │   └── selective_scan_launch.h    # tiling 结构/标志位/launch 声明
│       │   └── op_plugin/
│       │       └── selective_scan_plugin.cpp  # schema 注册 + Meta + NPU impl + launch
│       └── selective_scan_bwd/                # 反向算子
│           ├── CMakeLists.txt
│           ├── op_kernel/
│           │   ├── selective_scan_bwd_kernel.cpp  # bwd kernel（后缀扫描 + 原子加归约）
│           │   └── selective_scan_bwd_launch.h
│           └── op_plugin/
│               └── selective_scan_bwd_plugin.cpp # schema 注册 + Meta + NPU impl + launch
├── npu_shim_selective_scan.py                 # mamba_ssm 接入 shim（patch + polyfill + 诊断开关）
└── examples/
    ├── quickstart_npu.py                      # 最小示例：fwd 前向 + autograd 反向 + CPU 参考对比
    └── benchmark_generation_npu.py            # 上游生成 benchmark 的 NPU 适配版（§5.4 Example 用）
```

> 算子源码头注释中引用的 `.cannbot/` 方案/需求文档为开发期过程件，未随本包分发；
> 注释保留原样以便溯源（kernel 设计要点已内联于头注释本身）。

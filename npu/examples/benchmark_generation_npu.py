#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mamba 生成 benchmark 的 NPU 适配版（底本：上游 benchmarks/benchmark_generation_mamba_simple.py）。

上游脚本面向 CUDA 编写，本文件为其 Ascend NPU 最小适配版：device=npu、
torch.npu.synchronize/Event、cg=False、fused_add_norm 关闭 + RMSNorm 替换、
shim 挂载（selective_scan prefill 走 torch.ops.cann_bench NPU 算子）。
CLI 参数与上游完全一致，模型加载逻辑未改。

用法（在 mamba 仓根目录执行，本脚本对运行 cwd 鲁棒）：

    ASCEND_LAUNCH_BLOCKING=1 HF_ENDPOINT=https://hf-mirror.com \
        python npu/examples/benchmark_generation_npu.py --model-name "state-spaces/mamba-2.8b" --batch 32

环境变量：
    ASCEND_LAUNCH_BLOCKING=1          必加。dav-2201 平台 MTE 缺陷：mamba-2.8b 真实域在
                                      默认异步提交 / OpExecCache 重放路径下概率性 aivec 507035，
                                      该变量强制逐 kernel 同步提交规避（npu/README §7 第 1 条）
    HF_ENDPOINT=https://hf-mirror.com HuggingFace 下载镜像（直连 huggingface.co 超时的环境必加；
                                      已有本地 HF 缓存时命中缓存）
    ASCEND_RT_VISIBLE_DEVICES=<id>    共享环境选卡（默认 0）

batch 提示：共享小卡（~7 GB HBM 余量）下 mamba-2.8b `--batch 64` 加载/构造期即 OOM，
请用 `--batch 32`（实测可跑）或 `--batch 16`；独占大卡环境可自行尝试更大 batch。
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import torch_npu  # NPU 适配：CANN torch_npu 运行时初始化
# NPU 适配：decode()（mamba_ssm/utils/generation.py，包内禁改）内的
# torch.cuda.Event 构造在 CPU-torch + torch_npu 环境下直接报错
# （"Tried to instantiate dummy base class Event"），patch 为 npu.Event。
torch.cuda.Event = torch.npu.Event

# NPU 适配：shim 模块位于本脚本所在 npu/ 目录，按 __file__ 相对定位（对任意运行 cwd 鲁棒）。
# 追加（而非插入头部）到 sys.path：避免源码树 npu/cann_bench/（.so 为 gitignore 的本地产物，
# 全新 clone 中不存在）遮蔽已安装的 cann_bench wheel（同 npu/examples/quickstart_npu.py）。
sys.path.append(str(Path(__file__).resolve().parent.parent))
import npu_shim_selective_scan  # noqa: E402  NPU 适配：import 时完成 triton 兼容 polyfill（须先于 import mamba_ssm）

# NPU 适配：透明化输出——mamba_ssm 实际解析路径与 NPU 算子注册状态。
# 注意：CANN ascend-toolkit set_env.sh 导出的 PYTHONPATH 以冒号结尾（尾随空条目），
# Python 将空条目解释为 cwd，故从含 mamba_ssm/ 源码的仓根运行时，本地源码会遮蔽
# site-packages（两者内容一致、行为等价；从中性目录运行则解析到 site-packages）。
# 此处打印实际解析路径，防止再次混淆（见 npu/README §7 与问题记录 #25）。
import mamba_ssm  # noqa: E402  polyfill 已生效，import 安全
import cann_bench  # noqa: E402,F401  import 即注册 torch.ops.cann_bench.*
print(f"[npu-bench] mamba_ssm resolved: {mamba_ssm.__file__}")
print(f"[npu-bench] mamba_ssm version: {getattr(mamba_ssm, '__version__', 'n/a')}")
print(f"[npu-bench] torch.ops.cann_bench.selective_scan_fwd registered: "
      f"{torch.ops.cann_bench.selective_scan_fwd is not None}")

npu_shim_selective_scan.install_npu_shim()  # noqa: E402  NPU 适配：selective_scan_cuda → NPU 算子 shim（prefill 走 NPU kernel）
npu_shim_selective_scan.enable_triton_cuda_ctx_compat()  # noqa: E402  NPU 适配：triton kernels 内 torch.cuda.device 上下文 → torch.npu.device

from einops import rearrange  # noqa: E402

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # noqa: E402


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()

repeats = 3
device = "npu"  # NPU 适配（原文 "cuda"）
dtype = torch.float16

print(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("state-spaces/mamba") or args.model_name.startswith("state-spaces/transformerpp")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
    # NPU 适配：官方 config fused_add_norm=True 走 triton layer_norm kernel，
    # 其内部硬编码 torch.cuda.device 上下文（CPU-torch + torch_npu 环境不可用）；
    # RMSNorm wrapper 亦无条件走 triton。关闭融合改走等价非融合 torch 路径，
    # 并将 triton RMSNorm wrapper 替换为 torch 原生 RMSNorm（权重原位拷贝）。
    model.backbone.fused_add_norm = False
    def _to_torch_rmsnorm(norm):
        new_norm = torch.nn.RMSNorm(norm.weight.shape[0], eps=norm.eps, device=device, dtype=dtype)
        new_norm.weight.data.copy_(norm.weight.data)
        return new_norm
    model.backbone.norm_f = _to_torch_rmsnorm(model.backbone.norm_f)
    for _block in model.backbone.layers:
        _block.fused_add_norm = False
        _block.norm = _to_torch_rmsnorm(_block.norm)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

torch.random.manual_seed(0)
if args.prompt is None:
    input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device=device)  # NPU 适配（原文 device="cuda"）
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)  # NPU 适配（原文 device="cuda"）
else:
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
max_length = input_ids.shape[1] + args.genlen

if is_mamba:
    fn = lambda: model.generate(
        input_ids=input_ids,
        max_length=max_length,
        cg=False,  # NPU 适配（原文 cg=True）：CUDA Graph 为 CUDA 专有特性（torch.cuda.CUDAGraph/Stream），NPU 环境不可用
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        min_p=args.minp,
        repetition_penalty=args.repetition_penalty,
    )
else:
    fn = lambda: model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=max_length,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
    )
out = fn()
if args.prompt is not None:
    print(tokenizer.batch_decode(out.sequences.tolist()))

torch.npu.synchronize()  # NPU 适配（原文 torch.cuda.synchronize）
start = time.time()
for _ in range(repeats):
    fn()
torch.npu.synchronize()  # NPU 适配（原文 torch.cuda.synchronize）
print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")

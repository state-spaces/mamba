# Mamba

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

![Mamba](assets/selection.png "选择性状态空间模型 (Selective State Space)")
> **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**\
> Albert Gu*, Tri Dao*\
> 论文地址: https://arxiv.org/abs/2312.00752

![Mamba-2](assets/ssd_algorithm.png "状态空间对偶模型 (State Space Dual Model)")
> **Transformers are SSMs: Generalized Models and Efficient Algorithms**\
>     **Through Structured State Space Duality**\
> Tri Dao*, Albert Gu*\
> 论文地址: https://arxiv.org/abs/2405.21060

![Mamba-3](assets/mamba3.png "推理优先状态空间模型 (Inference-first State Space Model)")
> **Mamba-3: Improved Sequence Modeling using State Space Principles**\
>     **Through Structured State Space Duality**\
> Aakash Lahoti*, Kevin Y. Li*, Berlin Chen*, Caitlin Wang*, Aviv Bick, J. Zico Kolter, Tri Dao†, Albert Gu†\
> 论文地址: https://arxiv.org/abs/2603.15569

## 项目简介

Mamba 是一种全新的状态空间模型（State Space Model, SSM）架构，在语言建模等信息密集型任务中展现出卓越的性能，成功打破了以往亚二次复杂度（Subquadratic）模型性能逊于 Transformer 的瓶颈。
该架构建立在[结构化状态空间模型（S4 系列）](https://github.com/state-spaces/s4)的持续演进基础之上，并延续了 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 的软硬件协同优化（Hardware-Aware）设计理念与高效实现。

## 安装指南

请首先安装 PyTorch。默认情况下，`mamba-ssm` 仅安装纯 Python 核心包，不会自动编译 `selective_scan_cuda` 原生扩展，也不会下载预先缓存的 CUDA Wheel 包。

| 安装模式 | 执行命令 | 行为说明 |
| --- | --- | --- |
| 核心包安装 | `pip install mamba-ssm --no-build-isolation` | 安装 `mamba-ssm` 基础包，不包含 `selective_scan_cuda` 扩展，且不编译 CUDA 模块。 |
| 核心包 + causal-conv1d | `pip install "mamba-ssm[causal-conv1d]" --no-build-isolation` | 安装核心包以及 `causal-conv1d` 额外依赖，仍不包含 `selective_scan_cuda`。 |
| 强制本地构建核心包 | `MAMBA_FORCE_BUILD=TRUE pip install mamba-ssm --no-build-isolation` | 在本地强制编译默认的纯 Python Wheel，仍不包含 `selective_scan_cuda`。 |
| 启用 CUDA selective scan | `MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation` | 安装 `selective_scan_cuda`；pip 将首先尝试拉取匹配的预编译 CUDA/HIP Wheel，若无匹配版本则在本地进行编译。 |
| 强制本地编译 CUDA selective scan | `MAMBA_FORCE_BUILD=TRUE MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation` | 跳过在线缓存 Wheel，强制在本地从源码编译 `selective_scan_cuda`。 |

> **注意**：进行 CUDA 构建时，**必须添加** `--no-build-isolation` 参数，以确保 pip 使用当前环境中已支持 CUDA 的现有 PyTorch，而不是在隔离的构建沙箱中重新安装不带 CUDA 支持的 torch-cpu。

从源码安装遵循相同的默认规则与环境变量开关：

| 安装模式 | 执行命令 |
| --- | --- |
| 源码默认安装（不含 `selective_scan_cuda`） | `pip install . --no-build-isolation` |
| 直接从 GitHub 源码安装（不含 `selective_scan_cuda`） | `pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation` |
| 源码强制本地核心构建（不含 `selective_scan_cuda`） | `MAMBA_FORCE_BUILD=TRUE pip install . --no-build-isolation` |
| 源码启用 CUDA selective scan | `MAMBA_KEEP_CUDA_BUILD=TRUE pip install . --no-build-isolation` |
| 源码强制本地编译 CUDA selective scan | `MAMBA_FORCE_BUILD=TRUE MAMBA_KEEP_CUDA_BUILD=TRUE pip install . --no-build-isolation` |

**特别提示**：若要在最新源码树中使用 Mamba-3，需通过源码安装。若仅安装不带 `selective_scan_cuda` 的默认源码版本，请执行 `pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation`；若需启用 CUDA selective scan 扩展，请添加环境变量 `MAMBA_KEEP_CUDA_BUILD=TRUE`；若需在本地强制编译 CUDA 模块，请同时添加 `MAMBA_FORCE_BUILD=TRUE`。

核心基础运行环境：
- Linux 操作系统
- Python 3.10+
- PyTorch 1.12+

构建 CUDA `selective_scan_cuda` 原生扩展并进行 GPU 执行的额外环境要求：
- NVIDIA GPU 显卡
- CUDA 11.6+

若使用 AMD 显卡，请查阅下方的 AMD 平台额外配置要求。

## 使用指引

本代码库提供了面向 Mamba 模型的多个层级调用接口。

### 选择性状态空间层 (Selective SSM)

Mamba 的核心基础是选择性 SSM 层（论文第 3 节；算法 2）。

源码实现：[ops/selective_scan_interface.py](mamba_ssm/ops/selective_scan_interface.py)。

### Mamba 架构块 (Mamba Block)

本仓库的主要模型模块是封装了选择性 SSM 的 Mamba 核心架构块。

源码实现：[modules/mamba_simple.py](mamba_ssm/modules/mamba_simple.py)。

调用示例：
```python
import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # 该模块参数量大约为 3 * expand * d_model^2
    d_model=dim, # 模型隐层维度 d_model
    d_state=16,  # SSM 状态扩展因子
    d_conv=4,    # 局部一维卷积核宽度
    expand=2,    # 块扩展倍数
).to("cuda")
y = model(x)
assert y.shape == x.shape
```

### Mamba-2

Mamba-2 核心架构块实现于 [modules/mamba2.py](mamba_ssm/modules/mamba2.py)。

极简版本实现于 [modules/mamba2_simple.py](mamba_ssm/modules/mamba2_simple.py)。

使用方式与 Mamba(-1) 类似：
```python
from mamba_ssm import Mamba2
model = Mamba2(
    # 该模块参数量大约为 3 * expand * d_model^2
    d_model=dim, # 模型隐层维度 d_model
    d_state=64,  # SSM 状态扩展因子，通常取 64 或 128
    d_conv=4,    # 局部一维卷积核宽度
    expand=2,    # 块扩展倍数
).to("cuda")
y = model(x)
assert y.shape == x.shape
```

#### SSD (结构化状态空间对偶性)

内部 SSD 核心模块的极简实现（对应 Mamba-2 论文 Listing 1），包含“离散”与“连续”SSM 之间的等价转换逻辑，参见 [modules/ssd_minimal.py](mamba_ssm/modules/ssd_minimal.py)。

### Mamba-3

Mamba-3 核心架构块实现于 [modules/mamba3.py](mamba_ssm/modules/mamba3.py)。

调用示例：
```python
from mamba_ssm import Mamba3
batch, length, dim = 2, 2048, 768
x = torch.randn(batch, length, dim).to(torch.bfloat16).to("cuda")
model = Mamba3(
    # 该模块参数量大约为 6 * d_model^2
    d_model=dim, # 模型隐层维度 d_model
    d_state=128,  # SSM 状态尺寸
    headdim=64, # SSM 头维度 (headdim)
    is_mimo=True, # 启用多输入多输出 (MIMO) 模式
    mimo_rank=4, # MIMO 模式下的 Rank
    chunk_size=16, # 若 x 为 bf16 则为 64/mimo_rank，否则为 32/mimo_rank
    is_outproj_norm=False, # 是否在输出投影前进行额外归一化
    dtype=torch.bfloat16,
).to("cuda")
y = model(x)
assert y.shape == x.shape
```

### Mamba 完整语言模型

最后，我们提供了一个完整的端到端语言模型结构范例：深层序列主干骨干网络（由多个重复的 Mamba 块堆叠而成）+ 语言模型预测头（LM Head）。

源码实现：[models/mixer_seq_simple.py](mamba_ssm/models/mixer_seq_simple.py)。

该示例展示了如何将 Mamba 模块无缝集成至端到端深度神经网络中，并直接应用于下文的文本生成与评测脚本。

## 预训练模型

官方预训练模型权重已上传至 [Hugging Face Hub](https://huggingface.co/state-spaces)：涵盖在 The Pile 数据集（300B Tokens）上预训练的 `mamba-130m`、`mamba-370m`、`mamba-790m`、`mamba-1.4b`、`mamba-2.8b`、`mamba2-130m`、`mamba2-370m`、`mamba2-780m`、`mamba2-1.3b`、`mamba2-2.7b`、`transformerpp-2.7b`、`mamba2attn-2.7b`，以及在 SlimPajama 数据集（600B Tokens）上训练的 `mamba-2.8b-slimpj`。

下方的推理生成脚本会自动按需拉取模型权重。

这些模型均在 [The Pile 数据集](https://huggingface.co/datasets/EleutherAI/pile) 上完成训练，并遵循 GPT-3 架构所定义的标准模型维度规范，这也是众多主流开源模型所采用的标准：

| 参数量 | 网络层数 | 模型维度 |
|------------|--------|------------|
| 130M       | 24     | 768        |
| 370M       | 48     | 1024       |
| 790M       | 48     | 1536       |
| 1.4B       | 48     | 2048       |
| 2.8B       | 64     | 2560       |

（Mamba 的网络层数大约是相同规模 Transformer 的两倍，因为 Transformer 的每个标准层包含一个 MHA 多头注意力块和一个 MLP 前馈块，而在 Mamba 中每层对应两个 Mamba 块。）

> **说明**：上述模型均为仅在 300B Tokens 上训练的基座模型（Base Models），未经过任何下游任务修改（如指令微调等）。其性能预期等同或优于在类似数据量上训练的其他模型架构，但无法直接与更大规模或经过指令微调的模型对比。

## 效果评测

运行模型的零样本（Zero-Shot）效果评测（对应原论文表 3），我们使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 评测工具套件。

1. 通过 pip 安装评测库：`pip install lm-eval==0.4.2`。
2. 执行评测脚本（更多文档参见 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor) 仓库）：
```sh
lm_eval --model mamba_ssm --model_args pretrained=state-spaces/mamba-130m --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa --device cuda --batch_size 256
python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-160m --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande --device cuda --batch_size 64
```

复现官方博客中报告的 `mamba-2.8b-slimpj` 模型基准结果：
```sh
lm_eval --model mamba_ssm --model_args pretrained=state-spaces/mamba-2.8b-slimpj --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,race,truthfulqa_mc2 --device cuda --batch_size 256
lm_eval --model mamba_ssm --model_args pretrained=state-spaces/mamba-2.8b-slimpj --tasks mmlu --num_fewshot 5 --device cuda --batch_size 256
```

评测 Mamba-2 系列模型时，只需替换对应的模型标识符：
```sh
lm_eval --model mamba_ssm --model_args pretrained=state-spaces/mamba2-2.7b --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa --device cuda --batch_size 256
lm_eval --model mamba_ssm --model_args pretrained=state-spaces/transformerpp-2.7b --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa --device cuda --batch_size 256
lm_eval --model mamba_ssm --model_args pretrained=state-spaces/mamba2attn-2.7b --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa --device cuda --batch_size 256
```

> **注意**：由于评估流程中的随机波动，各任务的实际得分与论文报告值可能会有 0.1-0.3 的正常浮动。

## 推理基准测试

评测脚本 [benchmarks/benchmark_generation_mamba_simple.py](benchmarks/benchmark_generation_mamba_simple.py) 支持：
1. 自动从 Hugging Face Hub 下载指定模型；
2. 根据用户自定义 Prompt 执行续写生成；
3. 测试并输出该生成过程的端到端推理速度基准。

脚本支持设置 Top-p（核采样）概率阈值以及 Softmax 温度系数（Temperature）。

### 使用示例

测试不同采样策略下的单批次生成延迟（例如 Batch Size = 1）：

```sh
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-2.8b" --prompt "My cat wrote all this CUDA code for a new language model and" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
python benchmarks/benchmark_generation_mamba_simple.py --model-name "EleutherAI/pythia-2.8b" --prompt "My cat wrote all this CUDA code for a new language model and" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-2.8b" --prompt "My cat wrote all this CUDA code for a new language model and" --minp 0.05 --topk 0 --temperature 0.7 --repetition-penalty 1.2
```

测试使用随机 Prompt 时的生成吞吐量（例如大 Batch Size 场景）：
```sh
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-2.8b" --batch 64
python benchmarks/benchmark_generation_mamba_simple.py --model-name "EleutherAI/pythia-2.8b" --batch 64
```

测试 Mamba-2 模型时，只需切换模型名称即可：
```sh
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba2-2.7b" --prompt "My cat wrote all this CUDA code for a new language model and" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
```

## 疑难排查 (Troubleshooting)

### 计算精度 (Precision)
我们的预训练模型使用 PyTorch [AMP (自动混合精度)](https://pytorch.org/docs/stable/amp.html) 进行训练。AMP 会将主模型参数保留在 float32 精度，并在前向/反向计算时按需动态转换为半精度。
与此相对，部分其他框架（如 DeepSpeed）会将模型参数直接存储在 float16 中，仅在必要时（如优化器累加）进行上采样提升精度。

我们在实验中发现，为模型主参数维持更高的存储精度是至关重要的，因为状态空间模型（SSM）对其循环动态演进（Recurrent Dynamics）极其敏感。如果您在训练中遇到数值不稳定问题，作为第一步，建议尝试在像 AMP 这样将主参数保存在 fp32 的框架下运行。

### 参数初始化 (Initialization)
模型的部分组件沿用了此前 S4 系列模型的研究成果进行参数初始化。
[例如](https://github.com/state-spaces/mamba/blob/f0affcf69f06d1d06cef018ff640bf080a11c421/mamba_ssm/modules/mamba_simple.py#L102)，$\Delta$ 参数通过初始化其线性投影偏置项使其处于特定的目标数值范围内。
然而，部分训练框架可能会包含额外的后初始化钩子（例如将所有 `nn.Linear` 层的偏置强制置零）。
若遇到此类情况，您可能需要加入特定于该训练框架的适配逻辑（例如在官方训练代码中通过[这行代码](https://github.com/state-spaces/mamba/blob/f0affcf69f06d1d06cef018ff640bf080a11c421/mamba_ssm/modules/mamba_simple.py#L104)关闭二次初始化，而在其他框架中该行代码则等同于空操作）。

## AMD 显卡额外配置要求

### 打补丁适配 ROCm

若您使用的是 ROCm 6.0，请按照以下步骤应用补丁以避免编译报错。ROCm 6.1 及以上版本已无需此操作。

1. 确认您的 ROCm 安装路径（通常位于 `/opt/rocm/`，具体取决于您的系统配置）。

2. 应用补丁文件（如遇权限问题请配合 `sudo` 执行）：
   ```bash
   patch /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h < rocm_patch/rocm6_0.patch 
   ```

## 学术引用

如果您在研究中使用了本代码库，或认为本项目对您的工作有所启发，请按如下格式引用 Mamba 相关论文：

```bibtex
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@inproceedings{mamba2,
  title={Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}

@misc{lahoti2026mamba3improvedsequencemodeling,
      title={Mamba-3: Improved Sequence Modeling using State Space Principles}, 
      author={Aakash Lahoti and Kevin Y. Li and Berlin Chen and Caitlin Wang and Aviv Bick and J. Zico Kolter and Tri Dao and Albert Gu},
      year={2026},
      eprint={2603.15569},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.15569}, 
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月13日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！

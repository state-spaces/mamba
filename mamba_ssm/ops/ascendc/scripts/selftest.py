# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""mamba3_decode_step G1 自测: 逐 case 跑 NPU 算子 vs CPU float64 golden。

判据（任务书 §3.2 字面规则；§3.2 第 2 条声明阈值按生态算子开源精度标准
``docs/zh/ops_precision_standard/mixed_tolerance_standard.md`` 执行）:

  rtol / atol / required_matched_ratio 与任务书 §3.2 阈值表一致（fp16 1e-3/1e-3/0.99，
  bf16 7.8e-3/7.8e-3/0.99，fp32 1.2e-4/1.2e-4/0.99）。

  整体判定（标准 §2.1.2 + §2.3，与任务书 §3.2 的「matched_ratio 且聚合 max_abs_error」一致）:
    1) matched_ratio = 逐元素 |a-g| <= atol + rtol*|g| 的通过点占比；±inf 点按标准 §2.3 的收窄
       规则比对（golden 已按输出 dtype 收窄，同号 ±inf 记为通过；NaN 点记为不通过）。
    2) max_abs_error = 所有【有限值比对点】的最大绝对误差（±inf/NaN 点不参与取最大值）。
    3) max_abs_error_limit = max(兜底值, 32 * ULP(g_low))，兜底值与 ULP 指数按任务书 §3.2 表；
       g_low 为最大绝对误差点的 golden 值（按输出 dtype 收窄后的值）。「或」取两项中较大者。
    4) 通过 = matched_ratio >= required_matched_ratio 且 max_abs_error <= max_abs_error_limit。

  仅披露、不参与判定:
    - verdict_fixed_only: 把 max_abs_error_limit 退化为固定兜底值（1e-1 / 1e-2）时的判定；
    - verdict_contract_v2_disclosure: PRECISION_CONTRACT-v2 §1.2 的逐元素硬上限形式。

用法:
  python scripts/selftest.py --out selftest_results.json
  python scripts/selftest.py --dump-cases cases.json
  python scripts/selftest.py --list
  python scripts/selftest.py --filter p0001
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mamba3_step as _step_mod  # noqa: E402
from mamba3_step import mamba3_step_fn  # noqa: E402
from golden import mamba3_decode_step_ref  # noqa: E402

SEED = 20261027
PRECISION_CASES_TOTAL = 1000
PERF_CASES_TOTAL = 200
GENERATOR_VERSION = "g1-cases-v2"

ULP_MULTIPLIER = 32.0

THRESHOLDS = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3, "ratio": 0.99, "fixed": 1e-1, "rank": 1,
                    "mantissa": 10, "min_subnormal": 2.0 ** -24},
    torch.bfloat16: {"atol": 7.8e-3, "rtol": 7.8e-3, "ratio": 0.99, "fixed": 1e-1, "rank": 0,
                     "mantissa": 7, "min_subnormal": 2.0 ** -133},
    torch.float32: {"atol": 1.2e-4, "rtol": 1.2e-4, "ratio": 0.99, "fixed": 1e-2, "rank": 2,
                    "mantissa": 23, "min_subnormal": 2.0 ** -149},
}


def _ulp(value_abs: float, row_dtype: torch.dtype) -> float:
    """ULP(x) = 2^(floor(log2 x) - 尾数位)；x = 0 取该 dtype 最小次正规数（契约 §1.2 口径）。"""
    th = THRESHOLDS[row_dtype]
    if not math.isfinite(value_abs) or value_abs == 0.0:
        return th["min_subnormal"]
    exponent = math.floor(math.log2(value_abs))
    return max(th["min_subnormal"], 2.0 ** (exponent - th["mantissa"]))


def _limit(value_abs: float, row_dtype: torch.dtype) -> float:
    """max_abs_error_limit = max(兜底值, 32 * ULP(g_low))（任务书 §3.2 表 + 标准 §2.1.2）。"""
    return max(THRESHOLDS[row_dtype]["fixed"], ULP_MULTIPLIER * _ulp(value_abs, row_dtype))


def _elementwise_ulp_limit(g_abs: torch.Tensor, row_dtype: torch.dtype) -> torch.Tensor:
    """契约 v2 §1.2 第 2 条逐元素硬上限（仅披露）: max(固定值, 32 * ULP_judge(|g_i|))。"""
    th = THRESHOLDS[row_dtype]
    g64 = g_abs.double()
    with torch.no_grad():
        exponent = torch.floor(torch.log2(torch.clamp(g64, min=1e-300)))
        ulp = torch.pow(torch.tensor(2.0, dtype=torch.float64), exponent - th["mantissa"])
        ulp = torch.where(g64 == 0, torch.full_like(ulp, th["min_subnormal"]), ulp)
        ulp = torch.clamp(ulp, min=th["min_subnormal"])
    return torch.clamp(ULP_MULTIPLIER * ulp, min=th["fixed"])


def judge_output(actual: torch.Tensor, golden: torch.Tensor, row_dtype: torch.dtype) -> Dict:
    """按任务书 §3.2 gate（matched_ratio + 聚合 max_abs_error ≤ max(兜底值, 32×ULP)）判定。"""
    th = THRESHOLDS[row_dtype]
    a = actual.detach().float().reshape(-1)
    g = golden.detach().float().reshape(-1)
    atol, rtol = th["atol"], th["rtol"]

    finite_a = torch.isfinite(a)
    finite_g = torch.isfinite(g)
    both = finite_a & finite_g
    inf_a = torch.isinf(a)
    inf_g = torch.isinf(g)
    same_inf = inf_a & inf_g & (torch.sign(a) == torch.sign(g))
    nan_a = torch.isnan(a)
    nan_g = torch.isnan(g)

    diff = torch.where(both, (a - g).abs(), torch.zeros_like(a))
    close = both & (diff <= atol + rtol * g.abs())
    matched = close | same_inf

    numel = int(matched.numel())
    ratio = float(matched.float().mean().item()) if numel else float("nan")

    n_nonfinite = int((~both).sum().item())
    n_nonfinite_matched = int(same_inf.sum().item())
    n_nan = int((nan_a | nan_g).sum().item())

    if bool(both.any().item()):
        flat_idx = int(torch.argmax(diff).item())
        max_abs = float(diff[flat_idx].item())
        g_low = float(abs(g[flat_idx].item()))
        ulp = _ulp(g_low, row_dtype)
        limit = max(th["fixed"], ULP_MULTIPLIER * ulp)
    else:
        max_abs, g_low, ulp, limit = None, None, None, th["fixed"]

    ok = (
        numel > 0
        and ratio >= th["ratio"]
        and (max_abs is None or max_abs <= limit)
    )
    fixed_ok = (
        numel > 0
        and ratio >= th["ratio"]
        and (max_abs is None or max_abs <= th["fixed"])
    )

    # 披露: 契约 v2 §1.2 逐元素硬上限（不计入 gate，不替代任务书判据）
    limit_elem = _elementwise_ulp_limit(g.abs().double(), row_dtype).float()
    n_hard_cap = int(((diff > limit_elem) & both).sum().item())
    v2_ok = bool(
        numel > 0
        and ratio >= th["ratio"]
        and n_hard_cap == 0
        and int((~both & ~same_inf).sum().item()) == 0
    )

    return {
        "storage_dtype": str(actual.dtype).replace("torch.", ""),
        "judge_dtype": str(row_dtype).replace("torch.", ""),
        "numel": numel,
        "matched_ratio": round(ratio, 8),
        "matched_ratio_required": th["ratio"],
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": None if max_abs is None else round(max_abs, 12),
        "golden_at_max_abs_error": None if g_low is None else round(g_low, 10),
        "golden_abs_max": round(float(g.abs().max().item()), 6) if numel else None,
        "ulp_at_max_abs_error": None if ulp is None else ulp,
        "max_abs_error_limit": round(limit, 12),
        "max_abs_error_limit_fixed": th["fixed"],
        "limit_from_32ulp": bool(max_abs is not None and ULP_MULTIPLIER * ulp > th["fixed"]),
        "nonfinite_count": n_nonfinite,
        "nonfinite_matched_count": n_nonfinite_matched,
        "nan_count": n_nan,
        "n_hard_cap_violations_contract_v2": n_hard_cap,
        "verdict": "PASS" if ok else "FAIL",
        "verdict_fixed_only": "PASS" if fixed_ok else "FAIL",
        "verdict_contract_v2_disclosure": "PASS" if v2_ok else "FAIL",
    }


def judge_row(*dtypes: torch.dtype) -> torch.dtype:
    return min(dtypes, key=lambda d: THRESHOLDS[d]["rank"])


def _mixed_dist(shape, gen: torch.Generator) -> torch.Tensor:
    """任务书 §3.5: 均匀分布 [-5,5] 占 50%, 正态分布 mu in [-5,5], sigma in [0.1,2] 占 50%。"""
    uni = torch.empty(shape).uniform_(-5.0, 5.0, generator=gen)
    mu = torch.empty(shape).uniform_(-5.0, 5.0, generator=gen)
    sigma = torch.empty(shape).uniform_(0.1, 2.0, generator=gen)
    nor = torch.normal(mu, sigma, generator=gen)
    mask = torch.rand(shape, generator=gen) < 0.5
    return torch.where(mask, uni, nor)


def _make_inputs(case, gen: torch.Generator) -> Dict[str, torch.Tensor]:
    B, H, D, N, R = case.B, case.H, case.D, case.N, case.R

    u = torch.empty((B, H)).uniform_(0.0, 1.0, generator=gen)
    log_lo = torch.log(torch.tensor(1e-3))
    log_hi = torch.log(torch.tensor(1.0))
    dt_t = torch.exp(log_lo + u * (log_hi - log_lo)).clamp(1e-3, 1.0)
    dt_t[0, 0] = 1e-3
    dt_t[-1, -1] = 1.0
    trap_t = torch.sigmoid(torch.randn((B, H), generator=gen))
    trap_t[0, 0] = 0.0
    trap_t[-1, -1] = 1.0
    A_t = (-F.softplus(torch.randn((B, H), generator=gen) * 3.0)).clamp(min=-20.0, max=0.0)

    ds, do = case.dtype_state, case.dtype_other
    tensors = {
        "state": _mixed_dist((B, H, D, N), gen).to(ds),
        "Bstate": _mixed_dist((B, R, H, N), gen).to(do),
        "Xstate": _mixed_dist((B, H, D), gen).to(do),
        "A": A_t.to(do),
        "B": _mixed_dist((B, R, H, N), gen).to(do),
        "C": _mixed_dist((B, R, H, N), gen).to(do),
        "D": _mixed_dist((H,), gen).to(do),
        "x": _mixed_dist((B, H, D), gen).to(do),
        "dt": dt_t.to(do),
        "trap": trap_t.to(do),
        "xproj": _mixed_dist((R, H, D), gen).to(do),
    }
    if case.has_outproj:
        tensors["outproj"] = _mixed_dist((R, H, D), gen).to(do)
    if case.has_z:
        tensors["z"] = _mixed_dist((B, H, D), gen).to(do)
        tensors["zproj"] = _mixed_dist((R, H, D), gen).to(do)
    return tensors


@dataclass
class Case:
    cid: str
    B: int
    H: int
    D: int
    N: int
    R: int
    dtype_state: torch.dtype
    dtype_other: torch.dtype
    has_outproj: bool
    has_z: bool
    tile_D: int
    inplace: bool = False
    bcast_bc: bool = False
    num_warps: int = 2
    tag: str = ""
    data_seed: int = 0

    def describe(self) -> Dict:
        return {
            "id": self.cid,
            "shape": {"B": self.B, "H": self.H, "D": self.D, "N": self.N, "R": self.R},
            "dtype_state": str(self.dtype_state).replace("torch.", ""),
            "dtype_other": str(self.dtype_other).replace("torch.", ""),
            "branch": {"outproj": self.has_outproj, "z": self.has_z},
            "tile_D": self.tile_D,
            "num_warps": self.num_warps,
            "state_out_inplace": self.inplace,
            "bcast_bc_head_stride0_view": self.bcast_bc,
            "data_seed": self.data_seed,
            "tag": self.tag,
        }


# 任务书 §3.5 shape 边界点: B=1 / H=1 / D=tile_D / D=2×tile_D / N=32 / N=128 / R=1 / R=4，
# 另含 H≠D、N≠D、奇尺寸与 §3.3 规模档；D 的取值覆盖 32/64/96/128（可整除 32/64/128 子集）。
SHAPES: List[Tuple[int, int, int, int, int]] = [
    (1, 1, 32, 32, 1),
    (1, 1, 64, 32, 4),
    (1, 1, 64, 128, 1),
    (1, 1, 128, 128, 4),
    (1, 2, 64, 32, 2),
    (1, 2, 128, 32, 1),
    (1, 4, 32, 64, 4),
    (1, 4, 128, 64, 2),
    (1, 8, 128, 32, 1),
    (1, 8, 64, 128, 4),
    (1, 16, 64, 128, 2),
    (1, 32, 32, 128, 4),
    (1, 64, 128, 128, 4),
    (2, 1, 32, 32, 1),
    (2, 1, 128, 32, 1),
    (2, 1, 64, 128, 4),
    (2, 2, 32, 128, 1),
    (2, 2, 64, 64, 2),
    (2, 3, 96, 48, 3),
    (2, 4, 128, 128, 4),
    (2, 4, 64, 32, 1),
    (2, 8, 32, 64, 2),
    (2, 16, 64, 32, 4),
    (3, 1, 64, 64, 2),
    (3, 2, 32, 32, 1),
    (3, 5, 96, 32, 4),
    (4, 3, 96, 48, 3),
    (4, 4, 128, 32, 2),
    (4, 8, 64, 128, 1),
    (5, 1, 32, 64, 4),
    (5, 2, 64, 32, 1),
    (6, 4, 32, 128, 2),
    (8, 2, 96, 64, 1),
    (8, 8, 64, 64, 4),
    (16, 1, 128, 32, 2),
    (16, 4, 32, 32, 1),
    (32, 2, 64, 128, 4),
    (32, 8, 32, 64, 1),
    (64, 1, 64, 32, 2),
    (128, 1, 64, 128, 1),
    (128, 4, 32, 32, 2),
]

# §3.3 性能子集最大档；精度全量每个 dtype 面各覆盖 1 条（合计 3 条）
HEAVY_SHAPE = (128, 64, 64, 128, 4)

FACES = [
    ("fp32", torch.float32, torch.float32, 334),
    ("fp16", torch.float16, torch.float16, 333),
    ("bf16", torch.bfloat16, torch.bfloat16, 333),
]

BRANCHES = [
    ("opz", True, True),
    ("op", True, False),
    ("none", False, False),
    ("z", False, True),
]

PERF_SHAPES = [
    (1, 1, 32, 32, 1),
    (1, 1, 64, 32, 1),
    (128, 64, 64, 128, 1),
    (1, 64, 128, 128, 4),
    (128, 1, 64, 128, 1),
    (128, 64, 64, 128, 4),
]


def _valid_tiles(D: int) -> List[int]:
    return [t for t in (32, 64, 128) if D % t == 0]


def _data_seed(cid: str) -> int:
    return SEED + zlib.crc32(cid.encode()) % 100000


def _combos(shape_pool) -> List[Tuple[Tuple[int, int, int, int, int], Tuple[str, bool, bool], int]]:
    out = []
    for shape in shape_pool:
        for branch in BRANCHES:
            for tile in _valid_tiles(shape[2]):
                out.append((shape, branch, tile))
    return out


def build_cases() -> List[Case]:
    """任务书 §3.5 精度全量 1000 条：3 个 dtype 面（334/333/333）× shape×分支×tile_D 正交遍历。"""
    combos = _combos(SHAPES)
    cases: List[Case] = []
    for face_name, ds, do, count in FACES:
        body = list(itertools.islice(itertools.cycle(combos), count - 1))
        for i, (shape, (bname, has_op, has_z), tile) in enumerate(body):
            B, H, D, N, R = shape
            cid = f"p{len(cases):04d}_{face_name}_s{SHAPES.index(shape):02d}_{bname}_t{tile}"
            cases.append(
                Case(
                    cid=cid, B=B, H=H, D=D, N=N, R=R,
                    dtype_state=ds, dtype_other=do,
                    has_outproj=has_op, has_z=has_z, tile_D=tile,
                    inplace=(i % 2 == 0),
                    bcast_bc=(i % 17 == 3),
                    num_warps=4 if i % 3 == 0 else 2,
                    tag="precision",
                    data_seed=_data_seed(cid),
                )
            )
        B, H, D, N, R = HEAVY_SHAPE
        cid = f"p{len(cases):04d}_{face_name}_heavy_zz_opz_t64"
        cases.append(
            Case(
                cid=cid, B=B, H=H, D=D, N=N, R=R,
                dtype_state=ds, dtype_other=do,
                has_outproj=True, has_z=True, tile_D=64,
                inplace=False, bcast_bc=False, num_warps=2,
                tag="precision-perf-shape",
                data_seed=_data_seed(cid),
            )
        )
    assert len(cases) == PRECISION_CASES_TOTAL, len(cases)
    return cases


def build_perf_cases() -> List[Dict]:
    """任务书 §3.5 性能子集 200 条：dtype × 规模档分层抽取（本 G1 子单只产出用例清单，不执行）。"""
    rows: List[Dict] = []
    tiles = {32: [32], 64: [32, 64], 128: [32, 64, 128], 96: [32]}
    for i in range(PERF_CASES_TOTAL):
        face_name, ds, _do, _cnt = FACES[i % len(FACES)]
        shape = PERF_SHAPES[(i // len(FACES)) % len(PERF_SHAPES)]
        B, H, D, N, R = shape
        bname, has_op, has_z = BRANCHES[(i // 3) % len(BRANCHES)]
        tile = tiles[D][(i // 7) % len(tiles[D])]
        rows.append({
            "id": f"perf{i:04d}_{face_name}_s{PERF_SHAPES.index(shape):02d}_{bname}_t{tile}",
            "shape": {"B": B, "H": H, "D": D, "N": N, "R": R},
            "dtype": face_name,
            "dtype_state": str(ds).replace("torch.", ""),
            "branch": {"outproj": has_op, "z": has_z},
            "tile_D": tile,
            "num_warps": 2 if i % 2 == 0 else 4,
            "data_seed": _data_seed(f"perf{i:04d}"),
        })
    return rows


def run_negative_cases(device) -> List[Dict]:
    g = torch.Generator().manual_seed(SEED + 7)
    results: List[Dict] = []

    def base(**over):
        B, H, D, N, R = 2, 3, 32, 32, 1
        kw = dict(
            state=_mixed_dist((B, H, D, N), g).to(torch.float32).to(device),
            Bstate=_mixed_dist((B, R, H, N), g).to(torch.float32).to(device),
            Xstate=_mixed_dist((B, H, D), g).to(torch.float32).to(device),
            A=(-F.softplus(torch.randn((B, H), generator=g))).to(device),
            B=_mixed_dist((B, R, H, N), g).to(torch.float32).to(device),
            C=_mixed_dist((B, R, H, N), g).to(torch.float32).to(device),
            D=_mixed_dist((H,), g).to(torch.float32).to(device),
            x=_mixed_dist((B, H, D), g).to(torch.float32).to(device),
            dt=torch.empty((B, H)).uniform_(1e-3, 1.0, generator=g).to(device),
            trap=torch.rand((B, H), generator=g).to(device),
            xproj=_mixed_dist((R, H, D), g).to(torch.float32).to(device),
            outproj=_mixed_dist((R, H, D), g).to(torch.float32).to(device),
            tile_D=32,
        )
        kw["out"] = torch.empty((B, H, D), dtype=torch.float32, device=device)
        kw.update(over)
        return kw

    def expect_raise(nid, fn, why):
        try:
            fn()
        except ValueError as exc:
            results.append({"id": nid, "expect": "ValueError", "why": why, "result": "PASS",
                            "message": str(exc)[:200]})
        except Exception as exc:  # noqa: BLE001
            results.append({"id": nid, "expect": "ValueError", "why": why, "result": "FAIL",
                            "message": f"{type(exc).__name__}: {exc}"[:200]})
        else:
            results.append({"id": nid, "expect": "ValueError", "why": why, "result": "FAIL",
                            "message": "no exception raised"})

    expect_raise("neg01_state_rank", lambda: mamba3_step_fn(**{**base(), "state": base()["state"][0]}),
                 "state 秩不为 4")
    expect_raise("neg02_empty", lambda: mamba3_step_fn(**base(state=torch.empty((0, 3, 32, 32), device=device))),
                 "numel==0")
    expect_raise("neg03_tile_d", lambda: mamba3_step_fn(**base(tile_D=48)), "tile_D 不在 {32,64,128}")
    expect_raise("neg04_tile_div", lambda: mamba3_step_fn(**base(tile_D=64)), "D%tile_D != 0")
    expect_raise("neg05_z_no_zproj", lambda: mamba3_step_fn(**base(z=base()["x"])), "z 非空但 zproj 为 None")
    expect_raise(
        "neg06_out_shape_with_outproj",
        lambda: mamba3_step_fn(**{**base(), "out": torch.empty((2, 1, 3, 32), device=device)}),
        "outproj 非空时 out 必须是 [B,H,D]",
    )
    expect_raise(
        "neg07_out_shape_no_outproj",
        lambda: mamba3_step_fn(**base(outproj=None)),
        "outproj 为 None 时 out 必须是 [B,R,H,D]",
    )
    expect_raise(
        "neg08_dtype_group",
        lambda: mamba3_step_fn(**{**base(), "C": base()["C"].to(torch.float16)}),
        "B/Bstate/C 组内 dtype 不一致",
    )
    big = torch.empty((2, 3, 32, 64), device=device)
    expect_raise(
        "neg09_partial_overlap",
        lambda: mamba3_step_fn(**{**base(), "state": big[..., :32], "state_out": big[..., 32:]}),
        "state_out 与 state 部分重叠",
    )
    expect_raise("neg10_num_warps", lambda: mamba3_step_fn(**base(num_warps=0)), "num_warps 非正整数")
    expect_raise(
        "neg11_xproj_shape",
        lambda: mamba3_step_fn(**{**base(), "xproj": torch.empty((2, 3, 32), device=device)}),
        "xproj 形状不为 [R,H,D]",
    )
    expect_raise(
        "neg12_dtype_unsupported",
        lambda: mamba3_step_fn(**{**base(), "x": base()["x"].to(torch.int32)}),
        "x dtype 不在支持集合",
    )
    expect_raise("neg13_out_none", lambda: mamba3_step_fn(**{**base(), "out": None}), "out 未预分配")
    expect_raise(
        "neg14_outproj_shape",
        lambda: mamba3_step_fn(**{**base(), "outproj": torch.empty((2, 3, 32), device=device)}),
        "outproj 形状不为 [R,H,D]",
    )
    expect_raise(
        "neg15_z_shape",
        lambda: mamba3_step_fn(**{**base(), "z": torch.empty((2, 3, 16), device=device),
                                  "zproj": torch.empty((1, 3, 32), device=device)}),
        "z 形状不为 [B,H,D]",
    )
    expect_raise(
        "neg16_D_len",
        lambda: mamba3_step_fn(**{**base(), "D": torch.empty((4,), device=device)}),
        "D 长度不为 H",
    )
    expect_raise(
        "neg17_A_shape",
        lambda: mamba3_step_fn(**{**base(), "A": torch.empty((2, 1), device=device)}),
        "A 形状不为 [B,H]",
    )
    # 18) 第五组 dtype：A / dt / trap / D 在 tiling 里共用同一个 scalar dtype 编码（取自 A），
    #     kernel 按该宽度读四个 buffer。混搭必须被 C++ 校验器拒绝（rc=34），否则会被读成垃圾值
    #     且不抛异常（状态相对 L2 误差 6e-1 量级）。
    kw = base()
    rc, _ = _step_mod._validator()(
        kw["state"], kw["Bstate"], kw["Xstate"], kw["A"].to(torch.bfloat16),
        kw["B"], kw["C"], kw["D"], kw["x"], kw["dt"], kw["trap"], kw["xproj"],
        kw["outproj"], None, kw["out"], None, None, kw["tile_D"],
    )
    results.append({"id": "neg18_scalar_dtype_mismatch", "expect": "rc=34",
                    "why": "A/dt/trap/D 组内 dtype 不一致", "result": "PASS" if rc == 34 else "FAIL",
                    "message": f"rc={rc}"})
    # 19) 反向：Python 入口把标量侧半精度归一到 fp32（模块路径的真实情形），归一后的结果必须与
    #     显式传 fp32 逐位一致 —— 若归一失效，上面的混搭就会被 kernel 误读，这里立刻暴露。
    kw = base()
    kw["dt"] = kw["dt"].to(torch.bfloat16)
    kw["trap"] = kw["trap"].to(torch.bfloat16)
    # state 是就地更新的，两次调用各自拿一份独立副本，否则第二次的初值已被第一次改写。
    out_mixed = torch.empty_like(kw["out"])
    mamba3_step_fn(**{**kw, "state": kw["state"].clone(), "out": out_mixed})
    out_fp32 = torch.empty_like(kw["out"])
    mamba3_step_fn(**{**kw, "state": kw["state"].clone(),
                      "dt": kw["dt"].to(torch.float32), "trap": kw["trap"].to(torch.float32),
                      "out": out_fp32})
    same = torch.equal(out_mixed.cpu(), out_fp32.cpu())
    diff = 0.0 if same else float((out_mixed.cpu() - out_fp32.cpu()).abs().max().item())
    results.append({"id": "neg19_scalar_dtype_normalized", "expect": "bf16 标量侧 == 显式 fp32",
                    "why": "入口归一到 fp32 后与 fp32 输入逐位一致", "result": "PASS" if same else "FAIL",
                    "message": "bit-identical" if same else f"max|diff|={diff:.3e}"})
    return results


def run_case(case: Case, device) -> Dict:
    gen = torch.Generator().manual_seed(case.data_seed)
    inputs = _make_inputs(case, gen)

    # golden 与 kernel 必须看到同一份数据：B/C 的 head 轴 stride 0 视图在 CPU 侧建视图给 golden，
    # 在 device 侧同样建视图（不物化副本）给 kernel。
    ref_inputs = dict(inputs)
    if case.bcast_bc:
        ref_inputs["B"] = inputs["B"][:, :, :1, :].expand(case.B, case.R, case.H, case.N)
        ref_inputs["C"] = inputs["C"][:, :, :1, :].expand(case.B, case.R, case.H, case.N)

    golden_out, golden_state = mamba3_decode_step_ref(
        ref_inputs["state"], ref_inputs["Bstate"], ref_inputs["Xstate"], ref_inputs["A"], ref_inputs["B"],
        ref_inputs["C"], ref_inputs["D"], ref_inputs["x"], ref_inputs["dt"], ref_inputs["trap"],
        ref_inputs["xproj"],
        outproj=ref_inputs.get("outproj"), z=ref_inputs.get("z"), zproj=ref_inputs.get("zproj"),
    )

    dev = {k: v.to(device) for k, v in inputs.items()}
    if case.bcast_bc:
        # B/C 沿 head 轴 stride 0 的视图 (在 device 上做视图, 不物化副本)
        dev["B"] = dev["B"][:, :, :1, :].expand(case.B, case.R, case.H, case.N)
        dev["C"] = dev["C"][:, :, :1, :].expand(case.B, case.R, case.H, case.N)

    B, H, D, R = case.B, case.H, case.D, case.R
    if case.has_outproj:
        out = torch.empty((B, H, D), dtype=case.dtype_other, device=device)
    else:
        out = torch.empty((B, R, H, D), dtype=case.dtype_other, device=device)
    state_out = None if case.inplace else torch.empty((B, H, D, case.N), dtype=case.dtype_state, device=device)
    state_before = dev["state"].clone()

    mamba3_step_fn(
        dev["state"], dev["Bstate"], dev["Xstate"], dev["A"], dev["B"], dev["C"], dev["D"], dev["x"], dev["dt"],
        dev["trap"], dev["xproj"],
        outproj=dev.get("outproj"), state_out=state_out, out=out, z=dev.get("z"), zproj=dev.get("zproj"),
        tile_D=case.tile_D, num_warps=case.num_warps,
    )
    torch.npu.synchronize()

    actual_state = dev["state"] if case.inplace else state_out
    row = judge_row(case.dtype_state, case.dtype_other)
    st = judge_output(actual_state.cpu(), golden_state, row)
    ot = judge_output(out.cpu(), golden_out, row)
    verdict = "PASS" if (st["verdict"] == "PASS" and ot["verdict"] == "PASS") else "FAIL"

    # 参考链自身精度上限（仅失败 case 采集）: CPU f32 参考链相对 CPU f64 参考链的偏差
    ref_chain = None
    if verdict != "PASS":
        ref32_out, ref32_state = mamba3_decode_step_ref(
            ref_inputs["state"], ref_inputs["Bstate"], ref_inputs["Xstate"], ref_inputs["A"], ref_inputs["B"],
            ref_inputs["C"], ref_inputs["D"], ref_inputs["x"], ref_inputs["dt"], ref_inputs["trap"],
            ref_inputs["xproj"],
            outproj=ref_inputs.get("outproj"), z=ref_inputs.get("z"), zproj=ref_inputs.get("zproj"),
            compute_dtype=torch.float32,
        )
        ref_chain = {
            "out_max_abs_error": round(float((ref32_out.float() - golden_out.float()).abs().max().item()), 12),
            "state_out_max_abs_error": round(
                float((ref32_state.float() - golden_state.float()).abs().max().item()), 12
            ),
        }

    extra = {}
    if case.inplace:
        extra["inplace_changed"] = bool((state_before != dev["state"]).any().item())

    result = case.describe()
    result.update({
        "state_out": st,
        "out": ot,
        "ref_chain_f32_vs_f64": ref_chain,
        "verdict": verdict,
        **extra,
    })
    return result


def determinism_check(device, case_ids: List[str]) -> List[Dict]:
    """任务书 §2.5: 同一输入/设备/构建重复执行结果逐位一致。"""
    out: List[Dict] = []
    by_id = {c.cid: c for c in build_cases()}
    for cid in case_ids:
        case = by_id[cid]
        results = []
        for _ in range(2):
            gen = torch.Generator().manual_seed(case.data_seed)
            inputs = _make_inputs(case, gen)
            dev = {k: v.to(device) for k, v in inputs.items()}
            if case.has_outproj:
                out_t = torch.empty((case.B, case.H, case.D), dtype=case.dtype_other, device=device)
            else:
                out_t = torch.empty((case.B, case.R, case.H, case.D), dtype=case.dtype_other, device=device)
            mamba3_step_fn(
                dev["state"], dev["Bstate"], dev["Xstate"], dev["A"], dev["B"], dev["C"], dev["D"], dev["x"],
                dev["dt"], dev["trap"], dev["xproj"], outproj=dev.get("outproj"), out=out_t,
                z=dev.get("z"), zproj=dev.get("zproj"), tile_D=case.tile_D, num_warps=case.num_warps,
            )
            torch.npu.synchronize()
            results.append((out_t.cpu().clone(), dev["state"].cpu().clone()))
        bitwise_out = bool(torch.equal(results[0][0], results[1][0]))
        bitwise_state = bool(torch.equal(results[0][1], results[1][1]))
        out.append({
            "id": cid,
            "out_bitwise_equal": bitwise_out,
            "state_out_bitwise_equal": bitwise_state,
            "result": "PASS" if (bitwise_out and bitwise_state) else "FAIL",
        })
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coverage(cases: List[Case]) -> Dict:
    fields = {
        "B=1": lambda c: c.B == 1,
        "H=1": lambda c: c.H == 1,
        "D=tile_D": lambda c: c.D == c.tile_D,
        "D=2*tile_D": lambda c: c.D == 2 * c.tile_D,
        "N=32": lambda c: c.N == 32,
        "N=128": lambda c: c.N == 128,
        "R=1": lambda c: c.R == 1,
        "R=4": lambda c: c.R == 4,
        "tile_D=32": lambda c: c.tile_D == 32,
        "tile_D=64": lambda c: c.tile_D == 64,
        "tile_D=128": lambda c: c.tile_D == 128,
        "state_out inplace": lambda c: c.inplace,
        "state_out explicit": lambda c: not c.inplace,
        "B/C head-stride-0 view": lambda c: c.bcast_bc,
        "num_warps=4": lambda c: c.num_warps == 4,
        "num_warps=2": lambda c: c.num_warps == 2,
        "outproj+z": lambda c: c.has_outproj and c.has_z,
        "outproj only": lambda c: c.has_outproj and not c.has_z,
        "no outproj/z": lambda c: not c.has_outproj and not c.has_z,
        "no outproj+z (gate dropped)": lambda c: not c.has_outproj and c.has_z,
    }
    return {k: sum(1 for c in cases if f(c)) for k, f in fields.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="selftest_results.json")
    ap.add_argument("--dump-cases", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--skip-negative", action="store_true")
    ap.add_argument("--skip-determinism", action="store_true")
    args = ap.parse_args()

    cases = build_cases()
    perf_cases = build_perf_cases()

    if args.dump_cases:
        payload = {
            "op": "mamba3_decode_step",
            "generator_version": GENERATOR_VERSION,
            "seed": SEED,
            "precision_full": {"total": len(cases), "cases": [c.describe() for c in cases]},
            "perf_subset": {"total": len(perf_cases), "cases": perf_cases,
                            "note": "任务书 §3.5 性能子集清单；NPU 性能实测属后续阶段，本 G1 子单未执行"},
            "coverage": _coverage(cases),
        }
        Path(args.dump_cases).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"cases -> {args.dump_cases} ({len(cases)} precision / {len(perf_cases)} perf)")
        return 0

    if args.filter:
        cases = [c for c in cases if args.filter in c.cid]
    if args.list:
        for c in cases:
            print(c.cid, c.B, c.H, c.D, c.N, c.R, c.dtype_state, c.dtype_other,
                  c.has_outproj, c.has_z, c.tile_D, c.inplace, c.num_warps, c.bcast_bc)
        return 0

    import torch_npu  # noqa: F401  (注册 PrivateUse1 后端)

    device = torch.device("npu:0")
    lib = Path(__file__).resolve().parents[1] / "build" / "libmamba3_decode_step_ops.so"

    rows: List[Dict] = []
    for case in cases:
        try:
            rows.append(run_case(case, device))
        except Exception:  # noqa: BLE001
            row = case.describe()
            row.update({"verdict": "ERROR", "traceback": traceback.format_exc()})
            rows.append(row)
        print(f"{rows[-1]['id']:<52} {rows[-1]['verdict']}")
        sys.stdout.flush()

    neg = [] if args.skip_negative else run_negative_cases(device)
    for n in neg:
        print(f"{n['id']:<52} {n['result']}")

    det_ids = [c.cid for c in build_cases() if c.tag == "precision-perf-shape"]
    det_ids = [cases[0].cid] + det_ids[:2] if cases else det_ids
    det = [] if (args.skip_negative or args.skip_determinism) else determinism_check(device, det_ids)
    for d in det:
        print(f"determinism {d['id']:<40} {d['result']}")

    passed = sum(1 for r in rows if r["verdict"] == "PASS")
    failed = [r["id"] for r in rows if r["verdict"] != "PASS"]
    neg_failed = [n["id"] for n in neg if n["result"] != "PASS"]
    det_failed = [d["id"] for d in det if d["result"] != "PASS"]

    def _key(r, key):
        if key == "dtype":
            return r["dtype_state"]
        if key == "branch":
            return f"outproj={r['branch']['outproj']},z={r['branch']['z']}"
        return r.get(key)

    def _group(rows_, key, field="verdict"):
        out = {}
        for r in rows_:
            k = _key(r, key)
            out.setdefault(k, {"total": 0, field: 0})
            out[k]["total"] += 1
            if "state_out" in r and r["state_out"][field] == "PASS" and r["out"][field] == "PASS":
                out[k][field] += 1
        return out

    per_dtype = _group(rows, "dtype")
    per_dtype_fixed = _group(rows, "dtype", field="verdict_fixed_only")
    per_branch = _group(rows, "branch")

    payload = {
        "op": "mamba3_decode_step",
        "phase": "G1 self-test",
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "device": torch.npu.get_device_name(0),
        "lib_path": str(lib),
        "lib_sha256": sha256_file(lib) if lib.is_file() else None,
        "gate": {
            "source": "任务书 docs/plan/mamba3_decode_step_task_doc.md §3.2（阈值表 + 第 2 条声明的"
                      "生态算子开源精度标准 mixed_tolerance_standard.md §2.1.2/§2.3）",
            "rule": "matched_ratio >= required_matched_ratio 且 max_abs_error <= max(兜底值, 32*ULP(g_low))",
            "matched_element": "|a-g| <= atol + rtol*|g|；±inf 按标准 §2.3 收窄后同号即通过；NaN 不通过",
            "max_abs_error": "仅统计 actual 与 golden 均为有限值的比对点",
            "max_abs_error_limit": "兜底值 fp16/bf16 1e-1、fp32 1e-2；与 32*ULP(g_low) 取较大者",
            "ulp": "ULP(x)=2^(floor(log2 x)-尾数位)，尾数 10/7/23；x=0 取最小次正规数",
            "g_low": "最大绝对误差点的 golden 值（已按输出 dtype 收窄）",
        },
        "thresholds": {str(k).replace("torch.", ""): v for k, v in THRESHOLDS.items()},
        "summary": {
            "total": len(rows),
            "passed": passed,
            "failed": len(failed),
            "failed_ids": failed,
            "by_dtype": per_dtype,
            "by_dtype_fixed_only": per_dtype_fixed,
            "by_branch": per_branch,
            "fixed_only_limit_pass": sum(
                1 for r in rows if "state_out" in r
                and r["state_out"]["verdict_fixed_only"] == "PASS"
                and r["out"]["verdict_fixed_only"] == "PASS"
            ),
            "contract_v2_disclosure_pass": sum(
                1 for r in rows if "state_out" in r
                and r["state_out"]["verdict_contract_v2_disclosure"] == "PASS"
                and r["out"]["verdict_contract_v2_disclosure"] == "PASS"
            ),
            "negative_total": len(neg),
            "negative_failed_ids": neg_failed,
            "determinism_failed_ids": det_failed,
            "coverage": _coverage(build_cases()),
        },
        "cases": rows,
        "negative_cases": neg,
        "determinism": det,
        "perf_subset": {"total": len(perf_cases), "executed": False,
                        "note": "任务书 §3.5 性能子集 200 条已生成（见 cases.json）；NPU 性能实测属后续阶段"},
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== {passed}/{len(rows)} PASS, {len(failed)} FAIL; negative "
          f"{len(neg) - len(neg_failed)}/{len(neg)}; determinism {len(det) - len(det_failed)}/{len(det)} ===")
    print(f"results -> {args.out}")
    return 0 if (not failed and not neg_failed and not det_failed) else 1


if __name__ == "__main__":
    raise SystemExit(main())

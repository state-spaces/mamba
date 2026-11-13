# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend C 直调算子包（NPU）。

当前提供两个入口：

* ``mamba3_step_fn``（mamba3_decode_step）——与
  ``mamba_ssm.ops.cute.mamba3.mamba3_step_fn`` 同名同签名，调用方只替换导入即可；
* ``mamba3_siso_step``——SISO decode 的单 token 递推，与
  ``mamba_ssm.ops.triton.mamba3.mamba3_siso_step.mamba3_siso_step`` 同名同签名，
  内部由 ``mamba3_step_fn``（R=1）承接状态递推。
"""

from .mamba3_siso_step import mamba3_siso_step
from .mamba3_step import mamba3_step_fn

__all__ = ["mamba3_step_fn", "mamba3_siso_step"]

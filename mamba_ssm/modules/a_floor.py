import math
from typing import Optional

import torch


def apply_a_floor(A: torch.Tensor, A_floor: Optional[float]) -> torch.Tensor:
    """Keep the SSM decay ``A`` away from zero.

    ``A`` is negative (``A = -exp(A_log)``). If it drifts too close to zero the
    recurrence becomes a pure accumulator, the state can grow without bound,
    and training can overflow to NaN. Clamping ``A`` to at most ``-A_floor``
    enforces a minimum decay, matching Mamba-3's ``A_floor`` behavior.

    ``A_floor=None`` disables the clamp for exact checkpoint compatibility.
    """
    if A_floor is None:
        return A
    if not math.isfinite(A_floor) or A_floor <= 0:
        raise ValueError(
            f"A_floor must be a positive finite number or None, got {A_floor}"
        )
    return torch.clamp(A, max=-A_floor)

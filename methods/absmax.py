from __future__ import annotations

import torch
from core.base import BaseQuantizer

class AbsMaxQuantizer(BaseQuantizer):
    """
    Symmetric weight-only quantization.

    Each group uses one scale from the absolute maximum value and stores signed
    integer codes. The testbed immediately reconstructs FP32 weights so the
    forward pass stays CPU-friendly and comparable across methods.
    """
    method_name = "absmax"

    def __init__(self, bits: int = 8, block_size: int = 64, granularity: str = "layer"):
        super().__init__(bits=bits, block_size=block_size, granularity=granularity)

    def _quantize_impl(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self.granularity == "global" or self.granularity == "layer":
            scale = torch.clamp(torch.max(torch.abs(tensor)) / max(self.qmax, 1), min=1e-12)
            q = torch.clamp(torch.round(tensor / scale), self.qmin, self.qmax)
            return q * scale, {"scale_count": 1, "bpw": self.bits}

        if self.granularity != "block":
            raise ValueError(f"Unsupported granularity for AbsMax: {self.granularity}")

        flat = tensor.flatten()
        out = torch.empty_like(flat)
        scale_count = 0
        for start in range(0, flat.numel(), self.block_size):
            block = flat[start : start + self.block_size]
            scale = torch.clamp(torch.max(torch.abs(block)) / max(self.qmax, 1), min=1e-12)
            q = torch.clamp(torch.round(block / scale), self.qmin, self.qmax)
            out[start : start + self.block_size] = q * scale
            scale_count += 1

        # Approximate BPW includes one FP32 scale per block.
        bpw = self.bits + (32.0 * scale_count / max(flat.numel(), 1))
        return out.reshape_as(tensor), {"scale_count": scale_count, "bpw": bpw}

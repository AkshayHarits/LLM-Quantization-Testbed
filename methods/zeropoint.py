from __future__ import annotations

import torch
from core.base import BaseQuantizer

class ZeroPointQuantizer(BaseQuantizer):
    """
    Asymmetric min-max weight-only quantization with a zero point.
    """
    method_name = "zeropoint"

    def __init__(self, bits: int = 8, block_size: int = 64, granularity: str = "layer"):
        super().__init__(bits=bits, block_size=block_size, granularity=granularity)
        self.qmin = 0
        self.qmax = (2**self.bits) - 1

    def _quantize_block(self, block: torch.Tensor) -> torch.Tensor:
        t_min = torch.min(block)
        t_max = torch.max(block)
        scale = torch.clamp((t_max - t_min) / max(self.qmax - self.qmin, 1), min=1e-12)
        zero_point = torch.clamp(torch.round(self.qmin - t_min / scale), self.qmin, self.qmax)
        q = torch.clamp(torch.round(block / scale + zero_point), self.qmin, self.qmax)
        return (q - zero_point) * scale

    def _quantize_impl(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self.granularity == "global" or self.granularity == "layer":
            return self._quantize_block(tensor), {"scale_count": 1, "bpw": self.bits}

        if self.granularity != "block":
            raise ValueError(f"Unsupported granularity for ZeroPoint: {self.granularity}")

        flat = tensor.flatten()
        out = torch.empty_like(flat)
        scale_count = 0
        for start in range(0, flat.numel(), self.block_size):
            out[start : start + self.block_size] = self._quantize_block(flat[start : start + self.block_size])
            scale_count += 1

        # One FP32 scale plus one FP32 zero point per block in this simulation estimate.
        bpw = self.bits + (64.0 * scale_count / max(flat.numel(), 1))
        return out.reshape_as(tensor), {"scale_count": scale_count, "bpw": bpw}

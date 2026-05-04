from __future__ import annotations

import math

import torch
from core.base import BaseQuantizer


class MXFPQuantizer(BaseQuantizer):
    """
    CPU simulation of MXFP-style microscaling.

    Real MXFP formats use a shared exponent per block and compact floating
    mantissas. This implementation keeps the same experimental idea using a
    power-of-two block scale and signed low-bit integer mantissas, then
    reconstructs FP32 weights for ordinary CPU inference.
    """

    method_name = "mxfp"

    def __init__(self, bits: int = 4, block_size: int = 32, granularity: str = "block"):
        super().__init__(bits=bits, block_size=block_size, granularity=granularity)
        self.qmax = (2 ** (self.bits - 1)) - 1
        self.qmin = -(2 ** (self.bits - 1))

    def _quantize_block(self, block: torch.Tensor) -> torch.Tensor:
        absmax = torch.max(torch.abs(block))
        if absmax <= 0:
            return torch.zeros_like(block)
        raw_scale = absmax / max(self.qmax, 1)
        pow2_scale = 2.0 ** torch.floor(torch.log2(torch.clamp(raw_scale, min=1e-30)))
        q = torch.clamp(torch.round(block / pow2_scale), self.qmin, self.qmax)
        return q * pow2_scale

    def _quantize_impl(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self.granularity in ("global", "layer"):
            return self._quantize_block(tensor), {"scale_count": 1, "bpw": self.bits}

        if self.granularity != "block":
            raise ValueError(f"Unsupported granularity for MXFP: {self.granularity}")

        flat = tensor.flatten()
        out = torch.empty_like(flat)
        scale_count = 0
        for start in range(0, flat.numel(), self.block_size):
            out[start : start + self.block_size] = self._quantize_block(flat[start : start + self.block_size])
            scale_count += 1

        # Real MXFP stores compact scale metadata; 8 bits/block is a conservative estimate.
        bpw = self.bits + (8.0 * scale_count / max(flat.numel(), 1))
        return out.reshape_as(tensor), {"scale_count": scale_count, "bpw": bpw}

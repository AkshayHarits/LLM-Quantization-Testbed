from __future__ import annotations

import torch
from core.base import BaseQuantizer, flatten_weight_rows, restore_weight_rows


class GPTQQuantizer(BaseQuantizer):
    """
    Lightweight GPTQ-style CPU approximation.

    GPTQ uses second-order activation statistics and compensates later columns
    after each rounding decision. This implementation follows that principle for
    fair testbed comparisons, but omits the production GPU batching/kernels from
    the paper. If calibration Hessians are unavailable it falls back to symmetric
    per-row quantization and records that in metadata.
    """

    method_name = "gptq"

    def __init__(
        self,
        bits: int = 4,
        block_size: int = 128,
        granularity: str = "row",
        damp_percent: float = 0.01,
    ):
        super().__init__(bits=bits, block_size=block_size, granularity=granularity)
        self.damp_percent = damp_percent

    def _row_scales(self, rows: torch.Tensor) -> torch.Tensor:
        max_abs = torch.max(torch.abs(rows), dim=1, keepdim=True).values
        return torch.clamp(max_abs / max(self.qmax, 1), min=1e-12)

    def _round_with_scales(self, rows: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        q = torch.clamp(torch.round(rows / scales), self.qmin, self.qmax)
        return q * scales

    def _fallback_absmax(self, rows: torch.Tensor) -> torch.Tensor:
        scales = self._row_scales(rows)
        return self._round_with_scales(rows, scales)

    def _quantize_impl(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
        rows, original_shape = flatten_weight_rows(tensor)
        rows = rows.clone()
        out_features, in_features = rows.shape

        if self.hessian is None or self.hessian.shape[-1] != in_features:
            reconstructed = self._fallback_absmax(rows)
            return restore_weight_rows(reconstructed, original_shape), {
                "bpw": self.bits + 32.0 / max(in_features, 1),
                "gptq_fallback": True,
            }

        H = self.hessian.to(device=rows.device, dtype=torch.float32).clone()
        diag_mean = torch.mean(torch.diag(H)).clamp(min=1e-8)
        H = H + torch.eye(in_features, device=rows.device) * (self.damp_percent * diag_mean)

        try:
            chol = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(chol)
        except RuntimeError:
            H_inv = torch.linalg.pinv(H)

        scales = self._row_scales(rows)
        working = rows.clone()
        quantized = torch.zeros_like(working)

        for col_start in range(0, in_features, self.block_size):
            col_end = min(col_start + self.block_size, in_features)
            for i in range(col_start, col_end):
                q_col = self._round_with_scales(working[:, i : i + 1], scales).squeeze(1)
                quantized[:, i] = q_col
                denom = torch.clamp(H_inv[i, i], min=1e-8)
                err = (working[:, i] - q_col) / denom
                if i + 1 < in_features:
                    working[:, i + 1 :] -= err[:, None] * H_inv[i, i + 1 :][None, :]

        bpw = self.bits + 32.0 / max(in_features, 1)
        return restore_weight_rows(quantized, original_shape), {"bpw": bpw, "gptq_fallback": False}

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any

import torch


def flatten_weight_rows(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """
    View a weight tensor as rows while preserving the original shape.

    Linear weights are already [out_features, in_features]. Conv kernels become
    [out_channels, in_channels * kh * kw], which lets the same weight-only
    quantizers operate on CNN and transformer layers.
    """
    original_shape = tensor.shape
    if tensor.ndim == 1:
        return tensor.reshape(1, -1), original_shape
    return tensor.reshape(tensor.shape[0], -1), original_shape


def restore_weight_rows(rows: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return rows.reshape(shape)


def tensor_mse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    return torch.mean((reference.float() - estimate.float()) ** 2).item()


def tensor_snr_db(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    signal = torch.sum(reference.float() ** 2).item()
    noise = torch.sum((reference.float() - estimate.float()) ** 2).item()
    if noise <= 0.0:
        return float("inf")
    return 10.0 * math.log10(max(signal, 1e-30) / noise)


class BaseQuantizer(ABC):
    """
    Common interface for CPU-only fake weight quantization.

    The testbed intentionally performs:
        FP32 weight -> quantized representation -> reconstructed FP32 weight

    The reconstructed tensor is then used in an ordinary PyTorch forward pass.
    This isolates algorithmic quantization error, but it does not measure real
    packed-kernel latency or hardware speedup.
    """

    method_name = "base"

    def __init__(self, bits: int = 8, block_size: int = 64, granularity: str = "layer"):
        self.bits = int(bits)
        self.block_size = int(block_size)
        self.granularity = granularity
        self.qmax = (2 ** (self.bits - 1)) - 1
        self.qmin = -(2 ** (self.bits - 1))
        self.layer_name: str | None = None
        self.hessian: torch.Tensor | None = None
        self.x_var: float | None = None
        self.reports: list[dict[str, Any]] = []

    def set_layer_context(
        self,
        layer_name: str,
        hessian: torch.Tensor | None = None,
        x_var: float | None = None,
    ) -> None:
        self.layer_name = layer_name
        self.hessian = hessian
        self.x_var = x_var

    def reset_reports(self) -> None:
        self.reports = []

    def fake_quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        start = time.perf_counter()
        reconstructed, metadata = self._quantize_impl(tensor.float())
        runtime = time.perf_counter() - start

        reconstructed = reconstructed.to(dtype=tensor.dtype, device=tensor.device)
        metadata = dict(metadata)
        metadata.setdefault("simulated", True)
        metadata.setdefault("method", self.method_name)
        metadata.setdefault("bits", self.bits)
        metadata.setdefault("block_size", self.block_size)
        metadata.setdefault("granularity", self.granularity)
        metadata["quant_runtime_sec"] = runtime

        report = {
            "layer": self.layer_name or "<unknown>",
            "method": self.method_name,
            "bits": self.bits,
            "num_weights": tensor.numel(),
            "mse": tensor_mse(tensor, reconstructed),
            "snr_db": tensor_snr_db(tensor, reconstructed),
            "bpw": float(metadata.get("bpw", self.bits)),
            "compression_ratio": 32.0 / max(float(metadata.get("bpw", self.bits)), 1e-9),
            "quant_runtime_sec": runtime,
            "overload_rate": float(metadata.get("overload_rate", 0.0)),
        }
        for key in (
            "num_scales",
            "k_utilization",
            "rescued_rate",
            "dp_error",
            "greedy_error",
            "dp_vs_greedy_error_reduction",
            "selected_betas",
        ):
            if key in metadata:
                report[key] = metadata[key]
        self.reports.append(report)
        return reconstructed, metadata

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.fake_quantize(tensor)

    def dequantize(self, q_tensor: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
        return q_tensor

    @abstractmethod
    def _quantize_impl(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        pass

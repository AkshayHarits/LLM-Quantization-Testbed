from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from core.base import BaseQuantizer


def is_hf_conv1d(module: nn.Module) -> bool:
    return module.__class__.__name__ == "Conv1D" and hasattr(module, "weight")


def is_quantizable_weight_module(module: nn.Module, quantize_conv: bool = True) -> bool:
    return (
        isinstance(module, nn.Linear)
        or (quantize_conv and isinstance(module, nn.Conv2d))
        or is_hf_conv1d(module)
    )


def clone_model(model: nn.Module) -> nn.Module:
    return deepcopy(model)


def snapshot_weight_tensors(module: nn.Module, skip_layers: list[str] | None = None) -> dict[str, torch.Tensor]:
    skip_layers = skip_layers or []
    weights: dict[str, torch.Tensor] = {}
    for name, child in module.named_modules():
        if not name or any(skip in name for skip in skip_layers):
            continue
        if is_quantizable_weight_module(child):
            weights[name] = child.weight.detach().float().cpu().clone()
    return weights


def quantize_model_weights(
    module: nn.Module,
    quantizer: BaseQuantizer,
    skip_layers: list[str] | None = None,
    hessian_dict: dict[str, torch.Tensor] | None = None,
    x_var_dict: dict[str, float] | None = None,
    quantize_conv: bool = True,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """
    In-place CPU fake weight quantization.

    Each eligible Linear/Conv2d weight is quantized and reconstructed once.
    Biases are left untouched. The module class is not replaced, so normal
    PyTorch CPU forward passes remain fast and transparent.
    """
    skip_layers = skip_layers or ["lm_head"]
    quantizer.reset_reports()

    for name, child in module.named_modules():
        if not name or any(skip in name for skip in skip_layers):
            continue
        eligible = is_quantizable_weight_module(child, quantize_conv=quantize_conv)
        if not eligible:
            continue

        hessian = hessian_dict.get(name) if hessian_dict else None
        x_var = x_var_dict.get(name) if x_var_dict else None
        quantizer.set_layer_context(name, hessian=hessian, x_var=x_var)

        if verbose:
            print(f"Quantizing {name} with {quantizer.method_name}...")
        with torch.no_grad():
            if is_hf_conv1d(child):
                # Hugging Face GPT-2 Conv1D stores weights as [in_features, out_features].
                # Quantizers expect Linear-style rows [out_features, in_features].
                source = child.weight.data.t().contiguous()
                q_weight, _ = quantizer.quantize(source)
                child.weight.data.copy_(q_weight.t().to(dtype=child.weight.dtype, device=child.weight.device))
            else:
                q_weight, _ = quantizer.quantize(child.weight.data)
                child.weight.data.copy_(q_weight.to(dtype=child.weight.dtype, device=child.weight.device))

    return module, list(quantizer.reports)


def swap_linear_layers(
    module: nn.Module,
    quantizer: BaseQuantizer,
    skip_layers: list[str] | None = None,
    hessian_dict: dict[str, torch.Tensor] | None = None,
    x_var_dict: dict[str, float] | None = None,
    name_prefix: str = "",
) -> nn.Module:
    """
    Backward-compatible alias for the old API.
    """
    quantize_model_weights(module, quantizer, skip_layers, hessian_dict, x_var_dict)
    return module

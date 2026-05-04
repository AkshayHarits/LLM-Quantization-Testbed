from __future__ import annotations

import os

import torch
import torch.nn as nn
from datasets import load_dataset


def is_hf_conv1d(module: nn.Module) -> bool:
    return module.__class__.__name__ == "Conv1D" and hasattr(module, "weight")


def get_calibration_data(
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    num_samples: int = 128,
    seq_len: int = 512,
    seed: int = 1234,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Build a fixed set of token blocks for LLM calibration.
    """
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text_blocks = [s["text"] for s in dataset if len(s["text"].strip()) > 0]
    combined_text = " ".join(text_blocks)
    enc = tokenizer(combined_text, return_tensors="pt")
    ids = enc.input_ids[0]

    max_start = max(ids.numel() - seq_len - 1, 1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    starts = torch.randint(0, max_start, (num_samples,), generator=generator)
    samples = torch.stack([ids[start : start + seq_len] for start in starts], dim=0)
    return samples.to(device)


def capture_statistics(
    model: nn.Module,
    input_ids: torch.Tensor,
    cache_path: str | None = None,
    use_cache: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """
    Capture uncentered activation covariance H = E[XX^T] for each Linear layer.

    These Hessians are used by GPTQ and the NestQuant + LDLQ ablation. Conv2d
    Hessians are not captured here; CNN conv layers use weight-only metrics.
    """
    if use_cache and cache_path and os.path.exists(cache_path):
        payload = torch.load(cache_path, map_location="cpu")
        return payload["hessians"], payload["x_vars"]

    hessians: dict[str, torch.Tensor] = {}
    x_vars: dict[str, float] = {}
    counts: dict[str, int] = {}
    hooks = []

    def get_hook(name: str):
        def hook(module, inputs, output):
            x = inputs[0].detach().float()
            x = x.reshape(-1, x.shape[-1]).cpu()
            n_tokens = x.shape[0]
            h_batch = (x.t() @ x) / max(n_tokens, 1)
            var_batch = torch.var(x, dim=0, unbiased=False).mean().item()

            if name not in hessians:
                hessians[name] = h_batch
                x_vars[name] = var_batch
                counts[name] = 1
            else:
                counts[name] += 1
                alpha = 1.0 / counts[name]
                hessians[name] = (1.0 - alpha) * hessians[name] + alpha * h_batch
                x_vars[name] = (1.0 - alpha) * x_vars[name] + alpha * var_batch

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or is_hf_conv1d(module):
            hooks.append(module.register_forward_hook(get_hook(name)))

    model.eval()
    with torch.no_grad():
        for sample in input_ids:
            batch = sample.unsqueeze(0).to(next(model.parameters()).device)
            try:
                model(input_ids=batch, labels=batch)
            except TypeError:
                model(input_ids=batch)

    for hook in hooks:
        hook.remove()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"hessians": hessians, "x_vars": x_vars}, cache_path)

    return hessians, x_vars

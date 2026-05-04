from __future__ import annotations

import math
from typing import Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append((correct_k / target.numel()).item())
    return results


def evaluate_cnn_accuracy(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device = torch.device("cpu"),
    max_samples: int | None = None,
) -> dict[str, float]:
    model.eval()
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating CNN"):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))
            batch = labels.numel()
            top1_sum += top1 * batch
            top5_sum += top5 * batch
            total += batch
            if max_samples is not None and total >= max_samples:
                break
    total = max(total, 1)
    return {"top1": top1_sum / total, "top5": top5_sum / total}


def calculate_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "validation",
    seq_len: int = 512,
    num_samples: int = 100,
    device: torch.device = torch.device("cpu"),
) -> float:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text_blocks = [s["text"] for s in dataset if len(s["text"].strip()) > 0]
    combined_text = " ".join(text_blocks)
    encodings = tokenizer(combined_text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    nsamples = min(input_ids.numel() // seq_len, num_samples)
    if nsamples <= 0:
        raise ValueError("Not enough tokens to compute perplexity with the requested seq_len.")

    nlls = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(nsamples), desc="Evaluating PPL"):
            chunk = input_ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)
            outputs = model(input_ids=chunk, labels=chunk)
            nlls.append(outputs.loss.detach().float().item())

    return float(torch.exp(torch.tensor(sum(nlls) / len(nlls))).item())


def aggregate_layer_reports(reports: list[dict]) -> dict[str, float]:
    if not reports:
        return {}
    total_weights = sum(r["num_weights"] for r in reports)
    weighted_mse = sum(r["mse"] * r["num_weights"] for r in reports) / max(total_weights, 1)
    weighted_bpw = sum(r["bpw"] * r["num_weights"] for r in reports) / max(total_weights, 1)
    runtime = sum(r.get("quant_runtime_sec", 0.0) for r in reports)
    signal_noise = [r["snr_db"] for r in reports if math.isfinite(r["snr_db"])]
    return {
        "weight_mse": weighted_mse,
        "snr_db_mean": sum(signal_noise) / max(len(signal_noise), 1),
        "bpw": weighted_bpw,
        "compression_ratio": 32.0 / max(weighted_bpw, 1e-9),
        "quant_runtime_sec": runtime,
        "overload_rate_mean": sum(r.get("overload_rate", 0.0) for r in reports) / len(reports),
    }

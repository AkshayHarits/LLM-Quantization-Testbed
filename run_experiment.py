from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from core.surgery import clone_model, quantize_model_weights
from eval.metrics import aggregate_layer_reports, calculate_perplexity, evaluate_cnn_accuracy
from eval.plot import plot_beta_distribution, plot_layer_sensitivity, plot_metric_vs_bits, plot_pareto
from methods.absmax import AbsMaxQuantizer
from methods.gptq import GPTQQuantizer
from methods.mxfp import MXFPQuantizer
from methods.nestquant.calibration import capture_statistics, get_calibration_data
from methods.nestquant.quantizer import NestQuantizer
from methods.zeropoint import ZeroPointQuantizer


@dataclass
class ExperimentConfig:
    task: str = "llm"
    model: str = "gpt2"
    methods: str = "fp32,absmax,zeropoint,mxfp,gptq,nestquant,nestquant_ldlq"
    bits: int = 4
    block_size: int = 64
    granularity: str = "block"
    k: int = 4
    m: int = 32
    grid_type: str = "log"
    seq_len: int = 512
    calib_samples: int = 128
    eval_samples: int = 100
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    dataset_split: str = "validation"
    cnn_model: str = "resnet18"
    cnn_dataset: str = "imagenet"
    cnn_data_dir: str | None = None
    download_cnn_data: bool = True
    cnn_checkpoint: str | None = None
    cnn_pretrained: bool = True
    cnn_pretrained_repo: str = "Phoenix21/resnet18-cifar10-baseline"
    cnn_pretrained_file: str = "resnet18_cifar10_baseline.pth"
    cnn_cifar_resnet_stem: str = "auto"
    train_cnn: bool = False
    cnn_epochs: int = 10
    cnn_lr: float = 0.01
    cnn_momentum: float = 0.9
    cnn_weight_decay: float = 0.0005
    cnn_train_samples: int = 0
    batch_size: int = 32
    seed: int = 1234
    output_dir: str = "outputs"
    skip_layers: str = "lm_head"
    cache_dir: str = "outputs/cache"
    quantize_conv: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="CPU fake weight-only quantization testbed")
    for field, value in ExperimentConfig().__dict__.items():
        arg = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=value)
        elif value is None:
            parser.add_argument(arg, default=value)
        else:
            parser.add_argument(arg, type=type(value), default=value)
    return ExperimentConfig(**vars(parser.parse_args()))


def build_quantizer(method: str, cfg: ExperimentConfig):
    if method == "absmax":
        return AbsMaxQuantizer(bits=cfg.bits, block_size=cfg.block_size, granularity=cfg.granularity)
    if method == "zeropoint":
        return ZeroPointQuantizer(bits=cfg.bits, block_size=cfg.block_size, granularity=cfg.granularity)
    if method == "mxfp":
        return MXFPQuantizer(bits=cfg.bits, block_size=cfg.block_size, granularity="block")
    if method == "gptq":
        return GPTQQuantizer(bits=cfg.bits, block_size=max(cfg.block_size, 1))
    if method == "nestquant":
        return NestQuantizer(bits=cfg.bits, k=cfg.k, m=cfg.m, grid_type=cfg.grid_type, block_size=cfg.block_size)
    if method == "nestquant_ldlq":
        q = NestQuantizer(bits=cfg.bits, k=cfg.k, m=cfg.m, grid_type=cfg.grid_type, block_size=cfg.block_size, use_ldlq=True)
        q.method_name = "nestquant_ldlq"
        return q
    raise ValueError(f"Unknown method: {method}")


def load_llm(cfg: ExperimentConfig, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model).to(device)
    return model, tokenizer


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    def _is_tensor_dict(obj: Any) -> bool:
        return isinstance(obj, Mapping) and len(obj) > 0 and all(torch.is_tensor(v) for v in obj.values())

    def _find_tensor_dict(obj: Any) -> dict[str, torch.Tensor] | None:
        if _is_tensor_dict(obj):
            return dict(obj)
        if isinstance(obj, Mapping):
            for preferred in ("state_dict", "model_state_dict", "model", "net", "weights", "ema"):
                if preferred in obj:
                    found = _find_tensor_dict(obj[preferred])
                    if found is not None:
                        return found
            for value in obj.values():
                found = _find_tensor_dict(value)
                if found is not None:
                    return found
        return None

    payload = _find_tensor_dict(payload)
    if payload is None:
        raise ValueError("Checkpoint is not a state-dict-like object.")
    if payload and all(isinstance(k, str) and k.startswith("module.") for k in payload.keys()):
        payload = {k[len("module.") :]: v for k, v in payload.items()}
    return payload


def load_checkpoint_with_match_guard(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    model_state = model.state_dict()
    matched = 0
    total = len(model_state)

    for key, tensor in model_state.items():
        if key in state_dict and torch.is_tensor(state_dict[key]) and state_dict[key].shape == tensor.shape:
            matched += 1

    match_ratio = matched / max(total, 1)
    if match_ratio < 0.75:
        raise RuntimeError(
            f"Checkpoint/model mismatch is too large (matched {matched}/{total} = {match_ratio:.1%}). "
            f"checkpoint={checkpoint_path}"
        )

    compatible_state_dict = {
        key: tensor
        for key, tensor in state_dict.items()
        if key in model_state and torch.is_tensor(tensor) and tensor.shape == model_state[key].shape
    }
    load_result = model.load_state_dict(compatible_state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            f"Checkpoint loaded with non-strict key match. missing={len(load_result.missing_keys)}, "
            f"unexpected={len(load_result.unexpected_keys)}, matched_ratio={match_ratio:.1%}"
        )
    else:
        print(f"Checkpoint loaded with full key match. matched_ratio={match_ratio:.1%}")


def infer_resnet_stem_from_checkpoint(checkpoint_path: str, device: torch.device) -> str | None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = _extract_state_dict(checkpoint)
        conv1 = state_dict.get("conv1.weight")
        if torch.is_tensor(conv1) and conv1.ndim == 4:
            k = conv1.shape[-1]
            if k == 7:
                return "imagenet"
            if k == 3:
                return "cifar"
    except Exception:
        return None
    return None


def maybe_download_pretrained_cifar10_checkpoint(cfg: ExperimentConfig) -> str | None:
    if not cfg.cnn_pretrained or cfg.cnn_dataset.lower() != "cifar10":
        return None
    if cfg.cnn_checkpoint:
        return cfg.cnn_checkpoint
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except Exception as exc:
        print(f"Warning: huggingface_hub unavailable for checkpoint download: {exc}")
        return None
    try:
        target_dir = os.path.join(cfg.cache_dir, "hf_cnn_checkpoints")
        os.makedirs(target_dir, exist_ok=True)
        if cfg.cnn_pretrained_file:
            path = hf_hub_download(
                repo_id=cfg.cnn_pretrained_repo,
                filename=cfg.cnn_pretrained_file,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )
            print(f"Loaded pretrained checkpoint from {path}")
            return path
        repo_dir = snapshot_download(repo_id=cfg.cnn_pretrained_repo, local_dir=target_dir, local_dir_use_symlinks=False)
        candidates = []
        for root, _, files in os.walk(repo_dir):
            for name in files:
                if name.lower().endswith((".pth", ".pt", ".bin")):
                    candidates.append(os.path.join(root, name))
        if not candidates:
            print(f"Warning: no checkpoint file found in repo {cfg.cnn_pretrained_repo}")
            return None
        candidates.sort()
        print(f"Loaded pretrained checkpoint from {candidates[0]}")
        return candidates[0]
    except Exception as exc:
        print(f"Warning: failed to download checkpoint from {cfg.cnn_pretrained_repo}: {exc}")
        return None


def get_cifar10_normalize(cfg: ExperimentConfig):
    from torchvision import transforms

    if cfg.cnn_model.lower() == "resnet18":
        return transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))


def build_cnn_model(cfg: ExperimentConfig, stem_mode: str | None = None):
    import torch.nn as nn
    from torchvision import models

    cnn_model = cfg.cnn_model.lower()
    if cnn_model == "alexnet":
        if cfg.cnn_dataset.lower() == "imagenet":
            return models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if cfg.cnn_pretrained else None)
        return models.alexnet(weights=None, num_classes=10)
    if cnn_model == "resnet18":
        if cfg.cnn_dataset.lower() == "imagenet":
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if cfg.cnn_pretrained else None)
        if stem_mode is None:
            stem_mode = "cifar"
        model = models.resnet18(weights=None)
        if stem_mode == "cifar":
            # CIFAR-style ResNet stem.
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        return model
    raise ValueError("--cnn-model must be either 'alexnet' or 'resnet18'.")


def load_cnn_model_and_loader(cfg: ExperimentConfig, device: torch.device):
    from torchvision import datasets, transforms
    from torchvision.models import AlexNet_Weights, ResNet18_Weights

    cnn_dataset = cfg.cnn_dataset.lower()
    cnn_model = cfg.cnn_model.lower()
    data_root = cfg.cnn_data_dir or os.path.join(cfg.output_dir, "data")
    stem_mode = "imagenet"
    if cnn_dataset == "cifar10" and cnn_model == "resnet18":
        stem_setting = cfg.cnn_cifar_resnet_stem.lower()
        if stem_setting == "auto":
            checkpoint_path = cfg.cnn_checkpoint or maybe_download_pretrained_cifar10_checkpoint(cfg)
            if checkpoint_path and cfg.cnn_checkpoint is None:
                cfg.cnn_checkpoint = checkpoint_path
            inferred = infer_resnet_stem_from_checkpoint(cfg.cnn_checkpoint, device) if cfg.cnn_checkpoint else None
            stem_mode = inferred or "cifar"
        elif stem_setting in ("cifar", "imagenet"):
            stem_mode = stem_setting
        else:
            raise ValueError("--cnn-cifar-resnet-stem must be one of: auto, cifar, imagenet")
    model = build_cnn_model(cfg, stem_mode=stem_mode).to(device)

    if cnn_dataset == "imagenet":
        if not cfg.cnn_data_dir:
            raise ValueError("ImageNet evaluation requires --cnn-data-dir pointing to an ImageNet-style validation folder.")
        if cnn_model == "alexnet" and cfg.cnn_pretrained:
            transform = AlexNet_Weights.IMAGENET1K_V1.transforms()
        elif cnn_model == "resnet18" and cfg.cnn_pretrained:
            transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        dataset = datasets.ImageFolder(cfg.cnn_data_dir, transform=transform)
    elif cnn_dataset == "cifar10":
        normalize = get_cifar10_normalize(cfg)
        if cnn_model == "alexnet":
            transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
        else:
            transform = transforms.Compose([transforms.ToTensor(), normalize])
        dataset = datasets.CIFAR10(root=data_root, train=False, download=cfg.download_cnn_data, transform=transform)
        if cfg.cnn_checkpoint is None:
            cfg.cnn_checkpoint = maybe_download_pretrained_cifar10_checkpoint(cfg)
        if cfg.cnn_checkpoint is None and cfg.cnn_pretrained:
            print(
                "Warning: no pretrained CIFAR-10 checkpoint available. "
                "Model is randomly initialized; run with --train-cnn to train first."
            )
    else:
        raise ValueError("--cnn-dataset must be either 'imagenet' or 'cifar10'.")

    if cfg.cnn_checkpoint:
        load_checkpoint_with_match_guard(model, cfg.cnn_checkpoint, device)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return model, dataloader


def get_cifar10_loaders(cfg: ExperimentConfig):
    from torchvision import datasets, transforms

    data_root = cfg.cnn_data_dir or os.path.join(cfg.output_dir, "data")
    normalize = get_cifar10_normalize(cfg)
    stem_setting = cfg.cnn_cifar_resnet_stem.lower()
    use_imagenet_geometry = cfg.cnn_model.lower() == "alexnet"

    if use_imagenet_geometry:
        train_transform = transforms.Compose(
            [transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
        )
        eval_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
    else:
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        eval_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = datasets.CIFAR10(root=data_root, train=True, download=cfg.download_cnn_data, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=cfg.download_cnn_data, transform=eval_transform)
    if cfg.cnn_train_samples and cfg.cnn_train_samples > 0:
        train_set = torch.utils.data.Subset(train_set, list(range(min(cfg.cnn_train_samples, len(train_set)))))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def default_cifar_checkpoint_path(cfg: ExperimentConfig) -> str:
    return os.path.join(cfg.output_dir, "checkpoints", f"{cfg.cnn_model.lower()}_cifar10.pt")


def train_cnn_cifar10(cfg: ExperimentConfig, device: torch.device) -> str:
    checkpoint_path = cfg.cnn_checkpoint or default_cifar_checkpoint_path(cfg)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model = build_cnn_model(cfg).to(device)
    train_loader, test_loader = get_cifar10_loaders(cfg)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.cnn_lr,
        momentum=cfg.cnn_momentum,
        weight_decay=cfg.cnn_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.cnn_epochs, 1))

    best_top1 = -1.0
    for epoch in range(cfg.cnn_epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.numel()
            total += labels.numel()
        scheduler.step()

        metrics = evaluate_cnn_accuracy(model, test_loader, device=device, max_samples=cfg.eval_samples)
        avg_loss = running_loss / max(total, 1)
        print(
            f"Epoch {epoch + 1}/{cfg.cnn_epochs} "
            f"loss={avg_loss:.4f} top1={metrics['top1']:.4f} top5={metrics['top5']:.4f}"
        )
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            torch.save({"state_dict": model.state_dict(), "top1": best_top1, "config": asdict(cfg)}, checkpoint_path)

    print(f"Saved CIFAR-10 {cfg.cnn_model} checkpoint to {checkpoint_path}")
    return checkpoint_path


def save_rows(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def generate_plots(cfg: ExperimentConfig, task: str) -> None:
    results_csv = os.path.join(cfg.output_dir, "results.csv")
    layer_csv = os.path.join(cfg.output_dir, "layer_metrics.csv")
    plot_dir = os.path.join(cfg.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    try:
        if task == "llm":
            plot_metric_vs_bits(results_csv, "perplexity", os.path.join(plot_dir, "perplexity_vs_bits.png"))
            plot_pareto(results_csv, "perplexity", os.path.join(plot_dir, "perplexity_vs_compression.png"))
        else:
            plot_metric_vs_bits(results_csv, "top1", os.path.join(plot_dir, "top1_vs_bits.png"))
            plot_metric_vs_bits(results_csv, "top5", os.path.join(plot_dir, "top5_vs_bits.png"))
            plot_pareto(results_csv, "top1", os.path.join(plot_dir, "top1_vs_compression.png"))

        if os.path.exists(layer_csv):
            plot_layer_sensitivity(layer_csv, os.path.join(plot_dir, "layer_mse_sensitivity.png"))
            plot_beta_distribution(layer_csv, os.path.join(plot_dir, "nestquant_beta_distribution.png"))
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")


def write_config(cfg: ExperimentConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def run_llm(cfg: ExperimentConfig, device: torch.device) -> None:
    model, tokenizer = load_llm(cfg, device)
    methods = [m.strip().lower() for m in cfg.methods.split(",") if m.strip()]
    skip_layers = [s.strip() for s in cfg.skip_layers.split(",") if s.strip()]

    needs_hessian = any(m in {"gptq", "nestquant_ldlq"} for m in methods)
    hessians = {}
    x_vars = {}
    if needs_hessian:
        calib_ids = get_calibration_data(
            tokenizer,
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            num_samples=cfg.calib_samples,
            seq_len=cfg.seq_len,
            seed=cfg.seed,
            device=device,
        )
        cache_path = os.path.join(cfg.cache_dir, f"{cfg.model.replace('/', '_')}_hessian_{cfg.calib_samples}_{cfg.seq_len}.pt")
        hessians, x_vars = capture_statistics(model, calib_ids, cache_path=cache_path)

    baseline_ppl = calculate_perplexity(
        model,
        tokenizer,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=cfg.dataset_split,
        seq_len=cfg.seq_len,
        num_samples=cfg.eval_samples,
        device=device,
    )

    result_rows = [
        {
            "task": "llm",
            "model": cfg.model,
            "method": "fp32",
            "bits": 32,
            "perplexity": baseline_ppl,
            "ppl_ratio": 1.0,
            "loss_increase": 0.0,
            "compression_ratio": 1.0,
            "bpw": 32.0,
        }
    ]
    layer_rows: list[dict[str, Any]] = []

    for method in methods:
        if method == "fp32":
            continue
        q_model = clone_model(model).to(device)
        quantizer = build_quantizer(method, cfg)
        q_model, reports = quantize_model_weights(q_model, quantizer, skip_layers=skip_layers, hessian_dict=hessians, x_var_dict=x_vars)
        if not reports:
            raise RuntimeError(f"No layers were quantized for method '{method}'. Check model module types and skip_layers.")
        q_ppl = calculate_perplexity(
            q_model,
            tokenizer,
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            split=cfg.dataset_split,
            seq_len=cfg.seq_len,
            num_samples=cfg.eval_samples,
            device=device,
        )
        summary = aggregate_layer_reports(reports)
        result_rows.append(
            {
                "task": "llm",
                "model": cfg.model,
                "method": method,
                "bits": cfg.bits,
                "perplexity": q_ppl,
                "ppl_ratio": q_ppl / baseline_ppl,
                "loss_increase": float(torch.log(torch.tensor(q_ppl / baseline_ppl)).item()),
                **summary,
            }
        )
        for report in reports:
            layer_rows.append({"task": "llm", "model": cfg.model, **report})

    save_rows(os.path.join(cfg.output_dir, "results.csv"), result_rows)
    save_rows(os.path.join(cfg.output_dir, "layer_metrics.csv"), layer_rows)
    generate_plots(cfg, "llm")


def run_cnn(cfg: ExperimentConfig, device: torch.device) -> None:
    if cfg.train_cnn:
        if cfg.cnn_dataset.lower() != "cifar10":
            raise ValueError("--train-cnn is currently supported for --cnn-dataset cifar10 only.")
        checkpoint_path = train_cnn_cifar10(cfg, device)
        cfg.cnn_checkpoint = checkpoint_path
        write_config(cfg)

    model, dataloader = load_cnn_model_and_loader(cfg, device)
    methods = [m.strip().lower() for m in cfg.methods.split(",") if m.strip()]
    skip_layers = [s.strip() for s in cfg.skip_layers.split(",") if s.strip()]
    model_label = f"{cfg.cnn_model.lower()}_{cfg.cnn_dataset.lower()}"

    baseline = evaluate_cnn_accuracy(model, dataloader, device=device, max_samples=cfg.eval_samples)
    result_rows = [
        {
            "task": "cnn",
            "model": model_label,
            "method": "fp32",
            "bits": 32,
            "top1": baseline["top1"],
            "top5": baseline["top5"],
            "top1_drop": 0.0,
            "top5_drop": 0.0,
            "compression_ratio": 1.0,
            "bpw": 32.0,
        }
    ]
    layer_rows: list[dict[str, Any]] = []

    for method in methods:
        if method == "fp32":
            continue
        q_model = clone_model(model).to(device)
        quantizer = build_quantizer(method, cfg)
        q_model, reports = quantize_model_weights(q_model, quantizer, skip_layers=skip_layers, quantize_conv=cfg.quantize_conv)
        if not reports:
            raise RuntimeError(f"No layers were quantized for method '{method}'. Check model module types and skip_layers.")
        metrics = evaluate_cnn_accuracy(q_model, dataloader, device=device, max_samples=cfg.eval_samples)
        summary = aggregate_layer_reports(reports)
        result_rows.append(
            {
                "task": "cnn",
                "model": model_label,
                "method": method,
                "bits": cfg.bits,
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "top1_drop": baseline["top1"] - metrics["top1"],
                "top5_drop": baseline["top5"] - metrics["top5"],
                **summary,
            }
        )
        for report in reports:
            layer_rows.append({"task": "cnn", "model": model_label, **report})

    save_rows(os.path.join(cfg.output_dir, "results.csv"), result_rows)
    save_rows(os.path.join(cfg.output_dir, "layer_metrics.csv"), layer_rows)
    generate_plots(cfg, "cnn")


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    write_config(cfg)
    device = torch.device("cpu")
    print("CPU-only fake quantization: weights are quantized, reconstructed to FP32, then evaluated with normal PyTorch CPU kernels.")
    print(json.dumps(asdict(cfg), indent=2))

    if cfg.task == "llm":
        run_llm(cfg, device)
    elif cfg.task == "cnn":
        run_cnn(cfg, device)
    else:
        raise ValueError("--task must be either llm or cnn")

    print(f"Saved results to {cfg.output_dir}")


if __name__ == "__main__":
    main()

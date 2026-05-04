from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def _save(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_metric_vs_bits(results_csv: str, metric: str, output_path: str) -> None:
    df = pd.read_csv(results_csv)
    plt.figure(figsize=(7, 4.5))
    sns.lineplot(data=df, x="bits", y=metric, hue="method", marker="o")
    plt.xlabel("Bits per weight code")
    plt.ylabel(metric)
    _save(output_path)


def plot_pareto(results_csv: str, performance_metric: str, output_path: str) -> None:
    df = pd.read_csv(results_csv)
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(data=df, x="compression_ratio", y=performance_metric, hue="method", style="method", s=80)
    plt.xlabel("Estimated compression ratio vs FP32")
    plt.ylabel(performance_metric)
    _save(output_path)


def plot_layer_sensitivity(layer_csv: str, output_path: str, top_n: int = 25) -> None:
    df = pd.read_csv(layer_csv).sort_values("mse", ascending=False).head(top_n)
    plt.figure(figsize=(8, max(4, top_n * 0.22)))
    sns.barplot(data=df, y="layer", x="mse", hue="method", dodge=False)
    plt.xlabel("Per-layer weight MSE")
    plt.ylabel("")
    _save(output_path)


def plot_beta_distribution(layer_csv: str, output_path: str) -> None:
    df = pd.read_csv(layer_csv)
    if "selected_betas" not in df.columns:
        raise ValueError("No selected_betas column found. Run NestQuant first.")
    betas = []
    for raw in df["selected_betas"].dropna():
        for token in str(raw).strip("[]").split(","):
            token = token.strip()
            if token:
                betas.append(float(token))
    plt.figure(figsize=(7, 4.5))
    sns.histplot(betas, bins=30)
    plt.xlabel("Selected beta")
    plt.ylabel("Count")
    _save(output_path)


def plot_weight_error(original_weights: torch.Tensor, quantized_weights: torch.Tensor, layer_name: str, output_dir: str = "outputs/plots"):
    real = original_weights.flatten().float()
    approx = quantized_weights.flatten().float()
    relative_error = (approx - real).abs() / (real.abs() + 1e-5)
    relative_error = relative_error[relative_error > 1e-8]
    log_relative_error = torch.log10(relative_error + 1e-10)

    plt.figure(figsize=(8, 5))
    sns.histplot(log_relative_error.detach().cpu().numpy(), bins=100, kde=True)
    plt.xlabel("log10 relative error")
    plt.ylabel("Frequency")
    plt.title(f"Quantization error: {layer_name}")
    filename = f"error_plot_{layer_name.replace('.', '_')}.png"
    _save(os.path.join(output_dir, filename))


def plot_weight_distribution(original_weights: torch.Tensor, quantized_weights: torch.Tensor, layer_name: str, output_dir: str = "outputs/plots"):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(original_weights.flatten().detach().cpu().numpy(), label="FP32", fill=True, alpha=0.25)
    sns.kdeplot(quantized_weights.flatten().detach().cpu().numpy(), label="Quantized", linestyle="--")
    plt.xlabel("Weight value")
    plt.ylabel("Density")
    plt.title(f"Weight distribution: {layer_name}")
    plt.legend()
    filename = f"dist_plot_{layer_name.replace('.', '_')}.png"
    _save(os.path.join(output_dir, filename))

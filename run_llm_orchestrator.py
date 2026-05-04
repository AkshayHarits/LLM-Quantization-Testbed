from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class OrchestratorConfig:
    model: str = "gpt2"
    output_dir: str = "outputs/llm_orchestrator"
    bits_list: str = "2,3,4,8"
    k_list: str = "1,2,4,8,16"
    m_list: str = "8,16,32,64,128"
    grid_types: str = "linear,log"
    methods_main: str = "fp32,absmax,zeropoint,mxfp,nestquant"
    methods_nest: str = "fp32,nestquant"
    include_gptq: bool = False
    include_nest_ldlq: bool = False
    seq_len: int = 512
    calib_samples: int = 128
    eval_samples: int = 100
    block_size: int = 64
    granularity: str = "block"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    dataset_split: str = "validation"
    seed: int = 1234
    cache_dir: str = "outputs/cache"
    dry_run: bool = False
    skip_existing: bool = True
    python_executable: str = sys.executable


def parse_args() -> OrchestratorConfig:
    parser = argparse.ArgumentParser(description="LLM sweep orchestrator for quantization experiments")
    for field, value in OrchestratorConfig().__dict__.items():
        arg = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(arg, type=type(value), default=value)
    return OrchestratorConfig(**vars(parser.parse_args()))


def parse_int_csv(raw: str) -> list[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def parse_str_csv(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compose_methods(cfg: OrchestratorConfig, nest_only: bool) -> str:
    if nest_only:
        methods = parse_str_csv(cfg.methods_nest)
        if cfg.include_nest_ldlq and "nestquant_ldlq" not in methods:
            methods.append("nestquant_ldlq")
        return ",".join(methods)

    methods = parse_str_csv(cfg.methods_main)
    if cfg.include_gptq and "gptq" not in methods:
        methods.append("gptq")
    return ",".join(methods)


def run_one(cfg: OrchestratorConfig, run_tag: str, overrides: dict[str, Any], nest_only: bool = False) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_experiment_path = os.path.join(script_dir, "run_experiment.py")
    run_output_dir = os.path.join(cfg.output_dir, "runs", run_tag)
    ensure_dir(run_output_dir)

    if cfg.skip_existing and os.path.exists(os.path.join(run_output_dir, "results.csv")):
        print(f"[skip] {run_tag} already exists")
        return run_output_dir

    methods = compose_methods(cfg, nest_only=nest_only)
    cmd = [
        cfg.python_executable,
        run_experiment_path,
        "--task",
        "llm",
        "--model",
        cfg.model,
        "--methods",
        methods,
        "--seq-len",
        str(cfg.seq_len),
        "--calib-samples",
        str(cfg.calib_samples),
        "--eval-samples",
        str(cfg.eval_samples),
        "--block-size",
        str(cfg.block_size),
        "--granularity",
        cfg.granularity,
        "--dataset-name",
        cfg.dataset_name,
        "--dataset-config",
        cfg.dataset_config,
        "--dataset-split",
        cfg.dataset_split,
        "--seed",
        str(cfg.seed),
        "--cache-dir",
        cfg.cache_dir,
        "--output-dir",
        run_output_dir,
    ]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        cmd.extend([flag, str(value)])

    print("[run]", " ".join(cmd))
    if not cfg.dry_run:
        subprocess.run(cmd, check=True)
    return run_output_dir


def load_run_results(run_dir: str, sweep_name: str, sweep_value: str, params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_path = os.path.join(run_dir, "results.csv")
    layer_path = os.path.join(run_dir, "layer_metrics.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing results.csv in {run_dir}")

    results = pd.read_csv(results_path)
    results["sweep_name"] = sweep_name
    results["sweep_value"] = sweep_value
    results["run_dir"] = run_dir
    for key, value in params.items():
        results[key] = value

    if os.path.exists(layer_path):
        layers = pd.read_csv(layer_path)
        layers["sweep_name"] = sweep_name
        layers["sweep_value"] = sweep_value
        layers["run_dir"] = run_dir
        for key, value in params.items():
            layers[key] = value
    else:
        layers = pd.DataFrame()
    return results, layers


def save_master_tables(output_dir: str, results_frames: list[pd.DataFrame], layer_frames: list[pd.DataFrame]) -> tuple[str, str]:
    ensure_dir(output_dir)
    master_results = pd.concat(results_frames, ignore_index=True) if results_frames else pd.DataFrame()
    master_layers = pd.concat(layer_frames, ignore_index=True) if layer_frames else pd.DataFrame()

    results_path = os.path.join(output_dir, "master_results.csv")
    layers_path = os.path.join(output_dir, "master_layer_metrics.csv")
    master_results.to_csv(results_path, index=False)
    master_layers.to_csv(layers_path, index=False)
    return results_path, layers_path


def _save_plot(path: str) -> None:
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def generate_master_plots(results_csv: str, output_dir: str) -> None:
    df = pd.read_csv(results_csv)
    if df.empty:
        return

    plot_dir = os.path.join(output_dir, "plots")
    ensure_dir(plot_dir)

    bit_df = df[df["sweep_name"] == "bit_width"].copy()
    if not bit_df.empty:
        plt.figure(figsize=(7.5, 4.8))
        sns.lineplot(data=bit_df, x="bits", y="perplexity", hue="method", marker="o")
        plt.title("Perplexity vs Bit-Width")
        plt.xlabel("Bits")
        plt.ylabel("Perplexity")
        _save_plot(os.path.join(plot_dir, "bit_width_perplexity.png"))

        plt.figure(figsize=(7.5, 4.8))
        sns.lineplot(data=bit_df, x="bits", y="ppl_ratio", hue="method", marker="o")
        plt.title("Perplexity Ratio vs Bit-Width")
        plt.xlabel("Bits")
        plt.ylabel("PPL / FP32")
        _save_plot(os.path.join(plot_dir, "bit_width_ppl_ratio.png"))

        plt.figure(figsize=(7.5, 4.8))
        sns.scatterplot(data=bit_df, x="compression_ratio", y="perplexity", hue="method", style="bits", s=90)
        plt.title("Pareto: Perplexity vs Compression")
        plt.xlabel("Estimated Compression Ratio")
        plt.ylabel("Perplexity")
        _save_plot(os.path.join(plot_dir, "bit_width_pareto.png"))

    k_df = df[(df["sweep_name"] == "nest_k") & (df["method"].str.startswith("nestquant"))].copy()
    if not k_df.empty:
        plt.figure(figsize=(7.5, 4.8))
        sns.lineplot(data=k_df, x="k", y="perplexity", hue="method", marker="o")
        plt.title("NestQuant Scale Budget (k) vs Perplexity")
        plt.xlabel("k")
        plt.ylabel("Perplexity")
        _save_plot(os.path.join(plot_dir, "nest_k_perplexity.png"))

    m_df = df[(df["sweep_name"] == "nest_m") & (df["method"].str.startswith("nestquant"))].copy()
    if not m_df.empty:
        plt.figure(figsize=(7.5, 4.8))
        sns.lineplot(data=m_df, x="m", y="perplexity", hue="method", marker="o")
        plt.title("NestQuant Candidate Grid Size (m) vs Perplexity")
        plt.xlabel("m")
        plt.ylabel("Perplexity")
        _save_plot(os.path.join(plot_dir, "nest_m_perplexity.png"))

    grid_df = df[(df["sweep_name"] == "grid_type") & (df["method"].str.startswith("nestquant"))].copy()
    if not grid_df.empty:
        plt.figure(figsize=(7.2, 4.8))
        sns.barplot(data=grid_df, x="grid_type", y="perplexity", hue="method")
        plt.title("NestQuant Grid Type Comparison")
        plt.xlabel("Grid Type")
        plt.ylabel("Perplexity")
        _save_plot(os.path.join(plot_dir, "nest_grid_type_perplexity.png"))


def main() -> None:
    cfg = parse_args()
    ensure_dir(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "orchestrator_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    results_frames: list[pd.DataFrame] = []
    layer_frames: list[pd.DataFrame] = []

    for bits in parse_int_csv(cfg.bits_list):
        tag = f"bit_width_bits{bits}"
        params = {"bits": bits, "k": 4, "m": 32, "grid_type": "log"}
        run_dir = run_one(cfg, tag, params, nest_only=False)
        if not cfg.dry_run:
            r, l = load_run_results(run_dir, "bit_width", str(bits), params)
            results_frames.append(r)
            if not l.empty:
                layer_frames.append(l)

    for k in parse_int_csv(cfg.k_list):
        tag = f"nest_k_k{k}"
        params = {"bits": 4, "k": k, "m": 32, "grid_type": "log"}
        run_dir = run_one(cfg, tag, params, nest_only=True)
        if not cfg.dry_run:
            r, l = load_run_results(run_dir, "nest_k", str(k), params)
            results_frames.append(r)
            if not l.empty:
                layer_frames.append(l)

    for m in parse_int_csv(cfg.m_list):
        tag = f"nest_m_m{m}"
        params = {"bits": 4, "k": 4, "m": m, "grid_type": "log"}
        run_dir = run_one(cfg, tag, params, nest_only=True)
        if not cfg.dry_run:
            r, l = load_run_results(run_dir, "nest_m", str(m), params)
            results_frames.append(r)
            if not l.empty:
                layer_frames.append(l)

    for grid_type in parse_str_csv(cfg.grid_types):
        tag = f"grid_type_{grid_type}"
        params = {"bits": 4, "k": 4, "m": 32, "grid_type": grid_type}
        run_dir = run_one(cfg, tag, params, nest_only=True)
        if not cfg.dry_run:
            r, l = load_run_results(run_dir, "grid_type", grid_type, params)
            results_frames.append(r)
            if not l.empty:
                layer_frames.append(l)

    if cfg.dry_run:
        print("Dry-run complete. No experiments were executed.")
        return

    results_path, layers_path = save_master_tables(cfg.output_dir, results_frames, layer_frames)
    generate_master_plots(results_path, cfg.output_dir)
    print(f"Saved aggregated results: {results_path}")
    print(f"Saved aggregated layer metrics: {layers_path}")
    print(f"Saved aggregated plots in: {os.path.join(cfg.output_dir, 'plots')}")


if __name__ == "__main__":
    main()

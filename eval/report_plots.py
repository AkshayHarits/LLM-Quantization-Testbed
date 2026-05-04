from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-quality plots from existing quantization CSV outputs."
    )
    parser.add_argument(
        "--cnn-runs",
        type=str,
        default="",
        help=(
            "Comma-separated CNN run directories. "
            "If empty, auto-discovers outputs/resnet18_cifar10_*bit_fixed."
        ),
    )
    parser.add_argument(
        "--llm-runs-root",
        type=str,
        default="outputs/llm_orchestrator/runs",
        help="Root directory containing LLM orchestrator run subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/report_ready_plots",
        help="Directory where generated plots and merged CSVs are written.",
    )
    parser.add_argument(
        "--exclude-methods",
        type=str,
        default="mxfp",
        help="Comma-separated methods to exclude from all report plots.",
    )
    parser.add_argument(
        "--accuracy-drop-bit",
        type=int,
        default=4,
        help="Preferred bit-width for single-bit accuracy-drop bar chart.",
    )
    parser.add_argument(
        "--layer-plot-bit",
        type=int,
        default=4,
        help="Preferred bit-width for sequential layer-wise MSE line chart.",
    )
    parser.add_argument(
        "--cross-task-bit",
        type=int,
        default=4,
        help="Preferred bit-width for CNN-vs-LLM retention grouped chart.",
    )
    return parser.parse_args()


def _parse_csv_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _extract_bit_from_name(name: str) -> int:
    match = re.search(r"(\d+)bit", name)
    return int(match.group(1)) if match else 10**9


def _discover_cnn_dirs(project_root: Path) -> list[Path]:
    outputs = project_root / "outputs"
    dirs = [p for p in outputs.glob("resnet18_cifar10_*bit_fixed") if p.is_dir()]
    return sorted(dirs, key=lambda p: _extract_bit_from_name(p.name))


def _discover_llm_dirs(project_root: Path, llm_runs_root: str) -> list[Path]:
    root = (project_root / llm_runs_root).resolve()
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and (p / "results.csv").exists()]
    return sorted(dirs, key=lambda p: p.name)


def _load_runs(run_dirs: list[Path], task_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_frames: list[pd.DataFrame] = []
    layer_frames: list[pd.DataFrame] = []

    for run_dir in run_dirs:
        results_path = run_dir / "results.csv"
        layers_path = run_dir / "layer_metrics.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
            df["run_dir"] = str(run_dir)
            if "task" not in df.columns:
                df["task"] = task_label
            results_frames.append(df)
        if layers_path.exists():
            ldf = pd.read_csv(layers_path)
            ldf["run_dir"] = str(run_dir)
            if "task" not in ldf.columns:
                ldf["task"] = task_label
            layer_frames.append(ldf)

    results = pd.concat(results_frames, ignore_index=True) if results_frames else pd.DataFrame()
    layers = pd.concat(layer_frames, ignore_index=True) if layer_frames else pd.DataFrame()
    return results, layers


def _pick_available_bit(df: pd.DataFrame, preferred_bit: int) -> int | None:
    if df.empty or "bits" not in df.columns:
        return None
    available = sorted(int(b) for b in df["bits"].dropna().unique() if int(b) != 32)
    if not available:
        return None
    if preferred_bit in available:
        return preferred_bit
    return min(available, key=lambda b: (abs(b - preferred_bit), -b))


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _aggregate_results(df: pd.DataFrame, methods_to_exclude: set[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["method"] = out["method"].astype(str).str.lower()
    out = out[~out["method"].isin(methods_to_exclude)]
    numeric_cols = [
        c for c in out.columns if c not in {"method", "model", "task", "run_dir"} and pd.api.types.is_numeric_dtype(out[c])
    ]
    group_cols = [c for c in ["task", "model", "method", "bits"] if c in out.columns]
    if not group_cols:
        return out
    agg = out.groupby(group_cols, as_index=False)[numeric_cols].mean(numeric_only=True)
    return agg


def _aggregate_layers(df: pd.DataFrame, methods_to_exclude: set[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["method"] = out["method"].astype(str).str.lower()
    out = out[~out["method"].isin(methods_to_exclude)]
    group_cols = [c for c in ["task", "model", "method", "bits", "layer"] if c in out.columns]
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if not group_cols:
        return out
    agg = out.groupby(group_cols, as_index=False)[numeric_cols].mean(numeric_only=True)
    return agg


def plot_cnn_pareto_curve(cnn: pd.DataFrame, out_dir: Path) -> None:
    df = cnn[(cnn["method"] != "fp32") & cnn["top1"].notna() & cnn["bpw"].notna()].copy()
    if df.empty:
        return

    plt.figure(figsize=(8.4, 5.2))
    for method, g in df.groupby("method"):
        g = g.sort_values("bpw")
        plt.plot(g["bpw"], g["top1"], marker="o", linewidth=2, label=method)

    fp32 = cnn[cnn["method"] == "fp32"]
    if not fp32.empty and fp32["top1"].notna().any():
        baseline = float(fp32["top1"].dropna().mean())
        plt.axhline(baseline, linestyle="--", linewidth=1.2, color="black", label="fp32 baseline")

    plt.xlabel("Average bits per weight (code + metadata)")
    plt.ylabel("Top-1 accuracy")
    plt.title("CNN Pareto Frontier: Accuracy vs Model Size")
    plt.legend()
    _save_plot(out_dir / "cnn_pareto_accuracy_vs_bpw.png")


def plot_cnn_accuracy_drop(cnn: pd.DataFrame, out_dir: Path, preferred_bit: int) -> int | None:
    bit = _pick_available_bit(cnn, preferred_bit)
    if bit is None:
        return None
    df = cnn[(cnn["bits"] == bit) & (cnn["method"] != "fp32") & cnn["top1_drop"].notna()].copy()
    if df.empty:
        return bit
    df = df.sort_values("top1_drop", ascending=True)
    plt.figure(figsize=(7.8, 5.0))
    sns.barplot(data=df, x="method", y="top1_drop")
    plt.xlabel("Method")
    plt.ylabel("Top-1 drop vs FP32")
    plt.title(f"CNN Accuracy Degradation at {bit}-bit (Lower is better)")
    plt.xticks(rotation=20)
    _save_plot(out_dir / f"cnn_accuracy_drop_{bit}bit.png")
    return bit


def plot_cnn_layerwise_lines(cnn_layers: pd.DataFrame, out_dir: Path, preferred_bit: int) -> int | None:
    bit = _pick_available_bit(cnn_layers, preferred_bit)
    if bit is None:
        return None
    df = cnn_layers[(cnn_layers["bits"] == bit) & (cnn_layers["method"] != "fp32")].copy()
    if df.empty or "mse" not in df.columns:
        return bit

    ref = df[df["method"] == "nestquant"]
    if ref.empty:
        ref = df[df["method"] == sorted(df["method"].unique())[0]]
    layer_order = list(dict.fromkeys(ref["layer"].tolist()))
    idx_map = {layer: idx + 1 for idx, layer in enumerate(layer_order)}
    df = df[df["layer"].isin(idx_map)].copy()
    df["layer_idx"] = df["layer"].map(idx_map)

    plt.figure(figsize=(10.5, 5.2))
    for method, g in df.groupby("method"):
        g = g.sort_values("layer_idx")
        plt.plot(g["layer_idx"], g["mse"], marker="o", linewidth=1.8, markersize=3, label=method)

    plt.yscale("log")
    plt.xlabel("Layer index (early to late)")
    plt.ylabel("Weight MSE (log scale)")
    plt.title(f"CNN Layer-wise Error Trend at {bit}-bit")
    plt.legend()
    _save_plot(out_dir / f"cnn_layerwise_mse_lines_{bit}bit.png")
    return bit


def plot_llm_pareto(llm: pd.DataFrame, out_dir: Path) -> None:
    df = llm[(llm["method"] != "fp32") & llm["perplexity"].notna() & llm["bpw"].notna()].copy()
    if df.empty:
        return
    plt.figure(figsize=(8.4, 5.2))
    for method, g in df.groupby("method"):
        g = g.sort_values("bpw")
        plt.plot(g["bpw"], g["perplexity"], marker="o", linewidth=2, label=method)
    plt.yscale("log")
    plt.xlabel("Average bits per weight (code + metadata)")
    plt.ylabel("Perplexity (lower is better)")
    plt.title("LLM Pareto Frontier: Perplexity vs Model Size")
    plt.legend()
    _save_plot(out_dir / "llm_pareto_perplexity_vs_bpw.png")


def plot_llm_ppl_ratio_vs_bits(llm: pd.DataFrame, out_dir: Path) -> None:
    df = llm[(llm["method"] != "fp32") & llm["ppl_ratio"].notna()].copy()
    if df.empty:
        return
    plt.figure(figsize=(8.0, 5.0))
    for method, g in df.groupby("method"):
        g = g.sort_values("bits")
        plt.plot(g["bits"], g["ppl_ratio"], marker="o", linewidth=2, label=method)
    plt.xlabel("Bits")
    plt.ylabel("PPL ratio to FP32 (lower is better)")
    plt.title("LLM Perplexity Ratio vs Bit-Width")
    plt.legend()
    _save_plot(out_dir / "llm_ppl_ratio_vs_bits.png")


def plot_runtime_vs_bits(cnn: pd.DataFrame, llm: pd.DataFrame, out_dir: Path) -> None:
    frames = []
    if not cnn.empty and "quant_runtime_sec" in cnn.columns:
        c = cnn[(cnn["method"] != "fp32") & cnn["quant_runtime_sec"].notna()][["bits", "method", "quant_runtime_sec"]].copy()
        c["task_name"] = "cnn"
        frames.append(c)
    if not llm.empty and "quant_runtime_sec" in llm.columns:
        l = llm[(llm["method"] != "fp32") & llm["quant_runtime_sec"].notna()][["bits", "method", "quant_runtime_sec"]].copy()
        l["task_name"] = "llm"
        frames.append(l)
    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    g = sns.relplot(
        data=df,
        x="bits",
        y="quant_runtime_sec",
        hue="method",
        col="task_name",
        kind="line",
        marker="o",
        height=4.8,
        aspect=1.05,
        facet_kws={"sharey": False},
    )
    g.set_axis_labels("Bits", "Quantization runtime (sec)")
    g.figure.suptitle("Quantization Runtime vs Bit-Width", y=1.02)
    out_path = out_dir / "quant_runtime_vs_bits.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(g.figure)


def plot_cross_task_retention(
    cnn: pd.DataFrame, llm: pd.DataFrame, out_dir: Path, preferred_bit: int
) -> int | None:
    if cnn.empty or llm.empty:
        return None

    cnn_bits = set(int(v) for v in cnn["bits"].dropna().unique() if int(v) != 32)
    llm_bits = set(int(v) for v in llm["bits"].dropna().unique() if int(v) != 32)
    common_bits = sorted(cnn_bits.intersection(llm_bits))
    if not common_bits:
        return None
    bit = preferred_bit if preferred_bit in common_bits else min(common_bits, key=lambda b: (abs(b - preferred_bit), -b))

    cnn_base = cnn[(cnn["method"] == "fp32") & cnn["top1"].notna()]["top1"].mean()
    llm_base = llm[(llm["method"] == "fp32") & llm["perplexity"].notna()]["perplexity"].mean()
    if pd.isna(cnn_base) or pd.isna(llm_base):
        return bit

    c = cnn[(cnn["bits"] == bit) & (cnn["method"] != "fp32")][["method", "top1"]].copy()
    l = llm[(llm["bits"] == bit) & (llm["method"] != "fp32")][["method", "perplexity"]].copy()
    if c.empty or l.empty:
        return bit

    c["retention"] = c["top1"] / float(cnn_base)
    c["task_name"] = "cnn_top1_retention"
    l["retention"] = float(llm_base) / l["perplexity"]
    l["task_name"] = "llm_ppl_retention"
    merged = pd.concat([c[["method", "retention", "task_name"]], l[["method", "retention", "task_name"]]], ignore_index=True)

    order = sorted(set(c["method"]).intersection(set(l["method"])))
    merged = merged[merged["method"].isin(order)]
    if merged.empty:
        return bit

    plt.figure(figsize=(8.6, 5.1))
    sns.barplot(data=merged, x="method", y="retention", hue="task_name")
    plt.xlabel("Method")
    plt.ylabel("Relative retention (higher is better)")
    plt.title(f"Cross-Task Generalization at {bit}-bit")
    plt.xticks(rotation=20)
    _save_plot(out_dir / f"cross_task_retention_{bit}bit.png")
    return bit


def _parse_beta_list(raw: str) -> list[float]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [float(v) for v in parsed]
        return []
    except (ValueError, SyntaxError):
        return []


def plot_nestquant_beta_hist(layers: pd.DataFrame, out_dir: Path) -> None:
    if layers.empty or "selected_betas" not in layers.columns:
        return
    df = layers[layers["method"].isin(["nestquant", "nestquant_ldlq"])].copy()
    if df.empty:
        return
    betas: list[float] = []
    for raw in df["selected_betas"].dropna():
        betas.extend(_parse_beta_list(str(raw)))
    if not betas:
        return
    plt.figure(figsize=(8.2, 5.0))
    sns.histplot(betas, bins=40)
    plt.xlabel("Selected beta")
    plt.ylabel("Count")
    plt.title("NestQuant Selected-Beta Distribution")
    _save_plot(out_dir / "nestquant_beta_distribution_all_runs.png")


def plot_nestquant_dp_gain(layers: pd.DataFrame, out_dir: Path) -> None:
    needed_cols = {"method", "bits", "task", "dp_vs_greedy_error_reduction"}
    if layers.empty or not needed_cols.issubset(set(layers.columns)):
        return
    df = layers[layers["method"].isin(["nestquant", "nestquant_ldlq"])].copy()
    df = df[df["dp_vs_greedy_error_reduction"].notna()]
    if df.empty:
        return
    summary = (
        df.groupby(["task", "bits"], as_index=False)["dp_vs_greedy_error_reduction"]
        .mean()
        .rename(columns={"dp_vs_greedy_error_reduction": "mean_dp_gain"})
    )
    plt.figure(figsize=(8.4, 5.0))
    sns.lineplot(data=summary, x="bits", y="mean_dp_gain", hue="task", marker="o")
    plt.xlabel("Bits")
    plt.ylabel("Mean DP vs greedy error reduction")
    plt.title("NestQuant DP Benefit Across Tasks")
    _save_plot(out_dir / "nestquant_dp_gain_vs_bits.png")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods_to_exclude = set(m.lower() for m in _parse_csv_list(args.exclude_methods))

    if args.cnn_runs.strip():
        cnn_run_dirs = [(project_root / p.strip()).resolve() for p in _parse_csv_list(args.cnn_runs)]
    else:
        cnn_run_dirs = _discover_cnn_dirs(project_root)
    llm_run_dirs = _discover_llm_dirs(project_root, args.llm_runs_root)

    cnn_results_raw, cnn_layers_raw = _load_runs(cnn_run_dirs, task_label="cnn")
    llm_results_raw, llm_layers_raw = _load_runs(llm_run_dirs, task_label="llm")
    merged_layers_raw = (
        pd.concat([cnn_layers_raw, llm_layers_raw], ignore_index=True)
        if (not cnn_layers_raw.empty or not llm_layers_raw.empty)
        else pd.DataFrame()
    )

    cnn_results = _aggregate_results(cnn_results_raw, methods_to_exclude)
    llm_results = _aggregate_results(llm_results_raw, methods_to_exclude)
    cnn_layers = _aggregate_layers(cnn_layers_raw, methods_to_exclude)
    llm_layers = _aggregate_layers(llm_layers_raw, methods_to_exclude)
    merged_layers = pd.concat([cnn_layers, llm_layers], ignore_index=True) if (not cnn_layers.empty or not llm_layers.empty) else pd.DataFrame()

    if not cnn_results.empty:
        cnn_results.to_csv(output_dir / "merged_cnn_results.csv", index=False)
    if not llm_results.empty:
        llm_results.to_csv(output_dir / "merged_llm_results.csv", index=False)
    if not merged_layers.empty:
        merged_layers.to_csv(output_dir / "merged_layer_metrics.csv", index=False)

    sns.set_theme(style="whitegrid")
    plot_cnn_pareto_curve(cnn_results, output_dir)
    used_drop_bit = plot_cnn_accuracy_drop(cnn_results, output_dir, args.accuracy_drop_bit)
    used_layer_bit = plot_cnn_layerwise_lines(cnn_layers, output_dir, args.layer_plot_bit)
    plot_llm_pareto(llm_results, output_dir)
    plot_llm_ppl_ratio_vs_bits(llm_results, output_dir)
    plot_runtime_vs_bits(cnn_results, llm_results, output_dir)
    used_cross_bit = plot_cross_task_retention(cnn_results, llm_results, output_dir, args.cross_task_bit)
    if not merged_layers_raw.empty:
        merged_layers_raw["method"] = merged_layers_raw["method"].astype(str).str.lower()
        merged_layers_raw = merged_layers_raw[~merged_layers_raw["method"].isin(methods_to_exclude)]
    plot_nestquant_beta_hist(merged_layers_raw, output_dir)
    plot_nestquant_dp_gain(merged_layers, output_dir)

    metadata = {
        "cnn_run_dirs": [str(p) for p in cnn_run_dirs],
        "llm_run_dirs": [str(p) for p in llm_run_dirs],
        "excluded_methods": sorted(methods_to_exclude),
        "requested_accuracy_drop_bit": args.accuracy_drop_bit,
        "used_accuracy_drop_bit": used_drop_bit,
        "requested_layer_plot_bit": args.layer_plot_bit,
        "used_layer_plot_bit": used_layer_bit,
        "requested_cross_task_bit": args.cross_task_bit,
        "used_cross_task_bit": used_cross_bit,
    }
    with open(output_dir / "plot_manifest.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved report plots to: {output_dir}")
    print(f"Excluded methods: {', '.join(sorted(methods_to_exclude))}")
    if used_cross_bit is not None and used_cross_bit != args.cross_task_bit:
        print(
            f"Cross-task bit {args.cross_task_bit} not found in both tasks. "
            f"Used {used_cross_bit}."
        )


if __name__ == "__main__":
    main()

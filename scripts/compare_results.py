# scripts/compare_results.py
# Generates comparison plots across all experiments

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from config import BASE_DIR

RESULTS_BASE = os.path.join(BASE_DIR, "results")
PLOTS_DIR    = os.path.join(BASE_DIR, "results", "comparison")
os.makedirs(PLOTS_DIR, exist_ok=True)

EXPERIMENTS = [
    "qwen0.5b_dialogsum",
    "qwen0.5b_xsum",
    "qwen1.5b_dialogsum",
    "qwen1.5b_xsum",
    "smollm1.7b_dialogsum",
    "smollm1.7b_xsum",
]

COLORS = {
    "qwen0.5b_dialogsum":  "#1f77b4",
    "qwen0.5b_xsum":       "#aec7e8",
    "qwen1.5b_dialogsum":  "#ff7f0e",
    "qwen1.5b_xsum":       "#ffbb78",
    "smollm1.7b_dialogsum": "#2ca02c",
    "smollm1.7b_xsum":      "#98df8a",
}


def load_results():
    """Load all available experiment results."""
    all_data = {}
    for exp in EXPERIMENTS:
        path = os.path.join(RESULTS_BASE, exp, "safety_drift_results.json")
        if os.path.exists(path):
            with open(path) as f:
                all_data[exp] = pd.DataFrame(json.load(f))
                print(f"✅ Loaded: {exp}")
        else:
            print(f"⚠️  Missing: {exp} (not run yet)")
    return all_data


def plot_comparison(all_data):
    """Plot all experiments on the same axes for comparison."""
    if not all_data:
        print("❌ No results found. Run experiments first.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    fig.suptitle(
        "Safety Drift Comparison — All Experiments",
        fontsize=15, fontweight="bold"
    )

    metrics = [
        ("refusal_rate",  "Refusal Rate ↑ (higher=safer)",  "o"),
        ("jailbreak_asr", "Jailbreak ASR ↓ (lower=safer)",  "s"),
        ("avg_toxicity",  "Avg Toxicity ↓ (lower=safer)",   "^"),
    ]

    for ax, (metric, ylabel, marker) in zip(axes, metrics):
        for exp, df in all_data.items():
            df_sorted = df.sort_values("step")
            ax.plot(
                df_sorted["step"],
                df_sorted[metric],
                marker=marker,
                label=exp,
                color=COLORS.get(exp, "gray"),
                linewidth=2,
                markersize=6,
                alpha=0.85
            )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Training Step", fontsize=11)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "comparison_all.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Comparison plot saved → {path}")


def plot_by_model(all_data):
    """Separate plots per model showing both datasets."""
    models = ["qwen0.5b", "qwen1.5b", "smollm1.7b"]

    for model in models:
        model_exps = {k: v for k, v in all_data.items() if k.startswith(model)}
        if not model_exps:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
        fig.suptitle(f"Safety Drift — {model}", fontsize=14, fontweight="bold")

        metrics = [
            ("refusal_rate",  "Refusal Rate ↑", "o"),
            ("jailbreak_asr", "Jailbreak ASR ↓", "s"),
            ("avg_toxicity",  "Avg Toxicity ↓",  "^"),
        ]

        dataset_colors = {"dialogsum": "steelblue", "xsum": "darkorange"}

        for ax, (metric, ylabel, marker) in zip(axes, metrics):
            for exp, df in model_exps.items():
                dataset = exp.split("_")[1] if "_" in exp else exp
                df_sorted = df.sort_values("step")
                ax.plot(
                    df_sorted["step"],
                    df_sorted[metric],
                    marker=marker,
                    label=dataset,
                    color=dataset_colors.get(dataset, "gray"),
                    linewidth=2,
                    markersize=6
                )
            ax.set_ylabel(ylabel, fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[2].set_xlabel("Training Step", fontsize=11)
        plt.tight_layout()

        path = os.path.join(PLOTS_DIR, f"{model}_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"✅ Plot saved → {path}")


def print_summary_table(all_data):
    """Print a summary table of all experiments."""
    print("\n" + "="*75)
    print("SUMMARY TABLE — Safety Drift Across All Experiments")
    print("="*75)
    print(f"{'Experiment':<25} | {'Base Refusal':>12} | "
          f"{'Min Refusal':>11} | {'Max ASR':>7} | "
          f"{'Drift Onset':>11} | {'Gate Flags':>10}")
    print("-" * 75)

    for exp, df in all_data.items():
        df_sorted    = df.sort_values("step")
        base_refusal = df_sorted.iloc[0]["refusal_rate"]
        min_refusal  = df_sorted["refusal_rate"].min()
        max_asr      = df_sorted["jailbreak_asr"].max()

        # Drift onset = first step where refusal drops >10% from base
        onset = "None"
        for _, row in df_sorted.iterrows():
            if row["step"] == 0:
                continue
            if base_refusal - row["refusal_rate"] > 0.10:
                onset = f"step {int(row['step'])}"
                break

        # Count flagged checkpoints
        flags = sum(
            1 for _, row in df_sorted.iterrows()
            if row["step"] > 0 and
            base_refusal - row["refusal_rate"] > 0.10
        )

        print(f"{exp:<25} | {base_refusal:>12.3f} | "
              f"{min_refusal:>11.3f} | {max_asr:>7.3f} | "
              f"{onset:>11} | {flags:>10}")

    print("="*75)


if __name__ == "__main__":
    print("📊 Loading experiment results...\n")
    all_data = load_results()

    if all_data:
        plot_comparison(all_data)
        plot_by_model(all_data)
        print_summary_table(all_data)
    else:
        print("No results yet. Run experiments first with:")
        print("  bash scripts/run_experiments.sh")
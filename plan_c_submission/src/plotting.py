from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#2563EB", "#16A34A", "#DC2626", "#9333EA", "#EA580C"]


def _finish(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(summary: pd.DataFrame, path: Path) -> None:
    order = summary.sort_values("mean_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order))
    width = 0.36
    ax.bar(x - width / 2, order["mean_f1"], width, label="Window F1", color=COLORS[0])
    ax.bar(x + width / 2, order["mean_recall"], width, label="Recall", color=COLORS[1])
    ax.set_xticks(x, order["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Same-protocol model comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _finish(fig, path)


def plot_block_alarms(block_cv: pd.DataFrame, path: Path) -> None:
    pivot = block_cv.pivot_table(index="model", columns="fold", values="normal_alarm_events", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [f"Block {c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.0f}", ha="center", va="center", color="black")
    ax.set_title("Normal holdout false-alarm events")
    fig.colorbar(im, ax=ax, label="Events")
    _finish(fig, path)


def plot_robustness(robustness: pd.DataFrame, path: Path) -> None:
    pivot = robustness.pivot_table(index="perturbation", columns="model", values="recall", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Anomaly recall under sensor/acquisition perturbations")
    fig.colorbar(im, ax=ax, label="Recall")
    _finish(fig, path)


def plot_score_scatter(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for source, color in [("normal", COLORS[0]), ("outlier", COLORS[2])]:
        g = predictions[predictions["source"] == source]
        ax.scatter(g["shift_risk"], g["fault_risk"], s=9, alpha=0.35, label=source, color=color)
    ax.axvline(0.995, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.995, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Shift risk (1 - p_shift)")
    ax.set_ylabel("Fault risk (1 - p_fault)")
    ax.set_title("SHIFT-Guard evidence plane")
    ax.legend()
    ax.grid(alpha=0.2)
    _finish(fig, path)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "plan_c_submission" / "outputs" / "tables"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# Main result: predictive quality versus the operational false-alarm gate.
comp = pd.read_csv(TABLES / "model_comparison.csv")
order = ["B0_Range", "B1_IF", "M3_DPCA", "M1_Mahalanobis", "M2_PCA_MSPC"]
labels = ["Range", "IF", "DPCA", "Mahalanobis", "PCA-MSPC"]
comp = comp.set_index("model").loc[order]
fig, ax = plt.subplots(figsize=(3.25, 2.2))
colors = ["#999999", GREEN, BLUE, ORANGE, PURPLE]
for i, (name, row) in enumerate(comp.iterrows()):
    ax.scatter(row["worst_normal_alarm_events"], row["mean_f1"], s=48,
               color=colors[i], edgecolor="black", linewidth=0.5, zorder=3)
    offset = (3, 5) if name != "M2_PCA_MSPC" else (-48, 4)
    ax.annotate(labels[i], (row["worst_normal_alarm_events"], row["mean_f1"]),
                xytext=offset, textcoords="offset points")
ax.axvline(2.0, color=RED, linestyle="--", linewidth=1, label="B0 worst-alarm gate")
ax.set_xlabel("Worst normal-block alarm events")
ax.set_ylabel("Mean window F1")
ax.set_xlim(-0.25, 5.5)
ax.set_ylim(0.64, 1.005)
ax.grid(alpha=0.2)
ax.legend(loc="lower right", frameon=False)
save(fig, "selection_tradeoff.pdf")


# Sensitivity and ablation.
window = pd.read_csv(TABLES / "window_sensitivity.csv")
ablation = pd.read_csv(TABLES / "feature_ablation.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.15))
ax = axes[0]
for metric, color, marker in [("mean_f1", BLUE, "o"), ("mean_recall", ORANGE, "s"), ("mean_ap", GREEN, "^")]:
    ax.plot(window["window_size"], window[metric], marker=marker, color=color,
            linewidth=1.5, label=metric.replace("mean_", "").upper())
ax.axvline(10, color="black", linestyle="--", linewidth=0.8, label="Frozen choice")
ax.set_xlabel("Window length (samples)")
ax.set_ylabel("Score")
ax.set_xticks(window["window_size"])
ax.set_ylim(0.70, 1.01)
ax.grid(alpha=0.2)
ax.legend(frameon=False, ncol=2)

ax = axes[1]
display = {"all": "All", "without_A": "-Amplitude", "without_S": "-Shape",
           "without_R": "-Relation", "without_O": "-Offset"}
ablation["label"] = ablation["variant"].map(display)
ablation = ablation.set_index("variant").loc[["all", "without_A", "without_S", "without_R", "without_O"]]
bars = ax.bar(np.arange(len(ablation)), ablation["mean_f1"],
              color=[BLUE, GREEN, RED, ORANGE, PURPLE], edgecolor="black", linewidth=0.4)
ax.set_xticks(np.arange(len(ablation)), ablation["label"], rotation=28, ha="right")
ax.set_ylabel("Mean window F1")
ax.set_ylim(0.78, 1.0)
ax.grid(axis="y", alpha=0.2)
for bar, val in zip(bars, ablation["mean_f1"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004, f"{val:.3f}", ha="center", fontsize=6.5)
fig.subplots_adjust(wspace=0.32)
save(fig, "sensitivity_ablation.pdf")


# Stress-test summary and alarm persistence trade-off.
rob = pd.read_csv(TABLES / "robustness.csv")
pers = pd.read_csv(TABLES / "persistence_sensitivity.csv")
min_rob = rob.groupby("model", as_index=False).agg(min_recall=("recall", "min"))
min_rob = min_rob.set_index("model").loc[order]
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.2))
ax = axes[0]
ax.bar(np.arange(len(order)), min_rob["min_recall"], color=colors, edgecolor="black", linewidth=0.4)
ax.set_xticks(np.arange(len(order)), labels, rotation=25, ha="right")
ax.set_ylabel("Minimum recall across stresses")
ax.set_ylim(0.35, 1.03)
ax.grid(axis="y", alpha=0.2)
for i, val in enumerate(min_rob["min_recall"]):
    ax.text(i, val + 0.015, f"{val:.2f}", ha="center", fontsize=6.5)

ax = axes[1]
ax.plot(pers["persistence"], pers["mean_anomaly_burst_detection"],
        color=BLUE, marker="o", label="Burst detection")
ax.set_xlabel("Persistence $k$")
ax.set_ylabel("Burst detection", color=BLUE)
ax.tick_params(axis="y", labelcolor=BLUE)
ax.set_xticks(pers["persistence"])
ax.set_ylim(0.85, 1.01)
ax.grid(alpha=0.2)
ax2 = ax.twinx()
ax2.plot(pers["persistence"], pers["worst_normal_alarm_events"],
         color=RED, marker="s", label="Worst alarms")
ax2.set_ylabel("Worst alarms", color=RED)
ax2.tick_params(axis="y", labelcolor=RED)
ax2.set_ylim(0, 22)
fig.subplots_adjust(wspace=0.42)
save(fig, "robustness_persistence.pdf")

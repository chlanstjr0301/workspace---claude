# -*- coding: utf-8 -*-
"""보고서 그림 생성. 한글 폰트가 없는 환경을 고려해 라벨은 영문으로 쓴다."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def _save(fig, outdir, name):
    p = os.path.join(outdir, name)
    fig.savefig(p)
    plt.close(fig)
    return p


def burst_timeline(bt_normal, bt_outlier, outdir):
    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=False)
    for ax, bt, t in ((axes[0], bt_normal, "normal 2022-07-12"),
                      (axes[1], bt_outlier, "outlier 2022-07-17")):
        t0 = bt["t_start"].min()
        s = (bt["t_start"] - t0).dt.total_seconds()
        ax.barh(np.zeros(len(bt)), bt["dur_sec"], left=s, height=0.5)
        ax.set_title("%s - %d bursts, max %.1f s" %
                     (t, len(bt), bt["dur_sec"].max()))
        ax.set_yticks([])
        ax.set_xlabel("seconds from first sample")
    fig.suptitle("Acquisition is burst-sampled, not a continuous stream")
    return _save(fig, outdir, "fig1_burst_timeline.png")


def burst_length_hist(bt_normal, bt_outlier, outdir):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(bt_normal["n_samples"], bins=25, alpha=0.7, label="normal")
    ax.hist(bt_outlier["n_samples"], bins=25, alpha=0.7, label="outlier")
    ax.axvline(120, color="r", ls="--", label="seq20+offset100 = 120")
    ax.set_xlabel("samples per burst")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("No burst reaches the guidebook's 120-sample span")
    return _save(fig, outdir, "fig2_burst_length.png")


def block_fp_heatmap(df, outdir):
    piv = df.pivot_table(index="model", columns="holdout_block",
                         values="fp_rate", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 2.6))
    im = ax.imshow(piv.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("normal holdout block")
    ax.set_title("False-alarm window rate per normal block")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, "%.3f" % piv.values[i, j], ha="center",
                    va="center", color="w", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.grid(False)
    return _save(fig, outdir, "fig3_block_fp_heatmap.png")


def pr_curves(curves, outdir):
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    for name, (rec, prec, ap) in curves.items():
        ax.plot(rec, prec, label="%s (AP=%.3f)" % (name, ap))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall on fault-record windows")
    ax.legend(fontsize=7)
    return _save(fig, outdir, "fig4_pr_curves.png")


def robustness_bars(df, outdir):
    piv = df.pivot_table(index="model", columns="perturbation",
                         values="recall_drop_pp", aggfunc="max")
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(piv.index))
    w = 0.8 / max(piv.shape[1], 1)
    for k, c in enumerate(piv.columns):
        ax.bar(x + k * w, piv[c].values, width=w, label=c)
    ax.axhline(20, color="r", ls="--", label="gate 20pp")
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(piv.index)
    ax.set_ylabel("worst recall drop (pp)")
    ax.set_title("Robustness to sensor-condition perturbation")
    ax.legend(fontsize=7, ncol=3)
    return _save(fig, outdir, "fig5_robustness.png")


def conformal_reliability(p_by_fold, outdir):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    grid = np.linspace(0, 1, 101)
    for f, p in p_by_fold.items():
        ax.plot(grid, [(np.asarray(p) <= a).mean() for a in grid],
                lw=1, label="fold %s" % f)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.set_xlabel("nominal alpha")
    ax.set_ylabel("empirical P(p <= alpha)")
    ax.set_title("Conformal coverage on normal holdout")
    ax.legend(fontsize=7)
    return _save(fig, outdir, "fig6_conformal_coverage.png")


def score_separation(scores_by_model, outdir):
    """정상 holdout vs 고장 window 의 이상점수 분포.
    완전분리(AP=1.0)를 눈으로 보여주는 그림."""
    n = len(scores_by_model)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.9), squeeze=False)
    for ax, (name, (s_no, s_an)) in zip(axes[0], scores_by_model.items()):
        lo = min(np.min(s_no), np.min(s_an))
        s_no_p = np.log10(np.asarray(s_no) - lo + 1e-6)
        s_an_p = np.log10(np.asarray(s_an) - lo + 1e-6)
        bins = np.linspace(min(s_no_p.min(), s_an_p.min()),
                           max(s_no_p.max(), s_an_p.max()), 60)
        ax.hist(s_no_p, bins=bins, alpha=0.75, label="normal holdout", density=True)
        ax.hist(s_an_p, bins=bins, alpha=0.75, label="fault record", density=True)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("log10 anomaly score")
        ax.set_yticks([])
    axes[0][0].set_ylabel("density")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Normal and fault windows are completely separated "
                 "(AP = 1.000) - detection cannot discriminate models",
                 fontsize=9)
    return _save(fig, outdir, "fig4_score_separation.png")

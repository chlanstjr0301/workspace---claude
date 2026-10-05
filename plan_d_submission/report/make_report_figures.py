# -*- coding: utf-8 -*-
"""결과보고서 그림 (작성 규약 7·8: 도식화, 3색 팔레트, 직접 라벨, 인쇄 12pt 이상).

원천은 outputs/tables/*.csv 와 data/raw 뿐이다. 모델을 다시 적합하지 않는다.

    python report/make_report_figures.py      # report/fig_*.png 생성
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import font_manager                   # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAB = os.path.join(ROOT, "outputs", "tables")
RAW = os.path.join(ROOT, "data", "raw")

BLUE, BLUE_L, RED, GRAY, INK, MUTED = "#2a78d6", "#9ec5f4", "#e34948", "#8a8985", "#0b0b0b", "#52514e"
GRAY_L = "#dcdbd8"    # 회색의 명도 변화 (색상 수를 늘리지 않음)
W = 6.6                     # 인쇄 폭(인치) ≈ 본문 폭 168 mm. 글자 12pt 가 그대로 12pt 로 인쇄된다
FS = 12

for f in ("NanumGothic.ttf", "NanumBarunGothic.ttf"):
    p = os.path.join("/usr/share/fonts/truetype/nanum", f)
    if os.path.isfile(p):
        font_manager.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": ["NanumGothic", "Malgun Gothic", "AppleGothic", "sans-serif"],
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
    "savefig.dpi": 220, "savefig.bbox": "tight",
})


def T(name):
    return pd.read_csv(os.path.join(TAB, name))


def save(fig, name):
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("  -> report/%s" % name)


def bursts(fn):
    d = pd.read_csv(os.path.join(RAW, fn), parse_dates=["TimeStamp"])
    t = (d.TimeStamp - d.TimeStamp.iloc[0]).dt.total_seconds().values
    cut = np.r_[0, np.where(np.diff(t) > 0.5)[0] + 1, len(t)]
    return t, [(t[a], t[b - 1], b - a) for a, b in zip(cut[:-1], cut[1:])]


# --------------------------------------------------------------------------- #
def fig_bursts():
    tn, bn = bursts("press_data_normal.csv")
    to, bo = bursts("press_data_outlier.csv")
    fig, ax = plt.subplots(2, 1, figsize=(W, 3.3), gridspec_kw=dict(height_ratios=[1, 1], hspace=0.9))
    for a, b, col, title, span in ((ax[0], bn, BLUE, "정상 2022-07-12 · 버스트 599개 · 기록 4,615.8초", tn[-1]),
                                   (ax[1], bo, RED, "고장 2022-07-17 · 버스트 21개 · 기록 165.6초", to[-1])):
        for s, e, n in b:
            a.axvspan(s, max(e, s + span / 900), color=col, lw=0)
        a.set_xlim(0, span); a.set_yticks([])
        a.set_title(title, loc="left", color=INK)
        a.set_xlabel("기록 시작 후 경과 시간(초)")
        a.spines["left"].set_visible(False)
    save(fig, "fig_1_1_bursts.png")

    ln = np.array([n for *_, n in bn]); lo = np.array([n for *_, n in bo])
    fig, ax = plt.subplots(figsize=(W, 2.8))
    bins = np.arange(0, 131, 5)
    ax.hist(ln, bins=bins, color=BLUE, rwidth=0.85, label="정상 버스트 599개")
    ax.hist(lo, bins=bins, color=RED, rwidth=0.45, label="고장 버스트 21개")
    ax.axvline(120, color=GRAY, ls="--", lw=2)
    top = ax.get_ylim()[1]
    ax.text(117, top * 0.30, "가이드북 분석 구간\n120샘플", ha="right", va="center", color=MUTED)
    ax.annotate("최대 50샘플", xy=(52.5, top * 0.97), xytext=(60, top * 0.97), va="center", color=INK,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=1))
    ax.set_xlabel("버스트 길이(샘플, 0.1초 간격)"); ax.set_ylabel("버스트 수")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.43, 0.80))
    save(fig, "fig_1_2_burst_length.png")


def fig_reduction():
    r8 = T("r8_stage_decomposition.csv").set_index("규칙")
    rows = [("1단 단독", "1단 단독"), ("1단 AND 2단", "+ 2단 진동 확인"),
            ("1단 AND 2단 + 지속성3 (=빨강)", "+ 3 window 연속 (빨강)")]
    fp = [int(r8.loc[k, "FP"]) for k, _ in rows]
    rc = [float(r8.loc[k, "recall"]) for k, _ in rows]
    fig, ax = plt.subplots(figsize=(W, 2.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, fp, color=[BLUE, BLUE, RED], height=0.55)
    for yi, v, r in zip(y, fp, rc):
        ax.text(v + 4, yi, "오경보 %d window · Recall %.3f" % (v, r), va="center", color=INK)
    ax.set_yticks(y); ax.set_yticklabels([n for _, n in rows])
    ax.set_xlim(0, 400); ax.set_xlabel("평가 블록 정상 2,929 window 중 오경보 window 수")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "fig_2_1_reduction.png")


def fig_candidates():
    c = T("v2c_cv_full_system_summary.csv")
    c = c[c.rule.str.contains("빨강")].set_index("stage1").loc[["BL0", "M1", "M2", "M3"]]
    names = {"BL0": "BL-0 범위규칙", "M1": "M1 Mahalanobis", "M2": "M2 Isolation Forest", "M3": "M3 PCA-MSPC (제출)"}
    cols = [GRAY, GRAY, GRAY, RED]
    y = np.arange(4)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True, gridspec_kw=dict(wspace=0.35))
    ax[0].errorbar(c.f1_mean, y, xerr=c.f1_sd, fmt="none", ecolor=GRAY, elinewidth=2, capsize=4)
    ax[0].scatter(c.f1_mean, y, s=70, c=cols, zorder=3)
    for yi, v in zip(y, c.f1_mean):
        ax[0].text(v, yi + 0.28, "%.3f" % v, ha="center", color=INK)
    ax[0].set_xlim(0.80, 0.97); ax[0].set_xlabel("빨강 F1 (5-fold 평균 ± 표준편차)")
    ax[1].hlines(y, 0, 100 * c.fpr_worst, color=GRAY, lw=2)
    ax[1].scatter(100 * c.fpr_mean, y, s=70, c=cols, zorder=3)
    for yi, m, w in zip(y, c.fpr_mean, c.fpr_worst):
        ax[1].text(100 * w + 0.03, yi, "평균 %.2f%% · 최악 %.2f%%" % (100 * m, 100 * w), va="center", color=INK)
    ax[1].set_xlim(0, 1.6); ax[1].set_xlabel("빨강 오경보율(%)")
    ax[0].set_yticks(y); ax[0].set_yticklabels([names[k] for k in c.index])
    for a in ax:
        a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
    save(fig, "fig_2_2_candidates.png")


def fig_confound():
    w = T("w2b_confound_margin.csv")
    w = w[(w.preproc == "R0") & (w.model == "M1")].set_index("subset")
    order = [("G-vibA", "진동 진폭 6"), ("G-vib", "진동 전체 15"), ("G-all", "전체 23")]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for yi, (k, lab) in zip(y, order):
        p, r = w.loc[k, "pseudo_auroc_max"], w.loc[k, "real_auroc_frozen"]
        ax.plot([p, r], [yi, yi], color=GRAY_L, lw=3, zorder=1)
        ax.scatter([p], [yi], s=80, color=GRAY, zorder=3)
        ax.scatter([r], [yi], s=80, color=RED, zorder=3)
        ax.text(p, yi - 0.38, "가짜 고장 최대 %.3f" % p, ha="left", color=MUTED)
        ax.text(r, yi + 0.2, "실제 고장 %.3f" % r, ha="right", color=INK)
    ax.axvline(0.5, color=GRAY, ls=":", lw=1.5)
    ax.text(0.505, y[-1] - 0.45, "무작위 0.5", color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in order]); ax.set_ylim(-0.7, len(order) - 0.4)
    ax.set_xlim(0.40, 1.02); ax.set_xlabel("AUROC (M1 진단 모델, 같은 학습·보정 블록)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "fig_3_1_confound.png")


def fig_stress():
    r6 = T("r6_frozen_system_stress.csv")
    pick = [("none", "-", "섭동 없음"), ("offset_AI2_only", "0.5", "전류 오프셋 +0.5σ"),
            ("gain_AI2_only", "1.25", "전류 이득 ×1.25"), ("gain_all", "1.25", "전 채널 이득 ×1.25"),
            ("jitter", "0.05", "샘플링 흔들림 0.05초"), ("gain_AI0_only", "1.25", "상부 진동 이득 ×1.25"),
            ("offset_AI1_only", "0.5", "하부 진동 오프셋 +0.5σ"),
            ("offset_all", "0.5", "전 채널 오프셋 +0.5σ")]
    vals = []
    for p, a, lab in pick:
        r = r6[(r6.perturbation == p) & (r6.amount.astype(str) == a)].iloc[0]
        vals.append((lab, 100 * r.normal_red_rate, r.normal_red_ratio_vs_none))
    y = np.arange(len(vals))[::-1]
    fig, ax = plt.subplots(figsize=(W, 3.6))
    cols = [GRAY] + [BLUE] * (len(vals) - 3) + [RED] * 2
    ax.barh(y, [v for _, v, _ in vals], color=cols, height=0.6)
    for yi, (_, v, k) in zip(y, vals):
        ax.text(v + 0.12, yi, "%.2f%% (%.1f배)" % (v, k), va="center", color=INK)
    ax.set_yticks(y); ax.set_yticklabels([l for l, _, _ in vals])
    ax.set_xlim(0, 13.5); ax.set_xlabel("정상 평가 블록 빨강 오경보율(%) · 괄호는 섭동 없음 대비")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "fig_3_2_stress.png")


def fig_flow():
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
    bw, bh = 15, 11
    xs = [0, 28.3, 56.6, 85]

    def box(x, y, text, fc, tc="white", w=bw, h=bh):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6))

    yc = 22
    box(xs[0], yc, "window\n10샘플", BLUE)
    box(xs[1], yc, "1단 M3\n특징 23", BLUE)
    box(xs[2], yc, "2단 M1\n진동 15", BLUE)
    box(xs[3], yc, "빨강\n3연속", RED)
    for i in range(3):
        arrow(xs[i] + bw, yc + bh / 2, xs[i + 1], yc + bh / 2)
    ax.text((xs[1] + bw + xs[2]) / 2, yc + bh / 2 + 2.5, "p1≤0.01", ha="center", color=MUTED)
    ax.text((xs[2] + bw + xs[3]) / 2, yc + bh / 2 + 2.5, "p2≤0.01", ha="center", color=MUTED)
    box(xs[1], 2, "초록\np1>0.01", GRAY_L, INK)
    box(xs[2], 2, "노랑 · 기록\n미확인·연속 미달", GRAY_L, INK, w=bw + 10)
    ax.add_patch(plt.Rectangle((xs[0], 39), bw + 23, 9, fc="white", ec=GRAY, ls="--", lw=1.2))
    ax.text(xs[0] + (bw + 23) / 2, 43.5, "관측부족·보류 (운영 제안)", ha="center", va="center", color=MUTED)
    arrow(xs[1] + bw / 2, yc, xs[1] + bw / 2, 13)
    arrow(xs[2] + bw / 2, yc, xs[2] + bw / 2, 13)
    arrow(xs[0] + bw / 2, yc + bh, xs[0] + bw / 2, 39)
    ax.text(xs[0] + bw / 2 + 1.5, 35.5, "window<3", color=MUTED)
    save(fig, "fig_4_1_flow.png")


def main():
    print("보고서 그림 생성")
    fig_bursts(); fig_reduction(); fig_candidates(); fig_confound(); fig_stress(); fig_flow()


if __name__ == "__main__":
    main()

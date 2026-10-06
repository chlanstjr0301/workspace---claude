# -*- coding: utf-8 -*-
"""
논문 Problem Formulation 절을 뒷받침하는 프로토콜 실험 2종

E-A  무작위 점수 베이스라인 (Kim et al., AAAI 2022 의 권고)
     "개선은 적절한 베이스라인 대비로 평가되어야 한다." 균등 무작위 이상점수가
     같은 임계값 절차에서 얻는 F1 을 보고한다.

E-B  Point adjustment (PA) 의 영향
     PA 는 이상 구간 S_m 안에서 한 번이라도 임계값을 넘으면 그 구간 전체를
     탐지로 간주하는 관행이다. 본 연구는 PA 를 쓰지 않는다. 쓰면 어떻게 되는지를
     우리 데이터에서 직접 보인다 — 무작위 점수조차 F1 이 1 에 가까워진다.

구간 S_m 은 고장 버스트 17개로 정의한다(이 데이터에서 자연스러운 이상 구간 단위).

usage: python make_protocol_tables.py
출력:  outputs/tables/d5_random_baseline.csv, d6_point_adjustment.csv
"""
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "tables")
PRED = os.path.join(HERE, "outputs", "predictions.csv")
N_SEED = 20


def prf(yhat, y):
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def apply_pa(yhat, y, seg):
    """Point adjustment: 구간 seg 안에서 한 번이라도 1이면 그 구간 전체를 1로."""
    out = yhat.copy()
    for s in np.unique(seg[y == 1]):
        m = (seg == s) & (y == 1)
        if out[m].any():
            out[m] = 1
    return out


def best_f1(score, y, seg=None, pa=False, n_grid=400):
    """임계값을 전수 탐색해 달성 가능한 최대 F1 (논문들이 흔히 보고하는 상한)."""
    qs = np.quantile(score, np.linspace(0.0, 1.0, n_grid))
    best = (0.0, None)
    for d in np.unique(qs):
        yh = (score > d).astype(int)
        if pa:
            yh = apply_pa(yh, y, seg)
        f1 = prf(yh, y)[2]
        if f1 > best[0]:
            best = (f1, d)
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(PRED)
    ev = d[((d.source == "normal") & (d.split == "holdout")) |
           (d.source == "outlier")].reset_index(drop=True)
    y = (ev.source == "outlier").astype(int).values
    # 이상 구간 S_m = 고장 버스트. 정상 window 는 구간에 속하지 않으므로 -1.
    seg = np.where(y == 1, ev.burst_id.values, -1)
    n_seg = len(np.unique(seg[y == 1]))
    print("평가 집합: 정상 holdout %d + 고장 %d window, 이상 구간 %d개"
          % ((y == 0).sum(), (y == 1).sum(), n_seg))

    ours = ev.score_stage1.values
    rng = np.random.default_rng(0)
    rand = [rng.random(len(ev)) for _ in range(N_SEED)]

    # --- E-A 무작위 베이스라인 ------------------------------------------- #
    rows = []
    f1_o, th_o = best_f1(ours, y)
    p_o, r_o, _ = prf((ours > th_o).astype(int), y)
    rows.append(dict(scorer="제출 1단 점수", protocol="PA 미적용",
                     f1_mean=f1_o, f1_sd=0.0, precision=p_o, recall=r_o))
    fr = [best_f1(s, y)[0] for s in rand]
    rows.append(dict(scorer="균등 무작위", protocol="PA 미적용",
                     f1_mean=float(np.mean(fr)), f1_sd=float(np.std(fr)),
                     precision=np.nan, recall=np.nan))

    # 운영 임계값(conformal alpha=0.01)에서의 제출본
    yh = (ev.p_normal_stage1 <= 0.01).astype(int).values
    p1, r1, f1c = prf(yh, y)
    rows.append(dict(scorer="제출 1단 (conformal α=0.01)", protocol="PA 미적용",
                     f1_mean=f1c, f1_sd=0.0, precision=p1, recall=r1))

    # --- E-B Point adjustment -------------------------------------------- #
    f1_opa, _ = best_f1(ours, y, seg, pa=True)
    rows.append(dict(scorer="제출 1단 점수", protocol="PA 적용",
                     f1_mean=f1_opa, f1_sd=0.0, precision=np.nan, recall=np.nan))
    fpa = [best_f1(s, y, seg, pa=True)[0] for s in rand]
    rows.append(dict(scorer="균등 무작위", protocol="PA 적용",
                     f1_mean=float(np.mean(fpa)), f1_sd=float(np.std(fpa)),
                     precision=np.nan, recall=np.nan))

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "d5_random_baseline.csv"), index=False, encoding="utf-8-sig")
    print("\n%-30s %-10s %8s %8s" % ("점수", "프로토콜", "F1", "sd"))
    for r in rows:
        print("%-30s %-10s %8.4f %8.4f" % (r["scorer"], r["protocol"], r["f1_mean"], r["f1_sd"]))

    # --- PA 가 구간 길이에 따라 얼마나 부풀리는가 ------------------------- #
    seg_len = pd.Series(seg[y == 1]).value_counts().sort_values()
    rows2 = []
    for frac in (0.25, 0.5, 1.0):
        keep = set(seg_len.index[:max(1, int(len(seg_len) * frac))])
        m = (y == 0) | np.isin(seg, list(keep))
        yy, ss = y[m], seg[m]
        f1n = np.mean([best_f1(s[m], yy)[0] for s in rand[:5]])
        f1p = np.mean([best_f1(s[m], yy, ss, pa=True)[0] for s in rand[:5]])
        rows2.append(dict(subset="짧은 구간 %d%%" % int(frac * 100),
                          n_segment=len(keep),
                          median_len=float(seg_len[list(keep)].median()),
                          random_f1=f1n, random_f1_pa=f1p,
                          inflation=f1p / f1n if f1n > 0 else np.nan))
    t2 = pd.DataFrame(rows2)
    t2.to_csv(os.path.join(OUT, "d6_point_adjustment.csv"), index=False, encoding="utf-8-sig")
    print("\n[PA 부풀림이 구간 길이에 따라 커지는가]")
    print(t2.to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\n저장: outputs/tables/d5_random_baseline.csv, d6_point_adjustment.csv")


if __name__ == "__main__":
    main()

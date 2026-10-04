# -*- coding: utf-8 -*-
"""
A6 — 시간적 통합 규칙 (N개 중 M개) 의 지연·미탐·오경보 맞바꿈 표

현재 제출본의 빨강 규칙은 "연속 3 window" 다. 이를 (N, M) 으로 일반화해
  · 이상 버스트 탐지 수
  · 빨강 window recall
  · 정상 holdout 오경보 (window / 이벤트)
  · 경보 지연 (버스트 시작 -> 첫 빨강, 초)
를 함께 보고한다.

판정 규칙은 제출본과 동일하다. window 가 '후보'가 되려면
  p_normal_stage1 <= red_p  AND  p_normal_stage2 <= red_p
이고, 버스트 내부에서 연속한 N개 window 중 M개 이상이 후보이면 그 시점에 빨강.
N 은 버스트를 넘지 않는다(버스트 경계에서 창이 끊긴다).

usage: python plan_d_submission/make_mofn_table.py
출력:  outputs/tables/d4_mofn_tradeoff.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "tables")
PRED = os.path.join(HERE, "outputs", "predictions.csv")
CFG = os.path.join(HERE, "config.yaml")

STRIDE_SEC = 0.1          # window stride = 1 sample = 0.1 s
GRID = [(1, 1), (2, 2), (3, 2), (3, 3), (4, 3), (5, 3), (5, 4), (5, 5)]


def fire_mofn(cand, n, m):
    """후보 불린 배열에 (N=n, M=m) 규칙 적용 -> 빨강 불린 배열. 버스트 내부에서만."""
    c = np.asarray(cand, dtype=int)
    out = np.zeros(len(c), dtype=bool)
    if len(c) < n:
        return out
    csum = np.concatenate(([0], np.cumsum(c)))
    for i in range(n - 1, len(c)):
        if csum[i + 1] - csum[i + 1 - n] >= m:
            out[i] = True
    return out


def apply_by_burst(df, n, m):
    red = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("burst_id").groups.items():
        pos = df.index.get_indexer(idx)
        red[pos] = fire_mofn(df.loc[idx, "cand"].values, n, m)
    return red


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    red_p = cfg["calibration"]["red_p"]
    cur_k = cfg["calibration"]["red_consecutive"]

    d = pd.read_csv(PRED)
    d["cand"] = (d.p_normal_stage1 <= red_p) & (d.p_normal_stage2 <= red_p)

    fault = d[d.source == "outlier"].reset_index(drop=True)
    norm = d[(d.source == "normal") & (d.split == "holdout")].reset_index(drop=True)
    for f in (fault, norm):
        f["t"] = pd.to_datetime(f.time_start)

    n_burst_fault = fault.burst_id.nunique()
    norm_hours = len(norm) * STRIDE_SEC / 3600.0
    print("고장 window %d / 버스트 %d · 정상 holdout window %d (%.3f h)"
          % (len(fault), n_burst_fault, len(norm), norm_hours))
    print("빨강 후보(1단∧2단 p<=%.2f): 고장 %d · 정상 %d\n"
          % (red_p, fault.cand.sum(), norm.cand.sum()))

    rows = []
    for n, m in GRID:
        rf = apply_by_burst(fault, n, m)
        rn = apply_by_burst(norm, n, m)

        det_bursts = fault.loc[rf, "burst_id"].nunique()
        # 경보 지연: 버스트별 첫 빨강 window 의 버스트 내 순번 x stride
        delays = []
        for b, g in fault.assign(red=rf).groupby("burst_id"):
            w = np.where(g.red.values)[0]
            if len(w):
                delays.append(w[0] * STRIDE_SEC)
        # 정상 오경보 이벤트 = 연속 빨강 구간 수 (버스트 내)
        ev = 0
        for b, g in norm.assign(red=rn).groupby("burst_id"):
            v = g.red.values
            ev += int(np.sum(v & ~np.concatenate(([False], v[:-1]))))

        rows.append(dict(
            N=n, M=m, rule="연속 %d" % n if n == m else "%d중 %d" % (n, m),
            burst_detected=det_bursts, burst_total=n_burst_fault,
            window_recall=rf.sum() / len(fault),
            fp_window=int(rn.sum()), fp_rate=rn.sum() / len(norm),
            fp_events=ev, far_per_h=ev / norm_hours if norm_hours else np.nan,
            delay_median_s=float(np.median(delays)) if delays else np.nan,
            delay_max_s=float(np.max(delays)) if delays else np.nan,
            is_submitted=(n == cur_k and m == cur_k),
        ))

    t = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(os.path.join(OUT, "d4_mofn_tradeoff.csv"), index=False, encoding="utf-8-sig")

    print("%-8s %8s %10s %9s %8s %9s %9s %9s"
          % ("규칙", "버스트", "win recall", "FP win", "FP율", "FP이벤트", "지연중앙", "지연최대"))
    for r in rows:
        mark = "  <= 제출본" if r["is_submitted"] else ""
        print("%-8s %5d/%-3d %9.3f %8d %8.4f %8d %9.1f %9.1f%s"
              % (r["rule"], r["burst_detected"], r["burst_total"], r["window_recall"],
                 r["fp_window"], r["fp_rate"], r["fp_events"],
                 r["delay_median_s"], r["delay_max_s"], mark))
    print("\n저장: outputs/tables/d4_mofn_tradeoff.csv")


if __name__ == "__main__":
    main()

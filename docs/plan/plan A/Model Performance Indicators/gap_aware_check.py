# -*- coding: utf-8 -*-
"""
gap-aware windowing 적용 후 BL-0 재측정 + windowing 선택지 ablation.

usage:
    python gap_aware_check.py --runs       # 연속 구간(burst) 구조 분석
    python gap_aware_check.py --ablation   # windowing 조합별 BL-0 재측정
    python gap_aware_check.py --burst      # burst 단위(L3 대용) 평가
    python gap_aware_check.py              # 전부

gap-aware windowing 규칙
    연속 샘플 간격(dt)이 GAP_SEC 를 초과하면 그 지점에서 시계열을 끊는다.
    window 는 끊긴 구간(run) 내부에서만 생성한다. run 을 가로지르는 window 는 버린다.
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "data", "raw"))

COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
SEQ = 20
OFFSET = 100       # 가이드북 label offset
GAP_SEC = 0.5


def load():
    n = pd.read_csv(os.path.join(RAW, "press_data_normal.csv"),
                    index_col=0, parse_dates=["TimeStamp"])
    o = pd.read_csv(os.path.join(RAW, "press_data_outlier.csv"),
                    index_col=0, parse_dates=["TimeStamp"])
    return n, o


def run_bounds(df):
    """dt > GAP_SEC 로 끊은 연속 구간 [(start, end), ...] (end 배타)."""
    dt = df.TimeStamp.diff().dt.total_seconds().values
    brk = np.where(dt > GAP_SEC)[0]
    edges = np.concatenate(([0], brk, [len(df)]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


# --------------------------------------------------------------------------- #
# 연속 구간 구조
# --------------------------------------------------------------------------- #
def analyse_runs(n, o):
    print("=" * 78)
    print("연속 구간(burst) 구조")
    print("=" * 78)
    for name, d in (("normal", n), ("outlier", o)):
        runs = np.array([b - a for a, b in run_bounds(d)])
        total_s = (d.TimeStamp.max() - d.TimeStamp.min()).total_seconds()
        dt = d.TimeStamp.diff().dt.total_seconds()
        gap_s = dt[dt > GAP_SEC].sum()
        print("\n### %s  rows=%d  runs=%d" % (name, len(d), len(runs)))
        print("  run 길이(샘플) : min=%d p25=%.0f med=%.0f p75=%.0f max=%d mean=%.1f"
              % (runs.min(), np.percentile(runs, 25), np.median(runs),
                 np.percentile(runs, 75), runs.max(), runs.mean()))
        print("  run 길이 == 50 : %d개 (%.1f%%)"
              % ((runs == 50).sum(), 100 * (runs == 50).mean()))
        print("  공백 총합      : %.0f s / 전체 %.0f s = %.1f%%"
              % (gap_s, total_s, 100 * gap_s / total_s))
        print("  실측 데이터 시간: %.0f s (= rows x 0.1 s)" % (0.1 * len(d)))
        for w in (20, 10, 5):
            ok = int(np.clip(runs - w + 1, 0, None).sum())
            naive = len(d) - w + 1
            print("  window=%2d -> gap-free %5d / naive %5d (잔존 %.1f%%), "
                  "run>=w 인 run %d개"
                  % (w, ok, naive, 100 * ok / naive, int((runs >= w).sum())))


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def windows_naive(df, seq=SEQ, offset=OFFSET):
    X = df[COLS].values
    n = len(X) - seq - offset
    idx = np.arange(max(n, 0))
    return (np.stack([X[i:i + seq] for i in idx]) if len(idx) else
            np.empty((0, seq, len(COLS)))), idx


def windows_gap_aware(df, seq=SEQ, offset=OFFSET):
    """run 내부에서만 window 생성. offset 은 전체 꼬리에서 잘라냄(가이드북 호환)."""
    X = df[COLS].values
    limit = len(X) - offset
    starts = []
    for a, b in run_bounds(df):
        b = min(b, limit)
        if b - a >= seq:
            starts.extend(range(a, b - seq + 1))
    starts = np.array(starts, dtype=int)
    W = (np.stack([X[i:i + seq] for i in starts]) if len(starts) else
         np.empty((0, seq, len(COLS))))
    return W, starts


def bl0_predict(W, lo, hi):
    """BL-0: window 내 어느 한 채널이라도 학습구간 min/max 를 벗어나면 이상."""
    if len(W) == 0:
        return np.zeros(0, dtype=int)
    return ((W < lo) | (W > hi)).any(axis=(1, 2)).astype(int)


def report(label, pred_n, pred_a, extra=""):
    tp = int(pred_a.sum()); fn = len(pred_a) - tp
    fp = int(pred_n.sum()); tn = len(pred_n) - fp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    print("%-42s N=%4d/%4d | TP=%4d FN=%3d FP=%4d TN=%4d | "
          "P=%.4f R=%.4f F1=%.4f FPR=%.4f %s"
          % (label, len(pred_n), len(pred_a), tp, fn, fp, tn, p, r, f1, fpr, extra))
    return f1


# --------------------------------------------------------------------------- #
# ablation
# --------------------------------------------------------------------------- #
def ablation(n, o):
    print("\n" + "=" * 78)
    print("BL-0 재측정 — windowing 선택지 ablation")
    print("=" * 78)
    print("BL-0 = window 내 한 채널이라도 학습구간 min/max 이탈 시 이상 (학습 없음)\n")

    cases = [
        # (라벨, windowing, offset, train_end, test_start, valid_n, valid_a)
        ("A 기존(naive, 가이드북 분할)", windows_naive, OFFSET, 15000, 15000, 880, 300),
        ("B naive, valid 예약 없음",     windows_naive, OFFSET, 15000, 15000, 0, 0),
        ("C gap-aware, offset=100",      windows_gap_aware, OFFSET, 15000, 15000, 0, 0),
        ("D gap-aware, offset=0",        windows_gap_aware, 0, 15000, 15000, 0, 0),
        ("E gap-aware, 프로토콜 6.1",     windows_gap_aware, 0, 12000, 15000, 0, 0),
    ]
    for label, wf, off, tr_end, te_start, vn, va in cases:
        tr = n.iloc[:tr_end]
        lo, hi = tr[COLS].min().values, tr[COLS].max().values
        Wn, _ = wf(n.iloc[te_start:], SEQ, off)
        Wa, _ = wf(o, SEQ, off)
        if va >= len(Wa):
            print("%-42s SKIP — valid 예약 %d개 > 가용 이상 window %d개"
                  % (label, va, len(Wa)))
            continue
        report(label, bl0_predict(Wn[vn:], lo, hi), bl0_predict(Wa[va:], lo, hi))

    print("\n%-42s N=4000/ 180 | TP= 154 FN= 26 FP=  78 TN=3922 | "
          "P=0.6638 R=0.8556 F1=0.7476 FPR=0.0195  (문헌값)"
          % "참고: 가이드북 LSTM-AE")


# --------------------------------------------------------------------------- #
# burst 단위 평가 (L3 대용)
# --------------------------------------------------------------------------- #
def burst_eval(n, o):
    print("\n" + "=" * 78)
    print("burst 단위 평가 — 연속 구간 1개 = 관측 1건 (L3 대용)")
    print("=" * 78)
    tr = n.iloc[:15000]
    lo, hi = tr[COLS].min().values, tr[COLS].max().values

    def burst_pred(df, offset_rows=0):
        X = df[COLS].values
        out = []
        for a, b in run_bounds(df):
            if b - a < SEQ:
                continue
            W = np.stack([X[i:i + SEQ] for i in range(a, b - SEQ + 1)])
            out.append(int(bl0_predict(W, lo, hi).any()))
        return np.array(out)

    pn = burst_pred(n.iloc[15000:])
    pa = burst_pred(o)
    print("\nburst 수: 정상 테스트 %d개 / 이상 %d개 (길이>=%d 인 burst만)"
          % (len(pn), len(pa), SEQ))
    report("BL-0 burst 단위", pn, pa)

    dur_h = 0.1 * len(n.iloc[15000:]) / 3600.0
    print("\n정상 테스트 실측 데이터 시간 = %.3f h" % dur_h)
    print("FAR/h (burst 단위 오경보) = %.2f 건/h" % (pn.sum() / dur_h if dur_h else 0))
    print("\n[주의] 이상 burst %d개는 모두 같은 1건의 고장 기록에서 나온 것이므로"
          " 독립 표본이 아니다. 실질 이벤트 수는 여전히 1." % len(pa))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--burst", action="store_true")
    a = ap.parse_args()
    allm = not (a.runs or a.ablation or a.burst)
    n, o = load()
    if a.runs or allm:
        analyse_runs(n, o)
    if a.ablation or allm:
        ablation(n, o)
    if a.burst or allm:
        burst_eval(n, o)


if __name__ == "__main__":
    main()

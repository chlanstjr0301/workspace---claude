# -*- coding: utf-8 -*-
"""
README.md 에 인용된 모든 수치를 재현하는 스크립트.

usage:
    python baseline_check.py --profile    # §1 데이터 사실
    python baseline_check.py --baseline   # §2, §7 베이스라인 보드
    python baseline_check.py              # 둘 다

가이드북(「소성가공 예지보전 AI 데이터셋」분석실습 가이드북)과 동일한 분할을 재현한다:
    train  = normal[:15000]
    window = 20 samples (2 s), stride 1, label offset +100 (10 s)
    valid  = 정상 window[:880]  + 이상 window[:300]
    test   = 정상 window[880:]  + 이상 window[300:]   -> 정상 4000 / 이상 180
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "data", "raw"))

COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
SEQ = 20          # window 길이 (샘플)
OFFSET = 100      # 가이드북의 label offset (10 s)
TRAIN_END = 15000
N_VALID_NORMAL = 880
N_VALID_ANOM = 300
GAP_SEC = 0.5     # 이 이상 벌어지면 결측구간으로 간주


def load():
    n = pd.read_csv(os.path.join(RAW, "press_data_normal.csv"),
                    index_col=0, parse_dates=["TimeStamp"])
    o = pd.read_csv(os.path.join(RAW, "press_data_outlier.csv"),
                    index_col=0, parse_dates=["TimeStamp"])
    return n, o


# --------------------------------------------------------------------------- #
# §1  데이터 프로파일
# --------------------------------------------------------------------------- #
def profile(n, o):
    print("=" * 78)
    print("§1  지표 설계 전에 확인한 데이터 사실")
    print("=" * 78)
    for name, d in (("press_data_normal.csv", n), ("press_data_outlier.csv", o)):
        dt = d.TimeStamp.diff().dt.total_seconds().dropna()
        dur = (d.TimeStamp.max() - d.TimeStamp.min()).total_seconds()
        print("\n### %s  shape=%s" % (name, d.shape))
        print("  Equipment_state  : %s" % d.Equipment_state.value_counts().to_dict())
        print("  기간             : %s ~ %s  (%.0f s)"
              % (d.TimeStamp.min(), d.TimeStamp.max(), dur))
        print("  dt 중앙값        : %.4f s  (min %.4f / max %.4f)"
              % (dt.median(), dt.min(), dt.max()))
        print("  %.1fs 초과 공백  : %d개" % (GAP_SEC, (dt > GAP_SEC).sum()))
        print("  결측치           : %d" % d.isna().sum().sum())
        print("  중복 타임스탬프  : %d" % d.TimeStamp.duplicated().sum())
        print(d[COLS].describe().T.to_string(float_format=lambda x: "%.4f" % x))


# --------------------------------------------------------------------------- #
# window 생성 / 지표
# --------------------------------------------------------------------------- #
def make_windows(df, seq=SEQ, offset=OFFSET):
    """가이드북과 동일: stride 1, 끝에서 offset 만큼 잘라냄 (gap 미고려)."""
    X = df[COLS].values
    n = len(X) - seq - offset
    return np.stack([X[i:i + seq] for i in range(n)])


def scores(name, pred_normal, pred_anom):
    tp = int(pred_anom.sum())
    fn = len(pred_anom) - tp
    fp = int(pred_normal.sum())
    tn = len(pred_normal) - fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if fp + tn else 0.0
    print("%-46s TP=%4d FN=%3d FP=%4d TN=%4d | P=%.4f R=%.4f F1=%.4f Acc=%.4f FPR=%.4f"
          % (name, tp, fn, fp, tn, prec, rec, f1, acc, fpr))
    return dict(name=name, tp=tp, fn=fn, fp=fp, tn=tn,
                precision=prec, recall=rec, f1=f1, accuracy=acc, fpr=fpr)


# --------------------------------------------------------------------------- #
# §2 / §7  베이스라인
# --------------------------------------------------------------------------- #
def baseline(n, o):
    print("\n" + "=" * 78)
    print("§2  Window F1 포화 증명  /  §7 베이스라인 보드")
    print("=" * 78)

    train = n.iloc[:TRAIN_END][COLS]
    lo, hi = train.min().values, train.max().values

    print("\n[학습구간 min/max]")
    for c, l, h in zip(COLS, lo, hi):
        print("  %-14s [%.4f, %.4f]" % (c, l, h))

    # --- 샘플 단위 범위 이탈률 (README §2 두 번째 표) ---------------------- #
    print("\n[학습구간 범위 이탈률]")
    test_normal_raw = n.iloc[TRAIN_END:]
    for i, c in enumerate(COLS):
        r_n = ((test_normal_raw[c] < lo[i]) | (test_normal_raw[c] > hi[i])).mean()
        r_o = ((o[c] < lo[i]) | (o[c] > hi[i])).mean()
        print("  %-14s 정상 테스트 %.3f%%   이상 %.3f%%" % (c, 100 * r_n, 100 * r_o))

    # --- window 구성 (가이드북 분할) -------------------------------------- #
    Wn = make_windows(test_normal_raw)
    Wo = make_windows(o)
    Wtr = make_windows(n.iloc[:TRAIN_END])
    print("\n[window] 정상 %d / 이상 %d  (학습 %d)" % (len(Wn), len(Wo), len(Wtr)))

    tn_w = Wn[N_VALID_NORMAL:]
    ta_w = Wo[N_VALID_ANOM:]
    print("[test ] 정상 %d / 이상 %d" % (len(tn_w), len(ta_w)))

    def out_of_range(W):
        return ((W < lo) | (W > hi)).any(axis=(1, 2)).astype(int)

    def win_std(W, ch):
        return W[:, :, ch].std(axis=1)

    print("\n[베이스라인 보드]")
    rows = [scores("BL-0  범위 이탈 1줄 규칙", out_of_range(tn_w), out_of_range(ta_w))]

    for q in (0.99, 0.995, 0.999):
        th = np.quantile(win_std(Wtr, 0), q)
        rows.append(scores("BL-1  AI0 win-std > train q%.3f" % q,
                           (win_std(tn_w, 0) > th).astype(int),
                           (win_std(ta_w, 0) > th).astype(int)))

    print("%-46s TP= 154 FN= 26 FP=  78 TN=3922 | P=0.6638 R=0.8556 F1=0.7476 "
          "Acc=0.9751 FPR=0.0195" % "BL-2  가이드북 LSTM-AE (문헌값)")

    # --- window 단위 범위 이탈률 (README §2 두 번째 표 마지막 행) --------- #
    print("\n[window(20) 어느 한 채널이라도 범위 이탈]")
    print("  정상 테스트 %.3f%%   이상 %.3f%%"
          % (100 * out_of_range(Wn).mean(), 100 * out_of_range(Wo).mean()))

    # --- §4.4 유병률 민감도 표 -------------------------------------------- #
    print("\n[§4.4  유병률에 따른 Precision 붕괴  (TPR=0.75, FPR=0.001)]")
    tpr, fpr = 0.75, 0.001
    for pi in (0.043, 0.01, 0.001, 0.0001):
        p = tpr * pi / (tpr * pi + fpr * (1 - pi))
        print("  유병률 %7.4f%%  ->  Precision %.3f" % (100 * pi, p))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    run_all = not (a.profile or a.baseline)

    n, o = load()
    if a.profile or run_all:
        profile(n, o)
    if a.baseline or run_all:
        baseline(n, o)


if __name__ == "__main__":
    main()

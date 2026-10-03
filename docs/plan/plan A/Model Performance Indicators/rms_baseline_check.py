# -*- coding: utf-8 -*-
"""
RMS 베이스라인 실험  (공격2 §8 전략 1 / 전략 3)

주장:
  "가이드북 LSTM-AE(F1 0.7476)는 학습 없는 윈도우 RMS 임계값 규칙으로 재현/상회된다.
   따라서 LSTM-AE의 재구성 오차는 시간 패턴이 아니라 진폭을 재고 있다."

부속 주장:
  RMS = sqrt(mean(x^2)) 는 abs() 에 대해 정확히 불변이다 (|x|^2 == x^2).
  즉 가이드북이 abs() 로 파괴한 정보(부호/위상)는 RMS 가 애초에 쓰지 않는 정보다.
  LSTM-AE 가 RMS 를 이길 유일한 근거였던 위상 구조를 전처리 단계에서 스스로 버렸다.

usage:
    python rms_baseline_check.py              # 전부
    python rms_baseline_check.py --absinv     # abs 불변성만
    python rms_baseline_check.py --board      # 베이스라인 보드만
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "data", "raw"))

COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
SHORT = ["AI0", "AI1", "CUR"]
SEQ = 20            # 2 s
OFFSET = 100        # 가이드북 label offset (10 s)
TRAIN_END = 15000
N_VALID_NORMAL = 880
N_VALID_ANOM = 300
GAP_SEC = 0.5

LSTM_AE = dict(tp=154, fn=26, fp=78, tn=3922,
               precision=0.6638, recall=0.8556, f1=0.7476, acc=0.9751, fpr=0.0195)


def load():
    n = pd.read_csv(os.path.join(RAW, "press_data_normal.csv"),
                    index_col=0, parse_dates=["TimeStamp"])
    o = pd.read_csv(os.path.join(RAW, "press_data_outlier.csv"),
                    index_col=0, parse_dates=["TimeStamp"])
    return n, o


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def windows_naive(df, seq=SEQ, offset=OFFSET):
    X = df[COLS].values
    n = len(X) - seq - offset
    return np.stack([X[i:i + seq] for i in range(n)]) if n > 0 else np.empty((0, seq, 3))


def windows_gap_aware(df, seq=SEQ, offset=OFFSET):
    """dt > GAP_SEC 지점에서 시계열을 끊고, 끊긴 구간을 넘지 않는 윈도우만 생성."""
    X = df[COLS].values
    dt = df.TimeStamp.diff().dt.total_seconds().values
    brk = np.where(dt > GAP_SEC)[0]
    bounds, s = [], 0
    for b in brk:
        bounds.append((s, b))
        s = b
    bounds.append((s, len(X)))
    out = []
    for s, e in bounds:
        n = (e - s) - seq - offset
        for i in range(max(n, 0)):
            out.append(X[s + i:s + i + seq])
    return np.stack(out) if out else np.empty((0, seq, 3))


# --------------------------------------------------------------------------- #
# 윈도우 특징
# --------------------------------------------------------------------------- #
def feat_rms(W):
    return np.sqrt((W ** 2).mean(axis=1))


def feat_std(W):
    return W.std(axis=1)


def feat_mean(W):
    """부호 있는 평균 = DC 오프셋. abs() 가 정확히 이것을 파괴한다."""
    return W.mean(axis=1)


def feat_absmean(W):
    """|평균| - 양/음 어느 쪽으로 치우쳤는지는 버리고 크기만."""
    return np.abs(W.mean(axis=1))


def feat_mav(W):
    return np.abs(W).mean(axis=1)


def feat_p2p(W):
    return W.max(axis=1) - W.min(axis=1)


def feat_peak(W):
    return np.abs(W).max(axis=1)


def feat_crest(W):
    return feat_peak(W) / (feat_rms(W) + 1e-12)


def feat_kurt(W):
    m = W.mean(axis=1, keepdims=True)
    d = W - m
    s = d.std(axis=1)
    return (d ** 4).mean(axis=1) / (s ** 4 + 1e-12)


FEATS = [("RMS", feat_rms), ("MEAN", feat_mean), ("AMEAN", feat_absmean),
         ("STD", feat_std), ("MAV", feat_mav),
         ("P2P", feat_p2p), ("PEAK", feat_peak),
         ("CREST", feat_crest), ("KURT", feat_kurt)]


# --------------------------------------------------------------------------- #
# 평가
# --------------------------------------------------------------------------- #
def scores(name, pred_n, pred_a, ap=None):
    tp = int(pred_a.sum())
    fn = len(pred_a) - tp
    fp = int(pred_n.sum())
    tn = len(pred_n) - fp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if fp + tn else 0.0
    aps = "  AP=%.4f" % ap if ap is not None else ""
    print("%-44s TP=%4d FN=%3d FP=%4d TN=%4d | P=%.4f R=%.4f F1=%.4f Acc=%.4f FPR=%.4f%s"
          % (name, tp, fn, fp, tn, p, r, f1, acc, fpr, aps))
    return dict(name=name, tp=tp, fn=fn, fp=fp, tn=tn, precision=p,
                recall=r, f1=f1, accuracy=acc, fpr=fpr, ap=ap)


def pr_cross_threshold(y, s):
    """가이드북 코드 29와 동일 취지: precision 곡선과 recall 곡선의 교점.
    정확히 p==r 인 점이 없을 수 있으므로 |p-r| 최소점을 택한다."""
    p, r, th = precision_recall_curve(y, s)
    i = int(np.argmin(np.abs(p[:-1] - r[:-1])))
    return th[i], p[i], r[i]


def skew(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std()
    return ((x - m) ** 3).mean() / (s ** 3 + 1e-12)


# --------------------------------------------------------------------------- #
# A. abs 불변성
# --------------------------------------------------------------------------- #
def absinv(n, o):
    print("=" * 104)
    print("A.  abs() 불변성 - 가이드북이 버린 정보는 RMS 가 애초에 안 쓰는 정보")
    print("=" * 104)
    W = windows_gap_aware(n.iloc[:TRAIN_END], offset=0)
    Wa = np.abs(W)
    print("  윈도우 %d개 기준, 특징별 |raw - abs| 최대 차이:" % len(W))
    for name, f in FEATS:
        d = np.abs(f(W) - f(Wa)).max()
        tag = "불변" if d < 1e-9 else "변형됨"
        print("    %-6s  max|delta| = %.3e   -> %s" % (name, d, tag))
    print()
    print("  해석:")
    print("    RMS/MAV/PEAK 는 abs 에 대해 정확히 불변 (|x|^2 == x^2).")
    print("    STD/P2P/CREST/KURT 는 변형됨 = abs 가 실제로 정보를 파괴한다는 증거.")
    print("    LSTM-AE 가 RMS 를 이길 근거는 위상/부호 구조뿐인데,")
    print("    가이드북은 그 구조를 코드 14(abs)에서 스스로 제거하고 LSTM 을 돌렸다.")
    print()
    print("  [이상 데이터 비대칭성 - abs 가 지우는 바로 그 신호]")
    for c in COLS:
        sn, so = n[c].values, o[c].values
        print("    %-14s 정상 skew=%+.4f (min %+.4f / max %+.4f)"
              % (c, skew(sn), sn.min(), sn.max()))
        print("    %-14s 이상 skew=%+.4f (min %+.4f / max %+.4f)"
              % ("", skew(so), so.min(), so.max()))


# --------------------------------------------------------------------------- #
# B. 베이스라인 보드
# --------------------------------------------------------------------------- #
def board(n, o, winfn, tag, offset=OFFSET, by_fraction=False):
    print()
    print("=" * 104)
    print("B.  RMS 베이스라인 보드  [%s]" % tag)
    print("=" * 104)

    Wtr = winfn(n.iloc[:TRAIN_END], offset=offset)
    Wn = winfn(n.iloc[TRAIN_END:], offset=offset)
    Wo = winfn(o, offset=offset)
    if by_fraction:
        # 가이드북의 valid 비율(정상 880/4880, 이상 300/480)을 그대로 유지
        kn = int(round(len(Wn) * N_VALID_NORMAL / 4880))
        ka = int(round(len(Wo) * N_VALID_ANOM / 480))
        print("[split] 가이드북 valid 비율 유지: 정상 %.1f%% / 이상 %.1f%%"
              % (100 * N_VALID_NORMAL / 4880, 100 * N_VALID_ANOM / 480))
    else:
        kn, ka = N_VALID_NORMAL, N_VALID_ANOM
    vn, tn_w = Wn[:kn], Wn[kn:]
    va, ta_w = Wo[:ka], Wo[ka:]
    print("[window] 학습 %d | 정상 valid %d / test %d | 이상 valid %d / test %d"
          % (len(Wtr), len(vn), len(tn_w), len(va), len(ta_w)))
    if len(ta_w) == 0:
        print("  ** 테스트 이상 윈도우 0개 - 이 설정은 평가 불가")
        return []

    y_test = np.r_[np.zeros(len(tn_w)), np.ones(len(ta_w))]
    y_val = np.r_[np.zeros(len(vn)), np.ones(len(va))]
    rows = []

    # --- 1) 단일 특징 x 단일 채널, 정책 (a) 가이드북 동형 임계값 ------------- #
    print()
    print("[정책 a] 임계값을 가이드북과 동일하게 결정 (valid 의 이상 라벨 사용, P-R 교점)")
    print("         -> LSTM-AE(F1 0.7476)와 '같은 조건' 비교")
    best = None
    for fname, f in FEATS:
        for ch in range(3):
            sv = np.r_[f(vn)[:, ch], f(va)[:, ch]]
            st = np.r_[f(tn_w)[:, ch], f(ta_w)[:, ch]]
            th, _, _ = pr_cross_threshold(y_val, sv)
            ap = average_precision_score(y_test, st)
            r = scores("BL-R  %-5s %s  > %.4g" % (fname, SHORT[ch], th),
                       (f(tn_w)[:, ch] > th).astype(int),
                       (f(ta_w)[:, ch] > th).astype(int), ap)
            rows.append(r)
            if best is None or r["f1"] > best["f1"]:
                best = r

    # --- 2) RMS 3채널 Mahalanobis (여전히 whitebox) ------------------------ #
    Ftr = feat_rms(Wtr)
    mu, S = Ftr.mean(axis=0), np.cov(Ftr.T)
    Si = np.linalg.inv(S)

    def maha(W):
        d = feat_rms(W) - mu
        return np.einsum("ij,jk,ik->i", d, Si, d)

    sv = np.r_[maha(vn), maha(va)]
    st = np.r_[maha(tn_w), maha(ta_w)]
    th, _, _ = pr_cross_threshold(y_val, sv)
    rows.append(scores("BL-M  RMS 3ch Mahalanobis > %.4g" % th,
                       (maha(tn_w) > th).astype(int), (maha(ta_w) > th).astype(int),
                       average_precision_score(y_test, st)))

    # --- 3) 정책 (b) 정상 데이터만으로 임계값 ------------------------------ #
    print()
    print("[정책 b] 임계값을 '학습 정상 윈도우 분위수'로만 결정 (이상 라벨 미사용)")
    print("         -> 공격2 2절에서 요구한 올바른 비지도 프로토콜")
    for ch, label in ((2, "CUR"), (0, "AI0")):
        for q in (0.99, 0.995, 0.999, 1.0):
            th = np.quantile(feat_rms(Wtr)[:, ch], q)
            rows.append(scores("BL-Q  RMS %s > train q%-5.3f = %.4g" % (label, q, th),
                               (feat_rms(tn_w)[:, ch] > th).astype(int),
                               (feat_rms(ta_w)[:, ch] > th).astype(int)))

    # --- 기준선 ----------------------------------------------------------- #
    print()
    print("[문헌값 / 기준선]")
    print("%-44s TP=%4d FN=%3d FP=%4d TN=%4d | P=%.4f R=%.4f F1=%.4f Acc=%.4f FPR=%.4f"
          % ("BL-AE 가이드북 LSTM-AE (문헌값)", LSTM_AE["tp"], LSTM_AE["fn"],
             LSTM_AE["fp"], LSTM_AE["tn"], LSTM_AE["precision"], LSTM_AE["recall"],
             LSTM_AE["f1"], LSTM_AE["acc"], LSTM_AE["fpr"]))
    prev = len(ta_w) / (len(ta_w) + len(tn_w))
    print("%-44s F1=0.0000  Acc=%.4f   (이상 유병률 %.2f%%)"
          % ("BL-MAJ 전부 정상이라 찍기", 1 - prev, 100 * prev))

    print()
    print("[요약] %s" % tag)
    print("  단일특징 최고 : %s" % best["name"].strip())
    print("                  F1=%.4f  vs  LSTM-AE %.4f   (차이 %+.4f)"
          % (best["f1"], LSTM_AE["f1"], best["f1"] - LSTM_AE["f1"]))
    return rows


def bursts(n, o):
    """수집이 연속 스트림이 아니라 짧은 burst 의 반복임을 보인다."""
    print("=" * 104)
    print("0.  수집 구조 - 이 데이터는 연속 시계열이 아니다")
    print("=" * 104)
    for name, d in (("press_data_normal.csv", n), ("press_data_outlier.csv", o)):
        dt = d.TimeStamp.diff().dt.total_seconds().values
        brk = np.where(dt > GAP_SEC)[0]
        runs = np.diff(np.r_[0, brk, len(d)])
        span = (d.TimeStamp.max() - d.TimeStamp.min()).total_seconds()
        print("  %s  rows=%d  span=%.0fs  (10Hz 연속이면 %.0fs 여야 함)"
              % (name, len(d), span, len(d) / 10))
        print("     dt 중앙값 %.3fs | %.1fs 초과 공백 %d개 | burst %d개"
              % (np.nanmedian(dt), GAP_SEC, len(brk), len(runs)))
        print("     burst 길이: min=%d  median=%d  max=%d  (= 최대 %.1f초)"
              % (runs.min(), int(np.median(runs)), runs.max(), runs.max() / 10))
        print("     seq=20+offset=100(=120샘플) 이 들어가는 burst: %d개"
              % (runs >= SEQ + OFFSET).sum())
    print()
    print("  결론: 어떤 burst 도 120 샘플에 못 미친다.")
    print("    -> 가이드북 코드 21의 'index+sequence+100 = 10초 뒤' 는 성립하지 않는다.")
    print("       행 인덱스를 균일 시간으로 착각한 것이며, +100 은 여러 burst 를 건너뛴다.")
    print("    -> naive windowing 의 20샘플 윈도우 상당수가 gap 을 가로지른다(= 가짜 2초).")
    print()


# --------------------------------------------------------------------------- #
# C. 핵심 특징 심층 - 순위력(AP) 과 임계값 선택을 분리
# --------------------------------------------------------------------------- #
SHORTLIST = [("MEAN", feat_mean, 2), ("AMEAN", feat_absmean, 2),
             ("KURT", feat_kurt, 2), ("RMS", feat_rms, 0),
             ("P2P", feat_p2p, 0), ("RMS", feat_rms, 2)]


def deepdive(n, o, winfn, tag, offset, by_fraction):
    print()
    print("=" * 104)
    print("C.  핵심 특징 심층  [%s]" % tag)
    print("=" * 104)
    Wtr = winfn(n.iloc[:TRAIN_END], offset=offset)
    Wn = winfn(n.iloc[TRAIN_END:], offset=offset)
    Wo = winfn(o, offset=offset)
    if by_fraction:
        kn = int(round(len(Wn) * N_VALID_NORMAL / 4880))
        ka = int(round(len(Wo) * N_VALID_ANOM / 480))
    else:
        kn, ka = N_VALID_NORMAL, N_VALID_ANOM
    tn_w, ta_w = Wn[kn:], Wo[ka:]
    if len(ta_w) == 0:
        print("  평가 불가")
        return
    y = np.r_[np.zeros(len(tn_w)), np.ones(len(ta_w))]

    print("  %-12s %8s %8s %8s %8s %8s" %
          ("특징", "AP", "oracleF1", "q.999F1", "q.999FPR", "q.999R"))
    print("  " + "-" * 60)
    for fname, f, ch in SHORTLIST:
        sn, sa = f(tn_w)[:, ch], f(ta_w)[:, ch]
        sc = np.r_[sn, sa]
        ap = average_precision_score(y, sc)
        # oracle: 테스트에서 달성 가능한 최대 F1 (상한, 보고용)
        pr, rc, _ = precision_recall_curve(y, sc)
        f1s = 2 * pr * rc / (pr + rc + 1e-12)
        oracle = f1s.max()
        # 정상만으로 정한 임계값
        th = np.quantile(f(Wtr)[:, ch], 0.999)
        tp = int((sa > th).sum()); fp = int((sn > th).sum())
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / len(sa)
        f1_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        print("  %-12s %8.4f %8.4f %8.4f %8.4f %8.4f"
              % ("%s %s" % (fname, SHORT[ch]), ap, oracle, f1_,
                 fp / len(sn), r_))
    print()
    print("  LSTM-AE 문헌값: F1=0.7476 (AP 보고 없음 - 가이드북은 단일 임계값만 제시)")
    print("  oracleF1 = 테스트에서 임계값을 최적으로 골랐을 때의 상한 (모델 비교용)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--absinv", action="store_true")
    ap.add_argument("--board", action="store_true")
    ap.add_argument("--bursts", action="store_true")
    a = ap.parse_args()
    run_all = not (a.absinv or a.board or a.bursts)

    n, o = load()
    if a.bursts or run_all:
        bursts(n, o)
    if a.absinv or run_all:
        absinv(n, o)
    if a.board or run_all:
        board(n, o, windows_naive, "naive windowing = 가이드북과 동일 조건",
              offset=OFFSET)
        deepdive(n, o, windows_naive, "naive windowing", OFFSET, False)
        board(n, o, windows_gap_aware,
              "gap-aware windowing, offset=0 (offset=100 은 윈도우 0개라 불가)",
              offset=0, by_fraction=True)
        deepdive(n, o, windows_gap_aware, "gap-aware windowing, offset=0",
                 0, True)


if __name__ == "__main__":
    main()

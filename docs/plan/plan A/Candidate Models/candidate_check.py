# -*- coding: utf-8 -*-
"""
후보모델 추천의 근거 수치를 재현하는 스크립트.

usage:
    python candidate_check.py --signal     # 신호 구조 (ACF 반전, 채널 상관 반전)
    python candidate_check.py --drift      # 정상 파일 내부 체제 변화
    python candidate_check.py --bench      # 후보 특징집합 / 검출기 벤치마크
    python candidate_check.py              # 전부

평가 조건 (Model Performance Indicators/README.md §6.1, §7.1-D 와 동일)
    gap-aware windowing, SEQ=20, offset=0
    train  normal[:12000] / valid normal[12000:15000] / test normal[15000:] + outlier 전체
    임계값 = 정상 검증구간 q99.9  (이상 라벨 미사용)
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "data", "raw"))

COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
SEQ = 20
GAP_SEC = 0.5
TRAIN_END, VALID_END = 12000, 15000

FEAT_NAMES = [f"{c[:3]}_{m}" for c in COLS
              for m in ("acf1", "acf2", "std", "ptp")] + ["corr01"]


def load():
    n = pd.read_csv(os.path.join(RAW, "press_data_normal.csv"),
                    index_col=0, parse_dates=["TimeStamp"])
    o = pd.read_csv(os.path.join(RAW, "press_data_outlier.csv"),
                    index_col=0, parse_dates=["TimeStamp"])
    return n, o


def runs(df):
    dt = df.TimeStamp.diff().dt.total_seconds().values
    e = np.concatenate(([0], np.where(dt > GAP_SEC)[0], [len(df)]))
    return [(int(a), int(b)) for a, b in zip(e[:-1], e[1:]) if b > a]


def _acf(x, lag):
    xc = x - x.mean()
    if xc.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(xc[:-lag], xc[lag:])[0, 1])


def window_features(df):
    """gap-aware window 별 13개 특징 + window 시작 행 인덱스."""
    X = df[COLS].values
    F, T = [], []
    for s, e in runs(df):
        for i in range(s, e - SEQ + 1):
            W = X[i:i + SEQ]
            f = []
            for k in range(3):
                f += [_acf(W[:, k], 1), _acf(W[:, k], 2),
                      float(W[:, k].std()), float(np.ptp(W[:, k]))]
            a, b = W[:, 0] - W[:, 0].mean(), W[:, 1] - W[:, 1].mean()
            f.append(float(np.corrcoef(a, b)[0, 1])
                     if a.std() > 1e-9 and b.std() > 1e-9 else 0.0)
            F.append(f)
            T.append(i)
    return np.array(F), np.array(T)


# --------------------------------------------------------------------------- #
def signal_structure(n, o):
    print("=" * 94)
    print("신호 구조 — 정상 vs 이상에서 시간상관·채널상관이 뒤집힌다")
    print("=" * 94)
    for nm, d in (("normal", n), ("outlier", o)):
        R = [(a, b) for a, b in runs(d) if b - a >= SEQ]
        print("\n### %s (run>=%d: %d개)" % (nm, SEQ, len(R)))
        X = d[COLS].values
        for k, c in enumerate(COLS):
            a1 = [_acf(X[s:e, k], 1) for s, e in R]
            a2 = [_acf(X[s:e, k], 2) for s, e in R]
            print("  %-14s lag1 ACF=%+.3f   lag2 ACF=%+.3f"
                  % (c, np.mean(a1), np.mean(a2)))
        print("  corr(AI0,AI1)=%+.3f   corr(AI0,AI2)=%+.3f   corr(AI1,AI2)=%+.3f"
              % (np.corrcoef(X[:, 0], X[:, 1])[0, 1],
                 np.corrcoef(X[:, 0], X[:, 2])[0, 1],
                 np.corrcoef(X[:, 1], X[:, 2])[0, 1]))


def drift(n, o):
    print("\n" + "=" * 94)
    print("정상 파일 내부 체제 변화 — 15000행 부근에서 운전 상태가 바뀐다")
    print("=" * 94)
    Fn, Tn = window_features(n)
    Fa, _ = window_features(o)
    show = ["AI0_acf1", "AI0_std", "AI1_acf1", "AI1_std",
            "AI2_acf1", "AI2_std", "corr01"]
    ix = [FEAT_NAMES.index(s) for s in show]
    print("\n구간(행)        " + "".join("%12s" % s for s in show))
    for lo, hi in [(0, 4000), (4000, 8000), (8000, 12000),
                   (12000, 15000), (15000, 20000)]:
        m = (Tn >= lo) & (Tn < hi)
        print("%6d-%-6d  " % (lo, hi)
              + "".join("%12.3f" % v for v in Fn[m][:, ix].mean(0)))
    print("이상 전체      " + "".join("%12.3f" % v for v in Fa[:, ix].mean(0)))
    print("\n  -> 학습구간(:15000)과 테스트구간(15000:)의 정상이 서로 다른 체제다.")
    print("     ACF·corr 기반 특징을 쓰는 단일기준 모델은 여기서 대량 오경보를 낸다.")
    j = FEAT_NAMES.index("AI2_acf1")
    print("\n[AI2_acf1] 정상 min=%.4f / 이상 max=%.4f -> %s"
          % (Fn[:, j].min(), Fa[:, j].max(),
             "완전분리(겹침 없음)" if Fn[:, j].min() > Fa[:, j].max() else "겹침 있음"))
    print("     정상 5개 구간 평균이 모두 0.927로 고정 — 체제 변화에도 꿈쩍 않는다.")
    print("     실제 운전변화에 불변인데 녹화 세션만 바뀌면 달라지는 값 = 세션 지문 의심.")


def report(name, s_norm, s_anom, th):
    pn, pa = (s_norm > th).astype(int), (s_anom > th).astype(int)
    tp, fn = int(pa.sum()), len(pa) - int(pa.sum())
    fp, tn = int(pn.sum()), len(pn) - int(pn.sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    print("  %-38s TP=%3d FN=%3d FP=%4d TN=%4d | P=%.3f R=%.3f F1=%.3f FPR=%.4f"
          % (name, tp, fn, fp, tn, P, R, F1, fp / (fp + tn) if fp + tn else 0))


def bench(n, o):
    from sklearn.ensemble import IsolationForest

    print("\n" + "=" * 94)
    print("후보 벤치마크 — 임계값은 모두 정상 검증구간 q99.9 (이상 라벨 미사용)")
    print("=" * 94)
    Fn, Tn = window_features(n)
    Fa, _ = window_features(o)
    tr = Fn[Tn < TRAIN_END]
    va = Fn[(Tn >= TRAIN_END) & (Tn < VALID_END)]
    te = Fn[Tn >= VALID_END]
    print("window: train=%d valid=%d test_normal=%d anomaly=%d"
          % (len(tr), len(va), len(te), len(Fa)))

    i = FEAT_NAMES.index
    print("\n[단일 특징 규칙]")
    for label, j, sign in [("C1  AI0_acf1 (저주파 상관 출현)", i("AI0_acf1"), +1),
                           ("C2  AI1_acf2", i("AI1_acf2"), +1),
                           ("C3  -corr01 (상·하부 역위상)", i("corr01"), -1),
                           ("C4  AI0_std (진폭)", i("AI0_std"), +1),
                           ("X   -AI2_acf1 << 세션 지문, 사용 금지", i("AI2_acf1"), -1)]:
            report(label, sign * te[:, j], sign * Fa[:, j],
                   np.quantile(sign * va[:, j], 0.999))

    print("\n[IsolationForest — 특징집합 비교 (n_estimators=300, seed=42)]")
    sets = {
        "전체 13특징": list(range(13)),
        "진동전용 9특징 (AI2 제외)": [i("AI0_acf1"), i("AI0_acf2"), i("AI0_std"),
                                i("AI0_ptp"), i("AI1_acf1"), i("AI1_acf2"),
                                i("AI1_std"), i("AI1_ptp"), i("corr01")],
        "진동 ACF+corr 5특징": [i("AI0_acf1"), i("AI0_acf2"), i("AI1_acf1"),
                            i("AI1_acf2"), i("corr01")],
        "진폭전용 6특징  <<추천": [i("AI0_std"), i("AI0_ptp"), i("AI1_std"),
                            i("AI1_ptp"), i("AI2_std"), i("AI2_ptp")],
    }
    for label, ix in sets.items():
        m = IsolationForest(n_estimators=300, random_state=42).fit(tr[:, ix])
        report("IF  " + label, -m.score_samples(te[:, ix]),
               -m.score_samples(Fa[:, ix]),
               np.quantile(-m.score_samples(va[:, ix]), 0.999))

    print("\n  참고  BL-0 범위규칙 (gap-aware)        "
          "                 | P=1.000 R=0.692 F1=0.818 FPR=0.0000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", action="store_true")
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    allm = not (a.signal or a.drift or a.bench)
    n, o = load()
    if a.signal or allm:
        signal_structure(n, o)
    if a.drift or allm:
        drift(n, o)
    if a.bench or allm:
        bench(n, o)


if __name__ == "__main__":
    main()

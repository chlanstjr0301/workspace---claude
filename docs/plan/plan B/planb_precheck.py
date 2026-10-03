# -*- coding: utf-8 -*-
"""
Plan B 사전실험 — 계획서(README.md)의 모든 수치를 재현한다.

usage:
    python planb_precheck.py            # 전부
    python planb_precheck.py --sep      # [P1] 특징별 분리력(AUROC)과 정상 내부 드리프트(E1)
    python planb_precheck.py --models   # [P2] 후보 모델 4종 + 센서 섭동(E3) + 블록 교차검증 FAR
    python planb_precheck.py --delay    # [P3] 진입시점 무작위화 탐지지연
    python planb_precheck.py --ablation # [P4] 특징군 ablation, 오경보 위치, 전류 진폭·기록 간격

공통 규칙
    gap-aware: 연속 샘플 간격 > 0.5 s 이면 끊고, window 는 끊긴 구간(run) 안에서만 만든다.
    window 길이 W = 10 샘플(1 s), stride 1.
    분할: train normal[:12000] / valid normal[12000:15000] / test normal[15000:] + outlier 전체
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("KAMP_RAW") or os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "data", "raw"))

COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
W = 10
GAP_SEC = 0.5
SEED = 42


# ----------------------------------------------------------------------------- #
def load():
    n = pd.read_csv(os.path.join(RAW, "press_data_normal.csv"), index_col=0,
                    parse_dates=["TimeStamp"])
    o = pd.read_csv(os.path.join(RAW, "press_data_outlier.csv"), index_col=0,
                    parse_dates=["TimeStamp"])
    return n.reset_index(drop=True), o.reset_index(drop=True)


def run_bounds(df):
    dt = df.TimeStamp.diff().dt.total_seconds().values
    brk = np.where(dt > GAP_SEC)[0]
    edges = np.concatenate(([0], brk, [len(df)]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def windows(df, w=W):
    """gap-aware window. 반환: X (n, w, 3), 시작 행, 끝 시각, run id"""
    X = df[COLS].values
    t = df.TimeStamp.values
    starts, rid = [], []
    for k, (a, b) in enumerate(run_bounds(df)):
        if b - a >= w:
            s = list(range(a, b - w + 1))
            starts += s
            rid += [k] * len(s)
    starts = np.array(starts, dtype=int)
    Xw = np.stack([X[i:i + w] for i in starts])
    tend = t[starts + w - 1]
    return Xw, starts, tend, np.array(rid)


def _ac1(x):
    a = x[:, :-1] - x[:, :-1].mean(1, keepdims=True)
    b = x[:, 1:] - x[:, 1:].mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), 0.0)


def _corr(x, y):
    a = x - x.mean(1, keepdims=True)
    b = y - y.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), 0.0)


# 특징군
#   A 진폭(amplitude): 센서 이득(gain)에 민감
#   S 형태(shape): lag-1 자기상관 — 이득·극성·오프셋에 불변
#   R 채널 관계(relation): 상·하부 진동 상관 — 극성에 민감
#   R' |x| 관계: 절댓값 상관 — 극성에 불변
#   O 전류 오프셋
FEATS = {
    "A_std_AI0": "A", "A_std_AI1": "A", "A_std_AI2": "A",
    "A_p2p_AI0": "A", "A_p2p_AI1": "A", "A_p2p_AI2": "A",
    "S_ac1_AI0": "S", "S_ac1_AI1": "S", "S_ac1_AI2": "S",
    "R_corr01": "R", "R_abscorr01": "R'",
    "O_mean_AI2": "O",
}


def features(Xw):
    f = {}
    for c, nm in enumerate(["AI0", "AI1", "AI2"]):
        x = Xw[:, :, c]
        f["A_std_" + nm] = x.std(1)
        f["A_p2p_" + nm] = x.max(1) - x.min(1)
        f["S_ac1_" + nm] = _ac1(x)
    f["R_corr01"] = _corr(Xw[:, :, 0], Xw[:, :, 1])
    f["R_abscorr01"] = _corr(np.abs(Xw[:, :, 0]), np.abs(Xw[:, :, 1]))
    f["O_mean_AI2"] = Xw[:, :, 2].mean(1)
    return pd.DataFrame(f)


def auroc_dir(neg, pos):
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    a = roc_auc_score(y, s)
    return max(a, 1 - a), ("+" if a >= 0.5 else "-")


def split(n):
    return n.iloc[:12000], n.iloc[12000:15000], n.iloc[15000:]


# ----------------------------------------------------------------------------- #
def sep(n, o):
    print("=" * 80)
    print("[P1] 특징별 분리력 — AUROC (0.5 = 구분 못 함, 1.0 = 완전 분리)")
    print("=" * 80)
    tr, va, te = split(n)
    Fte = features(windows(te)[0])
    Fo = features(windows(o)[0])
    Fh1 = features(windows(n.iloc[:10000])[0])
    Fh2 = features(windows(n.iloc[10000:])[0])
    print("window 수: 정상 test %d / 이상 %d / 정상 전반 %d / 정상 후반 %d\n"
          % (len(Fte), len(Fo), len(Fh1), len(Fh2)))
    print("%-14s %-3s | %-22s | %-22s | %s"
          % ("feature", "군", "정상test vs 이상", "E1 정상전반 vs 후반", "중앙값 정상→이상"))
    for k, g in FEATS.items():
        a, d = auroc_dir(Fte[k].values, Fo[k].values)
        b, e = auroc_dir(Fh1[k].values, Fh2[k].values)
        print("%-14s %-3s | AUROC %.3f (%s)        | AUROC %.3f (%s)        | %.3f → %.3f"
              % (k, g, a, d, b, e, Fte[k].median(), Fo[k].median()))

    # 정상 내부 운전상태 변화: 2000행 블록별 진동 std
    print("\n정상 데이터 2000행 블록별 window std 중앙값 (AI0 / AI1 / AI2)")
    for blk in range(10):
        F = features(windows(n.iloc[blk * 2000:(blk + 1) * 2000])[0])
        print("  블록 %d (행 %5d~%5d): %.3f / %.3f / %.1f"
              % (blk, blk * 2000, (blk + 1) * 2000 - 1, F.A_std_AI0.median(),
                 F.A_std_AI1.median(), F.A_std_AI2.median()))


# ----------------------------------------------------------------------------- #
GROUPS = {
    "M1 형태 불변(S+R')": [k for k, g in FEATS.items() if g in ("S", "R'")],
    "M2 전체 특징": list(FEATS),
}


class Maha:
    """정상 특징 분포의 마할라노비스 거리(= Hotelling T²). 화이트박스."""

    def fit(self, F):
        self.mu = F.mean(0)
        C = np.cov(F, rowvar=False) + 1e-6 * np.eye(F.shape[1])
        self.P = np.linalg.inv(C)
        return self

    def score(self, F):
        d = F - self.mu
        return np.einsum("ij,jk,ik->i", d, self.P, d)


def bl0_scores(tr):
    lo, hi = tr[COLS].min().values, tr[COLS].max().values

    def f(Xw):
        return ((Xw < lo) | (Xw > hi)).any(axis=(1, 2)).astype(float)
    return f


def build_models(tr):
    """학습구간 tr 로 4개 모델을 학습해 score 함수 dict 반환."""
    Xtr = windows(tr)[0]
    Ftr = features(Xtr)
    out = {"BL-0 범위 이탈 규칙": bl0_scores(tr)}
    for name, cols in GROUPS.items():
        mu, sd = Ftr[cols].mean(), Ftr[cols].std() + 1e-12
        m = Maha().fit(((Ftr[cols] - mu) / sd).values)
        out[name] = (lambda cols, mu, sd, m:
                     lambda Xw: m.score(((features(Xw)[cols] - mu) / sd).values))(cols, mu, sd, m)
    cols = list(FEATS)
    mu, sd = Ftr[cols].mean(), Ftr[cols].std() + 1e-12
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(((Ftr[cols] - mu) / sd).values)
    out["M3 IsolationForest"] = (lambda Xw: -iso.score_samples(((features(Xw)[cols] - mu) / sd).values))
    return out


def threshold(fn, va, q=0.999):
    s = fn(windows(va)[0])
    if set(np.unique(s)) <= {0.0, 1.0}:
        return 0.5
    return np.quantile(s, q)


def flip_ai1(o):
    """E3-a: 하부 진동 센서 극성이 바뀌었을 가능성 — AI1 부호 반전"""
    p = o.copy()
    p["AI1_Vibration"] = -p["AI1_Vibration"]
    return p


def models(n, o):
    print("\n" + "=" * 80)
    print("[P2] 후보 모델 — window 단위, 임계값 = 정상 valid 의 q99.9 (이상 데이터 미사용)")
    print("=" * 80)
    tr, va, te = split(n)
    ms = build_models(tr)
    Xte = windows(te)[0]
    Xo = windows(o)[0]
    # E3-b: 센서 이득·오프셋 차이 가능성 — 이상 데이터 채널별 평균·표준편차를 정상 학습구간과 같게 맞춤
    o_norm = o.copy()
    for c in COLS:
        o_norm[c] = (o[c] - o[c].mean()) / o[c].std() * tr[c].std() + tr[c].mean()
    o_flip = flip_ai1(o)
    Xo_flip = windows(o_flip)[0]
    Xo_norm = windows(o_norm)[0]
    print("window 수: 정상 test %d / 이상 %d\n" % (len(Xte), len(Xo)))
    print("%-22s | %6s %6s %6s %6s | %s"
          % ("모델", "Recall", "FPR", "F1", "AUROC", "E3 섭동 후 Recall (AI1 부호반전 / 진폭 정규화)"))
    for name, fn in ms.items():
        th = threshold(fn, va)
        sn, so = fn(Xte), fn(Xo)
        tp = (so > th).sum(); fn_ = len(so) - tp
        fp = (sn > th).sum(); tn = len(sn) - fp
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / len(so)
        f1 = 2 * p * r / (p + r) if p + r else 0
        auc = roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(so))], np.r_[sn, so])
        rf = (fn(Xo_flip) > th).mean()
        rn = (fn(Xo_norm) > th).mean()
        print("%-22s | %6.3f %6.4f %6.3f %6.3f | %.3f / %.3f"
              % (name, r, fp / len(sn), f1, auc, rf, rn))

    # 블록 교차검증 FAR: 정상 전체를 5블록으로 나누고 4블록 학습 → 1블록 오경보 측정
    print("\n블록 교차검증 오경보 (정상 5블록, 4블록 학습 중 마지막 25% 를 임계값용으로 사용)")
    print("알람 = window 점수 > 임계값이 연속 k=3개 (디바운스). 이벤트 = 알람 run 1개")
    blocks = np.array_split(np.arange(len(n)), 5)
    tot = {}
    hours = 0.0
    for b in range(5):
        held = n.iloc[blocks[b]]
        rest_idx = np.concatenate([blocks[j] for j in range(5) if j != b])
        rest = n.iloc[rest_idx]
        cut = int(len(rest) * 0.75)
        trb, vab = rest.iloc[:cut], rest.iloc[cut:]
        ms_b = build_models(trb)
        Xh, _, _, rid = windows(held)
        hours += 0.1 * len(held) / 3600
        for name, fn in ms_b.items():
            th = threshold(fn, vab)
            al = fn(Xh) > th
            tot[name] = tot.get(name, 0) + count_events(al, rid, k=3)
    print("정상 관측시간 합계 = %.3f h (실측 샘플 기준)" % hours)
    for name, e in tot.items():
        ub = poisson_ub(e)
        print("  %-22s 오경보 이벤트 %3d건 → FAR %.2f 건/h (95%% 상한 %.2f 건/h)"
              % (name, e, e / hours, ub / hours))


def count_events(alarm, rid, k=3):
    """같은 run 안에서 연속 k개 window 가 알람이면 이벤트 1건. run 이 바뀌면 끊는다."""
    ev, streak, fired, prev = 0, 0, False, None
    for a, r in zip(alarm, rid):
        if r != prev:
            streak, fired, prev = 0, False, r
        streak = streak + 1 if a else 0
        if not a:
            fired = False
        if streak >= k and not fired:
            ev += 1
            fired = True
    return ev


def poisson_ub(x, alpha=0.05):
    from scipy.stats import chi2
    return chi2.ppf(1 - alpha / 2, 2 * (x + 1)) / 2


# ----------------------------------------------------------------------------- #
def delay(n, o):
    print("\n" + "=" * 80)
    print("[P3] 진입시점 무작위화 탐지지연 — 이상 기록의 각 run 을 '고장 시작'으로 가정")
    print("=" * 80)
    print("지연 = 시작 run 첫 샘플 시각 → 연속 k=3 window 알람 완성 시각 (벽시계 초)")
    tr, va, _ = split(n)
    ms = build_models(tr)
    Xo, st, tend, rid = windows(o)
    t0s = o.TimeStamp.values
    runs = run_bounds(o)
    for name, fn in ms.items():
        th = threshold(fn, va)
        al = fn(Xo) > th
        ds = []
        for k_run, (a, b) in enumerate(runs):
            start_t = t0s[a]
            idx = np.where(rid >= k_run)[0]
            streak, prev, got = 0, None, None
            for i in idx:
                if rid[i] != prev:
                    streak, prev = 0, rid[i]
                streak = streak + 1 if al[i] else 0
                if streak >= 3:
                    got = (tend[i] - start_t) / np.timedelta64(1, "s")
                    break
            ds.append(got)
        ok = np.array([d for d in ds if d is not None])
        miss = sum(d is None for d in ds)
        print("  %-22s 시작점 %2d개 | 탐지 %2d / 끝까지 미탐 %d | 지연 중앙값 %.1f s, 최대 %.1f s"
              % (name, len(ds), len(ok), miss,
                 np.median(ok) if len(ok) else np.nan, ok.max() if len(ok) else np.nan))


ABL_SETS = {
    "전체 특징": list(FEATS),
    "형태 3채널 + R'": ["S_ac1_AI0", "S_ac1_AI1", "S_ac1_AI2", "R_abscorr01"],
    "전류 형태만 (S_ac1_AI2)": ["S_ac1_AI2"],
    "진동만 A+S+R' (전류 제외)": ["A_std_AI0", "A_std_AI1", "A_p2p_AI0", "A_p2p_AI1",
                                  "S_ac1_AI0", "S_ac1_AI1", "R_abscorr01"],
    "진동 형태만 S+R'": ["S_ac1_AI0", "S_ac1_AI1", "R_abscorr01"],
}


def ablation(n, o):
    print("\n" + "=" * 80)
    print("[P4] 특징군 ablation (마할라노비스, 임계값 = 정상 valid q99.9) + 오경보 위치")
    print("=" * 80)
    tr, va, te = split(n)
    Ftr = features(windows(tr)[0])
    Fva = features(windows(va)[0])
    Xte, st, _, rid = windows(te)
    Fte = features(Xte)
    Fo = features(windows(o)[0])
    print("%-26s | %6s %6s %6s | %-24s | %s"
          % ("특징군", "Recall", "FPR", "AUROC", "FP window 블록 7/8/9", "k=3 오경보 이벤트"))
    for name, c in ABL_SETS.items():
        mu, sd = Ftr[c].mean(), Ftr[c].std() + 1e-12
        m = Maha().fit(((Ftr[c] - mu) / sd).values)
        sc = lambda F: m.score(((F[c] - mu) / sd).values)
        th = np.quantile(sc(Fva), 0.999)
        sn, so = sc(Fte), sc(Fo)
        fp = sn > th
        blk = np.bincount((st[fp] + 15000) // 2000, minlength=10)[7:]
        auc = roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(so))], np.r_[sn, so])
        print("%-26s | %6.3f %6.4f %6.3f | %-24s | %d건"
              % (name, (so > th).mean(), fp.mean(), auc, str(blk.tolist()),
                 count_events(fp, rid, 3)))

    print("\n전류 진폭: |I| 99분위수  정상 %.1f  →  이상 %.1f"
          % (np.quantile(np.abs(n.AI2_Current), 0.99), np.quantile(np.abs(o.AI2_Current), 0.99)))
    for name, d in (("정상", n), ("이상", o)):
        dt = d.TimeStamp.diff().dt.total_seconds().values[1:]
        w = dt[dt <= GAP_SEC]
        print("기록 간격(run 내부) %s: 평균 %.6f s, 표준편차 %.6f s" % (name, w.mean(), w.std()))


def main():
    ap = argparse.ArgumentParser()
    for f in ("sep", "models", "delay", "ablation"):
        ap.add_argument("--" + f, action="store_true")
    a = ap.parse_args()
    allm = not (a.sep or a.models or a.delay or a.ablation)
    n, o = load()
    if a.sep or allm:
        sep(n, o)
    if a.models or allm:
        models(n, o)
    if a.delay or allm:
        delay(n, o)
    if a.ablation or allm:
        ablation(n, o)


if __name__ == "__main__":
    main()

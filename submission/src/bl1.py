# -*- coding: utf-8 -*-
"""BL-1 실행 루틴: G0(가이드북 그대로) / G1(Plan D 프로토콜).

G0 는 문헌값 재현용이며 최종 후보로 선정하지 않는다 (§4.1).
G1 은 다른 모델과 '동일 조건'에서 비교하기 위한 수정판이다.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from . import calibration as cal
from . import deep as DP
from . import evaluation as EV
from . import windows as WD

COLS3 = 3


# --------------------------------------------------------------------------- #
# G0 : 가이드북 재현
# --------------------------------------------------------------------------- #
def _naive_windows(df, channels, seq, offset):
    X = df[channels].values.astype(float)
    n = len(X) - seq - offset
    return np.stack([X[i:i + seq] for i in range(n)]) if n > 0 else \
        np.empty((0, seq, len(channels)))


def run_g0(cfg, normal, outlier, seed, epochs):
    """abs / naive window(seq=20, offset=100) / 마지막 시점 MSE /
    검증셋에 이상 300개를 넣고 P-R 교점으로 임계값 결정."""
    ch = cfg["data"]["channels"]
    seq, offset, train_end = 20, 100, 15000
    nv_n, nv_a = 880, 300

    n_abs = normal.copy()
    o_abs = outlier.copy()
    n_abs[ch] = n_abs[ch].abs()
    o_abs[ch] = o_abs[ch].abs()

    Xtr = _naive_windows(n_abs.iloc[:train_end], ch, seq, offset)
    Xn = _naive_windows(n_abs.iloc[train_end:], ch, seq, offset)
    Xo = _naive_windows(o_abs, ch, seq, offset)

    lo, rng = DP.minmax_fit(Xtr)
    Xtr, Xn, Xo = (DP.minmax_apply(a, lo, rng) for a in (Xtr, Xn, Xo))

    t0 = time.time()
    m, ran = DP.fit_lstm_ae(Xtr, cfg["deep"]["lstm_ae"], seed, epochs)
    fit_s = time.time() - t0

    e_n = DP.recon_error(m, Xn, "last")
    e_o = DP.recon_error(m, Xo, "last")
    vn, tn = e_n[:nv_n], e_n[nv_n:]
    va, ta = e_o[:nv_a], e_o[nv_a:]

    y_val = np.r_[np.zeros(len(vn)), np.ones(len(va))]
    p, r, th = precision_recall_curve(y_val, np.r_[vn, va])
    i = int(np.argmin(np.abs(p[:-1] - r[:-1])))
    thr = th[i]

    y = np.r_[np.zeros(len(tn)), np.ones(len(ta))]
    pred = (np.r_[tn, ta] > thr).astype(int)
    met = EV.window_metrics(y, pred, np.r_[tn, ta])
    return dict(variant="G0", seed=seed, epochs_ran=ran, fit_sec=round(fit_s, 1),
                threshold=float(thr), n_train=len(Xtr),
                n_test_normal=len(tn), n_test_anom=len(ta), **met)


# --------------------------------------------------------------------------- #
# G1 : Plan D 프로토콜 (다른 모델과 동일 조건)
# --------------------------------------------------------------------------- #
def run_g1(cfg, P, seed, epochs):
    """원신호 / gap-aware window / 전체 window MSE / 정상 전용 conformal."""
    mn = P["mn"]
    folds = WD.cv_folds(cfg["cv"]["n_blocks"])
    yellow = cfg["calibration"]["yellow_p"]
    rows = []
    for f in folds:
        tr = np.isin(mn["block"].values, f["train"])
        cl = mn["block"].values == f["cal"]
        ho = mn["block"].values == f["holdout"]
        lo, rng = DP.minmax_fit(P["Xn"][tr])
        Xtr = DP.minmax_apply(P["Xn"][tr], lo, rng)
        Xcl = DP.minmax_apply(P["Xn"][cl], lo, rng)
        Xho = DP.minmax_apply(P["Xn"][ho], lo, rng)
        Xo = DP.minmax_apply(P["Xo"], lo, rng)

        t0 = time.time()
        m, ran = DP.fit_lstm_ae(Xtr, cfg["deep"]["lstm_ae"], seed, epochs)
        fit_s = time.time() - t0

        s_cal = DP.recon_error(m, Xcl)
        s_ho = DP.recon_error(m, Xho)
        s_an = DP.recon_error(m, Xo)
        p_ho = cal.conformal_p(s_cal, s_ho)
        p_an = cal.conformal_p(s_cal, s_an)
        fl_ho, fl_an = p_ho <= yellow, p_an <= yellow

        fa = EV.false_alarm_profile(fl_ho, mn[ho].reset_index(drop=True))
        y = np.r_[np.zeros(ho.sum()), np.ones(len(Xo))]
        met = EV.window_metrics(y, np.r_[fl_ho, fl_an].astype(int),
                                np.r_[s_ho, s_an])
        det = EV.detection_delay(fl_an, P["mo"])
        rows.append(dict(model="BL1", pretty="BL-1 LSTM-AE (G1)",
                         variant="G1", fold=f["fold"],
                         holdout_block=f["holdout"], seed=seed,
                         epochs_ran=ran, fit_sec=round(fit_s, 1),
                         **fa, **met,
                         bursts_detected=int(sum(d["detected"] for d in det)),
                         bursts_total=len(det),
                         p_values=None))
    df = pd.DataFrame(rows).drop(columns=["p_values"])
    return df

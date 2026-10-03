# -*- coding: utf-8 -*-
import importlib.util, numpy as np
from sklearn.ensemble import IsolationForest
CM = r"C:\Users\cmsch\Desktop\대회\2026년 제6회 K-인공지능 제조데이터 분석 경진대회\docs\plan\plan A\Candidate Models\candidate_check.py"
spec = importlib.util.spec_from_file_location("cc", CM); cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
n, o = cc.load()
Fn, Tn = cc.window_features(n); Fa, _ = cc.window_features(o)
SEQ, TE, VE = 20, 12000, 15000

tr_m = Tn < TE; va_m = (Tn >= TE) & (Tn < VE); te_m = Tn >= VE
tr, va, te = Fn[tr_m], Fn[va_m], Fn[te_m]
Ttr, Tva, Tte = Tn[tr_m], Tn[va_m], Tn[te_m]
print("counts train=%d valid=%d test=%d anom=%d" % (len(tr), len(va), len(te), len(Fa)))

# --- boundary straddling windows -----------------------------------------
tr_straddle = Ttr[(Ttr + SEQ) > TE]        # train window reaching into valid rows
va_straddle = Tva[(Tva + SEQ) > VE]        # valid window reaching into test rows
print("\n[boundary leakage]")
print(" train windows overlapping valid rows: %d  starts=%s" % (len(tr_straddle), tr_straddle))
print(" valid windows overlapping test  rows: %d  starts=%s" % (len(va_straddle), va_straddle))

# --- does a straddling valid window determine the q99.9 threshold? -------
i = cc.FEAT_NAMES.index
single = [("C1 AI0_acf1", i("AI0_acf1"), +1), ("C2 AI1_acf2", i("AI1_acf2"), +1),
          ("C3 -corr01", i("corr01"), -1), ("C4 AI0_std", i("AI0_std"), +1),
          ("X  -AI2_acf1", i("AI2_acf1"), -1)]
print("\n[q99.9 determined by which valid windows? (np.quantile linear -> top2/top3)]")
pos = 0.999 * (len(va) - 1)
lo_i, hi_i = int(np.floor(pos)), int(np.ceil(pos))
print(" quantile position = %.2f -> uses sorted ranks %d and %d of %d (i.e. top %d/%d)"
      % (pos, lo_i, hi_i, len(va), len(va)-hi_i, len(va)-lo_i))
for lab, j, sg in single:
    v = sg * va[:, j]
    order = np.argsort(v)
    k = [Tva[order[lo_i]], Tva[order[hi_i]]]
    straddles = [int(x) for x in k if x + SEQ > VE]
    print("  %-14s threshold set by valid starts %s   straddling-test: %s"
          % (lab, k, straddles if straddles else "no"))
sets = {"13": list(range(13)),
        "9":  [i("AI0_acf1"),i("AI0_acf2"),i("AI0_std"),i("AI0_ptp"),i("AI1_acf1"),i("AI1_acf2"),i("AI1_std"),i("AI1_ptp"),i("corr01")],
        "5":  [i("AI0_acf1"),i("AI0_acf2"),i("AI1_acf1"),i("AI1_acf2"),i("corr01")],
        "6amp":[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp"),i("AI2_std"),i("AI2_ptp")]}
for lab, ix in sets.items():
    m = IsolationForest(n_estimators=300, random_state=42).fit(tr[:, ix])
    v = -m.score_samples(va[:, ix]); order = np.argsort(v)
    k = [Tva[order[lo_i]], Tva[order[hi_i]]]
    straddles = [int(x) for x in k if x + SEQ > VE]
    print("  IF %-11s threshold set by valid starts %s   straddling-test: %s"
          % (lab, k, straddles if straddles else "no"))

# --- all quantile call inputs: any anomaly rows? -------------------------
print("\n[quantile input audit] valid rows all from normal file, index range %d..%d (<%d): %s"
      % (Tva.min(), Tva.max(), VE, bool(Tva.max() < VE)))

# --- strict split: drop straddling windows, re-measure -------------------
def rep(nm, sn, sa, th):
    pn, pa = (sn > th).astype(int), (sa > th).astype(int)
    tp, fn_ = int(pa.sum()), len(pa)-int(pa.sum()); fp, tn = int(pn.sum()), len(pn)-int(pn.sum())
    P = tp/(tp+fp) if tp+fp else 0.; R = tp/(tp+fn_) if tp+fn_ else 0.
    F1 = 2*P*R/(P+R) if P+R else 0.
    print("  %-16s P=%.3f R=%.3f F1=%.3f FPR=%.4f" % (nm, P, R, F1, fp/(fp+tn)))
print("\n[strict: valid windows fully inside [12000,15000-20] only]")
va2_m = (Tn >= TE) & (Tn + SEQ <= VE)
va2 = Fn[va2_m]; print("  strict valid n=%d (was %d)" % (len(va2), len(va)))
for lab, j, sg in single:
    rep(lab, sg*te[:, j], sg*Fa[:, j], np.quantile(sg*va2[:, j], 0.999))
for lab, ix in sets.items():
    m = IsolationForest(n_estimators=300, random_state=42).fit(tr[:, ix])
    rep("IF "+lab, -m.score_samples(te[:, ix]), -m.score_samples(Fa[:, ix]),
        np.quantile(-m.score_samples(va2[:, ix]), 0.999))

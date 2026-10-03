# -*- coding: utf-8 -*-
import importlib.util, numpy as np
from sklearn.ensemble import IsolationForest
CM = r"C:\Users\cmsch\Desktop\대회\2026년 제6회 K-인공지능 제조데이터 분석 경진대회\docs\plan\plan A\Candidate Models\candidate_check.py"
s = importlib.util.spec_from_file_location("cc", CM); cc = importlib.util.module_from_spec(s); s.loader.exec_module(cc)
n, o = cc.load()
Fn, Tn = cc.window_features(n); Fa, _ = cc.window_features(o)
tr = Fn[Tn<12000]; va = Fn[(Tn>=12000)&(Tn<15000)]; te = Fn[Tn>=15000]
i = cc.FEAT_NAMES.index
sets = {"13": list(range(13)),
 "9": [i("AI0_acf1"),i("AI0_acf2"),i("AI0_std"),i("AI0_ptp"),i("AI1_acf1"),i("AI1_acf2"),i("AI1_std"),i("AI1_ptp"),i("corr01")],
 "5": [i("AI0_acf1"),i("AI0_acf2"),i("AI1_acf1"),i("AI1_acf2"),i("corr01")],
 "6amp": [i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp"),i("AI2_std"),i("AI2_ptp")]}
def f1(ix, seed, nest=300):
    m = IsolationForest(n_estimators=nest, random_state=seed).fit(tr[:, ix])
    th = np.quantile(-m.score_samples(va[:, ix]), 0.999)
    pa = -m.score_samples(Fa[:, ix]) > th; pn = -m.score_samples(te[:, ix]) > th
    tp=int(pa.sum()); fn_=len(pa)-tp; fp=int(pn.sum()); tn=len(pn)-fp
    P=tp/(tp+fp) if tp+fp else 0.; R=tp/(tp+fn_)
    return (2*P*R/(P+R) if P+R else 0.), P, R, fp/(fp+tn)
print("### IsolationForest seed sensitivity (n_estimators=300), seeds 0..29")
for lab, ix in sets.items():
    r = np.array([f1(ix, sd) for sd in range(30)])
    print("  %-5s F1 seed42=%.3f | seeds0-29 mean=%.3f sd=%.3f min=%.3f max=%.3f"
          % (lab, f1(ix,42)[0], r[:,0].mean(), r[:,0].std(), r[:,0].min(), r[:,0].max()))
    if lab=="6amp":
        print("        6amp FPR across seeds: mean=%.4f max=%.4f ; P min=%.3f"
              % (r[:,3].mean(), r[:,3].max(), r[:,1].min()))
print("\n### 6amp at other n_estimators (seed 42)")
for ne in (50,100,200,300,500,1000):
    F,P,R,fpr = f1(sets["6amp"], 42, ne); print("  n_est=%4d F1=%.3f P=%.3f R=%.3f FPR=%.4f" % (ne,F,P,R,fpr))

print("\n### signal_structure: channel corr computed across gaps vs run-wise")
for nm, d in (("normal", n), ("outlier", o)):
    X = d[cc.COLS].values
    glob = np.corrcoef(X[:,0], X[:,1])[0,1]
    R = [(a,b) for a,b in cc.runs(d) if b-a >= 20]
    rw = np.mean([np.corrcoef(X[a:b,0], X[a:b,1])[0,1] for a,b in R])
    print("  %-8s corr01 full-file(gaps crossed)=%+.4f   mean over runs>=20=%+.4f  (doc cites %s)"
          % (nm, glob, rw, "+0.368" if nm=="normal" else "-0.381"))

print("\n### wall-clock vs data-time of normal test segment")
t = n.TimeStamp
print("  n.iloc[15000:] wall-clock = %.1f s ; data-time (rows*0.1) = %.1f s"
      % ((t.iloc[-1]-t.iloc[15000]).total_seconds(), 0.1*5000))
print("  README §7.3 uses 0.139 h = %.0f s (data-time)" % (0.139*3600))
print("  README §4.1 uses 400 s for 78 FP -> %.0f alarms/h ; wall-clock basis -> %.0f/h"
      % (78/(400/3600), 78/((t.iloc[-1]-t.iloc[15880]).total_seconds()/3600)))

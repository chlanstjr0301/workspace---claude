# -*- coding: utf-8 -*-
import importlib.util, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
CM = r"C:\Users\cmsch\Desktop\대회\2026년 제6회 K-인공지능 제조데이터 분석 경진대회\docs\plan\plan A\Candidate Models\candidate_check.py"
s = importlib.util.spec_from_file_location("cc", CM); cc = importlib.util.module_from_spec(s); s.loader.exec_module(cc)
n, o = cc.load()

print("### 1. train[:15000] min/max  vs  full-file min/max (BL-0 threshold)")
for c in cc.COLS:
    a = n[c].iloc[:15000]; b = n[c]; t = n[c].iloc[15000:]
    print("  %-14s train[%.4f,%.4f]  full[%.4f,%.4f]  test[%.4f,%.4f]  identical=%s"
          % (c, a.min(), a.max(), b.min(), b.max(), t.min(), t.max(),
             (a.min()==b.min() and a.max()==b.max())))
    print("                 argmin row=%d argmax row=%d  (both <15000: %s)"
          % (b.idxmin(), b.idxmax(), b.idxmin()<15000 and b.idxmax()<15000))

print("\n### 2. gap median (README §1: normal 4.2s / outlier 5.4s)")
for nm, d in (("normal", n), ("outlier", o)):
    dt = d.TimeStamp.diff().dt.total_seconds().dropna()
    g = dt[dt > 0.5]
    print("  %-8s n=%d median=%.3f max=%.3f sum=%.1f" % (nm, len(g), g.median(), g.max(), g.sum()))

print("\n### 3. naive window wall-clock duration (README §1.1: 20-sample median 3.6s max 24.9s;")
print("        offset=100 median 21.7s range 13.2-47.4s)")
t = n.TimeStamp.values.astype("datetime64[ns]").astype(np.int64)/1e9
N = len(t)-20-100
dur = t[np.arange(N)+19] - t[np.arange(N)]
print("  20-sample window span : median=%.3f s  min=%.3f max=%.3f" % (np.median(dur), dur.min(), dur.max()))
off = t[np.arange(N)+19+100] - t[np.arange(N)+19]
print("  +100 offset span      : median=%.3f s  min=%.3f max=%.3f" % (np.median(off), off.min(), off.max()))
offb = t[np.arange(N)+20+100] - t[np.arange(N)+20]
print("  +100 offset (alt base): median=%.3f s  min=%.3f max=%.3f" % (np.median(offb), offb.min(), offb.max()))
print("  gap-aware 20-sample span (should be 1.9s fixed):")
_, S = None, None
Fn, Tn = cc.window_features(n)
d2 = t[Tn+19]-t[Tn]
print("    median=%.4f min=%.4f max=%.4f  unique=%s" % (np.median(d2), d2.min(), d2.max(), len(np.unique(np.round(d2,4)))))

print("\n### 4. univariate AUC (README §3: AI0_acf1 0.956 vs AI0_std 0.854)")
Fa, _ = cc.window_features(o)
te = Fn[Tn >= 15000]
for nm in cc.FEAT_NAMES:
    j = cc.FEAT_NAMES.index(nm)
    y = np.r_[np.zeros(len(te)), np.ones(len(Fa))]
    sc = np.r_[te[:, j], Fa[:, j]]
    a = roc_auc_score(y, sc)
    print("  %-10s AUC=%.4f  (|flip|=%.4f)" % (nm, a, max(a, 1-a)))

print("\n### 5. Mahalanobis 13-feature FPR (README §2.2 claims 20.5%)")
tr = Fn[Tn < 12000]; va = Fn[(Tn>=12000)&(Tn<15000)]
mu = tr.mean(0); S_ = np.cov(tr, rowvar=False)
Si = np.linalg.pinv(S_)
def md(F): 
    d = F-mu; return np.einsum("ij,jk,ik->i", d, Si, d)
th = np.quantile(md(va), 0.999)
pn = (md(te) > th); pa = (md(Fa) > th)
tp=int(pa.sum()); fn_=len(pa)-tp; fp=int(pn.sum()); tn=len(pn)-fp
P=tp/(tp+fp); R=tp/(tp+fn_)
print("  P=%.3f R=%.3f F1=%.3f FPR=%.4f  (FP=%d/%d)" % (P,R,2*P*R/(P+R),fp/(fp+tn),fp,len(pn)))

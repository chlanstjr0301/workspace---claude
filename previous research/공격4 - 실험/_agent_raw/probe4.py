# -*- coding: utf-8 -*-
import importlib.util, numpy as np
CM = r"C:\Users\cmsch\Desktop\대회\2026년 제6회 K-인공지능 제조데이터 분석 경진대회\docs\plan\plan A\Candidate Models\candidate_check.py"
s = importlib.util.spec_from_file_location("cc", CM); cc = importlib.util.module_from_spec(s); s.loader.exec_module(cc)
n, o = cc.load()
t = n.TimeStamp.values.astype("datetime64[ns]").astype(np.int64)/1e9
print("### window span variants (README §1.1 claims median 3.6 s, max 24.9 s)")
for lab, hi, lo, N in [("t[i+19]-t[i], N=len-120", 19, 0, len(t)-120),
                       ("t[i+20]-t[i], N=len-120", 20, 0, len(t)-120),
                       ("t[i+19]-t[i], N=len-19", 19, 0, len(t)-19),
                       ("t[i+20]-t[i], N=len-20", 20, 0, len(t)-20)]:
    i = np.arange(N); d = t[i+hi]-t[i+lo]
    print("  %-26s median=%.4f mean=%.4f max=%.4f" % (lab, np.median(d), d.mean(), d.max()))

print("\n### Mahalanobis FPR variants (README §2.2 claims 20.5%)")
Fn, Tn = cc.window_features(n); Fa, _ = cc.window_features(o)
tr = Fn[Tn<12000]; va = Fn[(Tn>=12000)&(Tn<15000)]; te = Fn[Tn>=15000]
for lab, inv, ddof in [("pinv, cov ddof=1", np.linalg.pinv, 1), ("inv, cov ddof=1", np.linalg.inv, 1),
                       ("pinv, cov ddof=0", np.linalg.pinv, 0), ("inv, cov ddof=0", np.linalg.inv, 0)]:
    mu = tr.mean(0); C = np.cov(tr, rowvar=False, ddof=ddof); Si = inv(C)
    f = lambda F: np.einsum("ij,jk,ik->i", F-mu, Si, F-mu)
    th = np.quantile(f(va), 0.999); pn=(f(te)>th); pa=(f(Fa)>th)
    fp=int(pn.sum()); print("  %-18s FPR=%.4f (FP=%d)  R=%.3f" % (lab, fp/len(pn), fp, pa.mean()))
    # also with train standardisation
for lab, q in [("q99.5",0.995),("q99.9",0.999),("q99.95",0.9995)]:
    mu = tr.mean(0); C = np.cov(tr, rowvar=False); Si = np.linalg.pinv(C)
    f = lambda F: np.einsum("ij,jk,ik->i", F-mu, Si, F-mu)
    th = np.quantile(f(va), q); print("  thresh %-7s FPR=%.4f" % (lab, (f(te)>th).mean()))

# -*- coding: utf-8 -*-
import sys, os, warnings, importlib.util
import numpy as np, pandas as pd
warnings.simplefilter("error")   # any RuntimeWarning -> exception

CM = r"C:\Users\cmsch\Desktop\대회\2026년 제6회 K-인공지능 제조데이터 분석 경진대회\docs\plan\plan A\Candidate Models\candidate_check.py"
GA = r"C:\Users\cmsch\Desktop\대회\2026년 제6회 K-인공지능 제조데이터 분석 경진대회\docs\plan\plan A\Model Performance Indicators\gap_aware_check.py"

def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

cc = imp("cc", CM); ga = imp("ga", GA)
n, o = cc.load()

print("=== A. runs() vs run_bounds() identity ===")
print(" normal identical:", cc.runs(n) == ga.run_bounds(n))
print(" outlier identical:", cc.runs(o) == ga.run_bounds(o))

print("\n=== B. window set identity: candidate window_features vs windows_gap_aware(offset=0) ===")
Fn, Tn = cc.window_features(n)
Fa, Ta = cc.window_features(o)
Wn0, Sn0 = ga.windows_gap_aware(n, 20, 0)
Wa0, Sa0 = ga.windows_gap_aware(o, 20, 0)
print(" normal  cc=%d ga=%d  starts equal=%s" % (len(Tn), len(Sn0), np.array_equal(Tn, Sn0)))
print(" outlier cc=%d ga=%d  starts equal=%s" % (len(Ta), len(Sa0), np.array_equal(Ta, Sa0)))

print("\n=== C. gap_aware case E test-normal windows vs candidate Tn>=15000 ===")
WnE, SnE = ga.windows_gap_aware(n.iloc[15000:], 20, 0)
cc_test = Tn[Tn >= 15000]
print(" ga case E n=%d, cc Tn>=15000 n=%d, equal(after +15000)=%s"
      % (len(SnE), len(cc_test), np.array_equal(SnE + 15000, cc_test)))

print("\n=== D. off-by-one: is break index the START of new run? ===")
dt = n.TimeStamp.diff().dt.total_seconds().values
brk = np.where(dt > 0.5)[0]
i = brk[0]
print(" first break idx=%d  dt[i]=%.4f  dt[i-1]=%.4f  dt[i+1]=%.4f" % (i, dt[i], dt[i-1], dt[i+1]))
rb = ga.run_bounds(n)
print(" run0=%s run1=%s -> run0 ends at %d (excl), run1 starts at %d" % (rb[0], rb[1], rb[0][1], rb[1][0]))
print(" => break index %d assigned as START of run1: %s" % (i, rb[1][0] == i))
# verify no window spans a gap
bad = 0
for s in Sn0:
    seg = dt[s+1:s+20]
    if (seg > 0.5).any(): bad += 1
print(" windows spanning a >0.5s gap (should be 0):", bad)

print("\n=== E. NaN / Inf in feature matrices ===")
for nm, F in (("normal", Fn), ("outlier", Fa)):
    print(" %s shape=%s  nan=%d  inf=%d" % (nm, F.shape, np.isnan(F).sum(), np.isinf(F).sum()))

print("\n=== F. _acf subarray-constant hazard ===")
# does any window have constant xc[:-lag] or xc[lag:] while full std >= 1e-9 ?
X = n[cc.COLS].values
Xo = o[cc.COLS].values
hits = 0; checked = 0
for (Xx, S) in ((X, Sn0), (Xo, Sa0)):
    for s in S:
        for k in range(3):
            w = Xx[s:s+20]; x = w[:, k]
            full = (x - x.mean()).std()
            for lag in (1, 2):
                checked += 1
                a, b = x[:-lag], x[lag:]
                if full >= 1e-9 and (a.std() < 1e-12 or b.std() < 1e-12):
                    hits += 1
print(" checked=%d  subarray-constant-but-full-varies cases=%d" % (checked, hits))
# how many windows are fully constant per channel (guard returns 0.0)
const = 0
for (Xx, S) in ((X, Sn0), (Xo, Sa0)):
    for s in S:
        for k in range(3):
            x = Xx[s:s+20]
            if (x[:, k] - x[:, k].mean()).std() < 1e-9: const += 1
print(" channel-windows hitting the 0.0 guard:", const)

print("\n=== G. duplicate timestamp ===")
d = n[n.TimeStamp.duplicated(keep=False)]
print(d[["TimeStamp"]].to_string())
print(" dt==0 count:", int((dt == 0).sum()), " dt<0 count:", int((dt < 0).sum()))

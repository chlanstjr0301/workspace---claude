# -*- coding: utf-8 -*-
import numpy as np, common
from sklearn.ensemble import IsolationForest
Fn,Tn,Bn,Fa,Ta,Ba = common.get()
i = common.FEAT_NAMES.index
tr = Fn[Tn<12000]; va = Fn[(Tn>=12000)&(Tn<15000)]; te = Fn[Tn>=15000]
AMP6 = [i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp"),i("AI2_std"),i("AI2_ptp")]
VIB4 = [i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp")]
ALL13= list(range(13))
VIB9 = [i("AI0_acf1"),i("AI0_acf2"),i("AI0_std"),i("AI0_ptp"),i("AI1_acf1"),i("AI1_acf2"),i("AI1_std"),i("AI1_ptp"),i("corr01")]
ACF5 = [i("AI0_acf1"),i("AI0_acf2"),i("AI1_acf1"),i("AI1_acf2"),i("corr01")]

def ifscore(ix, seed, ne=300):
    m = IsolationForest(n_estimators=ne, random_state=seed).fit(tr[:,ix])
    return -m.score_samples(va[:,ix]), -m.score_samples(te[:,ix]), -m.score_samples(Fa[:,ix])

print("### A1-1  IF 진폭6특징: seed 민감도 (n_estimators=300, q=0.999)")
seeds = [0,1,2,3,7,11,13,42,123,777,2024,2025,2026,31337,99991,5,17,23,55,101]
f1s=[]; rows=[]
for s in seeds:
    sv,st,sa = ifscore(AMP6,s)
    th = np.quantile(sv,0.999)
    P,R,F1,FPR = common.prf((st>th).astype(int),(sa>th).astype(int))
    f1s.append(F1); rows.append((s,P,R,F1,FPR))
for s,P,R,F1,FPR in rows: print("  seed=%-6d P=%.3f R=%.3f F1=%.4f FPR=%.4f %s"%(s,P,R,F1,FPR,"<-- 문서값" if s==42 else ("  ** BL-0(0.818) 미달" if F1<0.818 else "")))
f1s=np.array(f1s)
print("  => F1 mean=%.4f sd=%.4f min=%.4f max=%.4f | BL-0(0.8180) 이하 seed 수=%d/%d"%(f1s.mean(),f1s.std(),f1s.min(),f1s.max(),(f1s<=0.8180).sum(),len(f1s)))
print("  => 문서가 고른 seed=42 의 백분위: %.0f%% (상위 %d/%d)"%(100*(f1s<=0.8640).mean(), (f1s>0.8639).sum(), len(f1s)))

print("\n### A1-2  분위수 민감도 (seed=42, ne=300)  — 순위가 유지되는가")
qs=[0.95,0.99,0.995,0.999,0.9999,1.0]
print("  %-26s"%"후보" + "".join("%12s"%("q=%g"%q) for q in qs))
cands = {"IF 진폭6":AMP6,"IF 진동진폭4(AI2제외)":VIB4,"IF 전체13":ALL13,"IF 진동9":VIB9,"IF ACF5":ACF5}
res={}
for lab,ix in cands.items():
    sv,st,sa = ifscore(ix,42); line=[]
    for q in qs:
        th=np.quantile(sv,q); P,R,F1,FPR=common.prf((st>th).astype(int),(sa>th).astype(int)); line.append(F1)
    res[lab]=line
    print("  %-26s"%lab + "".join("%12.4f"%v for v in line))
# single features
for lab,j,sg in [("C2 AI1_acf2",i("AI1_acf2"),1),("C4 AI0_std",i("AI0_std"),1),("X -AI2_acf1",i("AI2_acf1"),-1),("C1 AI0_acf1",i("AI0_acf1"),1)]:
    line=[]
    for q in qs:
        th=np.quantile(sg*va[:,j],q); P,R,F1,FPR=common.prf((sg*te[:,j]>th).astype(int),(sg*Fa[:,j]>th).astype(int)); line.append(F1)
    res[lab]=line
    print("  %-26s"%lab + "".join("%12.4f"%v for v in line))
print("  %-26s"%"BL-0 (고정 min/max)" + "".join("%12.4f"%0.8180 for q in qs))
print("\n  분위수별 1위:")
for k,q in enumerate(qs):
    best=max(res.items(), key=lambda kv: kv[1][k])
    print("    q=%-8g 1위=%-24s F1=%.4f   (IF진폭6=%.4f, BL-0=0.8180 -> IF %s)"%(q,best[0],best[1][k],res["IF 진폭6"][k], "승" if res["IF 진폭6"][k]>0.818 else "패"))

print("\n### A1-3  n_estimators 민감도 (IF 진폭6, q=0.999), seed 10개 평균")
for ne in [50,100,200,300,500,1000,2000]:
    v=[]
    for s in seeds[:10]:
        sv,st,sa=ifscore(AMP6,s,ne); th=np.quantile(sv,0.999)
        v.append(common.prf((st>th).astype(int),(sa>th).astype(int))[2])
    v=np.array(v); print("  ne=%-5d F1 mean=%.4f sd=%.4f min=%.4f max=%.4f  (<=0.818: %d/10)"%(ne,v.mean(),v.std(),v.min(),v.max(),(v<=0.818).sum()))

print("\n### A1-4  seed x quantile 전수: IF진폭6 가 BL-0(0.818)을 이기는 비율")
grid=[]
for s in seeds:
    sv,st,sa=ifscore(AMP6,s)
    for q in [0.99,0.995,0.999,0.9999]:
        th=np.quantile(sv,q); F1=common.prf((st>th).astype(int),(sa>th).astype(int))[2]; grid.append((s,q,F1))
import collections
g=np.array([x[2] for x in grid])
print("  전체 %d개 설정: F1 mean=%.4f sd=%.4f  BL-0 초과=%d (%.0f%%)"%(len(g),g.mean(),g.std(),(g>0.818).sum(),100*(g>0.818).mean()))
for q in [0.99,0.995,0.999,0.9999]:
    sub=np.array([x[2] for x in grid if x[1]==q])
    print("    q=%-7g mean=%.4f  BL-0 초과 %d/%d"%(q,sub.mean(),(sub>0.818).sum(),len(sub)))

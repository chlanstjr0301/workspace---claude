# -*- coding: utf-8 -*-
import numpy as np, common, warnings; warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
Fn,Tn,Bn,Fa,Ta,Ba=common.get(); i=common.FEAT_NAMES.index
tr=Fn[Tn<12000]; vm=(Tn>=12000)&(Tn<15000); tm=Tn>=15000
va=Fn[vm]; te=Fn[tm]; Bte=Bn[tm]
sc=StandardScaler().fit(tr); A,V,T,X=sc.transform(tr),sc.transform(va),sc.transform(te),sc.transform(Fa)

print("### A6-1  PCA-SPE (BL-1, 문서가 '측정 안 한' 후보) — n_components 전수")
print("  nc  설명분산   | q=0.995         q=0.999        q=0.9999      | 비고")
for nc in range(1,13):
    p=PCA(n_components=nc).fit(A)
    spe=lambda z: ((z-p.inverse_transform(p.transform(z)))**2).sum(1)
    sv,st,sa=spe(V),spe(T),spe(X); out=[]
    for q in (0.995,0.999,0.9999):
        th=np.quantile(sv,q); P,R,F1,FPR=common.prf((st>th).astype(int),(sa>th).astype(int)); out.append((P,R,F1,FPR))
    print("  %2d  %7.4f  | "%(nc,p.explained_variance_ratio_.sum())
          +" ".join("F1=%.4f(FPR %.4f)"%(o[2],o[3]) for o in out))

print("\n### A6-2  nc=8, q=0.999 상세 + 블록 부트스트랩")
p=PCA(n_components=8).fit(A)
spe=lambda z: ((z-p.inverse_transform(p.transform(z)))**2).sum(1)
sv,st,sa=spe(V),spe(T),spe(X); th=np.quantile(sv,0.999)
pn,pa=(st>th).astype(int),(sa>th).astype(int)
print("  P=%.4f R=%.4f F1=%.4f FPR=%.4f  (TP=%d FN=%d FP=%d TN=%d)"%(common.prf(pn,pa)+(pa.sum(),len(pa)-pa.sum(),pn.sum(),len(pn)-pn.sum())))
ub_n=np.array(sorted(set(Bte.tolist()))); ub_a=np.array(sorted(set(Ba.tolist())))
idx_n={b:np.where(Bte==b)[0] for b in ub_n}; idx_a={b:np.where(Ba==b)[0] for b in ub_a}
rng=np.random.default_rng(7); B=4000; f=np.empty(B)
for b_ in range(B):
    s1=rng.choice(ub_n,len(ub_n),True); s2=rng.choice(ub_a,len(ub_a),True)
    jn=np.concatenate([idx_n[x] for x in s1]); ja=np.concatenate([idx_a[x] for x in s2])
    f[b_]=common.prf(pn[jn],pa[ja])[2]
print("  블록 부트스트랩 F1 mean=%.4f 95%%CI=[%.4f, %.4f]  (IF진폭6 CI=[0.753,0.958], BL-0 CI=[0.707,0.905])"%(f.mean(),*np.percentile(f,[2.5,97.5])))
print("  burst 단위: 이상 %d/13 검출, 정상 오경보 %d/113"%(sum(int(pa[idx_a[b]].any()) for b in ub_a), sum(int(pn[idx_n[b]].any()) for b in ub_n)))

print("\n### A6-3  Hotelling T2 (BL-1 의 나머지 절반) 와 T2+SPE 결합")
def t2(z,p): s=p.transform(z); return (s**2/p.explained_variance_).sum(1)
for nc in [4,6,8,10,12]:
    p=PCA(n_components=nc).fit(A)
    for nm,fn_ in [("T2",lambda z: t2(z,p)),("SPE",lambda z: ((z-p.inverse_transform(p.transform(z)))**2).sum(1))]:
        sv,st,sa=fn_(V),fn_(T),fn_(X); th=np.quantile(sv,0.999)
        P,R,F1,FPR=common.prf((st>th).astype(int),(sa>th).astype(int))
        print("  nc=%2d %-4s P=%.3f R=%.3f F1=%.4f FPR=%.4f"%(nc,nm,P,R,F1,FPR))
    # OR-combine with per-statistic q99.95 (Bonferroni-ish)
    p_=PCA(n_components=nc).fit(A)
    a1=t2(V,p_); a2=((V-p_.inverse_transform(p_.transform(V)))**2).sum(1)
    b1=t2(T,p_); b2=((T-p_.inverse_transform(p_.transform(T)))**2).sum(1)
    c1=t2(X,p_); c2=((X-p_.inverse_transform(p_.transform(X)))**2).sum(1)
    th1,th2=np.quantile(a1,0.9995),np.quantile(a2,0.9995)
    P,R,F1,FPR=common.prf(((b1>th1)|(b2>th2)).astype(int),((c1>th1)|(c2>th2)).astype(int))
    print("  nc=%2d T2|SPE (각 q99.95) P=%.3f R=%.3f F1=%.4f FPR=%.4f"%(nc,P,R,F1,FPR))

print("\n### A6-4  진폭6 특징집합에서의 PCA-SPE / Mahalanobis (문서 추천 특징)")
ix=[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp"),i("AI2_std"),i("AI2_ptp")]
sc2=StandardScaler().fit(tr[:,ix]); A2,V2,T2_,X2=sc2.transform(tr[:,ix]),sc2.transform(va[:,ix]),sc2.transform(te[:,ix]),sc2.transform(Fa[:,ix])
from sklearn.covariance import EmpiricalCovariance
cv=EmpiricalCovariance().fit(A2)
for q in (0.995,0.999,0.9999):
    th=np.quantile(cv.mahalanobis(V2),q)
    print("  Mahalanobis(진폭6) q=%-7g P=%.3f R=%.3f F1=%.4f FPR=%.4f"%((q,)+common.prf((cv.mahalanobis(T2_)>th).astype(int),(cv.mahalanobis(X2)>th).astype(int))))

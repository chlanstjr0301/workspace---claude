# -*- coding: utf-8 -*-
import numpy as np, common, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
Fn,Tn,Bn,Fa,Ta,Ba=common.get(); i=common.FEAT_NAMES.index
tr=Fn[Tn<12000]; va=Fn[(Tn>=12000)&(Tn<15000)]; te=Fn[Tn>=15000]
S={"전체13":list(range(13)),
   "진동9":[i("AI0_acf1"),i("AI0_acf2"),i("AI0_std"),i("AI0_ptp"),i("AI1_acf1"),i("AI1_acf2"),i("AI1_std"),i("AI1_ptp"),i("corr01")],
   "ACF5":[i("AI0_acf1"),i("AI0_acf2"),i("AI1_acf1"),i("AI1_acf2"),i("corr01")],
   "진폭6":[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp"),i("AI2_std"),i("AI2_ptp")]}

def scorers(ix):
    A,V,T,X=tr[:,ix],va[:,ix],te[:,ix],Fa[:,ix]
    sc=StandardScaler().fit(A); a,v,t,x=sc.transform(A),sc.transform(V),sc.transform(T),sc.transform(X)
    out={}
    m=IsolationForest(n_estimators=1000,random_state=42).fit(A)
    out["IF"]=(-m.score_samples(V),-m.score_samples(T),-m.score_samples(X))
    cov=EmpiricalCovariance().fit(a)
    out["Mahalanobis"]=(cov.mahalanobis(v),cov.mahalanobis(t),cov.mahalanobis(x))
    k=min(len(ix),max(1,len(ix)-1))
    p=PCA(n_components=min(len(ix)-1,max(1,int(np.ceil(len(ix)*0.6))))).fit(a) if len(ix)>1 else None
    if p is not None:
        def spe(z): return ((z-p.inverse_transform(p.transform(z)))**2).sum(1)
        out["PCA-SPE"]=(spe(v),spe(t),spe(x))
        def t2(z):
            s=p.transform(z); return (s**2/p.explained_variance_).sum(1)
        out["PCA-T2"]=(t2(v),t2(t),t2(x))
    l=LocalOutlierFactor(n_neighbors=20,novelty=True).fit(a)
    out["LOF"]=(-l.score_samples(v),-l.score_samples(t),-l.score_samples(x))
    s=OneClassSVM(kernel="rbf",gamma="scale",nu=0.01).fit(a[::3])
    out["OCSVM"]=(-s.score_samples(v),-s.score_samples(t),-s.score_samples(x))
    return out

print("### A5-1  '특징을 늘릴수록 나빠진다' 는 알고리즘 독립적인가 (F1, 임계=검증 q99.9)")
tab={}
for lab,ix in S.items():
    for alg,(sv,st,sa) in scorers(ix).items():
        th=np.quantile(sv,0.999); tab[(alg,lab)]=common.prf((st>th).astype(int),(sa>th).astype(int))[2]
algs=["IF","Mahalanobis","PCA-T2","PCA-SPE","LOF","OCSVM"]
print("  %-14s"%"알고리즘"+"".join("%10s"%k for k in S)+"   순서(문서주장: 13<9<5<6)")
for a_ in algs:
    row=[tab.get((a_,k),float('nan')) for k in S]
    order=sorted(S.keys(), key=lambda k: tab.get((a_,k),-1))
    ok = "O" if order==["전체13","진동9","ACF5","진폭6"] else "X  실측순서: "+"<".join(order)
    print("  %-14s"%a_+"".join("%10.4f"%v for v in row)+"   %s"%ok)
print("  * 문서는 IF 한 알고리즘(seed 1개) 결과로 '특징 추가는 비용'이라는 일반 결론을 썼다.")

print("\n### A5-2  알고리즘별 최고 F1 — IF 진폭6(0.864)는 최적인가")
best=sorted(tab.items(), key=lambda kv:-kv[1])[:8]
for (a_,l),v in best: print("  %-14s %-8s F1=%.4f"%(a_,l,v))

print("\n### A5-3  Conformal 의 'FPR <= alpha' 보장이 실제로 지켜지는가")
print("  calibration = 정상 valid window %d개 / 평가 = 정상 test window %d개"%(len(va),len(te)))
for lab,ix in S.items():
    sv,st,sa=scorers(ix)["IF"]
    m=len(sv)
    # conformal p-value
    p_t=(1+np.array([(sv>=s).sum() for s in st]))/(m+1)
    p_a=(1+np.array([(sv>=s).sum() for s in sa]))/(m+1)
    line=[]
    for al in [0.001,0.005,0.01,0.05,0.10]:
        fpr=(p_t<=al).mean(); line.append((al,fpr))
    print("  IF %-8s"%lab+"  ".join("a=%.3f -> 실측FPR=%.4f (x%.1f)"%(al,f,f/al if al else 0) for al,f in line))
print("\n  동일 검정을 '교환가능' 상황(valid 를 무작위 반분)에서 수행 — 보장이 성립하는 기준선:")
rng=np.random.default_rng(0)
sv_all,_,_=scorers(S["진폭6"])["IF"]
for rep in range(3):
    pm=rng.permutation(len(sv_all)); h=len(sv_all)//2
    c,e=sv_all[pm[:h]],sv_all[pm[h:]]
    p=(1+np.array([(c>=s).sum() for s in e]))/(len(c)+1)
    print("    rep%d  "%rep+"  ".join("a=%.3f -> %.4f"%(al,(p<=al).mean()) for al in [0.005,0.01,0.05,0.10]))

print("\n### A5-4  체제 이탈 구간(16000-19000)이 conformal 보장을 깨는 정도")
Tt=Tn[Tn>=15000]
sv,st,sa=scorers(S["전체13"])["IF"]
m=len(sv); p_t=(1+np.array([(sv>=s).sum() for s in st]))/(m+1)
for lo,hi in [(15000,16000),(16000,17000),(17000,18000),(18000,19000),(19000,20000)]:
    msk=(Tt>=lo)&(Tt<hi)
    print("  %5d-%-5d n=%4d | a=0.01 실측FPR=%.4f  a=0.05 실측FPR=%.4f"%(lo,hi,msk.sum(),(p_t[msk]<=0.01).mean(),(p_t[msk]<=0.05).mean()))

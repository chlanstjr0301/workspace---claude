# -*- coding: utf-8 -*-
import numpy as np, common, warnings; warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
Fn,Tn,Bn,Fa,Ta,Ba=common.get(); i=common.FEAT_NAMES.index
tr=Fn[Tn<12000]; vm=(Tn>=12000)&(Tn<15000); tm=Tn>=15000
va=Fn[vm]; te=Fn[tm]; Bte=Bn[tm]
ub_n=np.array(sorted(set(Bte.tolist()))); ub_a=np.array(sorted(set(Ba.tolist())))
idx_n={b:np.where(Bte==b)[0] for b in ub_n}; idx_a={b:np.where(Ba==b)[0] for b in ub_a}
def boot(pn,pa,B=3000,seed=7):
    rng=np.random.default_rng(seed); f=np.empty(B)
    for b_ in range(B):
        s1=rng.choice(ub_n,len(ub_n),True); s2=rng.choice(ub_a,len(ub_a),True)
        jn=np.concatenate([idx_n[x] for x in s1]); ja=np.concatenate([idx_a[x] for x in s2])
        f[b_]=common.prf(pn[jn],pa[ja])[2]
    return f.mean(), *np.percentile(f,[2.5,97.5])

def run(name, ix, kind, q=0.999, nc=None):
    sc=StandardScaler().fit(tr[:,ix]); A,V,T,X=[sc.transform(z[:,ix]) for z in (tr,va,te,Fa)]
    if kind=="SPE":
        p=PCA(n_components=nc).fit(A); f=lambda z: ((z-p.inverse_transform(p.transform(z)))**2).sum(1)
    elif kind=="T2":
        p=PCA(n_components=nc).fit(A); f=lambda z: ((p.transform(z)**2)/p.explained_variance_).sum(1)
    else:
        cv=EmpiricalCovariance().fit(A); f=cv.mahalanobis
    sv,st,sa=f(V),f(T),f(X); th=np.quantile(sv,q)
    pn,pa=(st>th).astype(int),(sa>th).astype(int)
    P,R,F1,FPR=common.prf(pn,pa); m,l,u=boot(pn,pa)
    bb=sum(int(pa[idx_a[b]].any()) for b in ub_a); fb=sum(int(pn[idx_n[b]].any()) for b in ub_n)
    print("  %-40s P=%.3f R=%.3f F1=%.4f FPR=%.4f | 95%%CI=[%.4f,%.4f] | burst %d/13, 오경보 %d/113"%(name,P,R,F1,FPR,l,u,bb,fb))
    return F1

ALL13=list(range(13)); NO_AI2ACF=[k for k in range(13) if k!=i("AI2_acf1")]
NO_AI2=[k for k in range(13) if not common.FEAT_NAMES[k].startswith("AI2")]
AMP6=[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp"),i("AI2_std"),i("AI2_ptp")]
VIB4=[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp")]

print("### A7-1  PCA-SPE F1=1.0 의 출처 — 금지변수 AI2_acf1 때문인가")
run("SPE 13특징 nc=8 (AI2_acf1 포함)",ALL13,"SPE",nc=8)
run("SPE 12특징 nc=8 (AI2_acf1 제외)",NO_AI2ACF,"SPE",nc=8)
run("SPE 12특징 nc=7 (AI2_acf1 제외)",NO_AI2ACF,"SPE",nc=7)
run("SPE 12특징 nc=9 (AI2_acf1 제외)",NO_AI2ACF,"SPE",nc=9)
run("SPE 9특징(AI2채널 전체 제외) nc=6",NO_AI2,"SPE",nc=6)
# 어느 PC 잔차가 AI2_acf1 을 담는가
sc=StandardScaler().fit(tr); A=sc.transform(tr)
p=PCA(n_components=8).fit(A)
res_load=np.abs(np.eye(13)-p.components_.T@p.components_).sum(0)
print("  nc=8 잔차공간 기여도(특징별):")
for k in np.argsort(-res_load): print("     %-10s %.4f"%(common.FEAT_NAMES[k],res_load[k]))

print("\n### A7-2  문서 추천 특징집합(진폭6) 만으로 — 금지변수 없음, 동일 프로토콜")
f_if=0.8642
run("Mahalanobis 진폭6  q=0.999",AMP6,"MAH")
run("Mahalanobis 진폭6  q=0.995",AMP6,"MAH",q=0.995)
run("Mahalanobis 진폭6  q=0.9999",AMP6,"MAH",q=0.9999)
run("PCA-T2 진폭6 nc=4 q=0.999",AMP6,"T2",nc=4)
run("Mahalanobis 진동진폭4(AI2제거) q=0.999",VIB4,"MAH")
print("  [문서 §3 1위]  IF 진폭6 seed42 q=0.999        F1=0.8642  95%CI=[0.7530,0.9581] | burst 13/13, 오경보 0/113")

print("\n### A7-3  Mahalanobis(진폭6) vs IF(진폭6) 대응 부트스트랩")
sc=StandardScaler().fit(tr[:,AMP6]); A,V,T,X=[sc.transform(z[:,AMP6]) for z in (tr,va,te,Fa)]
cv=EmpiricalCovariance().fit(A); th=np.quantile(cv.mahalanobis(V),0.999)
mp_n,mp_a=(cv.mahalanobis(T)>th).astype(int),(cv.mahalanobis(X)>th).astype(int)
from sklearn.ensemble import IsolationForest
m=IsolationForest(n_estimators=300,random_state=42).fit(tr[:,AMP6])
sv=-m.score_samples(va[:,AMP6]); thi=np.quantile(sv,0.999)
ip_n,ip_a=(-m.score_samples(te[:,AMP6])>thi).astype(int),(-m.score_samples(Fa[:,AMP6])>thi).astype(int)
rng=np.random.default_rng(11); d=np.empty(4000)
for b_ in range(4000):
    s1=rng.choice(ub_n,len(ub_n),True); s2=rng.choice(ub_a,len(ub_a),True)
    jn=np.concatenate([idx_n[x] for x in s1]); ja=np.concatenate([idx_a[x] for x in s2])
    d[b_]=common.prf(mp_n[jn],mp_a[ja])[2]-common.prf(ip_n[jn],ip_a[ja])[2]
print("  Δ(Mahalanobis - IF) mean=%+.4f 95%%CI=[%+.4f,%+.4f] P(Δ<=0)=%.4f"%(d.mean(),*np.percentile(d,[2.5,97.5]),(d<=0).mean()))

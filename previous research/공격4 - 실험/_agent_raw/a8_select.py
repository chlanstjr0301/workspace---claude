# -*- coding: utf-8 -*-
import numpy as np, common, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
Fn,Tn,Bn,Fa,Ta,Ba=common.get(); i=common.FEAT_NAMES.index
n,o=common.load(); COLS=common.COLS
tr=Fn[Tn<12000]; vm=(Tn>=12000)&(Tn<15000); tm=Tn>=15000
va=Fn[vm]; te=Fn[tm]; Bte=Bn[tm]
lo=n.iloc[:12000][COLS].min().values; hi=n.iloc[:12000][COLS].max().values
Xn=n[COLS].values; Xo=o[COLS].values
bl0_te=np.array([int(((Xn[t:t+20]<lo)|(Xn[t:t+20]>hi)).any()) for t in Tn[tm]])
bl0_a=np.array([int(((Xo[t:t+20]<lo)|(Xo[t:t+20]>hi)).any()) for t in Ta])

# ---- 문서 §3 의 10개 후보 재구성 (예측 벡터) ----
cands={}
for lab,j,sg in [("C1 AI0_acf1",i("AI0_acf1"),1),("C2 AI1_acf2",i("AI1_acf2"),1),
                 ("C3 -corr01",i("corr01"),-1),("C4 AI0_std",i("AI0_std"),1),
                 ("X -AI2_acf1",i("AI2_acf1"),-1)]:
    th=np.quantile(sg*va[:,j],0.999); cands[lab]=((sg*te[:,j]>th).astype(int),(sg*Fa[:,j]>th).astype(int))
sets={"IF 13":list(range(13)),
      "IF 진동9":[i("AI0_acf1"),i("AI0_acf2"),i("AI0_std"),i("AI0_ptp"),i("AI1_acf1"),i("AI1_acf2"),i("AI1_std"),i("AI1_ptp"),i("corr01")],
      "IF ACF5":[i("AI0_acf1"),i("AI0_acf2"),i("AI1_acf1"),i("AI1_acf2"),i("corr01")],
      "IF 진폭6":[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp"),i("AI2_std"),i("AI2_ptp")]}
for lab,ix in sets.items():
    m=IsolationForest(n_estimators=300,random_state=42).fit(tr[:,ix])
    th=np.quantile(-m.score_samples(va[:,ix]),0.999)
    cands[lab]=((-m.score_samples(te[:,ix])>th).astype(int),(-m.score_samples(Fa[:,ix])>th).astype(int))
cands["BL-0"]=(bl0_te,bl0_a)

ub_n=np.array(sorted(set(Bte.tolist()))); ub_a=np.array(sorted(set(Ba.tolist())))
idx_n={b:np.where(Bte==b)[0] for b in ub_n}; idx_a={b:np.where(Ba==b)[0] for b in ub_a}

print("### A8-1  테스트셋 선택 편향 — 이상 burst 13개를 반분해 '선택'과 '평가'를 분리")
rng=np.random.default_rng(3); R=2000
sel_f1=[]; hold_f1=[]; bl_hold=[]; chosen={}
names=list(cands)
for r in range(R):
    pm=rng.permutation(ub_a); Asel,Ahold=pm[:7],pm[7:]
    nm=rng.permutation(ub_n); Nsel,Nhold=nm[:len(ub_n)//2],nm[len(ub_n)//2:]
    jsa=np.concatenate([idx_a[x] for x in Asel]); jha=np.concatenate([idx_a[x] for x in Ahold])
    jsn=np.concatenate([idx_n[x] for x in Nsel]); jhn=np.concatenate([idx_n[x] for x in Nhold])
    sc=[(k,common.prf(cands[k][0][jsn],cands[k][1][jsa])[2]) for k in names]
    k,v=max(sc,key=lambda x:x[1])
    chosen[k]=chosen.get(k,0)+1
    sel_f1.append(v); hold_f1.append(common.prf(cands[k][0][jhn],cands[k][1][jha])[2])
    bl_hold.append(common.prf(bl0_te[jhn],bl0_a[jha])[2])
sel_f1=np.array(sel_f1); hold_f1=np.array(hold_f1); bl_hold=np.array(bl_hold)
print("  선택 반쪽의 1위 F1 평균 = %.4f  /  같은 모델의 보류 반쪽 F1 평균 = %.4f  => 낙관편향 %+.4f"%(sel_f1.mean(),hold_f1.mean(),sel_f1.mean()-hold_f1.mean()))
print("  보류 반쪽에서 선택모델이 BL-0 을 이긴 비율 = %.1f%%  (평균 Δ=%+.4f)"%(100*(hold_f1>bl_hold).mean(),(hold_f1-bl_hold).mean()))
print("  선택된 후보 분포:", dict(sorted(chosen.items(),key=lambda kv:-kv[1])))
print("  * 금지변수 X -AI2_acf1 를 선택지에서 빼면:")
names2=[k for k in names if k!="X -AI2_acf1"]
rng=np.random.default_rng(3); s2=[];h2=[];b2=[];ch2={}
for r in range(R):
    pm=rng.permutation(ub_a); Asel,Ahold=pm[:7],pm[7:]
    nm=rng.permutation(ub_n); Nsel,Nhold=nm[:len(ub_n)//2],nm[len(ub_n)//2:]
    jsa=np.concatenate([idx_a[x] for x in Asel]); jha=np.concatenate([idx_a[x] for x in Ahold])
    jsn=np.concatenate([idx_n[x] for x in Nsel]); jhn=np.concatenate([idx_n[x] for x in Nhold])
    sc=[(k,common.prf(cands[k][0][jsn],cands[k][1][jsa])[2]) for k in names2]
    k,v=max(sc,key=lambda x:x[1]); ch2[k]=ch2.get(k,0)+1
    s2.append(v); h2.append(common.prf(cands[k][0][jhn],cands[k][1][jha])[2]); b2.append(common.prf(bl0_te[jhn],bl0_a[jha])[2])
s2,h2,b2=map(np.array,(s2,h2,b2))
print("    선택 %.4f -> 보류 %.4f (편향 %+.4f) | BL-0 초과 %.1f%% | 선택분포 %s"%(s2.mean(),h2.mean(),s2.mean()-h2.mean(),100*(h2>b2).mean(),dict(sorted(ch2.items(),key=lambda kv:-kv[1]))))

print("\n### A8-2  §7.3 합격선 'FAR/h <= 1' 을 FPR 0.0000 으로 통과했다고 말할 수 있는가")
t_s=n.iloc[15000:]
wall=(t_s.TimeStamp.max()-t_s.TimeStamp.min()).total_seconds()/3600
data=0.1*len(t_s)/3600
print("  정상 테스트 구간: 벽시계 %.3f h / 실측 데이터시간 %.3f h"%(wall,data))
print("  FP=0 관측 -> FAR/h 점추정 0. 그러나 Poisson 상한(95%%, 0건=3.0) = %.1f 건/h (데이터시간 기준) / %.1f 건/h (벽시계)"%(3.0/data,3.0/wall))
print("  => 'FAR/h <= 1' 을 주장하려면 최소 %.1f h 의 무경보 정상 관측이 필요하다. 보유량은 %.2f h."%(3.0/1.0,wall))

print("\n### A8-3  §2.1 상관 반전은 고장 신호인가 진폭 효과인가")
j0,jc=i("AI0_std"),i("corr01")
for nm,F,T in [("정상 train(:12000)",Fn[Tn<12000],None),("정상 test(15000:)",Fn[Tn>=15000],None),("이상",Fa,None)]:
    r=np.corrcoef(F[:,j0],F[:,jc])[0,1]
    print("  %-20s corr(AI0_std, corr01)=%+.3f  | corr01 mean=%+.3f"%(nm,r,F[:,jc].mean()))
# 정상 window 중 진폭 상위 5% 의 corr01
q=np.quantile(Fn[:,j0],0.95); m=Fn[:,j0]>=q
print("  정상 window 중 AI0_std 상위5%% (n=%d): corr01 mean=%+.3f  (정상 전체 %+.3f, 이상 %+.3f)"%(m.sum(),Fn[m,jc].mean(),Fn[:,jc].mean(),Fa[:,jc].mean()))
q2=np.quantile(Fn[:,j0],0.05); m2=Fn[:,j0]<=q2
print("  정상 window 중 AI0_std 하위5%% (n=%d): corr01 mean=%+.3f"%(m2.sum(),Fn[m2,jc].mean()))
print("  이상 window 중 AI0_std 가 정상범위(<=%.3f)인 것 (n=%d): corr01 mean=%+.3f"%(np.quantile(Fn[:,j0],0.99),(Fa[:,j0]<=np.quantile(Fn[:,j0],0.99)).sum(),Fa[Fa[:,j0]<=np.quantile(Fn[:,j0],0.99),jc].mean()))
print("\n  AI0/AI1 포화(클리핑) 점검:")
for c in COLS:
    a=n[c].values; b=o[c].values
    print("    %-14s normal [%.4f, %.4f]  outlier [%.4f, %.4f]  outlier 가 normal 범위 밖인 비율 %.2f%%"
          %(c,a.min(),a.max(),b.min(),b.max(),100*np.mean((b<a.min())|(b>a.max()))))

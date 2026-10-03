# -*- coding: utf-8 -*-
import numpy as np, common
from sklearn.ensemble import IsolationForest
Fn,Tn,Bn,Fa,Ta,Ba=common.get(); i=common.FEAT_NAMES.index
n,o=common.load()
COLS=common.COLS
tr=Fn[Tn<12000]; vm=(Tn>=12000)&(Tn<15000); tm=Tn>=15000
va=Fn[vm]; te=Fn[tm]; Bte=Bn[tm]; Bva=Bn[vm]
AMP6=[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp"),i("AI2_std"),i("AI2_ptp")]
VIB4=[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp")]

# ---- BL-0 per-window predictions in the SAME window framework ----
lo=n.iloc[:12000][COLS].min().values; hi=n.iloc[:12000][COLS].max().values
def bl0(df, Tsel):
    X=df[COLS].values
    return np.array([int(((X[t:t+20]<lo)|(X[t:t+20]>hi)).any()) for t in Tsel])
bl0_te=bl0(n,Tn[tm]); bl0_va=bl0(n,Tn[vm]); bl0_a=bl0(o,Ta)
print("BL-0 재현: ", common.prf(bl0_te,bl0_a))

# BL-0 as a *score* so it can use the same q99.9 protocol (fair comparison)
def bl0_score(df,Tsel):
    X=df[COLS].values; rng=hi-lo; out=[]
    for t in Tsel:
        W=X[t:t+20]
        out.append(float(np.max(np.maximum((W-hi)/rng,(lo-W)/rng))))
    return np.array(out)
s_va=bl0_score(n,Tn[vm]); s_te=bl0_score(n,Tn[tm]); s_a=bl0_score(o,Ta)
print("\n### A4-1  BL-0 에 동일 프로토콜(검증 q99.9) 을 적용하면")
print("  문서 BL-0 (임계=학습 min/max, 즉 score>0): P=%.3f R=%.3f F1=%.4f FPR=%.4f"%common.prf((s_te>0).astype(int),(s_a>0).astype(int)))
for q in [0.99,0.995,0.999,0.9999]:
    th=np.quantile(s_va,q)
    print("  BL-0 score, 임계=검증 q%-7g (th=%+.4f): P=%.3f R=%.3f F1=%.4f FPR=%.4f"%((q,th)+common.prf((s_te>th).astype(int),(s_a>th).astype(int))))

# ---- block bootstrap over bursts ----
def f1_from(pn,pa): return common.prf(pn,pa)[2]
rng_=np.random.default_rng(7)
m=IsolationForest(n_estimators=300,random_state=42).fit(tr[:,AMP6])
sv=-m.score_samples(va[:,AMP6]); st=-m.score_samples(te[:,AMP6]); sa=-m.score_samples(Fa[:,AMP6])
th=np.quantile(sv,0.999)
if_te=(st>th).astype(int); if_a=(sa>th).astype(int)
print("\n점추정: IF진폭6 F1=%.4f  /  BL-0 F1=%.4f  /  Δ=%+.4f"%(f1_from(if_te,if_a),f1_from(bl0_te,bl0_a),f1_from(if_te,if_a)-f1_from(bl0_te,bl0_a)))

ub_n=np.array(sorted(set(Bte.tolist()))); ub_a=np.array(sorted(set(Ba.tolist())))
idx_n={b:np.where(Bte==b)[0] for b in ub_n}; idx_a={b:np.where(Ba==b)[0] for b in ub_a}
print("  부트스트랩 단위: 정상 테스트 burst %d개 / 이상 burst %d개"%(len(ub_n),len(ub_a)))

B=4000
f_if=np.empty(B); f_bl=np.empty(B); d=np.empty(B)
for b_ in range(B):
    sn=rng_.choice(ub_n,len(ub_n),replace=True); sa_=rng_.choice(ub_a,len(ub_a),replace=True)
    jn=np.concatenate([idx_n[x] for x in sn]); ja=np.concatenate([idx_a[x] for x in sa_])
    f_if[b_]=f1_from(if_te[jn],if_a[ja]); f_bl[b_]=f1_from(bl0_te[jn],bl0_a[ja]); d[b_]=f_if[b_]-f_bl[b_]
def ci(x): return np.percentile(x,[2.5,97.5])
print("\n### A4-2  블록 부트스트랩 (burst 리샘플, B=%d)"%B)
print("  IF 진폭6 F1 : mean=%.4f  95%%CI=[%.4f, %.4f]  (폭 %.4f)"%(f_if.mean(),*ci(f_if),np.diff(ci(f_if))[0]))
print("  BL-0     F1 : mean=%.4f  95%%CI=[%.4f, %.4f]  (폭 %.4f)"%(f_bl.mean(),*ci(f_bl),np.diff(ci(f_bl))[0]))
print("  CI 겹침 여부 : %s"%("겹친다 (유의하지 않음)" if ci(f_if)[0]<ci(f_bl)[1] else "겹치지 않음"))
print("  대응 Δ(IF-BL0): mean=%+.4f  95%%CI=[%+.4f, %+.4f]  P(Δ<=0)=%.3f"%(d.mean(),*ci(d),(d<=0).mean()))

# C2 AI1_acf2 (0.820) vs BL-0 (0.818)
j=i("AI1_acf2"); thc=np.quantile(va[:,j],0.999)
c2_te=(te[:,j]>thc).astype(int); c2_a=(Fa[:,j]>thc).astype(int)
d2=np.empty(B); f_c2=np.empty(B)
rng2=np.random.default_rng(7)
for b_ in range(B):
    sn=rng2.choice(ub_n,len(ub_n),replace=True); sa_=rng2.choice(ub_a,len(ub_a),replace=True)
    jn=np.concatenate([idx_n[x] for x in sn]); ja=np.concatenate([idx_a[x] for x in sa_])
    f_c2[b_]=f1_from(c2_te[jn],c2_a[ja]); d2[b_]=f_c2[b_]-f1_from(bl0_te[jn],bl0_a[ja])
print("\n  C2 AI1_acf2 F1=%.4f : 95%%CI=[%.4f, %.4f] | Δ vs BL-0 CI=[%+.4f,%+.4f] P(Δ<=0)=%.3f"
      %(f1_from(c2_te,c2_a),*ci(f_c2),*ci(d2),(d2<=0).mean()))

# ---- 유효 표본: 이상 window 자기상관 ----
print("\n### A4-3  이상 window 276개의 유효 독립 표본 수")
print("  이상 burst 13개, window/burst = %s"%[int((Ba==b).sum()) for b in ub_a])
print("  window stride=1, SEQ=20 -> 인접 window 95%% 중첩. burst 내부 window 는 사실상 1개 관측의 변형")
print("  문서 §1 자체 인정: '양성 이벤트 1건'. => F1 의 유효 분모는 276 이 아니라 13 (상한), 실질 1")
# event-level
print("  burst(=13) 단위 평가:")
for lab,p_te,p_a in [("IF 진폭6",if_te,if_a),("BL-0",bl0_te,bl0_a),("C2 AI1_acf2",c2_te,c2_a)]:
    bn=np.array([int(p_te[idx_n[x]].any()) for x in ub_n]); ba=np.array([int(p_a[x_].any()) for x_ in [idx_a[y] for y in ub_a]])
    P,R,F1,FPR=common.prf(bn,ba)
    print("    %-12s burst P=%.3f R=%.3f F1=%.3f  (이상burst 검출 %d/13, 정상burst 오경보 %d/%d)"%(lab,P,R,F1,ba.sum(),bn.sum(),len(bn)))

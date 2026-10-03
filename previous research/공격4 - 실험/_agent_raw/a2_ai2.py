# -*- coding: utf-8 -*-
import numpy as np, common
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
Fn,Tn,Bn,Fa,Ta,Ba = common.get(); i=common.FEAT_NAMES.index
tr=Fn[Tn<12000]; va=Fn[(Tn>=12000)&(Tn<15000)]; te=Fn[Tn>=15000]

print("### A2-1  AI2_Current 채널의 세션 간 수준 차이 (원시 신호)")
n,o = common.load()
for nm,d in (("normal[:12000] (train)",n.iloc[:12000]),("normal[12000:15000] (valid)",n.iloc[12000:15000]),
             ("normal[15000:] (test)",n.iloc[15000:]),("outlier 전체",o)):
    x=d["AI2_Current"].values
    print("  %-30s mean=%+9.3f std=%8.3f ptp=%9.3f  min=%+9.2f max=%+9.2f"%(nm,x.mean(),x.std(),np.ptp(x),x.min(),x.max()))

print("\n### A2-2  window 특징 수준: AI2_std / AI2_ptp 는 세션 분리 변수인가")
for f in ["AI2_std","AI2_ptp","AI2_acf1","AI0_std","AI1_std"]:
    j=i(f)
    a=Fn[Tn<15000][:,j]; b=Fn[Tn>=15000][:,j]; c=Fa[:,j]
    y=np.r_[np.zeros(len(b)),np.ones(len(c))]; s=np.r_[b,c]
    auc=roc_auc_score(y,s); auc=max(auc,1-auc)
    ov = (c.min()<=b.max()) and (b.min()<=c.max())
    print("  %-9s train구간 mean=%9.3f | test정상 mean=%9.3f | 이상 mean=%9.3f | |AUC|=%.4f | 정상test범위[%.3f,%.3f] 이상범위[%.3f,%.3f] 겹침=%s"
          %(f,a.mean(),b.mean(),c.mean(),auc,b.min(),b.max(),c.min(),c.max(),"O" if ov else "X(완전분리)"))

print("\n### A2-3  M-1(진폭6)의 성능은 AI2 채널이 떠받치고 있는가  (seed 10개 평균, q=0.999)")
seeds=[0,1,2,3,7,11,13,42,123,777]
sets={
 "진폭6 (문서 추천)":[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp"),i("AI2_std"),i("AI2_ptp")],
 "진동진폭4 (AI2 제거)":[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp")],
 "AI2진폭2 만":[i("AI2_std"),i("AI2_ptp")],
 "AI0_std 1개":[i("AI0_std")],
}
for lab,ix in sets.items():
    v=[]
    for s in seeds:
        m=IsolationForest(n_estimators=300,random_state=s).fit(tr[:,ix])
        sv,st,sa=-m.score_samples(va[:,ix]),-m.score_samples(te[:,ix]),-m.score_samples(Fa[:,ix])
        th=np.quantile(sv,0.999); v.append(common.prf((st>th).astype(int),(sa>th).astype(int)))
    v=np.array(v); print("  %-22s P=%.3f R=%.3f F1=%.4f(sd %.4f) FPR=%.4f"%(lab,v[:,0].mean(),v[:,1].mean(),v[:,2].mean(),v[:,2].std(),v[:,3].mean()))

print("\n### A2-4  단일특징 AI2_std / AI2_ptp 규칙 (문서 §3 표에 없는 행)")
for f,sg in [("AI2_std",-1),("AI2_ptp",-1),("AI2_std",+1)]:
    j=i(f); th=np.quantile(sg*va[:,j],0.999)
    P,R,F1,FPR=common.prf((sg*te[:,j]>th).astype(int),(sg*Fa[:,j]>th).astype(int))
    print("  %s%-9s P=%.3f R=%.3f F1=%.4f FPR=%.4f"%("-" if sg<0 else "+",f,P,R,F1,FPR))

print("\n### A2-5  AI2_acf1 = 순수 정현파의 수학적 귀결인가 (에일리어싱 검증)")
import numpy as np
for nm,d in (("normal",n),("outlier",o)):
    x=d["AI2_Current"].values
    # per-burst dominant apparent frequency via acf1 = cos(2*pi*f*dt)
    a1=[]; 
    for s,e in common.runs(d):
        if e-s<20: continue
        w=x[s:e]; a1.append(common._acf(w,1))
    a1=np.array(a1); ang=np.arccos(np.clip(a1,-1,1)); f_app=ang/(2*np.pi*0.1)
    print("  %-8s acf1 mean=%.4f sd=%.5f  -> 겉보기주파수 %.4f Hz (sd %.5f)"%(nm,a1.mean(),a1.std(),f_app.mean(),f_app.std()))
    # acf2 predicted by pure sinusoid: cos(2*theta) = 2*acf1^2-1
    a2=[]
    for s,e in common.runs(d):
        if e-s<20: continue
        a2.append(common._acf(x[s:e],2))
    a2=np.array(a2); pred=2*a1**2-1
    print("            acf2 실측 mean=%.4f / 순수정현파 예측 2*acf1^2-1 = %.4f  (잔차 %.4f)"%(a2.mean(),pred.mean(),np.abs(a2-pred).mean()))
print("  -> 60Hz 를 10Hz 로 샘플링하면 앨리어스 0Hz. 겉보기 0.6Hz 는 샘플링클럭/전원 주파수 차이 0.6Hz 로 설명된다.")
print("  -> 그러나 AI2_std(진폭)도 세션 간 120 -> 77 로 변한다: acf1 만 '지문'이라고 볼 근거 없음.")

# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, os, re
RAW = r"C:/Users/cmsch/Desktop/대회/2026년 제6회 K-인공지능 제조데이터 분석 경진대회/data/raw"
n = pd.read_csv(os.path.join(RAW,"press_data_normal.csv"), index_col=0, parse_dates=["TimeStamp"])
o = pd.read_csv(os.path.join(RAW,"press_data_outlier.csv"), index_col=0, parse_dates=["TimeStamp"])
C=["AI0_Vibration","AI1_Vibration","AI2_Current"]
def runs(d,m=1):
    dt=d.TimeStamp.diff().dt.total_seconds().values
    e=np.concatenate(([0],np.where(dt>0.5)[0],[len(d)]))
    return [(int(a),int(b)) for a,b in zip(e[:-1],e[1:]) if b-a>=m]

print("### J. 소수점 자릿수 — 파일 지문")
for nm,path in (("normal","press_data_normal.csv"),("outlier","press_data_outlier.csv")):
    txt=open(os.path.join(RAW,path),encoding='utf-8').read().split("\n")[1:]
    dec=[]
    for L in txt:
        p=L.split(",")
        if len(p)<6: continue
        for v in p[2:5]:
            dec.append(len(v.split(".")[1]) if "." in v else 0)
    dec=np.array(dec)
    print("%s: 소수자릿수 중앙값=%d  최대=%d  >7자리 비율=%.1f%%  유효숫자 중앙=%s"%(
        nm,np.median(dec),dec.max(),100*np.mean(dec>7),
        np.median([len(re.sub(r'[-.]','',v).lstrip('0')) for L in txt if len(L.split(','))>5 for v in L.split(',')[2:5]])))

print("\n### K. AI2 — 60Hz 캐리어선(0.5~0.75Hz 에일리어스 대역) 파워 비중")
for nm,d in (("normal",n),("outlier",o)):
    R=runs(d,40); fr=np.fft.rfftfreq(40,0.1); band=(fr>=0.4)&(fr<=0.9)
    rat=[]
    for s,e in R:
        v=d.AI2_Current.values[s:s+40]; v=(v-v.mean())*np.hanning(40)
        P=np.abs(np.fft.rfft(v))**2
        rat.append(P[band].sum()/P[1:].sum())
    print("%s: 캐리어대역 파워비중 중앙값=%.3f  10~90%%=%.3f~%.3f  (버스트 %d)"%(
        nm,np.median(rat),np.quantile(rat,.1),np.quantile(rat,.9),len(rat)))

print("\n### L. 버스트 단위 — 정상/이상 분포 겹침 (진짜 독립표본은 버스트)")
def bstats(d):
    out=[]
    for s,e in runs(d,30):
        a,b,c=d[C[0]].values[s:e],d[C[1]].values[s:e],d[C[2]].values[s:e]
        out.append(dict(corr=np.corrcoef(a,b)[0,1], a0=a.std(), a1=b.std(),
                        c=c.std(), ratio=a.std()/b.std()))
    return pd.DataFrame(out)
BN,BO=bstats(n),bstats(o)
print("정상 버스트 %d개 / 이상 버스트 %d개"%(len(BN),len(BO)))
for k in ["corr","a0","a1","c","ratio"]:
    print("  %-6s 정상 중앙=%8.4f [min %8.4f, max %8.4f] | 이상 중앙=%8.4f [min %8.4f, max %8.4f] | 겹침=%s"%(
        k,BN[k].median(),BN[k].min(),BN[k].max(),BO[k].median(),BO[k].min(),BO[k].max(),
        "없음" if (BN[k].min()>BO[k].max() or BN[k].max()<BO[k].min()) else "있음"))
print("  corr <= 이상중앙값(%.3f) 인 정상 버스트: %d/%d (%.1f%%)"%(
    BO.corr_ if False else BO['corr'].median(), (BN['corr']<=BO['corr'].median()).sum(), len(BN),
    100*(BN['corr']<=BO['corr'].median()).mean()))
print("  corr < 0 인 정상 버스트: %d/%d (%.1f%%)"%((BN['corr']<0).sum(),len(BN),100*(BN['corr']<0).mean()))

print("\n### M. IF 진폭 6특징에서 AI2를 빼면? (AI2_std/ptp 도 에일리어스 캐리어 유래)")
from sklearn.ensemble import IsolationForest
SEQ=20; FEAT=[f"{c[:3]}_{m}" for c in C for m in ("acf1","acf2","std","ptp")]+["corr01"]
def _acf(x,l):
    xc=x-x.mean()
    return 0.0 if xc.std()<1e-9 else float(np.corrcoef(xc[:-l],xc[l:])[0,1])
def wf(d):
    X=d[C].values; F,T=[],[]
    for s,e in runs(d,SEQ):
        for i in range(s,e-SEQ+1):
            W=X[i:i+SEQ]; f=[]
            for k in range(3): f+=[_acf(W[:,k],1),_acf(W[:,k],2),float(W[:,k].std()),float(np.ptp(W[:,k]))]
            a,b=W[:,0]-W[:,0].mean(),W[:,1]-W[:,1].mean()
            f.append(float(np.corrcoef(a,b)[0,1]) if a.std()>1e-9 and b.std()>1e-9 else 0.0)
            F.append(f); T.append(i)
    return np.array(F),np.array(T)
Fn,Tn=wf(n); Fa,_=wf(o)
tr=Fn[Tn<12000]; va=Fn[(Tn>=12000)&(Tn<15000)]; te=Fn[Tn>=15000]
i=FEAT.index
def rep(name,sn,sa,th):
    tp=int((sa>th).sum()); fn=len(sa)-tp; fp=int((sn>th).sum()); tn=len(sn)-fp
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn); F1=2*P*R/(P+R) if P+R else 0
    print("  %-34s P=%.3f R=%.3f F1=%.3f FPR=%.4f"%(name,P,R,F1,fp/(fp+tn)))
sets={"진폭6 (문서 추천, AI2 포함)":[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp"),i("AI2_std"),i("AI2_ptp")],
      "진폭4 (AI2 제거)":[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp")],
      "AI2 진폭 2특징만":[i("AI2_std"),i("AI2_ptp")],
      "AI0_std 단일":[i("AI0_std")]}
for lab,ix in sets.items():
    m=IsolationForest(n_estimators=300,random_state=42).fit(tr[:,ix])
    rep("IF "+lab,-m.score_samples(te[:,ix]),-m.score_samples(Fa[:,ix]),np.quantile(-m.score_samples(va[:,ix]),0.999))

print("\n### N. 이상 파일에서 AI1 = -k*AI0 인가 (고정위상 ~120~180°의 정체)")
for nm,d in (("normal",n),("outlier",o)):
    a=d[C[0]].values; b=d[C[1]].values
    k=np.polyfit(a,b,1)
    print("%s: AI1 = %+.3f*AI0 %+.3f,  R^2=%.3f,  std비 AI1/AI0=%.3f"%(nm,k[0],k[1],np.corrcoef(a,b)[0,1]**2,b.std()/a.std()))

print("\n### O. 버스트 길이·간격 구조 (프레스 1타 주기 비교)")
for nm,d in (("normal",n),("outlier",o)):
    R=runs(d); L=np.array([b-a for a,b in R])
    st=d.TimeStamp.values[[a for a,b in R]]
    per=np.diff(st).astype('timedelta64[ms]').astype(float)/1000
    print("%s: 버스트길이 중앙=%d  50샘플 비율=%.1f%%  <10샘플 비율=%.1f%%"%(nm,np.median(L),100*np.mean(L==50),100*np.mean(L<10)))
    print("   버스트 시작간격 분위수(s): "+", ".join("%.2f"%np.quantile(per,q) for q in [0,.1,.25,.5,.75,.9,1]))
    print("   수집 가동률(duty) = %.1f%%"%(100*len(d)*0.1/(d.TimeStamp.iloc[-1]-d.TimeStamp.iloc[0]).total_seconds()))

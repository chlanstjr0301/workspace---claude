# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, os
RAW = r"C:/Users/cmsch/Desktop/대회/2026년 제6회 K-인공지능 제조데이터 분석 경진대회/data/raw"
n = pd.read_csv(os.path.join(RAW,"press_data_normal.csv"), index_col=0, parse_dates=["TimeStamp"])
o = pd.read_csv(os.path.join(RAW,"press_data_outlier.csv"), index_col=0, parse_dates=["TimeStamp"])
C=["AI0_Vibration","AI1_Vibration","AI2_Current"]
def runs(d,m=1):
    dt=d.TimeStamp.diff().dt.total_seconds().values
    e=np.concatenate(([0],np.where(dt>0.5)[0],[len(d)]))
    return [(int(a),int(b)) for a,b in zip(e[:-1],e[1:]) if b-a>=m]

print("### A. AI2 정상 = 순수 정현파인가? (버스트별 사인 피팅)")
R=[r for r in runs(n,30)]
print("긴 버스트(>=30):",len(R))
rows=[]
for s,e in R[:400]:
    v=n.AI2_Current.values[s:e]; t=np.arange(len(v))
    best=None
    for f in np.arange(0.02,0.5,0.0005):   # cycles/sample
        A=np.c_[np.cos(2*np.pi*f*t),np.sin(2*np.pi*f*t),np.ones_like(t)]
        c,res,*_=np.linalg.lstsq(A,v,rcond=None)
        r2=1-np.sum((v-A@c)**2)/np.sum((v-v.mean())**2)
        if best is None or r2>best[0]: best=(r2,f,np.hypot(c[0],c[1]),c[2])
    rows.append(best)
rows=np.array(rows)
print("사인 1개 적합 R^2: 중앙값=%.4f  5%%=%.4f  min=%.4f"%(np.median(rows[:,0]),np.quantile(rows[:,0],0.05),rows[:,0].min()))
print("겉보기 주파수(Hz): 중앙값=%.4f  범위=%.4f~%.4f  (=주기 %.2f 샘플)"%(
    np.median(rows[:,1])*10, rows[:,1].min()*10, rows[:,1].max()*10, 1/np.median(rows[:,1])))
print("진폭: 중앙값=%.1f  5~95%%=%.1f~%.1f  min=%.1f max=%.1f"%(np.median(rows[:,2]),
    np.quantile(rows[:,2],0.05),np.quantile(rows[:,2],0.95),rows[:,2].min(),rows[:,2].max()))
print("DC오프셋: 중앙값=%.2f  범위=%.1f~%.1f"%(np.median(rows[:,3]),rows[:,3].min(),rows[:,3].max()))
print("→ 60Hz 전원이 fs로 에일리어싱: |60 - 6*fs| = %.3f Hz 이면 fs=%.4f Hz"%(np.median(rows[:,1])*10,(60-np.median(rows[:,1])*10)/6))

print("\n### B. 이상 AI2도 사인인가?")
Ro=[r for r in runs(o,30)]
rows2=[]
for s,e in Ro:
    v=o.AI2_Current.values[s:e]; t=np.arange(len(v))
    best=None
    for f in np.arange(0.02,0.5,0.0005):
        A=np.c_[np.cos(2*np.pi*f*t),np.sin(2*np.pi*f*t),np.ones_like(t)]
        c,*_=np.linalg.lstsq(A,v,rcond=None)
        r2=1-np.sum((v-A@c)**2)/np.sum((v-v.mean())**2)
        if best is None or r2>best[0]: best=(r2,f,np.hypot(c[0],c[1]),c[2])
    rows2.append(best)
rows2=np.array(rows2)
print("이상 버스트 수=%d  사인 R^2 중앙값=%.4f  범위=%.3f~%.3f"%(len(rows2),np.median(rows2[:,0]),rows2[:,0].min(),rows2[:,0].max()))
print("겉보기주파수 중앙값=%.3f Hz  진폭 중앙값=%.1f"%(np.median(rows2[:,1])*10,np.median(rows2[:,2])))

print("\n### C. 진동 채널 — 백색잡음인가 / 양자화 / 분포")
for nm,d in (("normal",n),("outlier",o)):
    for c in C[:2]:
        v=d[c].values
        dv=np.diff(np.sort(np.unique(v)))
        from scipy import stats
        print("%s %s: n_uniq=%d 최소간격=%.2e  왜도=%+.3f 첨도=%+.3f  정규성 p=%.1e"%(
            nm,c,len(np.unique(v)),dv.min(),stats.skew(v),stats.kurtosis(v),
            stats.normaltest(v[:5000]).pvalue))

print("\n### D. 진동 ACF lag1..8 (버스트 내, 평균)")
for nm,d in (("normal",n),("outlier",o)):
    R2=runs(d,30)
    for k,c in enumerate(C[:2]):
        acc=[]
        for s,e in R2:
            v=d[c].values[s:e]; v=v-v.mean()
            acc.append([np.corrcoef(v[:-L],v[L:])[0,1] for L in range(1,9)])
        m=np.mean(acc,0)
        print("%s %s: "%(nm,c)+" ".join("%+.3f"%x for x in m))

print("\n### E. 버스트 내 주기도 (평균 파워 스펙트럼, 0~5Hz)")
for nm,d in (("normal",n),("outlier",o)):
    R2=[r for r in runs(d,50)] or runs(d,40)
    print(nm,"버스트(>=50/40):",len(R2))
    for c in C:
        P=[]
        for s,e in R2[:300]:
            v=d[c].values[s:s+40]; v=v-v.mean()
            F=np.abs(np.fft.rfft(v*np.hanning(40)))**2
            P.append(F/F.sum())
        P=np.mean(P,0); fr=np.fft.rfftfreq(40,0.1)
        top=np.argsort(P)[::-1][:4]
        print("  %-14s 상위피크 Hz=%s  (파워비 %s)"%(c,
            ",".join("%.2f"%fr[i] for i in top), ",".join("%.2f"%P[i] for i in top)))

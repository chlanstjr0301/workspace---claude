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

print("### F. AI0-AI1 교차스펙트럼 위상/коherence (제대로 된 rocking 검정)")
for nm,d in (("normal",n),("outlier",o)):
    R=runs(d,40)
    Sxy=np.zeros(21,complex); Sxx=np.zeros(21); Syy=np.zeros(21)
    cnt=0
    for s,e in R:
        a=d[C[0]].values[s:s+40]; b=d[C[1]].values[s:s+40]
        a=(a-a.mean())*np.hanning(40); b=(b-b.mean())*np.hanning(40)
        A=np.fft.rfft(a); B=np.fft.rfft(b)
        Sxy+=A*np.conj(B); Sxx+=np.abs(A)**2; Syy+=np.abs(B)**2; cnt+=1
    fr=np.fft.rfftfreq(40,0.1)
    coh=np.abs(Sxy)**2/(Sxx*Syy); ph=np.degrees(np.angle(Sxy))
    print("%s (버스트 %d개)"%(nm,cnt))
    print("  f(Hz) :"+"".join("%7.2f"%x for x in fr[1:16]))
    print("  coher :"+"".join("%7.3f"%x for x in coh[1:16]))
    print("  위상° :"+"".join("%7.0f"%x for x in ph[1:16]))
    print("  AI0파워:"+"".join("%7.3f"%x for x in (Sxx/Sxx[1:].sum())[1:16]))
    print("  AI1파워:"+"".join("%7.3f"%x for x in (Syy/Syy[1:].sum())[1:16]))

print("\n### G. corr(AI0,AI1) — 유의성과 안정성")
from scipy import stats
for nm,d in (("normal",n),("outlier",o)):
    a=d[C[0]].values; b=d[C[1]].values
    r,p=stats.pearsonr(a,b); rs,_=stats.spearmanr(a,b)
    print("%s: 전체 Pearson r=%+.3f (p=%.1e, N=%d)  Spearman=%+.3f"%(nm,r,p,len(a),rs))
    # 20-샘플 윈도우 corr 분포
    R=runs(d,20); W=[]
    for s,e in R:
        for i in range(s,e-19):
            x=a[i:i+20]; y=b[i:i+20]
            W.append(np.corrcoef(x,y)[0,1])
    W=np.array(W)
    print("   20샘플 윈도우 corr: N=%d 평균=%+.3f sd=%.3f  |r|<0.44(p>.05) 비율=%.1f%%  부호-비율=%.1f%%"%(
        len(W),W.mean(),W.std(),100*np.mean(np.abs(W)<0.4438),100*np.mean(W<0)))
    print("   귀무가설 하 sd = 1/sqrt(17) = %.3f"%(1/np.sqrt(17)))
    # 버스트별 corr 분포
    B=[np.corrcoef(a[s:e],b[s:e])[0,1] for s,e in runs(d,30)]
    B=np.array(B)
    print("   버스트별 corr: N=%d 평균=%+.3f sd=%.3f 음수비율=%.1f%% 범위=%+.2f~%+.2f"%(
        len(B),B.mean(),B.std(),100*np.mean(B<0),B.min(),B.max()))

print("\n### H. 정상 파일 시간에 따른 AI2 진폭 / AI0 std (가동 여부)")
R=runs(n,30); rows=[]
for s,e in R:
    v=n.AI2_Current.values[s:e]; t=np.arange(len(v))
    f=0.06
    A=np.c_[np.cos(2*np.pi*f*t),np.sin(2*np.pi*f*t),np.ones_like(t)]
    c,*_=np.linalg.lstsq(A,v,rcond=None)
    rows.append((s,np.hypot(c[0],c[1]),n[C[0]].values[s:e].std(),n[C[1]].values[s:e].std()))
rows=np.array(rows)
print("버스트 수=%d"%len(rows))
for lo,hi in [(0,4000),(4000,8000),(8000,12000),(12000,15000),(15000,16000),(16000,18000),(18000,20000)]:
    m=(rows[:,0]>=lo)&(rows[:,0]<hi)
    if m.sum(): print("  행%5d-%-5d  n=%3d  AI2진폭=%6.1f  AI0std=%.4f  AI1std=%.4f"%(lo,hi,m.sum(),rows[m,1].mean(),rows[m,2].mean(),rows[m,3].mean()))
print("  상관 AI2진폭 vs AI0std : r=%+.3f"%np.corrcoef(rows[:,1],rows[:,2])[0,1])
print("  상관 AI2진폭 vs AI1std : r=%+.3f"%np.corrcoef(rows[:,1],rows[:,3])[0,1])
print("  AI2 진폭 분포: "+", ".join("%.0f"%np.quantile(rows[:,1],q) for q in [0,.1,.25,.5,.75,.9,1]))

print("\n### I. 두 파일 수치 포맷 비교 (세션 지문의 또 다른 흔적)")
import re
for nm,path in (("normal","press_data_normal.csv"),("outlier","press_data_outlier.csv")):
    L=open(os.path.join(RAW,path),encoding='utf-8').read().split("\n")[1:6]
    print(nm); [print("   ",x) for x in L]

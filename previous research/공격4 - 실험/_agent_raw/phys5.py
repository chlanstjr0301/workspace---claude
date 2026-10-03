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

print("### P. 50샘플 버스트 정밀 스펙트럼 (분해능 0.2 Hz) — 결정론적 선 스펙트럼인가")
for nm,d in (("normal",n),("outlier",o)):
    R=[r for r in runs(d,50)]
    print("%s  (50샘플 버스트 %d개)"%(nm,len(R)))
    fr=np.fft.rfftfreq(50,0.1)
    for c in C:
        P=[]
        for s,e in R:
            v=d[c].values[s:s+50]; v=(v-v.mean())*np.hanning(50)
            F=np.abs(np.fft.rfft(v))**2; P.append(F/F[1:].sum())
        P=np.mean(P,0)
        tops=np.argsort(P[1:])[::-1][:5]+1
        print("   %-14s 상위5: %s"%(c," | ".join("%.1fHz:%.0f%%"%(fr[i],100*P[i]) for i in tops)))

print("\n### Q. Ljung-Box — 정상 진동이 백색잡음인가")
from statsmodels.stats.diagnostic import acorr_ljungbox
for nm,d in (("normal",n),("outlier",o)):
    for c in C[:2]:
        R=runs(d,40); rej=0; tot=0; stats=[]
        for s,e in R:
            v=d[c].values[s:s+40]
            r=acorr_ljungbox(v-v.mean(), lags=[8], return_df=True)
            stats.append(r['lb_pvalue'].iloc[0]); tot+=1
            rej += r['lb_pvalue'].iloc[0]<0.05
        print("%s %-14s: lag1-8 Ljung-Box 기각 %d/%d (%.1f%%)  p중앙=%.2e"%(nm,c,rej,tot,100*rej/tot,np.median(stats)))

print("\n### R. 진동에도 전원 에일리어스(0.6Hz=60Hz, 1.2Hz=120Hz) 선이 있나")
R=[r for r in runs(n,50)]; fr=np.fft.rfftfreq(50,0.1)
for c in C:
    P=[]
    for s,e in R:
        v=n[c].values[s:s+50]; v=(v-v.mean())*np.hanning(50)
        F=np.abs(np.fft.rfft(v))**2; P.append(F/F[1:].sum())
    P=np.mean(P,0)
    print("  %-14s 0.6Hz=%.1f%%  1.2Hz=%.1f%%  1.8Hz=%.1f%%  (균등기대=%.1f%%)"%(
        c,100*P[3],100*P[6],100*P[9],100/25))

print("\n### S. MCSA 가능성 — 60Hz 대역은 에일리어싱으로 '평행이동'될 뿐 정보 보존")
print("   실측 캐리어 겉보기 주파수 f_a = 0.600 Hz")
print("   fs_true = (60 -+ 0.6)/6 = 9.900 또는 10.100 Hz  (타임스탬프 명목 10.000 Hz)")
print("   => 타임스탬프 클럭 오차 1.0%%. 60Hz 주변 ±fs/2(=±4.95Hz) 대역은")
print("      0~4.95Hz 로 선형 평행이동되어 '상대 간격이 보존'된다.")
print("      회전자봉 결함 측파대 60±2sf (s=0.02~0.04 -> 2.4~4.8Hz) 는 전부 관측대역 안.")
# 캐리어 제거 후 잔차 스펙트럼 = 측파대 후보
R=[r for r in runs(n,50)]
res=[]
for s,e in R:
    v=n.AI2_Current.values[s:s+50]; t=np.arange(50); f=0.06
    A=np.c_[np.cos(2*np.pi*f*t),np.sin(2*np.pi*f*t),np.ones(50)]
    c_,*_=np.linalg.lstsq(A,v,rcond=None); r_=v-A@c_
    F=np.abs(np.fft.rfft(r_*np.hanning(50)))**2; res.append(F/F[1:].sum())
res=np.mean(res,0); fr=np.fft.rfftfreq(50,0.1)
tops=np.argsort(res[1:])[::-1][:6]+1
print("   정상 AI2 캐리어 제거 후 잔차 상위선: "+" | ".join("%.1fHz:%.0f%%"%(fr[i],100*res[i]) for i in tops))
print("   (캐리어 잔차 에너지 비 = %.3f%% of 원신호)"%(100*np.median([
    (lambda v: (lambda t,A: (lambda c_: np.sum((v-A@c_)**2)/np.sum((v-v.mean())**2))(np.linalg.lstsq(A,v,rcond=None)[0]))(np.arange(50),np.c_[np.cos(2*np.pi*0.06*np.arange(50)),np.sin(2*np.pi*0.06*np.arange(50)),np.ones(50)]))(n.AI2_Current.values[s:s+50]) for s,e in R])))

print("\n### T. 상·하부 진폭비(AI0/AI1) — rocking보다 해석력이 큰 지표")
def br(d):
    out=[]
    for s,e in runs(d,30):
        out.append(d[C[0]].values[s:e].std()/d[C[1]].values[s:e].std())
    return np.array(out)
bn,bo=br(n),br(o)
print("  정상 AI0/AI1 진폭비: 중앙=%.3f  [%.3f, %.3f]  95분위=%.3f"%(np.median(bn),bn.min(),bn.max(),np.quantile(bn,.95)))
print("  이상 AI0/AI1 진폭비: 중앙=%.3f  [%.3f, %.3f]  정상95분위 초과 비율=%.0f%%"%(np.median(bo),bo.min(),bo.max(),100*np.mean(bo>np.quantile(bn,.95))))
print("  -> 상부(모터측) 진동이 하부(펌프측)보다 커졌다. 펌프 내부 결함이면 반대가 기대된다.")

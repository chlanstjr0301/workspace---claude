# -*- coding: utf-8 -*-
import numpy as np, common
Fn,Tn,Bn,Fa,Ta,Ba=common.get(); i=common.FEAT_NAMES.index
n,o=common.load()

print("### A3-0  AI2_acf1 이 '세션 상수'인가 — 세션 내부 변동")
j=i("AI2_acf1")
print("  정상 452 burst: mean=%.4f sd=%.5f min=%.4f max=%.4f range=%.4f"%(
   Fn[:,j].mean(),Fn[:,j].std(),Fn[:,j].min(),Fn[:,j].max(),np.ptp(Fn[:,j])))
print("  이상 13 burst 별 window 평균 AI2_acf1:")
for b in sorted(set(Ba.tolist())):
    m=Ba==b; print("    burst %2d (n=%3d): mean=%+.4f min=%+.4f max=%+.4f"%(b,m.sum(),Fa[m,j].mean(),Fa[m,j].min(),Fa[m,j].max()))
print("  이상 전체: sd=%.4f range=[%.4f, %.4f]  <- 단일 세션 내부에서 0.9 폭으로 변동"%(Fa[:,j].std(),Fa[:,j].min(),Fa[:,j].max()))
print("  => 세션 클럭 지문이라면 같은 세션 내부에서 상수여야 한다. 아니다.")

print("\n### A3-1  Equipment_state 라벨 확인")
print("  normal  Equipment_state 분포:", dict(n.Equipment_state.value_counts()))
print("  outlier Equipment_state 분포:", dict(o.Equipment_state.value_counts()))
print("  normal  TimeStamp [%s ~ %s]"%(n.TimeStamp.min(),n.TimeStamp.max()))
print("  outlier TimeStamp [%s ~ %s]"%(o.TimeStamp.min(),o.TimeStamp.max()))

print("\n### A3-2  15000행 '체제 변화'는 급격한가 점진적인가 — 1000행 단위")
show=["AI0_acf1","AI0_std","AI1_acf1","AI1_std","AI2_std","corr01"]
ix=[i(s) for s in show]
print("  %-14s %6s "%("구간","nwin")+"".join("%11s"%s for s in show)+"   burst수")
for lo in range(0,20000,1000):
    m=(Tn>=lo)&(Tn<lo+1000)
    if m.sum()==0: continue
    nb=len(set(Bn[m].tolist()))
    print("  %6d-%-7d %6d "%(lo,lo+1000,m.sum())+"".join("%11.3f"%v for v in Fn[m][:,ix].mean(0))+"   %d"%nb)

print("\n### A3-3  변화점 정확 위치 — burst 단위 AI0_std 이진분할 (CUSUM 최대)")
# burst-level series
bl=[]
for b in sorted(set(Bn.tolist())):
    m=Bn==b; bl.append((Tn[m].min(), Fn[m][:,i("AI0_std")].mean(), Fn[m][:,i("corr01")].mean(), Fn[m][:,i("AI1_std")].mean()))
bl=np.array(bl)
for k,nm in [(1,"AI0_std"),(2,"corr01"),(3,"AI1_std")]:
    x=bl[:,k]; x=(x-x.mean())/x.std(); c=np.cumsum(x)
    kstar=int(np.argmax(np.abs(c)))
    # exhaustive two-sample t-like split
    best=(None,-1)
    for s in range(5,len(x)-5):
        d=abs(x[:s].mean()-x[s:].mean())/np.sqrt(1/s+1/(len(x)-s))
        if d>best[1]: best=(s,d)
    s=best[0]
    print("  %-9s CUSUM argmax burst=%d(row~%d) | 최적분할 burst=%d(row~%d) t=%.1f | 이전평균=%.4f 이후평균=%.4f"%(
        nm,kstar,bl[kstar,0],s,bl[s,0],best[1],bl[:s,k].mean(),bl[s:,k].mean()))

print("\n### A3-4  시간 공백 구조 — 15000행 근처에 세션 경계가 있는가")
dt=n.TimeStamp.diff().dt.total_seconds().values
big=np.where(dt>5)[0]
print("  dt>5s 인 지점 %d개"%len(big))
for p in big[:40]:
    print("    row %5d  dt=%.1f s  (%s)"%(p,dt[p],n.TimeStamp.iloc[p]))
print("  상위 5개 공백:", sorted([(round(float(dt[p]),1),int(p)) for p in big],reverse=True)[:5])
rows=np.arange(len(n)); 
print("  row 14900~15100 의 dt 최대: %.3f s"%np.nanmax(dt[14900:15100]))
print("  정상파일 전체 dt 분포: median=%.3f p99=%.3f max=%.3f"%(np.nanmedian(dt),np.nanpercentile(dt,99),np.nanmax(dt)))

# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, os
RAW = r"C:/Users/cmsch/Desktop/대회/2026년 제6회 K-인공지능 제조데이터 분석 경진대회/data/raw"
n = pd.read_csv(os.path.join(RAW,"press_data_normal.csv"), index_col=0, parse_dates=["TimeStamp"])
o = pd.read_csv(os.path.join(RAW,"press_data_outlier.csv"), index_col=0, parse_dates=["TimeStamp"])
C=["AI0_Vibration","AI1_Vibration","AI2_Current"]

print("### 1. 버스트 구조")
for nm,d in (("normal",n),("outlier",o)):
    dt=d.TimeStamp.diff().dt.total_seconds().values
    e=np.concatenate(([0],np.where(dt>0.5)[0],[len(d)]))
    runs=[(int(a),int(b)) for a,b in zip(e[:-1],e[1:]) if b>a]
    L=np.array([b-a for a,b in runs])
    gaps=dt[np.where(dt>0.5)[0]]
    print(f"{nm}: runs={len(runs)} len uniq={np.unique(L)} total_rows={len(d)}")
    print(f"   gap(s): n={len(gaps)} min={gaps.min():.3f} med={np.median(gaps):.3f} max={gaps.max():.3f} mean={gaps.mean():.3f}")
    # 버스트 시작 간격
    st=d.TimeStamp.values[[a for a,b in runs]]
    per=np.diff(st).astype('timedelta64[ms]').astype(float)/1000
    print(f"   burst start period(s): med={np.median(per):.3f} min={per.min():.3f} max={per.max():.3f}")
    print(f"   전체 wall-clock: {(d.TimeStamp.iloc[-1]-d.TimeStamp.iloc[0]).total_seconds():.1f} s")
    # 샘플간격 (버스트 내부)
    ins=dt[(dt>0)&(dt<=0.5)]
    print(f"   내부 샘플간격: med={np.median(ins):.4f} uniq={np.unique(np.round(ins,3))[:8]}")

print("\n### 2. Equipment_state / 전류 수준")
for nm,d in (("normal",n),("outlier",o)):
    print(nm, "state uniq:", d.Equipment_state.unique(),
          " AI2 mean=%.3f std=%.3f min=%.2f max=%.2f"%(d.AI2_Current.mean(),d.AI2_Current.std(),d.AI2_Current.min(),d.AI2_Current.max()),
          " |AI2| mean=%.2f"%d.AI2_Current.abs().mean(),
          " AI2 rms=%.2f"%np.sqrt((d.AI2_Current**2).mean()))
    for c in C[:2]:
        print("   ",c,"std=%.4f ptp=%.4f rms=%.4f"%(d[c].std(), np.ptp(d[c]), np.sqrt((d[c]**2).mean())))

print("\n### 3. AI2 파형 — 정상 첫 버스트 50샘플 실제 값")
d=n
dt=d.TimeStamp.diff().dt.total_seconds().values
e=np.concatenate(([0],np.where(dt>0.5)[0],[len(d)]))
runs=[(int(a),int(b)) for a,b in zip(e[:-1],e[1:]) if b>a]
for ri in [0,1,2]:
    s,eo=runs[ri]
    v=d.AI2_Current.values[s:eo]
    print("run%d:"%ri, " ".join("%7.1f"%x for x in v[:50]))
    print("    부호:", "".join("+" if x>0 else "-" for x in v))
print("\n이상 첫 버스트:")
dt=o.TimeStamp.diff().dt.total_seconds().values
e=np.concatenate(([0],np.where(dt>0.5)[0],[len(o)]))
runso=[(int(a),int(b)) for a,b in zip(e[:-1],e[1:]) if b>a]
for ri in [0,1]:
    s,eo2=runso[ri]
    v=o.AI2_Current.values[s:eo2]
    print("run%d:"%ri," ".join("%7.1f"%x for x in v[:50]))
    print("    부호:", "".join("+" if x>0 else "-" for x in v))

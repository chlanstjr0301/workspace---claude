# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd
RAW = r"C:\Users\cmsch\Desktop\대회\2026년 제6회 K-인공지능 제조데이터 분석 경진대회\data\raw"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feat.npz")
COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
SEQ = 20; GAP_SEC = 0.5; TRAIN_END, VALID_END = 12000, 15000
FEAT_NAMES = [f"{c[:3]}_{m}" for c in COLS for m in ("acf1","acf2","std","ptp")] + ["corr01"]

def load():
    n = pd.read_csv(os.path.join(RAW,"press_data_normal.csv"), index_col=0, parse_dates=["TimeStamp"])
    o = pd.read_csv(os.path.join(RAW,"press_data_outlier.csv"), index_col=0, parse_dates=["TimeStamp"])
    return n,o

def runs(df):
    dt = df.TimeStamp.diff().dt.total_seconds().values
    e = np.concatenate(([0], np.where(dt>GAP_SEC)[0], [len(df)]))
    return [(int(a),int(b)) for a,b in zip(e[:-1],e[1:]) if b>a]

def _acf(x,lag):
    xc = x-x.mean()
    if xc.std()<1e-9: return 0.0
    return float(np.corrcoef(xc[:-lag],xc[lag:])[0,1])

def window_features(df):
    """returns F, T(start row), B(burst id)"""
    X = df[COLS].values
    F,T,B = [],[],[]
    for bi,(s,e) in enumerate(runs(df)):
        for i in range(s, e-SEQ+1):
            W = X[i:i+SEQ]; f=[]
            for k in range(3):
                f += [_acf(W[:,k],1), _acf(W[:,k],2), float(W[:,k].std()), float(np.ptp(W[:,k]))]
            a,b = W[:,0]-W[:,0].mean(), W[:,1]-W[:,1].mean()
            f.append(float(np.corrcoef(a,b)[0,1]) if a.std()>1e-9 and b.std()>1e-9 else 0.0)
            F.append(f); T.append(i); B.append(bi)
    return np.array(F), np.array(T), np.array(B)

def get():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["Fn"],d["Tn"],d["Bn"],d["Fa"],d["Ta"],d["Ba"]
    n,o = load()
    Fn,Tn,Bn = window_features(n); Fa,Ta,Ba = window_features(o)
    np.savez(CACHE, Fn=Fn,Tn=Tn,Bn=Bn,Fa=Fa,Ta=Ta,Ba=Ba)
    return Fn,Tn,Bn,Fa,Ta,Ba

def prf(pn, pa):
    tp=int(pa.sum()); fn=len(pa)-tp; fp=int(pn.sum()); tn=len(pn)-fp
    P = tp/(tp+fp) if tp+fp else 0.0
    R = tp/(tp+fn) if tp+fn else 0.0
    F1 = 2*P*R/(P+R) if P+R else 0.0
    return P,R,F1, fp/(fp+tn) if fp+tn else 0.0

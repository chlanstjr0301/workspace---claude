import importlib.util, numpy as np
from sklearn.ensemble import IsolationForest
CM = r"C:\Users\cmsch\Desktop\대회\2026년 제6회 K-인공지능 제조데이터 분석 경진대회\docs\plan\plan A\Candidate Models\candidate_check.py"
s = importlib.util.spec_from_file_location("cc", CM); cc = importlib.util.module_from_spec(s); s.loader.exec_module(cc)
n,o = cc.load(); Fn,Tn = cc.window_features(n); Fa,_ = cc.window_features(o)
tr=Fn[Tn<12000]; va=Fn[(Tn>=12000)&(Tn<15000)]; te=Fn[Tn>=15000]
def ev(ix,sd):
    m=IsolationForest(n_estimators=300,random_state=sd).fit(tr[:,ix])
    th=np.quantile(-m.score_samples(va[:,ix]),0.999)
    pa=-m.score_samples(Fa[:,ix])>th; pn=-m.score_samples(te[:,ix])>th
    tp=int(pa.sum()); fp=int(pn.sum()); P=tp/(tp+fp) if tp+fp else 0.; R=tp/len(pa)
    return (2*P*R/(P+R) if P+R else 0.),P,R,fp/len(pn)
ix13=list(range(13))
print("### 13-feature IF per seed (doc reports F1=0.753 FPR=0.0719 at seed 42)")
rows=[(sd,)+ev(ix13,sd) for sd in range(30)]
for sd,F,P,R,fpr in sorted(rows,key=lambda r:-r[1])[:6]:
    print("  seed=%2d F1=%.3f P=%.3f R=%.3f FPR=%.4f" % (sd,F,P,R,fpr))
print("  ...")
for sd,F,P,R,fpr in sorted(rows,key=lambda r:r[1])[:3]:
    print("  seed=%2d F1=%.3f P=%.3f R=%.3f FPR=%.4f" % (sd,F,P,R,fpr))
i=cc.FEAT_NAMES.index
amp=[i("AI0_std"),i("AI0_ptp"),i("AI1_std"),i("AI1_ptp"),i("AI2_std"),i("AI2_ptp")]
print("\n### does any seed make 13-feat beat 6amp(seed42)=0.864?  %d / 30 seeds"
      % sum(1 for r in rows if r[1] > 0.864))
print("### 13-feat seeds with F1>0.864 and their FPR:",
      [(r[0],round(r[1],3),round(r[4],4)) for r in rows if r[1]>0.864])

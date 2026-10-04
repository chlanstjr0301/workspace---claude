"""Check the hand-written paper's main table against frozen machine-readable results."""
from pathlib import Path
import json,re
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
text=(ROOT/'paper/manuscript_ko.tex').read_text();run=ROOT/'results/pca_20261003_cpu'
r=pd.read_csv(run/'tables/metrics_test.csv');r=r[(r.scope=='operational')&(r.target_fpr==.01)].set_index('key')
keys=['P0_W1_K2_Q','P1_W20_K2_Q','P2_W10_K8_QT','I0_W1_ensemble','I1_W5_ensemble','I2_W5_ensemble','I1_W20_ensemble']
body=text[text.index('모델 & 구성 & TP'):text.index('\\end{tabular}',text.index('모델 & 구성 & TP'))]
lines=[x for x in body.splitlines() if x.startswith(('P0 &','P1 &','P2 &','I0$','I1 &','I2$','I1$'))]
assert len(lines)==len(keys)
for line,key in zip(lines,keys):
 vals=line.split('&')[2:];got=[float(re.sub(r'\\.*','',v).strip()) for v in vals]
 a=r.loc[key];want=[a.TP,a.FN,a.FP,a.Recall*100,a.Precision*100,a.F1,a.F2,a.FPR*100,a.AP]
 tolerance=[0,0,0,.0051,.0051,.000051,.000051,.000051,.000051]
 for g,w,t in zip(got,want,tolerance):assert abs(g-w)<=t,(key,g,w,t)
common=pd.read_csv(run/'tables/metrics_test.csv');common=common[(common.scope=='common')&(common.target_fpr==.01)].set_index('key')
for suffix,expected in [('Q',[150,3,1]),('T2',[90,63,14]),('QT',[148,5,14])]:
 row=common.loc['P1_W20_K2_'+suffix];assert [row.TP,row.FN,row.FP]==expected
pred=pd.read_csv(run/'predictions/operational_P1_W20_K2_Q_target0p01_test.csv');pos=pred[pred.label==1]
assert (len(pos[~pos.fallback]),int(pos[~pos.fallback].prediction.sum()),len(pos[pos.fallback]),int(pos[pos.fallback].prediction.sum()))==(338,323,19,6)
post=pd.read_csv(run/'tables/postprocess_test.csv').set_index('policy')
assert post.FN.to_dict()=={'none':28,'consecutive2':46,'two_of_three':41}
assert post.FP.to_dict()=={'none':1,'consecutive2':0,'two_of_three':0}
print(json.dumps({'main_table_rows_checked':7,'Q_T2_QT_counts_checked':True,'coverage_identity_counts_checked':True,'postprocessing_counts_checked':True,'model_fit_performed':False},indent=2))

from core import *
from models import score
R=Path(__file__).resolve().parents[1];P=R.parents[1];old=P/'runs/pca_20261003_cpu';d=loadrows(R/'manifests/original_rows.csv');e=Engine(d,R,partition='split');cn=d.index[(d.split=='calibration')&(d.label==0)].to_numpy();te=d.index[d.split=='test'].to_numpy();ops=[];oldops=[];maxdiff=[]
for seed in [42,43,44]:
 per={};oldper={}
 for w,prefix in [(1,'I0'),(20,'I1')]:
  name=f'{prefix}_W{w}_S{seed}';b=joblib.load(old/'models'/f'{name}.joblib');meta=json.loads((old/'models/score_metadata.json').read_text())[name];idx=te[e.window(w,False)[te]>=0];x,_=e.x(idx,w,False,'full');s=score(b,x)['IF']/meta['reference'];arr=np.full(len(d),np.nan);arr[idx]=s;per[w]=arr[te]
  expected=np.load(old/'predictions'/f'raw_{name}_test.npz')['IF']/meta['reference'];oldper[w]=expected[te];maxdiff.append(float(np.nanmax(abs(arr[te]-expected[te]))))
 ops.append(np.where(np.isfinite(per[20]),per[20],per[1]));oldops.append(np.where(np.isfinite(oldper[20]),oldper[20],oldper[1]))
s=np.mean(ops,axis=0);ex=np.mean(oldops,axis=0);assert np.allclose(s,ex)
metrics_old=pd.read_csv(old/'tables/metrics_test.csv');row=metrics_old[(metrics_old.key=='I1_W20_ensemble')&(metrics_old.scope=='operational')&(metrics_old.target_fpr==.01)].iloc[0];m=metrics(d.loc[te,'label'],s,s>row.threshold);assert m['TP']==320 and m['FP']==6
js(R/'if_original_saved_reproduction.json',dict(metrics=m,max_normalized_score_error=max(maxdiff),seeds=[42,43,44],row_ids=d.loc[te,'row_id'].tolist(),note='saved model inference and stored raw score parity; new blocked calibration ensemble may have tiny ranking differences'))
# Aggregate same-length P1/P2 comparisons, all predeclared targets and both scopes.
t=pd.read_csv(R/'all_trials.csv');a=t[t.candidate=='P1_Q'];b=t[t.candidate=='P2_Q'];a.merge(b,on=['block','scope','target'],suffixes=('_P1','_P2')).to_csv(R/'tables/burst_comparison.csv',index=False)
# Quantile thresholds and empirical resolution including tie rule.
rec=[]
for path in (R/'models').glob('system_*_inner_*.joblib'):
 system=joblib.load(path);block='inner_early' if path.stem.endswith('inner_early') else 'inner_late';blocks=json.loads((R/'manifests/blocks.json').read_text());dd=loadrows(R/'manifests/review_rows.csv');ee=Engine(dd,R);r=ee.predict(system,blocks[block]['cal'])
 for target,h in system['thresholds'].items():rec.append(dict(candidate=system['spec']['id'],block=block,target=float(target),n=len(r),threshold=h,n_equal=int((r.score==h).sum()),actual_FPR=float((r.score>h).mean()),quantile='higher',decision='>',finite_sample_step=1/len(r),score_unique=r.score.nunique()))
pd.DataFrame(rec).to_csv(R/'tables/threshold_calibration_audit.csv',index=False)
# Original data hashes and complete analysis source snapshot; no final result changes.
js(R/'logs/execution_issues.json',[dict(stage='prepare',error='Existing virtualenv has no pip',fix='Use importlib.metadata to record installed versions; no environment mutation',affects_metrics=False),dict(stage='analysis',error='Local dict variable sys shadowed imported sys module',fix='Import sys as pysys in analysis_report.py and rerun analysis only',affects_metrics=False),dict(stage='design before fitting',change='Anomaly burst halves are 2/2 rather than row-balanced3/1; avoid a single-burst outer check',selection_scores_seen=False)])
print('Original IF reproduced',m['TP'],m['FN'],m['FP'],m['TN'])

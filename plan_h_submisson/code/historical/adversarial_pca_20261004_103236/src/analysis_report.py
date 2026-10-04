from core import *
import sys as pysys
import subprocess,tempfile,traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve,ConfusionMatrixDisplay
R=Path(__file__).resolve().parents[1];P=R.parents[1];d=loadrows(R/'manifests/review_rows.csv');lock=json.loads((R/'frozen_selection.json').read_text());proto=json.loads((R/'protocol.json').read_text());blocks=json.loads((R/'manifests/blocks.json').read_text());sel=lock['selected'];e=Engine(d,R);conditions=[];route_results=[];coverage=[];post_cost=[];complements=[];tail=[]
# Fixed train quantiles, never choose bins from evaluation errors.
train=d[d.split=='train'];edges={s:np.quantile(abs(train[s]),[.25,.5,.75]).tolist() for s in SENSORS};delta=d.groupby(['source_file','review_partition','burst_id'],sort=False).AI2_Current.diff().abs().fillna(0);edges['current_change']=np.quantile(delta[d.split=='train'],[.25,.5,.75]).tolist();js(R/'tables/error_bin_edges.json',edges)
def analyze(r,name,stage):
 r=r.copy();r.TimeStamp=pd.to_datetime(r.TimeStamp);nrm=int((r.label==0).sum());anm=int((r.label==1).sum())
 coverage.append(dict(candidate=name,stage=stage,normal=nrm,anomaly=anm,native_normal=int(((r.label==0)&(~r.fallback if name!='P0_Q' else True)).sum()),native_anomaly=int(((r.label==1)&(~r.fallback if name!='P0_Q' else True)).sum()),full20_normal=int(((r.label==0)&(r.route==20)).sum()),full20_anomaly=int(((r.label==1)&(r.route==20)).sum()),native_definition='any statistical route w>=3; P0 raw-only handled separately',fallback_normal=int(((r.label==0)&r.fallback).sum()),fallback_anomaly=int(((r.label==1)&r.fallback).sum()),unavailable=0))
 for route,g in r.groupby('route'):route_results.append(dict(candidate=name,stage=stage,route=route,**metrics(g.label,g.score,g.prediction)))
 masks={'first19':r.burst_pos<=19,'after19':r.burst_pos>19,'first_row':r.burst_pos==1,'short_burst_lt20':r.burst_length<20,'long_burst_ge20':r.burst_length>=20,'fallback':r.fallback,'main':~r.fallback,'cross_gap':r.crosses_gap,'within_burst':~r.crosses_gap,'elapsed_gt2p5':r.elapsed_seconds>2.5,'elapsed_le2p5':r.elapsed_seconds<=2.5}
 for s in SENSORS:
  bins=np.searchsorted(edges[s],abs(r[s]),side='right')
  for j in range(4):masks[s+'_train_abs_quartile'+str(j+1)]=bins==j
 dv=r.row_id.map(dict(zip(d.row_id,delta)));bins=np.searchsorted(edges['current_change'],dv,side='right')
 for j in range(4):masks['current_change_train_quartile'+str(j+1)]=bins==j
 for cond,mask in masks.items():
  g=r[np.asarray(mask)];y=g.label.to_numpy();pred=g.prediction.to_numpy();nn=int((y==0).sum());na=int((y==1).sum());fp=int(((y==0)&(pred==1)).sum());fn=int(((y==1)&(pred==0)).sum());conditions.append(dict(candidate=name,stage=stage,condition=cond,normal=nn,anomaly=na,FP=fp,FN=fn,FPR=fp/nn if nn else None,FNR=fn/na if na else None))
for spec in proto['candidate_specs']:
 p=R/'predictions'/f'dev_{spec["id"]}_all_targets.csv'
 if p.exists():
  r=loadrows(p);r=r[r.target==.01];analyze(r,spec['id'],'inner_target01')
for stage in ['outer_development','historical']:
 for name in ['baseline','selected','IF_same_W20','selected_consecutive2','selected_two_of_three']:
  r=loadrows(R/'predictions'/f'{stage}_{name}.csv');analyze(r,name,stage)
 # Cost of persistence: row failures and burst first-alarm delays; no point adjustment.
 raw=loadrows(R/'predictions'/f'{stage}_selected.csv');eps=alarm_summary(raw,raw.prediction).set_index('burst_id')
 for policy in ['consecutive2','two_of_three']:
  rp=loadrows(R/'predictions'/f'{stage}_selected_{policy}.csv');ep=alarm_summary(rp,rp.prediction).set_index('burst_id');delta_delay=ep.first_alarm_since_observed_burst_start_seconds-eps.first_alarm_since_observed_burst_start_seconds;post_cost.append(dict(stage=stage,policy=policy,raw_FN=int(((raw.label==1)&(raw.prediction==0)).sum()),post_FN=int(((rp.label==1)&(rp.prediction==0)).sum()),new_TP=int(((raw.label==1)&(raw.prediction==0)&(rp.prediction==1)).sum()),new_FN=int(((raw.label==1)&(raw.prediction==1)&(rp.prediction==0)).sum()),removed_FP=int(((raw.label==0)&(raw.prediction==1)&(rp.prediction==0)).sum()),new_FP=int(((raw.label==0)&(raw.prediction==0)&(rp.prediction==1)).sum()),short_burst_anomaly=int(((raw.label==1)&(raw.burst_length<20)).sum()),new_FN_short=int(((raw.label==1)&(raw.burst_length<20)&(raw.prediction==1)&(rp.prediction==0)).sum()),anomaly_bursts=int((ep.label==1).sum()),no_alarm_anomaly_bursts=int(((ep.label==1)&ep.no_alarm).sum()),additional_delay_median_among_both_detected=float(delta_delay[ep.label==1].median())))
 for kind in ['PR','ROC']:
  fig,ax=plt.subplots(figsize=(6,4))
  for name in ['baseline','selected','IF_same_W20']:
   r=loadrows(R/'predictions'/f'{stage}_{name}.csv')
   if kind=='PR':p,rc,_=precision_recall_curve(r.label,r.score);ax.plot(rc,p,label=name);ax.set(xlabel='Recall',ylabel='Precision')
   else:fpr,tpr,_=roc_curve(r.label,r.score);ax.plot(fpr,tpr,label=name);ax.set(xlabel='FPR',ylabel='Recall')
  ax.legend();fig.tight_layout();fig.savefig(R/'figures'/f'{stage}_{kind}.png',dpi=160);plt.close(fig)
 for name in ['baseline','selected']:
  r=loadrows(R/'predictions'/f'{stage}_{name}.csv');fig,ax=plt.subplots(figsize=(4,4));ConfusionMatrixDisplay.from_predictions(r.label,r.prediction,ax=ax,colorbar=False);ax.set_title(f'{stage}: {name}');fig.tight_layout();fig.savefig(R/'figures'/f'{stage}_{name}_confusion.png',dpi=160);plt.close(fig)
  fig,axs=plt.subplots(2,1,figsize=(11,5))
  for y,ax in enumerate(axs):
   q=r[r.label==y]
   for label,color,mask in [('correct','grey',q.error=='correct'),('FP','red',q.error=='FP'),('FN','orange',q.error=='FN')]:
    g=q[mask];ax.scatter(g.TimeStamp,g.score,s=9,c=color,label=label)
   ax.axhline(float(q.threshold.iloc[0]),ls='--',c='black');ax.set_ylabel(f'label={y} score');ax.legend(loc='upper right')
  fig.suptitle('Actual timestamps; points only, no lines across gaps');fig.autofmt_xdate();fig.tight_layout();fig.savefig(R/'figures'/f'{stage}_{name}_error_timeline.png',dpi=160);plt.close(fig)
# Q vs T2 vs recalibrated QT, and separate OR of individually calibrated decisions.
for target in [.001,.005,.01]:
 rr={key:loadrows(R/'predictions'/f'dev_{key}_all_targets.csv').query('target==@target').reset_index(drop=True) for key in ['P1_Q','P1_T2','P1_QT']};q=rr['P1_Q'];t=rr['P1_T2'];qt=rr['P1_QT'];assert q.row_id.tolist()==t.row_id.tolist()==qt.row_id.tolist();yp=q.label.to_numpy();p=q.prediction.to_numpy();pt=t.prediction.to_numpy();pc=qt.prediction.to_numpy();orr=(p|pt)
 complements.append(dict(target=target,T2_catches_Q_misses=int(((yp==1)&(p==0)&(pt==1)).sum()),Q_catches_T2_misses=int(((yp==1)&(p==1)&(pt==0)).sum()),QT_added_TP=int(((yp==1)&(p==0)&(pc==1)).sum()),QT_lost_TP=int(((yp==1)&(p==1)&(pc==0)).sum()),QT_added_FP=int(((yp==0)&(p==0)&(pc==1)).sum()),QT_removed_FP=int(((yp==0)&(p==1)&(pc==0)).sum()),QT_OR_disagreement=int((pc!=orr).sum()),OR_TP=int(((yp==1)&(orr==1)).sum()),OR_FP=int(((yp==0)&(orr==1)).sum())))
# Reference sample counts, tail resolution, routed fallback policy and calibration ties.
for p in (R/'models').glob('system_*_inner_*.joblib'):
 sys=joblib.load(p)
 for mi,member in enumerate(sys['members']):
  for w,m in member.items():
   n=m['pool_n'];tail.append(dict(system=p.name,member=mi,route=w,n=n,routed_n=m['routed_n'],pool_policy=m['pool_policy'],empirical_resolution=1/(n+1),reference=m['reference'],ecdf=bool(m['ecdf'] is not None),unique_reference_scores=len(np.unique(m['ecdf'])) if m['ecdf'] is not None else None,tail_saturation='no extrapolation' if m['ecdf'] is not None else 'Q95 multiplier unbounded'))
pd.DataFrame(conditions).to_csv(R/'tables/condition_errors.csv',index=False);pd.DataFrame(route_results).to_csv(R/'tables/route_metrics.csv',index=False);pd.DataFrame(coverage).to_csv(R/'tables/native_coverage.csv',index=False);pd.DataFrame(post_cost).to_csv(R/'tables/postprocessing_costs.csv',index=False);pd.DataFrame(complements).to_csv(R/'tables/Q_T2_QT_complements.csv',index=False);pd.DataFrame(tail).to_csv(R/'tables/calibration_tail_audit.csv',index=False)
# Per-burst observed duration excludes all between-burst gaps, model prep duration is recorded rowwise.
rows=[]
for (sp,bid),g in d.groupby(['split','burst_id'],sort=False):rows.append(dict(split=sp,burst_id=bid,label=int(g.label.iloc[0]),n=len(g),observed_span_seconds=float((g.TimeStamp.iloc[-1]-g.TimeStamp.iloc[0]).total_seconds())))
pd.DataFrame(rows).to_csv(R/'tables/burst_observed_duration.csv',index=False)
# Reopen saved final model in fresh process on unlabeled CSV and verify scores and decisions.
check=[]
with tempfile.TemporaryDirectory(dir=R/'logs') as tmp:
 for alias,name,target in [('baseline','P1_Q',.01),('selected',sel['candidate'],sel['target']),('IF_same_W20','IF_P1',.01)]:
  expected=loadrows(R/'predictions'/f'historical_{alias}.csv')
  for file,g in expected.groupby('source_file',sort=False):
   src=Path(tmp)/(alias+'_'+file);out=Path(tmp)/(alias+'_'+file+'.pred.csv');g[['TimeStamp']+SENSORS].to_csv(src,index=False)
   pr=subprocess.run([pysys.executable,str(R/'src/infer_review.py'),'--model',str(R/'models'/f'final_{name}.joblib'),'--csv',str(src),'--target',str(target),'--out',str(out)],capture_output=True,text=True);assert pr.returncode==0,pr.stderr
   a=pd.read_csv(out);assert np.allclose(a.score,g.score,atol=1e-10,rtol=1e-10);assert np.array_equal(a.prediction,g.prediction);check.append(dict(alias=alias,file=file,n=len(g),max_score_delta=float(np.max(abs(a.score.to_numpy()-g.score.to_numpy()))),identical_prediction=True))
js(R/'saved_inference_validation.json',check)
# Complete model metadata preserves logs from all engines and seed confirmations.
models=[]
for p in (R/'models').glob('*.joblib'):
 b=joblib.load(p)
 if isinstance(b,dict) and 'train_rows' in b:models.append(dict(file=p.name,sha256=sha(p),kind=b['kind'],n_train=len(b['train_rows']),fit_seconds=b['seconds'],rank=b['rank'],names=b['names'],removed=[n for n,k in zip(b['names'],b['keep']) if not k],eigenvalues=np.asarray(b['eigenvalues']).tolist()))
js(R/'manifests/model_inventory.json',models)
# Final pair count summaries.
pair=pd.read_csv(R/'paired_errors.csv');pair.groupby(['evaluation','comparison','change']).size().rename('rows').to_csv(R/'tables/paired_changes.csv')
js(R/'checkpoint.json',dict(stage='analysis_and_inference_verified',new_saved_inference_rows=sum(a['n'] for a in check),unchanged_selection_sha256=sha(R/'frozen_selection.json')))
print('Analysis complete; saved inference verified',sum(a['n'] for a in check))

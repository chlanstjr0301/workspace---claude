from v2 import *
from functools import cmp_to_key
import sys

def cmp(a,b):
 if abs(a['F2']-b['F2'])>1e-12:return -1 if a['F2']>b['F2'] else 1
 if a['FP']!=b['FP']:return -1 if a['FP']<b['FP'] else 1
 if a['rule']!=b['rule']:return -1 if a['rule']<b['rule'] else 1
 return (a['candidate']>b['candidate'])-(a['candidate']<b['candidate'])
def protocol():
 p=json.loads((R/'protocol_v2.json').read_text());assert sha(R/'protocol_v2.json')==(R/'protocol_v2.sha256').read_text().strip();assert all(sha(R/'models'/f)==v for f,v in p['model_hashes'].items());return p

def save_metrics(r,cid,block,base,control):
 m=measure(r);jud=judge(measure(base),m,'effect' if block=='S_pooled' else 'maintenance');pair=pair_counts(base,r);vscontrol=pair_counts(control,r) if control is not None else {};return dict(candidate=cid,block=block,**m,**jud,**{'vs_B0_'+k:v for k,v in pair.items()},**{'vs_C0_'+k:v for k,v in vscontrol.items()})

def selection():
 assert not (R/'selected_config.json').exists();d,e,b0,r5=context();p=protocol();assert p['baseline_feasible'],'B0 already violates required absolute FPR; OR search structurally impossible';started=time.perf_counter();js(R/'implementation_lock.json',dict(created=pd.Timestamp.now(tz='UTC').isoformat(),sources={f.name:sha(f) for f in (R/'src').glob('*.py')},protocol_sha256=sha(R/'protocol_v2.json')))
 ids=np.array(sorted(p['blocks']['S1']+p['blocks']['S2']));cal=d.index[(d.split=='calibration')&(d.label==0)].to_numpy();baseline=base_predictions(e,b0,ids);control=generate(e,b0,r5,ids,p['control_C0']['theta'],'G0');records=[];trials=[];pairedrows=[];seen={};predictions={};old=json.loads((R/'inputs/previous/selected_config.json').read_text());oldtheta={v['threshold']:k for k,v in old['configs'].items() if v['score']=='R5'}
 def add(r,cid):
  r.to_csv(R/'predictions'/f'{cid}_S_pooled.csv',index=False)
  for name,ii in [('S1',p['blocks']['S1']),('S2',p['blocks']['S2']),('S_pooled',ids)]:
   a=r.loc[ii];bb=baseline.loc[ii];cc=control.loc[ii];records.append(save_metrics(a,cid,name,bb,cc));a.to_csv(R/'predictions'/f'{cid}_{name}.csv',index=False)
  for label,ref in [('B0',baseline),('C0',control)]:
   z=r[r.prediction!=ref.prediction].copy();z['candidate']=cid;z['comparator']=label;z['comparator_prediction']=ref.loc[z.index,'prediction'];z['change']=np.where(z.label==1,np.where(z.prediction==1,'new_TP','new_FN'),np.where(z.prediction==1,'additional_FP','removed_FP'));pairedrows.append(z)
 for cid,r in [('B0',baseline),('C0',control),('C1',baseline.copy())]:add(r,cid)
 candidates={}
 for cfg in p['candidates']:
  tick=time.perf_counter();cid=cfg['id'];r=generate(e,b0,r5,ids,cfg['theta'],cfg['rule']);calr=generate(e,b0,r5,cal,cfg['theta'],cfg['rule']);r.to_csv(R/'predictions'/f'{cid}_S_pooled.csv',index=False);calr.to_csv(R/'predictions'/f'{cid}_C.csv',index=False);add(r,cid);m=measure(r);fail=[]
  for name,ii in [('S1',p['blocks']['S1']),('S2',p['blocks']['S2']),('S_pooled',ids)]:
   jud=judge(measure(baseline.loc[ii]),measure(r.loc[ii]),'effect' if name=='S_pooled' else 'maintenance')
   if not jud['passes']:fail.append(name+':'+jud['failed_rules'])
  signature=hashlib.sha256(np.r_[calr.prediction.to_numpy(np.uint8),r.prediction.to_numpy(np.uint8)].tobytes()).hexdigest();duplicate=None
  if cfg['rule']=='G0' and cfg['theta'] in oldtheta:duplicate='previous:'+oldtheta[cfg['theta']]
  elif signature in seen:duplicate=seen[signature]
  else:seen[signature]=cid
  trial=dict(candidate=cid,rule=cfg['rule'],q=cfg['q'],theta=cfg['theta'],**m,**{'vs_B0_'+k:v for k,v in pair_counts(baseline,r).items()},**{'vs_C0_'+k:v for k,v in pair_counts(control,r).items()},eligible=not fail and duplicate is None,status='duplicate_excluded' if duplicate else 'completed',duplicate_of=duplicate,failed_rules=' | '.join(fail),seconds=time.perf_counter()-tick);trials.append(trial);candidates[cid]=cfg;print(cid,dict(TP=m['TP'],FN=m['FN'],FP=m['FP']),trial['eligible'],duplicate,trial['failed_rules'],flush=True)
 best=sorted([x for x in trials if x['eligible']],key=cmp_to_key(cmp));chosen=best[0] if best else None;pd.DataFrame(records).to_csv(R/'block_metrics.csv',index=False);pd.DataFrame(trials).to_csv(R/'all_trials.csv',index=False);pd.concat(pairedrows).to_csv(R/'paired_selection.csv',index=False)
 lock=dict(created=pd.Timestamp.now(tz='UTC').isoformat(),selected=chosen['candidate'] if chosen else None,selected_config=candidates[chosen['candidate']] if chosen else None,selection_metrics=chosen,protocol_sha256=sha(R/'protocol_v2.json'),implementation_sha256=sha(R/'implementation_lock.json'),models=p['model_hashes'],evaluation_row_ids_sha256={n:sha(R/'manifests'/f'rows_{n}.csv') for n in ['S1','S2','V']},candidate_V_H_scores_seen=False,seconds=time.perf_counter()-started,reselection_allowed=False,baseline=dict(artifact='models/baseline.joblib',threshold=b0['threshold']),final_decision_pending_V=chosen is not None)
 js(R/'selected_config.json',lock);(R/'selected_config.sha256').write_text(sha(R/'selected_config.json')+'\n');js(R/'checkpoint.json',dict(stage='selection_frozen',selected=lock['selected']));print('Selected:',lock['selected'])

def evaluate():
 d,e,b0,r5=context();p=protocol();lock=json.loads((R/'selected_config.json').read_text());assert sha(R/'selected_config.json')==(R/'selected_config.sha256').read_text().strip();assert not (R/'followup_evaluation.json').exists();records=[];paired=[];outcomes={};cid=lock['selected'];cfg=lock['selected_config']
 if cid is None:
  js(R/'followup_evaluation.json',dict(selected=None,V='not_evaluated_no_selected_candidate',H='not_evaluated_no_selected_candidate',reselection=False));return
 for block,ids in [('V',np.array(p['blocks']['V'])),('H',d.index[d.split=='test'].to_numpy())]:
  b=base_predictions(e,b0,ids)
  # For V only, retain original P1's already observed selection-prefix context; R5/h reset at intact burst boundary.
  r=generate(e,b0,r5,ids,cfg['theta'],cfg['rule']);bm=measure(b);cm=measure(r);j=judge(bm,cm,'maintenance' if block=='V' else 'effect');outcomes[block]=j
  for name,a in [('B0',b),(cid,r)]:a.to_csv(R/'predictions'/f'{name}_{block}.csv',index=False);records.append(dict(candidate=name,block=block,**measure(a),**(j if name==cid else {})))
  z=r[r.prediction!=b.prediction].copy();z['block']=block;z['candidate']=cid;z['comparator']='B0';z['comparator_prediction']=b.loc[z.index,'prediction'];z['change']=np.where(z.label==1,np.where(z.prediction==1,'new_TP','new_FN'),np.where(z.prediction==1,'additional_FP','removed_FP'));paired.append(z);js(R/f'{block}_evaluation.json',dict(candidate=cid,base_metrics=bm,candidate_metrics=cm,judgment=j,paired=pair_counts(b,r),reselection=False,selection_sha256=sha(R/'selected_config.json')))
  print(block,'base',bm['TP'],bm['FN'],bm['FP'],'candidate',cm['TP'],cm['FN'],cm['FP'],'judge',j,flush=True)
  if block=='V' and not j['stability_pass']:outcomes['H']='not_evaluated_V_failed';break
 pd.DataFrame(records).to_csv(R/'followup_metrics.csv',index=False);pd.concat(paired).to_csv(R/'paired_followup.csv',index=False);js(R/'followup_evaluation.json',dict(selected=cid,outcomes=outcomes,reselection=False,new_independent_data=False));js(R/'checkpoint.json',dict(stage='followup_complete',selected=cid,outcomes=outcomes))
if __name__=='__main__':
 if sys.argv[1]=='select':selection()
 elif sys.argv[1]=='evaluate':evaluate()
 else:raise ValueError(sys.argv[1])

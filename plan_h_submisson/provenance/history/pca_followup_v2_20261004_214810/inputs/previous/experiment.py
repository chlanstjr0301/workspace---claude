from auxiliary import *
import sys
from functools import cmp_to_key

def eligible(b,c,strict=True,delta=.001):
 rules=dict(FN_decreased=c['FN']<b['FN'] if strict else c['FN']<=b['FN'],F2=c['F2']>b['F2']+1e-12 if strict else c['F2']>=b['F2']-1e-12,F1=c['F1']>=b['F1']-1e-12,FP_budget=c['FP']-b['FP']<=allowed_fp(b['TN']+b['FP'],delta),absolute_FPR=c['FP']<=allowed_fp(b['TN']+b['FP'],.01))
 return all(rules.values()),[k for k,v in rules.items() if not v]

def pick_res(rows):
 order={'R0':0,'R1':1,'R4':2,'R5':3,'R2':4,'R3':5}
 def cmp(a,b):
  for va,vb,sign in [(a['F2'],b['F2'],-1),(a['new_FP'],b['new_FP'],1),(a['FN'],b['FN'],1),(order[a['score_id']],order[b['score_id']],1)]:
   if abs(va-vb)>1e-12:return (-1 if va<vb else 1)*sign
  return (a['candidate']>b['candidate'])-(a['candidate']<b['candidate'])
 return sorted(rows,key=cmp_to_key(cmp))[0] if rows else None

def context():
 d=loadrows(R/'inputs/frozen_rows.csv');e=Engine(d);base=joblib.load(R/'models/baseline.joblib');p=json.loads((R/'preregistration.json').read_text());assert sha(R/'preregistration.json')==(R/'preregistration.sha256').read_text().strip()
 return d,e,base,p

def run_selection():
 assert not (R/'selected_config.json').exists(),'Selection already frozen; use new reproduction directory'
 d,e,base,p=context();t=time.perf_counter();js(R/'implementation_lock.json',dict(created=pd.Timestamp.now(tz='UTC').isoformat(),sources={f.name:sha(f) for f in (R/'src').glob('*.py')},preregistration_sha256=sha(R/'preregistration.json')))
 models,diag=fit_all(e,base,R/'models');blocks={k:np.asarray(v) for k,v in p['blocks'].items() if k!='V'};blocks['S_pooled']=np.sort(np.r_[blocks['S1'],blocks['S2']]);calids=d.index[(d.split=='calibration')&(d.label==0)].to_numpy();bc,_=evaluated(e.predict(base,calids),base['threshold']);baseline={k:evaluated(e.predict(base,ii),base['threshold']) for k,ii in blocks.items()};records=[];trials=[];thresholds=[];configs={};pairedrows=[]
 for name,(b,bm) in baseline.items():
  b.to_csv(R/'predictions'/f'baseline_{name}.csv',index=False);records.append(dict(candidate='baseline',block=name,**bm,new_TP=0,lost_TP=0,new_FP=0,removed_FP=0,allowed_additional_FP=allowed_fp(bm['FP']+bm['TN'],.001)))
 for key,m in models.items():
  tt=time.perf_counter();sc=aux_score(m,e,base,calids,bc.score);scores={name:aux_score(m,e,base,ii,baseline[name][0].score) for name,ii in blocks.items()}
  for cfg in [x for x in p['candidates'] if x['score']==key]:
   cid=cfg['id'];tau,td=cal_threshold(bc.prediction,sc,cfg['additional_calibration_FPR']);configs[cid]=dict(**cfg,threshold=tau,model=f'models/{key}.joblib');thresholds.append(dict(candidate=cid,score_id=key,threshold=tau,**td));res={};fail=[]
   for name,(b,bm) in baseline.items():
    out,cm=combine(b,scores[name],tau,key,base['threshold']);out['candidate']=cid;out.to_csv(R/'predictions'/f'{cid}_{name}.csv',index=False);res[name]=cm;ok,why=eligible(bm,cm,name=='S_pooled');fail.extend(name+':'+w for w in why);records.append(dict(candidate=cid,score_id=key,block=name,**cm,allowed_additional_FP=allowed_fp(bm['FP']+bm['TN'],.001),passes=ok,failed_rules=';'.join(why)))
    if name=='S_pooled':
     z=out[out.prediction!=out.base_prediction].copy();z['change']=np.where(z.label==1,'new_TP','new_FP');pairedrows.append(z)
     if key=='R0':
      for metric in ['AP','PR_AUC_trapezoid','ROC_AUC']:assert abs(cm[metric]-bm[metric])<1e-12
   strict=all(eligible(baseline[n][1],res[n],n=='S_pooled',0)[0] for n in blocks);exploratory=all(eligible(baseline[n][1],res[n],n=='S_pooled',.002)[0] for n in blocks)
   tr=dict(candidate=cid,score_id=key,threshold=tau,**res['S_pooled'],eligible=not fail and m['active'],strict_eligible=strict and m['active'],exploratory_002_only=exploratory and not (not fail),failures=';'.join(fail),status='completed' if m['active'] else 'disabled',failure_reason=m.get('reason',''),fit_seconds=diag[key]['fit_seconds'],evaluation_seconds=time.perf_counter()-tt);trials.append(tr)
   print(cid,'TP/FN/FP',tr['TP'],tr['FN'],tr['FP'],'eligible',tr['eligible'],flush=True)
 chosen=pick_res([z for z in trials if z['eligible']]);control=pick_res([z for z in trials if z['score_id']=='R0' and z['status']=='completed'])
 pd.DataFrame(trials).to_csv(R/'all_trials.csv',index=False);pd.DataFrame(records).to_csv(R/'block_metrics.csv',index=False);pd.DataFrame(thresholds).to_csv(R/'thresholds.csv',index=False);pd.concat(pairedrows).to_csv(R/'paired_selection.csv',index=False)
 selected=dict(created=pd.Timestamp.now(tz='UTC').isoformat(),selected=chosen['candidate'] if chosen else None,threshold_control=control['candidate'] if control else None,control_is_eligible=bool(control and control['eligible']),decision='selected subject to V' if chosen else 'no eligible candidate; retain original PCA',configs=configs,selection_metrics=chosen,control_metrics=control,baseline=dict(model='models/baseline.joblib',threshold=base['threshold'],postprocessing='none',fallback='original P0',score='Q/Q95'),preregistration_sha256=sha(R/'preregistration.json'),implementation_sha256=sha(R/'implementation_lock.json'),models={f.name:sha(f) for f in (R/'models').glob('*.joblib')},selection_row_ids_sha256={k:sha(R/'manifests'/f'rows_{k}.csv') for k in ['S1','S2','V']},V_scores_opened=False,old_test_new_candidates_evaluated=False,seconds=time.perf_counter()-t)
 js(R/'selected_config.json',selected);(R/'selected_config.sha256').write_text(sha(R/'selected_config.json')+'\n');js(R/'checkpoint.json',dict(stage='selection_frozen',selected=selected['selected'],V_opened=False));print('FROZEN',selected['selected'],'control',selected['threshold_control'],flush=True)

def run_evaluation():
 d,e,base,p=context();lock=json.loads((R/'selected_config.json').read_text());assert sha(R/'selected_config.json')==(R/'selected_config.sha256').read_text().strip();assert not (R/'historical_evaluation.json').exists()
 targets=list(dict.fromkeys(x for x in [lock['selected'],lock['threshold_control']] if x is not None));assert len(targets)<=2;allrecords=[];pairrows=[];diagnosis={};episode=[]
 for name,ids in [('V',np.asarray(p['blocks']['V'])),('historical',d.index[d.split=='test'].to_numpy())]:
  b,bm=evaluated(e.predict(base,ids),base['threshold']);b['integrated_score']=b.score/base['threshold'];b.to_csv(R/'predictions'/f'baseline_{name}.csv',index=False);allrecords.append(dict(candidate='baseline',block=name,**bm,new_TP=0,new_FP=0,lost_TP=0,removed_FP=0));ep=alarm_summary(b,b.prediction);ep['candidate']='baseline';ep['block']=name;episode.append(ep)
  for cid in targets:
   cfg=lock['configs'][cid];m=joblib.load(R/cfg['model']);s=aux_score(m,e,base,ids,b.score);out,cm=combine(b,s,cfg['threshold'],cfg['score'],base['threshold']);out['candidate']=cid;out.to_csv(R/'predictions'/f'{cid}_{name}.csv',index=False);ok,why=eligible(bm,cm,True);state='pass' if ok else 'improvement_unverifiable' if bm['FN']==0 or bm['TP']+bm['FN']==0 else 'fail';allrecords.append(dict(candidate=cid,block=name,**cm,allowed_additional_FP=allowed_fp(bm['TN']+bm['FP'],.001),passes=ok,result=state,failed_rules=';'.join(why)));diagnosis[f'{name}:{cid}']=dict(state=state,failed_rules=why);z=out[out.prediction!=out.base_prediction].copy();z['block']=name;z['change']=np.where(z.label==1,'new_TP','new_FP');pairrows.append(z);ep=alarm_summary(out,out.prediction);ep['candidate']=cid;ep['block']=name;episode.append(ep)
  if name=='V':js(R/'V_evaluation.json',dict(evaluated=pd.Timestamp.now(tz='UTC').isoformat(),selected=lock['selected'],results=diagnosis,selection_hash=sha(R/'selected_config.json'),reselection=False))
 pd.DataFrame(allrecords).to_csv(R/'baseline_vs_candidates.csv',index=False);pd.concat(pairrows).to_csv(R/'paired_evaluation.csv',index=False);pd.concat(episode).to_csv(R/'alarm_episodes.csv',index=False);js(R/'historical_evaluation.json',dict(evaluated=pd.Timestamp.now(tz='UTC').isoformat(),candidates=['baseline']+targets,single_restricted_batch=True,results=diagnosis,reselection=False));js(R/'checkpoint.json',dict(stage='historical_evaluated',selected=lock['selected'],retain_original=lock['selected'] is None or diagnosis.get('V:'+str(lock['selected']),{}).get('state')!='pass'));print(pd.DataFrame(allrecords)[['candidate','block','TP','FN','FP','F1','F2']].to_string(index=False))

if __name__=='__main__':
 if sys.argv[1]=='select':run_selection()
 elif sys.argv[1]=='evaluate':run_evaluation()
 else:raise ValueError(sys.argv[1])

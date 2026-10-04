from common import *
lock=json.loads((R/'frozen_selection.json').read_text());proto=json.loads((R/'protocol.json').read_text());d=loadrows(R/'inputs/frozen_rows.csv');e=Engine(d);results=[];pairrows=[];ep=[];rowmaps={};blockrows=[];blocks=json.loads((R/'manifests/diagnostic_blocks.json').read_text())
for name,h in lock['code_sha256'].items():assert sha(R/'src'/name)==h
for config in lock['final_configs']:assert sha(R/config['model_path'])==config['model_sha256']
for stage,split in [('selection','selection'),('historical','test')]:
 ids=d.index[d.split==split].to_numpy();base_raw=e.predict(joblib.load(R/'models/baseline.joblib'),ids);br,bm=evaluated(base_raw,lock['baseline_threshold']);common=ids[e.starts_for(20)[ids]>=0];d.loc[common,['row_id','label']].to_csv(R/'manifests'/f'common_W20_{stage}.csv',index=False)
 for cfg in lock['final_configs']:
  tic=time.perf_counter();r,m=evaluated(e.predict(joblib.load(R/cfg['model_path']),ids),cfg['threshold']);r['candidate']=cfg['candidate'];r['evaluation']=stage;r.to_csv(R/'predictions'/f'{stage}_frozen_{cfg["candidate"]}.csv',index=False);r[r.error!='correct'].to_csv(R/'predictions'/f'{stage}_errors_{cfg["candidate"]}.csv',index=False);rowmaps[stage,cfg['candidate']]=r
  row=dict(evaluation=stage,candidate=cfg['candidate'],system=cfg['system'],target=cfg['target'],threshold=cfg['threshold'],scope='operational',normal=int((r.label==0).sum()),anomaly=int((r.label==1).sum()),inference_seconds=time.perf_counter()-tic,**m,**paired(br,r))
  for rule,delta in proto['criteria'].items():row.update({rule+'_'+k:v for k,v in compare(bm,m,delta).items()})
  results.append(row)
  for scope,mask in [('common_W20',r.native_W20),('P0_route',r.fallback),('short_route',r.short_complement),('W20_route',r.route==20)]:
   q=r[mask]
   if len(q):results.append(dict(evaluation=stage,candidate=cfg['candidate'],scope=scope,normal=int((q.label==0).sum()),anomaly=int((q.label==1).sum()),**metrics(q.label,q.score,q.prediction)))
  a=alarm_summary(r,r.prediction);a['candidate']=cfg['candidate'];a['evaluation']=stage;ep.append(a)
  out=r.copy();out['baseline_prediction']=br.prediction;out['change']=np.select([(out.label==1)&(out.baseline_prediction==0)&(out.prediction==1),(out.label==1)&(out.baseline_prediction==1)&(out.prediction==0),(out.label==0)&(out.baseline_prediction==1)&(out.prediction==0),(out.label==0)&(out.baseline_prediction==0)&(out.prediction==1)],['new_TP','lost_TP','removed_FP','new_FP'],default='unchanged');pairrows.append(out)
  if stage=='selection' and cfg['candidate']=='baseline':
   for block,idx in blocks.items():
    q=r.loc[idx];mm=metrics(q.label,q.score,q.prediction);n=int((q.label==0).sum());rec=dict(candidate='baseline',stage='selection',block=block,normal=n,anomaly=int((q.label==1).sum()),normal_bursts=q[q.label==0].burst_id.nunique(),anomaly_bursts=q[q.label==1].burst_id.nunique(),baseline_Recall=mm['Recall'],baseline_FPR=mm['FPR'],baseline_FP=mm['FP'],**mm)
    for rule,delta in proto['criteria'].items():rec[rule+'_allowed_extra_FP']=allowed_fp(n,delta);rec[rule+'_FP_limit_met']=True
    blockrows.append(rec)
pd.DataFrame(results).to_csv(R/'baseline_vs_candidates.csv',index=False);pd.concat(pairrows).to_csv(R/'paired_errors.csv',index=False);pd.concat(ep).to_csv(R/'tables/alarm_episodes.csv',index=False);pd.concat([pd.read_csv(R/'block_metrics.csv'),pd.DataFrame(blockrows)]).to_csv(R/'block_metrics.csv',index=False)
js(R/'checkpoint.json',dict(stage='historical_evaluation_complete',unique_configs=len(lock['final_configs']),selection_lock_sha256=sha(R/'frozen_selection.json'),retuned=False));print(pd.DataFrame(results).query("scope=='operational'")[['evaluation','candidate','TP','FN','FP','TN','F1','F2','primary_eligible']].to_string(index=False))

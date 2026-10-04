from core import *
R=Path(__file__).resolve().parents[1];d=loadrows(R/'manifests/review_rows.csv');e=Engine(d,R);blocks=json.loads((R/'manifests/blocks.json').read_text());lock=json.loads((R/'frozen_selection.json').read_text());assert (R/'validation.json').exists();selected=lock['selected'];results=[];frames={};seeds=[];episodes=[];pairs=[]
for file,h in lock['code_sha256'].items():assert sha(R/'src'/file)==h,('Code changed since selection',file)
for name,m in lock['systems'].items():assert sha(R/m['path'])==m['sha256']
# Fix exactly which historical evaluations will be run before looking at any new historical score.
configs=[('baseline', 'P1_Q',.01,'none'),('baseline_threshold_0001','P1_Q',.001,'none'),('baseline_threshold_0005','P1_Q',.005,'none'),('selected',selected['candidate'],selected['target'],'none'),('selected_consecutive2',selected['candidate'],selected['target'],'consecutive2'),('selected_two_of_three',selected['candidate'],selected['target'],'two_of_three'),('IF_same_W20','IF_P1',.01,'none')]
if lock['if_selected']: configs.append(('IF_selected',lock['if_selected']['candidate'],lock['if_selected']['target'],'none'))
js(R/'evaluation_lock.json',dict(time=pd.Timestamp.now(tz='UTC').isoformat(),selection_sha256=sha(R/'frozen_selection.json'),configs=configs,code_sha256=sha(__file__),seed_confirmation=[45,46],new_model_selection=False))
for block in ['outer_development','historical']:
 for alias,name,target,policy in configs:
  sys=joblib.load(R/'models'/f'final_{name}.joblib');tic=time.perf_counter();raw=e.predict(sys,blocks[block]['eval']);r,m=evaluate(raw,sys['thresholds'][str(target)],policy);seconds=time.perf_counter()-tic;r['evaluation']=block;r['candidate']=name;r.to_csv(R/'predictions'/f'{block}_{alias}.csv',index=False);r[r.error!='correct'].to_csv(R/'predictions'/f'{block}_{alias}_errors.csv',index=False);frames[block,alias]=r
  results.append(dict(evaluation=block,alias=alias,candidate=name,target=target,policy=policy,scope='operational',inference_seconds=seconds,**m))
  common=set(pd.read_csv(R/'manifests'/f'common_{block}.csv').row_id);rc=r[r.row_id.isin(common)];results.append(dict(evaluation=block,alias=alias,candidate=name,target=target,policy=policy,scope='common_operational_threshold',**metrics(rc.label,rc.score,rc.prediction)))
  for route,g in r.groupby('route'):results.append(dict(evaluation=block,alias=alias,candidate=name,target=target,policy=policy,scope=f'route_{route}',**metrics(g.label,g.score,g.prediction)))
  ep=alarm_summary(r,r.prediction);ep['alias']=alias;ep['evaluation']=block;episodes.append(ep)
 base=frames[block,'baseline'];new=frames[block,'selected'];assert base.row_id.tolist()==new.row_id.tolist()
 for comparison,bb in [('controlled_baseline',base)]+([('original_paper_baseline',loadrows(R/'predictions/original_retrained.csv'))] if block=='historical' else []):
  assert bb.row_id.tolist()==new.row_id.tolist();out=new[['row_id','source_file','source_row','TimeStamp']+SENSORS+['label','burst_id','burst_pos','burst_length','route','fallback','elapsed_seconds']].copy();out['baseline_prediction']=bb.prediction.to_numpy();out['selected_prediction']=new.prediction.to_numpy();out['comparison']=comparison;out['evaluation']=block;out['change']=np.select([(out.label==1)&(out.baseline_prediction==0)&(out.selected_prediction==1),(out.label==1)&(out.baseline_prediction==1)&(out.selected_prediction==0),(out.label==0)&(out.baseline_prediction==1)&(out.selected_prediction==0),(out.label==0)&(out.baseline_prediction==0)&(out.selected_prediction==1)],['new_TP','lost_TP','removed_FP','new_FP'],default='unchanged');pairs.append(out)
 # All-normal/all-anomaly controls use same full denominator.
 for p in [0,1]:results.append(dict(evaluation=block,alias=f'constant_{p}',scope='operational',**metrics(base.label,np.full(len(base),p),np.full(len(base),p))))
# IF seed confirmation: fixed spec/threshold policy, individual calibration, no best-seed selection.
ifspec=next(s for s in json.loads((R/'protocol.json').read_text())['candidate_specs'] if s['id']=='IF_P1')
for seed in [42,43,44,45,46]:
 sys=e.fit_system(ifspec,blocks['historical']['cal'],seed=seed);joblib.dump(sys,R/'models'/f'IF_confirmation_{seed}.joblib',compress=3)
 for block in ['inner_early','inner_late','outer_development','historical']:
  # Early block must recalibrate only on early normal data.
  bs=e.fit_system(ifspec,blocks[block]['cal'],seed=seed) if block=='inner_early' else sys;r=e.predict(bs,blocks[block]['eval'])
  for target in [.001,.005,.01]:
   z,m=evaluate(r,bs['thresholds'][str(target)]);seeds.append(dict(seed=seed,evaluation=block,target=target,**m))
  if block=='historical':r.to_csv(R/'predictions'/f'historical_IF_seed{seed}_scores.csv',index=False)
pd.DataFrame(seeds).to_csv(R/'tables/if_seed_confirmation.csv',index=False);pd.DataFrame(seeds).groupby(['evaluation','target'])[['Recall','Precision','F1','F2','FPR','AP']].agg(['mean','std']).to_csv(R/'tables/if_five_seed_summary.csv')
pd.DataFrame(results).to_csv(R/'baseline_vs_improved.csv',index=False);pd.concat(pairs).to_csv(R/'paired_errors.csv',index=False);pd.concat(episodes).to_csv(R/'tables/alarm_episodes.csv',index=False)
# Add original paper numbers as a separate provenance row, not as new selection evidence.
r=loadrows(R/'predictions/original_retrained.csv');rows=[dict(evaluation='historical',alias='original_paper',scope='operational',**metrics(r.label,r.score,r.prediction))];pd.concat([pd.DataFrame(results),pd.DataFrame(rows)]).to_csv(R/'baseline_vs_improved.csv',index=False)
js(R/'checkpoint.json',dict(stage='final_evaluation_complete',selection_unchanged=sha(R/'frozen_selection.json')==json.loads((R/'evaluation_lock.json').read_text())['selection_sha256'],new_test_selection=False));print(pd.DataFrame(results).query("scope=='operational'")[['evaluation','alias','TP','FN','FP','TN','Recall','FPR','AP']].to_string(index=False))

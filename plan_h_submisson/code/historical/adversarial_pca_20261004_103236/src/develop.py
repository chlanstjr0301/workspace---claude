from core import *
import traceback
R=Path(__file__).resolve().parents[1];d=loadrows(R/'manifests/review_rows.csv');proto=json.loads((R/'protocol.json').read_text());blocks=json.loads((R/'manifests/blocks.json').read_text());e=Engine(d,R);trials=[];coverage=[];seedrows=[];store={};failed=[];start=time.monotonic()
# Evaluation intersection fixed before any score is computed.
for block in proto['selection_blocks']+['outer_development','historical']:
 ev=np.array(blocks[block]['eval']);common=ev[e.window(20,True)[ev]>=0];d.loc[common,['row_id','label','source_file','burst_id']].to_csv(R/'manifests'/f'common_{block}.csv',index=False)
js(R/'manifests/common_lock.json',dict(time=pd.Timestamp.now(tz='UTC').isoformat(),scores_seen=False,files={p.name:sha(p) for p in (R/'manifests').glob('common_*.csv')}))
# Training-only numerical diagnostics.
ids=np.flatnonzero((d.split=='train')&(e.window(20,False)>=0));a=e.raw[e.window(20,False)[ids,None]+np.arange(20)];x,names=stat_features(a,'full');z=StandardScaler().fit_transform(x)
sv=np.linalg.svd(z-z.mean(0),compute_uv=False);js(R/'tables/feature_redundancy.json',dict(identity='RMS^2 = mean^2 + population_std^2',max_identity_error=float(np.max(abs((a*a).mean(1)-a.mean(1)**2-a.std(1)**2))),singular_values=sv,condition=float(sv[0]/sv[-1]),note='Nonlinear deterministic redundancy does not imply exact linear rank deficiency; no outlier clipping'))
pd.DataFrame(np.corrcoef(z,rowvar=False),index=names,columns=names).to_csv(R/'tables/train_feature_correlations.csv')
for spec in proto['candidate_specs']:
 tic=time.monotonic()
 try:
  preds=[];per=[]
  for block in proto['selection_blocks']:
   b=blocks[block];system=e.fit_system(spec,b['cal']);joblib.dump(system,R/'models'/f'system_{spec["id"]}_{block}.joblib',compress=3);r=e.predict(system,b['eval']);r['block']=block
   r.to_csv(R/'predictions'/f'dev_{spec["id"]}_{block}_scores.csv',index=False)
   for target in proto['targets']:
    pred,m=evaluate(r,system['thresholds'][str(target)]);pred['block']=block;pred['target']=target;preds.append(pred)
    trials.append(dict(candidate=spec['id'],kind=spec['kind'],block=block,target=target,scope='operational',status='ok',threshold=system['thresholds'][str(target)],calibration_fpr=system['calibration_fpr'][str(target)],**m))
    ci=set(pd.read_csv(R/'manifests'/f'common_{block}.csv').row_id);rc=pred[pred.row_id.isin(ci)];trials.append(dict(candidate=spec['id'],kind=spec['kind'],block=block,target=target,scope='common_operational_threshold',status='ok',**metrics(rc.label,rc.score,rc.prediction)))
   for (label,route),g in r.groupby(['label','route']):coverage.append(dict(candidate=spec['id'],block=block,label=label,route=route,n=len(g),total_class=int((r.label==label).sum()),native_available=int(g.native_available.sum()),fallback_rows=int(g.fallback.sum()),elapsed_max=float(g.elapsed_seconds.max())))
   if spec['kind']=='IF':
    for seed,member in zip([42,43,44],system['members']):
     one=dict(system,members=[member]);cs=e.predict(one,b['cal']);sr=e.predict(one,b['eval'])
     for target in proto['targets']:
      h=threshold(cs.score,target);_,m=evaluate(sr,h);seedrows.append(dict(candidate=spec['id'],block=block,seed=seed,target=target,**m))
  pooled=pd.concat(preds);store[spec['id']]=pooled
  pooled.to_csv(R/'predictions'/f'dev_{spec["id"]}_all_targets.csv',index=False)
  for target,g in pooled.groupby('target'):
   vals=[x for x in trials if x['candidate']==spec['id'] and x['block'] in proto['selection_blocks'] and x['scope']=='operational' and x['target']==target]
   trials.append(dict(candidate=spec['id'],kind=spec['kind'],block='pooled_inner',target=target,scope='operational',status='ok',min_block_Recall=min(v['Recall'] for v in vals),max_block_FPR=max(v['FPR'] for v in vals),all_blocks_meet=all(v['FPR']<=.01 for v in vals),**metrics(g.label,g.score,g.prediction)))
  print('DONE',spec['id'],round(time.monotonic()-tic,2),'seconds',flush=True)
 except Exception as exc:
  failed.append(dict(candidate=spec['id'],error=repr(exc),traceback=traceback.format_exc()));trials.append(dict(candidate=spec['id'],status='failed',reason=repr(exc)));print('FAILED',spec['id'],repr(exc),flush=True)
 pd.DataFrame(trials).to_csv(R/'all_trials.csv',index=False);js(R/'logs/failures.json',failed);js(R/'checkpoint.json',dict(stage='development',completed_candidates=len(store),failed=len(failed),wall_seconds=time.monotonic()-start))
t=pd.DataFrame(trials);p=t[(t.block=='pooled_inner')&(t.scope=='operational')&(t.status=='ok')];good=p[(p.kind=='PCA')&(p.FPR<=.01)].sort_values(['Recall','F2','F1','AP','candidate','target'],ascending=[False,False,False,False,True,True]);status='eligible' if len(good) else 'TARGET_UNMET'
if not len(good):raise RuntimeError('No FPR eligible PCA; protocol requires target unmet, no relaxation')
selected=good.iloc[0].to_dict();g=p[(p.kind=='IF')&(p.FPR<=.01)].sort_values(['Recall','F2','F1','AP','candidate','target'],ascending=[False,False,False,False,True,True]);if_selected=g.iloc[0].to_dict() if len(g) else None
# Two additional policies are descriptive and cannot replace selected raw detector.
post=[]
for policy in ['consecutive2','two_of_three']:
 q=store[selected['candidate']];q=q[q.target==selected['target']].copy();parts=[]
 for block,z in q.groupby('block',sort=False):
  r,m=evaluate(z,float(z.threshold.iloc[0]),policy);parts.append(r);post.append(dict(candidate=selected['candidate'],target=selected['target'],block=block,policy=policy,**m))
 r=pd.concat(parts);post.append(dict(candidate=selected['candidate'],target=selected['target'],block='pooled_inner',policy=policy,**metrics(r.label,r.score,r.prediction)));r.to_csv(R/'predictions'/f'dev_selected_{policy}.csv',index=False)
pd.DataFrame(post).to_csv(R/'tables/postprocessing_development.csv',index=False);pd.DataFrame(coverage).to_csv(R/'tables/route_coverage_development.csv',index=False);pd.DataFrame(seedrows).to_csv(R/'tables/if_seeds_development.csv',index=False)
pd.DataFrame(seedrows).groupby(['candidate','block','target'])[['Recall','FPR','F1','AP']].agg(['mean','std']).to_csv(R/'tables/if_seed_summary_development.csv')
# Fit final normal calibration systems before freezing; no outer or historical score used.
final_systems={}
selected_spec=next(s for s in proto['candidate_specs'] if s['id']==selected['candidate'])
final_specs={s['id']:s for s in proto['candidate_specs'] if s['id'] in ['P1_Q','IF_P1',selected['candidate']]+([if_selected['candidate']] if if_selected else [])}
for name,spec in final_specs.items():
 sys=e.fit_system(spec,blocks['historical']['cal']);joblib.dump(sys,R/'models'/f'final_{name}.joblib',compress=3);final_systems[name]=dict(thresholds=sys['thresholds'],calibration_fpr=sys['calibration_fpr'],path=f'models/final_{name}.joblib',sha256=sha(R/'models'/f'final_{name}.joblib'))
locked=dict(time=pd.Timestamp.now(tz='UTC').isoformat(),status=status,selected=selected,if_selected=if_selected,primary_policy='none',systems=final_systems,protocol_sha256=sha(R/'protocol.json'),code_sha256={q.name:sha(q) for q in (R/'src').glob('*.py')},postprocessing='descriptive only; two predeclared policies, no replacement',new_operating_configurations=56,selection_used_only=proto['selection_blocks'],outer_and_historical_seen_for_new_selection=False,wall_seconds=time.monotonic()-start)
js(R/'frozen_selection.json',locked);print('LOCKED',json.dumps(selected),flush=True)

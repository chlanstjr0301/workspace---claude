"""Parse existing records only. No model imports, fitting, inference or metric recomputation."""
from pathlib import Path
import json,hashlib,csv,re,zipfile
import pandas as pd
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
L=ROOT/'runs/pca_literature_rescue_20261004_210504';F=ROOT/'runs/pca_followup_v2_20261004_214810';G=ROOT/'runs/pca_g1_frozen_validation_20261004_221914';A=ROOT/'runs/data_usage_audit_20261004_225053'
FR=next((F/'replays').iterdir());GR=Path(json.loads((G/'portable_replay.json').read_text())['replay_root']);LR=sorted((L/'replays').iterdir())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str))
def rel(p):
 p=Path(p);return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
E=[];hashes={}
def evidence(paths):
 for p in paths:
  p=Path(p)
  if p.exists() and p.is_file():hashes[rel(p)]=sha(p)
 return ';'.join(rel(p) for p in paths)
def event(id,round,category,stage,status,paths,**kw):
 E.append(dict(event_id=id,round=round,category=category,stage=stage,status=status,confirmation='minimum_confirmed_by_records',evidence=evidence(paths),**kw))
# Main logical rounds versus their actually executed replays.
for runid,path,kind,mainround in [('original',ROOT/'runs/pca_20261003_cpu','development','original'),('adversarial',ROOT/'runs/adversarial_pca_20261004_103236','development','adversarial'),('tradeoff',ROOT/'runs/pca_fn_fp_tradeoff_20261004_201516','development','tradeoff'),('literature',L,'development','literature'),('g1_v2',F,'development_selection','g1_v2'),('g1_frozen',G,'frozen_validation','g1_frozen')]:
 paths=[p for p in [path/'all_trials.csv',path/'selected_config.json',path/'frozen_selection.json',path/'selection_lock.json',path/'validation_protocol.json'] if p.exists()]
 event('round:'+runid,mainround,'logical_round',kind,'completed',paths,counts_as_G1_development=runid=='g1_v2',counts_as_G1_frozen=runid=='g1_frozen')
replays=[('adversarial',ROOT/'runs/adversarial_pca_20261004_195637_rerun_292928',ROOT/'runs/adversarial_pca_20261004_103236/logs/end_to_end_replay.log'),('tradeoff',ROOT/'runs/pca_fn_fp_tradeoff_20261004_202734_rerun_294980',ROOT/'runs/pca_fn_fp_tradeoff_20261004_201516/logs/end_to_end_replay.log')]+[('literature',x,L/'logs'/('end_to_end_replay.log' if i==0 else 'final_replay.log')) for i,x in enumerate(LR)]+[('g1_v2',FR,F/'logs/end_to_end_replay.log'),('g1_frozen',GR,G/'logs/portable_replay.log')]
for i,(family,path,log) in enumerate(replays):event('replay:'+str(i),family,'whole_replay','whole_pipeline','completed',[log,path/'end_to_end_replay.json' if (path/'end_to_end_replay.json').exists() else path/'portable_replay.json'],replay_root=rel(path),new_unique_candidate=False)
# Per-script stages with model computation. Nested online child invocations are separate entries, never summed with parents.
for n,path in [('original',F),('replay',FR)]:
 for stage,log,contains in [('diagnose','diagnose.log',False),('select','select.log' if n=='original' else 'experiment_v2_select.log',True),('evaluate','evaluate.log' if n=='original' else 'experiment_v2_evaluate.log',True),('validate','validate.log' if n=='original' else 'validate_v2.log',True)]:
  script={'diagnose':'diagnose.py','select':'experiment_v2.py','evaluate':'experiment_v2.py','validate':'validate_v2.py'}[stage]
  event('v2:'+n+':'+stage,'g1_v2','inference_parent_stage',stage,'completed',[path/'logs'/log,path/'src'/script],selected_G1_computed=contains,mode='script process; internal calls not counted individually')
 # analyze recalculates metrics from stored CSVs, with no model computation.
 event('v2:'+n+':analyze','g1_v2','stored_prediction_reaggregation','analyze','completed',[path/'logs'/('analyze.log' if n=='original' else 'analyze.log'),path/'src/analyze.py'],mode='metrics/PR from stored scores')
 event('v2:'+n+':online','g1_v2','online_batch_workflow','validate','completed',[path/'validation_report.json',path/'src/validate_v2.py'],parent_event='v2:'+n+':validate',dataset_comparisons=3,child_online_processes=6,child_note='2 source files each for S, full selection including V, H; overlaps explicit')
for n,path in [('original',G),('replay',GR)]:
 for stage,log,script in [('review','review.log','review.py'),('validate','validate.log','validate_frozen.py')]:event('frozen:'+n+':'+stage,'g1_frozen','inference_parent_stage',stage,'completed',[path/'logs'/log,path/'src'/script],selected_G1_computed=True)
 event('frozen:'+n+':report','g1_frozen','stored_prediction_reaggregation','make_report','completed',[path/'logs/report.log',path/'src/make_report.py'],mode='PR curve recomputed from saved continuous scores; most metrics re-cited')
 event('frozen:'+n+':online','g1_frozen','online_batch_workflow','validate','completed',[path/'validation_report.json',path/'src/validate_frozen.py'],parent_event='frozen:'+n+':validate',dataset_comparisons=2,child_online_processes=1,child_note='external API H replay in child, separate from 2 parent online/batch comparisons')
 # Auxiliary-only calibration reconfirmation is not a selected-G1 inference experiment.
 event('frozen:'+n+':audit_data','g1_frozen','B0_only_calibration_score_recalculation','audit_data','completed',[path/'logs/data_audit.log',path/'src/audit_data.py'],selected_G1_computed=False)
event('frozen:first_failed_review','g1_frozen','inference_parent_stage','review','failed_after_predictions',[G/'logs/attempt1_row_order_failure.log',G/'src/review.py',R/'session_commands.json'],selected_G1_computed=True,reason='S1/S2 concatenation order mismatch; scores/predictions written before row alignment assertion, retry sorted row IDs',session_line=1780)
# Report invocations: execution logs, not number of HTML/MD/ZIP files.
for id,path,log in [('v2:first',F,'report.log'),('v2:final',F,'final_report.log'),('v2:replay',FR,'report_v2.log'),('frozen:first',G,'report.log'),('frozen:replay',GR,'report.log')]:event('report:'+id,'g1_v2' if id.startswith('v2') else 'g1_frozen','report_generation_or_recitation','report','completed',[path/'logs'/log],may_overlap_reaggregation=id.startswith('frozen'))
# Training evidence. Minimum counts are separate from model/configuration uniqueness.
training=[]
def train_record(stage,family,n,kind,paths,detail):
 training.append(dict(stage=stage,model_family=family,minimum_fit_pipelines=n,kind=kind,confirmation='lower_bound_not_total',evidence=evidence(paths),detail=detail))
orig=ROOT/'runs/pca_20261003_cpu';logs=[]
for p in (orig/'logs').glob('fit_*.json'):
 j=json.loads(p.read_text())
 if j['kind']=='PCA' and (orig/'models'/(j['name']+'.joblib')).exists() and (orig/'manifests'/('train_'+j['name']+'.csv')).exists():logs.append(p)
train_record('original PCA','PCA',len(logs),'normal_detector_fit',logs+[orig/'source_final/experiment.py',orig/'source_final/models.py'],'each fit JSON is written after fit, matching model and training row manifest; full-spectrum diagnostic and retained PCA are two sklearn.PCA.fit calls per pipeline')
for path in [ROOT/'runs/adversarial_pca_20261004_103236',ROOT/'runs/adversarial_pca_20261004_195637_rerun_292928']:
 inv=json.loads((path/'manifests/model_inventory.json').read_text());pc=[x for x in inv if x['kind']=='PCA'];assert all(x['fit_seconds']>0 and x['n_train']>0 for x in pc)
 train_record(path.name,'PCA',len(pc),'distinct_cached_detector_minimum',[path/'manifests/model_inventory.json',path/'src/core.py',path/'logs'/('development.log' if path.name.endswith('103236') else 'develop.log')],'not file count alone: completed development log + stored fit time/training counts + core.fit no disk reload and fit-before-write. Repeated fits under same keys are not fully logged; minimum13 per run')
 train_record(path.name+':controlled','PCA',1,'validation_control_refit',[path/'src/validate.py',path/'logs'/('validation.log' if path.name.endswith('103236') else 'validate.log'),path/'models/controlled_pca_commonfit.joblib'],'explicit normal-only controlled fit_pca call in successful validator, absent from core model_inventory; no double count')
for path in [ROOT/'runs/pca_fn_fp_tradeoff_20261004_201516',ROOT/'runs/pca_fn_fp_tradeoff_20261004_202734_rerun_294980']:
 for file,script in [('refit_rows.csv','stage_a.py'),('new_model_fit_rows.csv','stage_b.py')]:
  m=pd.read_csv(path/'manifests'/file);n=m.model.nunique();train_record(path.name+':'+script,'PCA',int(n),'normal_refit_or_short_window_fit',[path/'manifests'/file,path/'src'/script,path/'logs'/script.replace('.py','.log')],'completed stage loop and written fit row manifest;2 fits per stage, not36 threshold configurations')
for path in [L]+LR:
 j=json.loads((path/'model_diagnostics.json').read_text());assert j['R5']['status']=='completed'
 train_record(rel(path),'R5',1,'normal_lag2_ridge_LW_fit',[path/'model_diagnostics.json',path/'src/experiment.py',path/'src/auxiliary.py',path/'logs'/('select.log' if path==L else 'experiment_select.log')],'fit_all invoked once per proven select execution; replay recreates auxiliaries despite same model bytes')
for path in [F,FR]:train_record(rel(path)+':validate','R5',1,'validation_only_parameter_reconstruction',[path/'src/validate_v2.py',path/'validation_report.json',path/'logs'/('validate.log' if path==F else 'validate_v2.log')],'line74 fits StandardScaler, solves ridge coefficient and fits LW covariance anew; not a new candidate and not persisted replacement')
pd.DataFrame(training).to_csv(R/'training_execution_minima.csv',index=False)
# Exact fixed model identities: semantic config != prediction equality.
basehash=sha(F/'models/baseline.joblib');r5hash=sha(F/'models/R5.joblib');datahash=sha(F/'inputs/frozen_rows.csv');corehash=sha(F/'src/v2.py');identities=[]
for path in [L]+LR+[F,FR,G,GR]:
 identities.append(dict(path=rel(path),B0_sha256=sha(path/'models/baseline.joblib'),R5_sha256=sha(path/'models/R5.joblib'),data_sha256=sha(path/'inputs/frozen_rows.csv'),v2_core_sha256=sha(path/'src/v2.py') if (path/'src/v2.py').exists() else None))
assert all(x['B0_sha256']==basehash and x['R5_sha256']==r5hash and x['data_sha256']==datahash for x in identities)
pd.DataFrame(identities).to_csv(R/'identity_chain.csv',index=False)
configs=[]
def cfg(cid,rule,theta,origin,category,path):
 identity=dict(B0=basehash,R5=r5hash,data=datahash,inference_core=corehash,rule=rule,theta=float(theta) if theta is not None else None)
 identityhash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();configs.append(dict(candidate=cid,rule=rule,theta=theta,origin=origin,category=category,identity_sha256=identityhash,evidence=evidence([path]),**{k:v for k,v in identity.items() if k not in ['theta','rule']}));return identityhash
ftr=pd.read_csv(F/'all_trials.csv');p=json.loads((F/'protocol_v2.json').read_text())
for x in p['candidates']:cfg(x['id'],x['rule'],x['theta'],'v2','selected_G1' if x['id']=='G1_Q999' else 'G1_family_development' if x['rule']=='G1' else 'non_G1_control',F/'protocol_v2.json')
gp=json.loads((G/'validation_protocol.json').read_text())
for x in gp['controls']:cfg(x['id'],x['rule'],x['theta'],'frozen','G1_control_previous_threshold' if x['id']=='D_old_threshold_G1' else 'selected_G1' if x['id']=='G1' else 'non_G1_control',G/'validation_protocol.json')
configs=pd.DataFrame(configs);configs.to_csv(R/'candidate_identity.csv',index=False)
# Prediction bytes and normalized semantic fingerprints read only: no scoring/metrics calculations.
proof=[]
for block,ff,gf in [('S','G1_Q999_S_pooled.csv','S_G1.csv'),('V','G1_Q999_V.csv','V_G1.csv'),('H','G1_Q999_H.csv','H_G1.csv')]:
 a=pd.read_csv(F/'predictions'/ff);b=pd.read_csv(G/'predictions'/gf)
 def fingerprint(x,cols):return hashlib.sha256(x[cols].sort_values('row_id').to_csv(index=False,float_format='%.12g').encode()).hexdigest()
 assert fingerprint(a,['row_id','label','prediction'])==fingerprint(b,['row_id','label','prediction'])
 proof.append(dict(block=block,source_file=rel(F/'predictions'/ff),copied_or_replayed_file=rel(G/'predictions'/gf),source_file_sha256=sha(F/'predictions'/ff),new_file_sha256=sha(G/'predictions'/gf),normalized_row_label_prediction_fingerprint=fingerprint(a,['row_id','label','prediction']),same_normalized_decisions=True,interpretation='same predictions, separate logs prove re-executions; wholeCSV hash may differ due metadata/serialization'))
js(R/'prediction_identity.json',proof)
# Data-use records and row-set overlaps, from saved artifacts only.
d=pd.read_csv(F/'inputs/frozen_rows.csv');lookup=d.set_index('row_id');bmap=p['blocks'];ds=[]
sets={'T':set(d[d.split=='train'].row_id),'C_normal':set(d[(d.split=='calibration')&(d.label==0)].row_id),'C_anomaly_original_sigmoid':set(d[(d.split=='calibration')&(d.label==1)].row_id),'S1':set(d.loc[bmap['S1'],'row_id']),'S2':set(d.loc[bmap['S2'],'row_id']),'V':set(d.loc[bmap['V'],'row_id']),'H':set(d[d.split=='test'].row_id)};sets['S']=sets['S1']|sets['S2'];assert not sets['S1']&sets['S2'];assert sets['S']|sets['V']==set(d[d.split=='selection'].row_id)
for part,ids in sets.items():
 q=lookup.loc[sorted(ids)];q[['source_file','source_row','TimeStamp','label']].to_csv(R/('rows_'+part+'.csv'))
 ds.append(dict(scope='canonical_original_rows',round='all_linked_rounds',partition=part,role='normal detector/scaler fit' if part=='T' else 'normal threshold calibration' if part=='C_normal' else 'supervised sigmoid calibration in original PCA; not current normal-only R5 threshold' if part.startswith('C_anomaly') else 'candidate selection + post-hoc analysis' if part in ['S','S1','S2'] else 'reused temporal validation' if part=='V' else 'already exposed historical evaluation',normal=int((q.label==0).sum()),anomaly=int((q.label==1).sum()),row_manifest='rows_'+part+'.csv',row_set_sha256=hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),start=str(q.TimeStamp.min()),end=str(q.TimeStamp.max()),dates=';'.join(sorted(q.TimeStamp.str[:10].unique())),same_original_acquisition=True,usage_status='used_confirmed',evidence=evidence([F/'protocol_v2.json',A/'row_usage_ledger.csv'])))
# Round/partition actual recorded candidate-set incidence; include G1 family and other controls separately.
for path,origin in [(F,'v2_original'),(FR,'v2_replay'),(G,'frozen_original'),(GR,'frozen_replay')]:
 for block in ['S','V','H']:
  names=['B0','C0']+ftr.candidate.tolist() if origin.startswith('v2') and block=='S' else ['B0','G1_Q999'] if origin.startswith('v2') else [x['id'] for x in gp['controls']]
  rowsids=sets[block]
  for cid in names:
   filename=f'{cid}_{"S_pooled" if block=="S" else block}.csv' if origin.startswith('v2') else f'{block}_{cid}.csv';fp=path/'predictions'/filename
   if not fp.exists():continue
   y=pd.read_csv(fp,usecols=['row_id']);assert set(y.row_id)==rowsids
   isfamily=cid.startswith('G1_') or cid in ['G1','D_old_threshold_G1'];ds.append(dict(scope='recorded_candidate_evaluation',round=origin,partition=block,role='selection' if block=='S' and origin.startswith('v2') else 'posthoc_control' if origin.startswith('frozen') else 'locked_temporal_or_historical_evaluation',candidate=cid,G1_family=isfamily,normal=sum(lookup.loc[v,'label']==0 for v in rowsids),anomaly=sum(lookup.loc[v,'label']==1 for v in rowsids),row_manifest='rows_'+block+'.csv',usage_status='used_confirmed',human_viewed='not implied by artifact; see specific conversation excerpts',evidence=evidence([fp])))
# Full original selection includes all current S and V. Prior four development rounds are distinct from G1 development.
prior=[('original',orig),('adversarial',ROOT/'runs/adversarial_pca_20261004_103236'),('tradeoff',ROOT/'runs/pca_fn_fp_tradeoff_20261004_201516'),('literature',L)]
for round,path in prior:
 for block in ['S','V','H']:
  ds.append(dict(scope='preceding_development_round',round=round,partition=block,role='same source rows used before G1, historical labels/calibration/selection scopes vary',usage_status='used_confirmed',G1_family=False,unique_candidates='not globally reconstructed across changing schemas; see prior_candidate_counts.csv',human_viewed='documented only where quoted subsequent user instruction discusses results',evidence=evidence([path/'all_trials.csv' if (path/'all_trials.csv').exists() else path/'tables/metrics_selection.csv',path/'review_report_ko.md' if (path/'review_report_ko.md').exists() else path/'selection_lock.json',A/'experiment_coverage.csv'])))
 # Forest is separate data-use evidence, not included in G1 counts.
 if round in ['original','adversarial']:ds.append(dict(scope='Forest_comparator_only',round=round,partition='original selection and historical test',role='IF comparator on shared data; never G1 candidate/training count',usage_status='used_confirmed',G1_family=False,evidence=evidence([path/'tables/metrics_selection.csv' if round=='original' else path/'all_trials.csv'])))
pd.DataFrame(ds).to_csv(R/'dataset_usage.csv',index=False)
# Counting prior configuration grids without equating CSV rows with candidates.
priorcounts=[]
a=pd.read_csv(ROOT/'runs/adversarial_pca_20261004_103236/all_trials.csv');priorcounts.append(dict(round='adversarial',CSV_rows=len(a),base_specs=a.candidate.nunique(),planned_operating_configs=56,main_operating_configs=len(a[['candidate','target']].drop_duplicates()),additional_postprocess=2,explanation='54 main points × (2 blocks ×2 scopes + pooled operational)=270 rows; +2 postprocessing policies =>56; IF6 main points kept separate from G1'))
t=pd.read_csv(ROOT/'runs/pca_fn_fp_tradeoff_20261004_201516/all_trials.csv');priorcounts.append(dict(round='tradeoff',CSV_rows=len(t),planned_operating_configs=36,main_operating_configs=len(t),explanation='12 fixed-model threshold points +2 short-window options ×12; no G1'))
t=pd.read_csv(L/'all_trials.csv');priorcounts.append(dict(round='literature',CSV_rows=len(t),planned_operating_configs=18,main_operating_configs=len(t),explanation='6 auxiliary scores ×3 calibration budgets; R5_A0/A1/A2 have no G1 confirmation; no selected candidate'))
for block in ['selection','test']:
 t=pd.read_csv(orig/'tables'/f'metrics_{block}.csv');q=t[t.scope=='operational'];pc=q[q.family.str.startswith('P')];iff=q[q.family.str.startswith('I')];priorcounts.append(dict(round='original_'+block,CSV_rows=len(t),main_operating_configs=len(q[['key','target_fpr']].drop_duplicates()),PCA_operating_points=len(pc[['key','threshold']].drop_duplicates()),IF_operating_points=len(iff[['key','threshold']].drop_duplicates()),explanation='key+threshold dedup within operational table; native/common scopes not extra configs; not all ancillary ablations/postprocessing aggregated'))
pd.DataFrame(priorcounts).to_csv(R/'prior_candidate_counts.csv',index=False)
# Implementation checks: repeated pass records and reused historical evidence distinguished.
checks=[]
for name,path in [('v2_original',F),('v2_replay',FR),('frozen_original',G),('frozen_replay',GR)]:
 j=json.loads((path/'validation_report.json').read_text());n=j.get('tests',len(j.get('new_checks',[])));checks.append(dict(run=name,reported_executed_check_items=n,reused_historical_check_items=j.get('reused_historical_checks',0),evidence=evidence([path/'validation_report.json']),independent_experiments=False))
pd.DataFrame(checks).to_csv(R/'implementation_check_counts.csv',index=False)
# ZIP members: not another run; files with matching content linked rather than counted.
zips=[]
for path in (ROOT/'deliverables').glob('*.zip'):
 with zipfile.ZipFile(path) as z:
  matches=[]
  for name in z.namelist():
   if name.endswith(('selected_config.json','model_manifest.json','protocol_v2.json','validation_protocol.json')) and ('pca_' in name):matches.append(dict(member=name,sha256=hashlib.sha256(z.read(name)).hexdigest()))
  zips.append(dict(path=rel(path),sha256=sha(path),members=matches,counts_as_execution=False,reason='archive/copy; execution only counted with distinct logs/commands'))
js(R/'archive_deduplication.json',zips)
pd.DataFrame(E).to_csv(R/'experiment_inventory.csv',index=False)
tr=pd.DataFrame(training);PCA=int(tr.loc[tr.model_family=='PCA','minimum_fit_pipelines'].sum());R5=int(tr.loc[tr.model_family=='R5','minimum_fit_pipelines'].sum());assert R5==5
family=configs[configs.rule=='G1'];assert family.identity_sha256.nunique()==3
counts=dict(scope='saved PCA project records + relevant local session until audit start; no global process ledger',selected_G1_development_selection_rounds=dict(count=1,status='confirmed_in_scope'),selected_G1_frozen_validation_rounds=dict(count=1,status='confirmed_in_scope'),selected_G1_whole_replays=dict(count=2,status='minimum_confirmed'),G1_family_unique_configurations=dict(count=3,status='confirmed_in_observed_artifacts',development=2,posthoc_previous_threshold_control=1,selected=1),v2_candidates=dict(planned=6,computed=6,computation_completed=6,execution_failed=0,status_completed_rows=int((ftr.status=='completed').sum()),status_duplicate_excluded_rows=int((ftr.status=='duplicate_excluded').sum()),structurally_distinct=6,new_relative_to_R5_A1=5,binary_C_and_S_signatures=4,seed_variants=0,unknown_extra_attempts=True),frozen_controls=dict(planned=5,computed=5,unique_in_round=5,new_structural_configuration=1,reselection=False),training=dict(PCA_fit_pipeline_minimum=PCA,PCA_sklearn_fit_call_minimum=2*PCA,R5_fit_minimum=R5,R5_saved_auxiliary_training=3,R5_validation_reconstruction=2,G1_round_operational_model_new_training=0,all_project_training_total='unknown: overwritten per-engine fit logs and nonexhaustive command history',Forest_excluded=True),inference=dict(selected_G1_parent_script_attempt_minimum=sum(x['category']=='inference_parent_stage' and x.get('selected_G1_computed',False) for x in E),selected_G1_parent_script_completed_minimum=sum(x['category']=='inference_parent_stage' and x.get('selected_G1_computed',False) and x['status']=='completed' for x in E),selected_G1_parent_script_failed_minimum=1,baseline_R5_diagnostic_parent_stages=2,unit='top-level stage script that computed selected-G1 predictions; nested child calls not added',exact_function_call_total='unknown'),evaluation=dict(new_prediction_metric_aggregation_parent_stages_minimum=6,stored_prediction_reaggregation_stages_minimum=4,unit='completed stage, not each metric or each S/S1/S2 report row',online_batch_workflows=4,online_batch_dataset_comparisons=10,nested_online_child_processes_minimum=14,report_generation_or_recitation_minimum=5,report_and_reaggregation_overlap=True),implementation=dict(v2_unique_reported_checks=151,v2_successful_checker_runs=2,frozen_new_reported_checks=76,frozen_checker_runs=2,reported_pass_items_across_4_runs=sum(x['reported_executed_check_items'] for x in checks),historical151_reuse_not_reexecution=True,ZIP_checks='separate integrity operations; not candidates or training',global_assertion_total='unknown'),preceding_shared_data_development_rounds=dict(count=4,status='minimum_confirmed',names=[x[0] for x in prior],additional_whole_replays=4),usage=dict(confirmed_used_unique_rows=20599,normal=19999,anomaly=600,duplicate_rows=1,confirmed_unused_unique_rows=0,unverifiable_unique_row_use=0,per_experiment_full_lineage='not universally verified',source_audit=rel(A/'audit_summary.json')),git='not a git repository; commit chronology unavailable',candidate_selection_chronology=dict(recorded_selection_UTC='2026-10-04T12:53:34.496235+00:00',session_evaluate_command_UTC='2026-10-04T12:53:55.854Z',session_source_line=1561,local_historical_logs_support_order=True,independent_tamperproof_timestamp_proof=False))
js(R/'count_summary.json',counts);js(R/'evidence_hashes.json',hashes);print(json.dumps(counts,ensure_ascii=False,indent=2))
if __name__=='__main__':pass

from common import *
import sys,subprocess,tempfile
lock=json.loads((R/'frozen_selection.json').read_text());d=loadrows(R/'inputs/frozen_rows.csv');e=Engine(d);selection=d.index[d.split=='selection'].to_numpy();checks={};saved=[]
for name,h in lock['code_sha256'].items():assert sha(R/'src'/name)==h
# Verify original raw CSV copies AND original source paths when available.
inv=json.loads((R/'inventory.json').read_text())
for f in inv['files']:
 assert sha(R/f['copy'])==f['sha256']
 if Path(f['source']).exists():assert sha(f['source'])==f['sha256']
checks['original_and_copied_hashes_preserved']=True
# All possible input windows remain within original roles. Reporting blocks do not reset windows.
for w in [1,3,5,20]:
 starts=e.starts_for(w);idx=np.flatnonzero(starts>=0)
 for col in ['source_file','split','train_role']:assert np.array_equal(d[col].to_numpy()[idx],d[col].to_numpy()[starts[idx]])
checks['unique_row_ids']=bool(d.row_id.is_unique);assert d.row_id.is_unique
checks['window_and_split_isolation']=True
# Every new model fit/calibration ID belongs to the permitted normal role.
train=set(d.loc[d.split=='train','row_id']);cal=set(d.loc[(d.split=='calibration')&(d.label==0),'row_id'])
for path in (R/'models').glob('warmup_*.joblib'):
 system=joblib.load(path)
 for w,en in system['entries'].items():
  b=en['bundle'];assert b['q_valid'] and b['k']<b['rank'];assert (b['model'].explained_variance_>b['tolerance']).all()
  if w not in [1,20]:assert set(en['train_ids'])<=train and set(en['normal_calibration_ids'])<=cal
checks['normal_only_fit_calibration_rank_eigenvalues']=True
# Future and forbidden metadata perturbations, evaluated on development only.
for cfg in lock['final_configs']:
 system=joblib.load(R/cfg['model_path']);before=e.predict(system,selection);dd=d.copy();prefix=[]
 for file,g in d.loc[selection].groupby('source_file',sort=False):
  cutoff=g.index[min(12,len(g)-2)];mask=(dd.source_file==file)&(dd.index>cutoff);dd.loc[mask,SENSORS]=dd.loc[mask,SENSORS]*-1000+777;prefix.extend(g.index[g.index<=cutoff])
 dd['burst_length']=999999;dd['label']=1-dd.label;dd['source_row']=987654;after=Engine(dd).predict(system,selection);assert np.array_equal(before.loc[prefix,'score'],after.loc[prefix,'score']);assert np.array_equal(before.loc[prefix,'route'],after.loc[prefix,'route']);checks[cfg['candidate']+'_future_invariant_rows']=len(prefix)
# Stored model inference via a fresh Python process, without labels or fitting.
with tempfile.TemporaryDirectory(dir=R/'logs') as temp:
 for cfg in lock['final_configs']:
  expected=loadrows(R/'predictions'/f'historical_frozen_{cfg["candidate"]}.csv')
  for file,g in expected.groupby('source_file',sort=False):
   ip=Path(temp)/(cfg['candidate']+'_'+file);op=Path(temp)/(cfg['candidate']+'_'+file+'.pred.csv');g[['TimeStamp']+SENSORS].to_csv(ip,index=False);pr=subprocess.run([sys.executable,str(R/'src/infer_system.py'),'--model',str(R/cfg['model_path']),'--threshold',str(cfg['threshold']),'--csv',str(ip),'--out',str(op)],capture_output=True,text=True);assert pr.returncode==0,pr.stderr;a=pd.read_csv(op);assert len(a)==len(g);assert np.array_equal(a.prediction,g.prediction);assert np.allclose(a.score,g.score,rtol=1e-10,atol=1e-10);saved.append(dict(candidate=cfg['candidate'],file=file,n=len(g),max_score_delta=float(np.max(abs(a.score.to_numpy()-g.score.to_numpy()))),identical_decisions=True))
checks['separate_inference']=saved
# Recalculate all final confusion matrices and named metrics from aligned rows.
s=pd.read_csv(R/'baseline_vs_candidates.csv')
for stage in ['selection','historical']:
 for cfg in lock['final_configs']:
  r=loadrows(R/'predictions'/f'{stage}_frozen_{cfg["candidate"]}.csv');m=metrics(r.label,r.score,r.prediction);x=s[(s.evaluation==stage)&(s.scope=='operational')&(s.candidate==cfg['candidate'])].iloc[0]
  assert r.row_id.tolist()==d.loc[d.split==('test' if stage=='historical' else 'selection'),'row_id'].tolist()
  for k,v in m.items():assert np.isclose(v,x[k],equal_nan=True)
  assert np.isclose(m['F1'],2*m['TP']/(2*m['TP']+m['FN']+m['FP']));assert np.isclose(m['F2'],5*m['TP']/(5*m['TP']+4*m['FN']+m['FP']))
checks['row_alignment_confusion_F1_F2']=True
sweep=pd.read_csv(R/'threshold_sweep.csv')
for k in ['AP','PR_AUC_trapezoid','ROC_AUC']:assert sweep[k].max()-sweep[k].min()<1e-14
checks['A_rank_metrics_invariant']=True;checks['integer_budget_cases']={str(n):[allowed_fp(n,x) for x in [0,.001,.002]] for n in [981,1005,1986,3990]};assert checks['integer_budget_cases']['981']==[0,0,1]
allr=pd.read_csv(R/'all_trials.csv');assert len(allr)<=36;checks['operating_configurations']=len(allr);checks['completed']=int((allr.status=='completed').sum());checks['failed']=int((allr.status=='failed').sum());checks['historical_unique_configs']=len(lock['final_configs']);assert len(lock['final_configs'])<=5
for rule,x in lock['criteria_selections'].items():
 chosen=pick(allr[(allr.status=='completed')&allr[rule+'_eligible']].to_dict('records'));assert (chosen['candidate'] if chosen else None)==(x['candidate'] if x else None)
checks['frozen_selection_recomputed_from_development_only']=True;checks['frozen_hash']=sha(R/'frozen_selection.json');checks['scientific_limitations']=['not independent validation','prior development and historical exposure','date-label confounding','no real failure onset','small routed calibration samples','no PCA seed pseudo-replication'];js(R/'validation_report.json',checks);print('Validation passed; new saved inference rows',sum(x['n'] for x in saved))

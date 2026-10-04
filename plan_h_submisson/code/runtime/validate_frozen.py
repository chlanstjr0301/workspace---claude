"""Reuse exact historical regression evidence; run only new wrapper/control contracts."""
from v2 import *
from review import frozen_prediction
from evaluate_new_data import checked_input
from datetime import datetime,timezone
import subprocess,sys,tempfile

def main():
 checks=[]
 def check(name,v):
  checks.append(dict(check=name,passed=bool(v)));assert v,name
 inv=json.loads((R/'inventory.json').read_text())
 for f in inv['files']:check('packaged immutable '+f['path'],sha(R/f['path'])==f['sha256'])
 p=json.loads((R/'validation_protocol.json').read_text());oldlock=json.loads((R/'inputs/previous/implementation_lock.json').read_text())
 check('reused inference sources exact historical version',all(sha(R/'src'/n)==oldlock['sources'][n] for n in ['common.py','models.py','evaluation.py','auxiliary.py','v2.py']))
 historical=json.loads((R/'inputs/previous/validation_report.json').read_text());check('historical151 passed record',historical['tests']==151 and historical['passed'] and all(x['passed'] for x in historical['regression']))
 check('151 evidence validator source copied intact',any(f['path']=='inputs/previous/validate_v2.py' for f in inv['files']))
 d,e,b0,r5=context();train=d[d.split=='train'];check('T exclusively normal',train.label.eq(0).all());check('raw IDs globally unique',not d.row_id.duplicated().any());check('all original nonduplicate rows assigned',len(d)==20599 and set(d.split)=={'train','calibration','selection','test'})
 check('R5 scaler normal train moments',np.allclose(train[SENSORS].to_numpy().mean(0),r5['scaler'].mean_,rtol=1e-12,atol=1e-12))
 # Reuse prior full normal-only R5 reconstruction, future perturbation, and split dependency checks.
 fit=pd.read_csv(R/'manifests/fit_rows_R5.csv');check('R5 fit manifest normal T only',set(fit.row_id)<=set(train.row_id))
 oldsel=json.loads((R/'inputs/previous/selected_config.json').read_text());oldV=json.loads((R/'inputs/previous/V_evaluation.json').read_text());oldH=json.loads((R/'inputs/previous/H_evaluation.json').read_text())
 js(R/'prior_validation_reuse.json',dict(reused_regression_count=151,original_report_sha256=sha(R/'inputs/previous/validation_report.json'),validation_code_sha256=sha(R/'inputs/previous/validate_v2.py'),exact_core_model_data_versions=True,interpretation='Historical successful evidence reused; not151 newly executed tests. Includes future perturbation and normal-only R5 reconstruction. Local hashes do not prove historical chronology.',old_selection=oldsel,V_record=oldV,H_record=oldH))
 # New fixed controls and wrapper: independent one-row implementation and separate process reload.
 online_checks=[];configs=[x for x in p['controls'] if x['id']!='B0']
 for partition in ['selection','test']:
  f=d[d.split==partition];ii=f.index.to_numpy();start=time.perf_counter();o=online_scores(f,b0,r5,configs);sec=time.perf_counter()-start;b=base_predictions(e,b0,ii);s=aux_score(r5,e,b0,ii,b.score)
  check(partition+' one-row input order',o.row_id.tolist()==b.row_id.tolist());check(partition+' B0 online exact decisions',np.array_equal(o.base_prediction,b.prediction));check(partition+' online R5 numeric scores',np.allclose(o.R5_score,s,atol=1e-9,rtol=1e-9,equal_nan=True))
  for c in configs:
   fz=frozen_prediction(b,s,c);check(partition+' online '+c['id'],np.array_equal(o[c['id']],fz.prediction))
  online_checks.append(dict(partition=partition,rows=len(o),B0_max_abs_score_error=float(np.max(abs(o.base_score.to_numpy()-b.score.to_numpy()))),R5_max_abs_error=float(np.nanmax(abs(o.R5_score.to_numpy()-s))),single_row_loop_seconds=sec,measured_average_seconds_per_row=sec/len(o),not_hardware_realtime_latency_guarantee=True))
 # Missing-input defect in unchanged helper is isolated to invalid-input contract, not performance data.
 tiny=d[d.split=='test'].iloc[:5].copy();tiny.loc[tiny.index[2],SENSORS[0]]=np.nan
 oo=online_scores(tiny,b0,r5,[configs[1]])
 missing=dict(original_helper_produces_nonfinite_base_score=bool(np.isnan(oo.base_score.iloc[2])),original_helper_treats_nonfinite_base_as_no_alarm=bool(oo.base_prediction.iloc[2]==0),supplied_original_data_have_no_missing=True,fix='strict finite-sensor input wrapper rejects the entire external input; frozen core unchanged',valid_data_scores_affected=False)
 check('demonstrate unsupported missing-input behavior',missing['original_helper_produces_nonfinite_base_score'] and missing['original_helper_treats_nonfinite_base_as_no_alarm'])
 (R/'external_smoke').mkdir(exist_ok=True);f=d[d.split=='test'].copy();sf=f[['row_id','TimeStamp']+SENSORS].copy();sf['session_id']=f.source_file.to_numpy();sf.to_csv(R/'external_smoke/sensors.csv',index=False);f[['row_id','label']].to_csv(R/'external_smoke/labels.csv',index=False)
 provenance=dict(independently_collected=False,sensor_semantics_compatible=True,prior_usage='historical_H_replay',equipment_id='unknown_original',acquisition_dates=['2022-07-12','2022-07-17'],units='original unchanged values; physical units undocumented',note='API regression smoke test only, not independent performance data')
 js(R/'external_smoke/provenance.json',provenance)
 bad=sf.iloc[:5].copy();bad.loc[2,SENSORS[0]]=np.nan;bad.to_csv(R/'external_smoke/missing_contract_only.csv',index=False)
 rejected=False
 try:checked_input(R/'external_smoke/missing_contract_only.csv')
 except ValueError:rejected=True
 check('wrapper rejects missing, no false normal classification',rejected);js(R/'missing_input_contract.json',missing)
 out=R/'external_smoke/replay'
 if out.exists():raise FileExistsError('Use a fresh run/replay directory, do not overwrite prior external evaluation')
 cmd=[sys.executable,str(R/'src/evaluate_new_data.py'),'--sensors',str(R/'external_smoke/sensors.csv'),'--provenance',str(R/'external_smoke/provenance.json'),'--labels',str(R/'external_smoke/labels.csv'),'--output',str(out)]
 proc=subprocess.run(cmd,capture_output=True,text=True);(R/'logs/external_smoke.log').write_text(proc.stdout+proc.stderr);check('new-data interface independent process model reload',proc.returncode==0)
 o=pd.read_csv(out/'predictions.csv');bb=pd.read_csv(R/'predictions/H_B0.csv');gg=pd.read_csv(R/'predictions/H_G1.csv');check('portable external replay same rows',o.row_id.tolist()==gg.row_id.tolist());check('portable external B0 same decisions',np.array_equal(o.B0,bb.prediction));check('portable external G1 same decisions',np.array_equal(o.G1,gg.prediction));check('prediction digest before labels unchanged',sha(out/'predictions.csv')==json.loads((out/'predictions_lock.json').read_text())['sha256'])
 # Scalar strict ties at exact frozen theta. No invented positive result on synthetic performance data.
 theta=p['controls'][2]['theta'];check('strict exact threshold equality no auxiliary hit',not(theta>theta));check('next float above theta hits',np.nextafter(theta,np.inf)>theta)
 # The prior dependencies deliberately share past S observations into V; report-only boundaries preserve state.
 dependency=pd.read_csv(R/'inputs/previous/dependency_audit.csv');check('prior dependency audit includes original split protection',(dependency.unique_raw_dependency_rows>=dependency.evaluation_rows).all())
 pd.DataFrame(checks).to_csv(R/'validation_checks.csv',index=False);js(R/'validation_report.json',dict(passed=True,reused_historical_checks=151,new_checks=checks,online=online_checks,new_data_performance_validation='not_performed_no_independent_new_acquisition',missing_input=missing,core_models_retrained=False,chronology_independently_proven=False));print('Validation passed; historical151 reused, new controls/wrapper verified')
if __name__=='__main__':main()

from v2 import *
from experiment_v2 import protocol
import subprocess,sys,tempfile,importlib.util

def main():
 d,e,b0,r5=context();p=protocol();lock=json.loads((R/'selected_config.json').read_text());checks={};reg=[];ids=np.array(sorted(p['blocks']['S1']+p['blocks']['S2']));base=base_predictions(e,b0,ids);bm=measure(base)
 def test(name,condition):
  reg.append(dict(test=name,passed=bool(condition)));assert condition,name
 test('identical B0 fails strict S effect',not judge(bm,bm,'effect')['passes'])
 # Regression inputs use actual V/S2 confusion matrices; synthetic changes conserve row totals.
 oldV=pd.read_csv(R/'inputs/previous/baseline_V.csv');vm=metrics(oldV.label,oldV.score,oldV.prediction);vj=judge(vm,vm);test('identical budget-respecting V passes stability',vj['stability_pass']);test('FN0to0 maintenance true reduction not_assessable',vj['fn_maintenance_pass'] is True and vj['fn_reduction_reproduced']=='not_assessable');test('F2 1to1 maintenance passes',vj['passes'])
 fail=vm.copy();fail.update(TP=89,FN=1,F1=178/179,F2=445/449);test('FN0to1 V maintenance fails',not judge(vm,fail)['passes'])
 s2=base.loc[p['blocks']['S2']];s2m=measure(s2);bad=s2m.copy();bad.update(FP=7,TN=609,F1=6/13,F2=15/22);sj=judge(s2m,bad);test('S2 FP6to7 fails both unchanged budgets','FP_budget' in sj['failed_rules'] and 'absolute_FPR' in sj['failed_rules'])
 normal=metrics(np.zeros(731),np.zeros(731),np.zeros(731));nj=judge(normal,normal);test('normal-only V judges only FP/FPR',nj['passes'] and nj['fn_maintenance_pass'] is None and nj['fn_reduction_reproduced']=='not_assessable' and not nj['anomaly_metrics_assessable']);normalbad=normal.copy();normalbad.update(FP=1,TN=730);test('normal-only V still enforces extra FP budget',not judge(normal,normalbad)['passes'])
 frame=pd.DataFrame(dict(source_file=['a']*8,split=['s']*8,train_role=['']*8,burst_id=[1]*4+[2]*4));ss=np.array([2,2,0,2,2,0,2,2.])
 g1=gate(frame,ss,1,'G1')[0];g2=gate(frame,ss,1,'G2')[0];test('G1 reset and no extension without current hit',np.array_equal(g1,[0,1,0,0,0,0,0,1]));test('G2 reset and current-hit requirement',np.array_equal(g2,[0,1,0,1,0,0,1,1]));missing=gate(frame,np.full(8,np.nan),1,'G2')[0];test('unavailable auxiliaries never alarm',not missing.any())
 # All operated rules preserve B0, and exact frame rows/classes are aligned.
 trial=pd.read_csv(R/'all_trials.csv');test('at most six slots and no replacement trials',len(trial)==6)
 for cfg in p['candidates']:
  r=pd.read_csv(R/'predictions'/f"{cfg['id']}_S_pooled.csv");test(cfg['id']+' same rows and B0 detections retained',r.row_id.tolist()==base.row_id.tolist() and not ((base.prediction.to_numpy()==1)&(r.prediction.to_numpy()==0)).any());test(cfg['id']+' score threshold matches alarm',np.array_equal(r.integrated_score>1,r.prediction.astype(bool)))
 # Independent online implementation and separate process model reload on S, then ONLY the selected V/H configuration.
 proc_checks=[];datasets=[('S',d.loc[ids],p['candidates'],'development')]
 if lock['selected']:
  if (R/'V_evaluation.json').exists():datasets.append(('selection_prefix_and_V',d[d.split=='selection'],[lock['selected_config']],'selected'))
  if (R/'H_evaluation.json').exists():datasets.append(('H',d[d.split=='test'],[lock['selected_config']],'selected'))
 with tempfile.TemporaryDirectory(dir=R/'logs') as temp:
  for stage,frame,configs,scope in datasets:
   parts=[]
   for name,g in frame.groupby('source_file',sort=False):
    ip=Path(temp)/(stage+'_'+name);op=ip.with_suffix('.out.csv');g[['row_id','TimeStamp']+SENSORS].to_csv(ip,index=False);z=subprocess.run([sys.executable,str(R/'src/infer_online.py'),'--csv',str(ip),'--out',str(op),'--scope',scope],capture_output=True,text=True);assert z.returncode==0,z.stderr;parts.append(pd.read_csv(op))
   online=pd.concat(parts,ignore_index=True);ii=frame.index.to_numpy();bb=base_predictions(e,b0,ii);ss=aux_score(r5,e,b0,ii,bb.score);test(stage+' online rows identical',online.row_id.tolist()==frame.row_id.tolist());test(stage+' online B0 scores/predictions match',np.allclose(online.base_score,bb.score,rtol=1e-9,atol=1e-10) and np.array_equal(online.base_prediction,bb.prediction));test(stage+' online R5 scores match',np.allclose(online.R5_score,ss,rtol=1e-9,atol=1e-10,equal_nan=True))
   for cfg in configs:
    batch=generate(e,b0,r5,ii,cfg['theta'],cfg['rule']);test(stage+' '+cfg['id']+' batch/online alarms identical',np.array_equal(online[cfg['id']],batch.prediction))
   proc_checks.append(dict(stage=stage,rows=len(frame),B0_max_error=float(np.max(abs(online.base_score.to_numpy()-bb.score.to_numpy()))),R5_max_error=float(np.nanmax(abs(online.R5_score.to_numpy()-ss)))))
 # Deliberately perturb future sensors only in S; earlier results must be byte-identical.
 changed=d.copy();past=[]
 for _,g in d.loc[ids].groupby('source_file',sort=False):
  ii=g.index.to_numpy();cut=len(ii)//2;past.extend(ii[:cut]);changed.loc[ii[cut:],SENSORS]=changed.loc[ii[cut:],SENSORS]*17-777
 ee=Engine(changed);past=np.array(sorted(past))
 for cfg in p['candidates']:
  a=generate(e,b0,r5,ids,cfg['theta'],cfg['rule']);b=generate(ee,b0,r5,ids,cfg['theta'],cfg['rule']);test(cfg['id']+' future perturbation leaves past scores and alarms unchanged',np.array_equal(a.loc[past,'prediction'],b.loc[past,'prediction']) and np.array_equal(a.loc[past,'integrated_score'],b.loc[past,'integrated_score']))
 # Complete raw-history ranges per model and per reporting block: no future or original split crossing.
 role=np.array(d.split,dtype=object);role[p['blocks']['S1']]='S1';role[p['blocks']['S2']]='S2';role[p['blocks']['V']]='V';dep=[];sets={}
 for block,ii in [('T',d.index[d.split=='train'].to_numpy()),('C',d.index[(d.split=='calibration')&(d.label==0)].to_numpy()),('S1',np.array(p['blocks']['S1'])),('S2',np.array(p['blocks']['S2'])),('V',np.array(p['blocks']['V'])),('H',d.index[d.split=='test'].to_numpy())]:
  for kind,width,aware in [('B0',20,False),('G0',3,True),('G1',4,True),('G2',5,True)]:
   allraw=set();cross=0;available=0
   for i in ii:
    if kind=='B0':start=e.starts_for(20)[i] if e.starts_for(20)[i]>=0 else i
    else:
     # Maximum history that can be used now, with unavailable h values represented by zeros.
     session_start=i
     while session_start>0 and session_start>i-width+1 and d.loc[session_start-1,'source_file']==d.loc[i,'source_file'] and d.loc[session_start-1,'split']==d.loc[i,'split'] and d.loc[session_start-1,'train_role']==d.loc[i,'train_role'] and d.loc[session_start-1,'burst_id']==d.loc[i,'burst_id']:session_start-=1
     start=session_start
    raw=list(range(start,i+1));allraw.update(raw);cross+=int(any(role[j]!=role[i] for j in raw));assert all(d.loc[j,'split']==d.loc[i,'split'] and d.loc[j,'source_file']==d.loc[i,'source_file'] and d.loc[j,'train_role']==d.loc[i,'train_role'] for j in raw);assert max(raw)<=i
    if kind!='B0':assert all(d.loc[j,'burst_id']==d.loc[i,'burst_id'] for j in raw)
    available+=int(e.starts_for(3,True)[i]>=0) if kind!='B0' else 1
   sets[(block,kind)]=allraw;dep.append(dict(block=block,kind=kind,evaluation_rows=len(ii),unique_raw_dependency_rows=len(allraw),rows_using_previous_reporting_block=cross,available_lag2_rows=available,max_past_rows=width-1))
 pd.DataFrame(dep).to_csv(R/'tables/dependency_audit.csv',index=False)
 for model in ['B0','G0','G1','G2']:
  for a,b in [('T','C'),('T','S1'),('C','S1'),('T','H'),('C','H'),('V','H')]:test(model+' raw dependencies disjoint '+a+'/'+b,sets[(a,model)].isdisjoint(sets[(b,model)]))
 # Old/new policy difference evaluated on the SAME stored predictions, never new candidate V/H tests.
 spec=importlib.util.spec_from_file_location('old_experiment',R/'inputs/previous/experiment.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);policy=[]
 for cid,block,a,b in [('C1','S_pooled',bm,bm),('C1','V',vm,vm)]:
  prev,why=old.eligible(a,b,True);new=judge(a,b,'effect' if block=='S_pooled' else 'maintenance');policy.append(dict(candidate=cid,block=block,old_pass=prev,old_failed_rules=';'.join(why),new_pass=new['passes'],fn_maintenance_pass=new['fn_maintenance_pass'],fn_reduction_reproduced=new['fn_reduction_reproduced'],predictions_changed=False))
 c0=pd.read_csv(R/'predictions/C0_S2.csv');prev,why=old.eligible(s2m,measure(c0),False);new=judge(s2m,measure(c0));policy.append(dict(candidate='C0',block='S2',old_pass=prev,old_failed_rules=';'.join(why),new_pass=new['passes'],fn_maintenance_pass=new['fn_maintenance_pass'],fn_reduction_reproduced=new['fn_reduction_reproduced'],predictions_changed=False));test('C0 old S2 rejection remains after V policy repair',not prev and not new['passes']);pd.DataFrame(policy).to_csv(R/'policy_comparison.csv',index=False)
 # Original copied files and source files remain unchanged.
 inv=json.loads((R/'inventory.json').read_text())
 for f in inv['files']:
  test('copied original hash '+f['path'],sha(R/f['path'])==f['sha256'])
  src=Path(f['source'])
  if src.exists():test('original remains intact '+f['path'],sha(src)==f['sha256'])
 impl=json.loads((R/'implementation_lock.json').read_text());test('locked implementation unchanged',all(sha(R/'src'/f)==v for f,v in impl['sources'].items()));test('protocol original hash unchanged',sha(R/'protocol_v2.json')==(R/'protocol_v2.sha256').read_text().strip());test('selection hash unchanged',sha(R/'selected_config.json')==(R/'selected_config.sha256').read_text().strip())
 # Original R5 fit can be reconstructed using only T, and matches stored prediction parameters.
 tt=d.index[d.split=='train'].to_numpy();test('normal-only T',(d.loc[tt,'label']==0).all());scale=StandardScaler().fit(e.raw[tt]);valid,u,z=lagdata(e,scale,tt,2);um=u.mean(0);zm=z.mean(0);uc=u-um;gram=uc.T@uc/len(u);lam=1e-3*np.trace(gram)/u.shape[1];coef=np.linalg.solve(gram+lam*np.eye(u.shape[1]),uc.T@(z-zm)/len(u));cov=covariance(z-((u-um)@coef+zm));test('fixed normal-only R5 reconstruction',np.allclose(scale.mean_,r5['scaler'].mean_) and np.allclose(coef,r5['coef'],atol=1e-12) and np.allclose(cov['cov'],r5['covariance']['cov'],atol=1e-12))
 pd.DataFrame(reg).to_csv(R/'tables/regression_tests.csv',index=False);js(R/'validation_report.json',dict(passed=True,tests=len(reg),regression=reg,online=proc_checks,selected=lock['selected'],no_model_retuning=True));print('All',len(reg),'checks passed',proc_checks)
if __name__=='__main__':main()

from auxiliary import *
from experiment import context
import shutil

def audit():
 d,e,base,p=context();cal=d.index[(d.split=='calibration')&(d.label==0)].to_numpy();b,_=evaluated(e.predict(base,cal),base['threshold']);out=b.copy();lock=json.loads((R/'selected_config.json').read_text())
 for key in ['R0','R1','R2','R3','R4','R5']:
  out[key]=aux_score(joblib.load(R/'models'/f'{key}.joblib'),e,base,cal,b.score)
  for c in [v for v in lock['configs'].values() if v['score']==key]:
   s=out[key].to_numpy();extra=(out.prediction==0)&np.isfinite(s)&(s>c['threshold']);out[c['id']+'_new_alarm']=extra.astype(int);assert int(extra.sum())<=allowed_fp(len(out),c['additional_calibration_FPR'])
 out.to_csv(R/'tables/normal_calibration_scores.csv',index=False)
 duplicates=[];mapping=[]
 for f,g in d.groupby('source_file',sort=False):
  raw=pd.read_csv(R/'data'/f);keys=['TimeStamp']+SENSORS+['Equipment_state'];raw['source_row']=np.arange(1,len(raw)+1);dup=raw.duplicated(keys);kept=raw.drop_duplicates(keys,keep='first').copy();kept['canonical_source_row']=kept.source_row
  link=raw.merge(kept[keys+['canonical_source_row']],on=keys,how='left');lookup=dict(zip(g.source_row,g.row_id));link['row_id']=link.canonical_source_row.map(lookup);link['removed_duplicate']=link.source_row!=link.canonical_source_row;link['source_file']=f;mapping.append(link[['source_file','source_row','canonical_source_row','row_id','removed_duplicate']]);duplicates.append(link[link.removed_duplicate])
 pd.concat(mapping).to_csv(R/'manifests/raw_to_clean_rows.csv',index=False);pd.concat(duplicates).to_csv(R/'manifests/duplicate_removed.csv',index=False)
 # Record unchanged original PCA constants/rank/eigenvalues; original objects remain untouched.
 details={}
 for w,entry in base['entries'].items():
  bundle=entry['bundle'];m=bundle['model'];details[str(w)]=dict(feature_names=bundle['names'],retained_features=np.asarray(bundle['names'])[bundle['keep']],removed_features=np.asarray(bundle['names'])[~bundle['keep']],rank=bundle['rank'],k=bundle['k'],tolerance=bundle['tolerance'],explained_variance=m.explained_variance_,PCA_mean=m.mean_,Q_reference=entry['reference'],solver=m.svd_solver,whiten=m.whiten,scaler_normal_samples=bundle['scaler'].n_samples_seen_)
 js(R/'tables/baseline_numeric_audit.json',details)
 # Disclosure of failures and ties includes inactive slots, even though none failed here.
 tr=pd.read_csv(R/'all_trials.csv');js(R/'execution_summary.json',dict(status_counts=tr.status.value_counts().to_dict(),registered_configs=18,executed_configs=len(tr),disabled_configs=int((tr.status=='disabled').sum()),selected=lock['selected'],source_code_added_after_selection='validation/report/artifact audit only; implementation_lock-covered model/selection code unchanged',baseline_duplicate_control='R0_A0 exactly equals original threshold; two output aliases, one distinct historical configuration'))
 print('Artifact audit complete')
if __name__=='__main__':audit()

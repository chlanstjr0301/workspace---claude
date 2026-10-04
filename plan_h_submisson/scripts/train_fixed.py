"""Reconstruct only the selected two PCA models and lag-2 R5. Not a search.
New models are written only under --out and never replace selected artifacts.
"""
from pathlib import Path
import sys,os,argparse,json,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code/runtime'))
from v2 import *
def main(out):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 d=loadrows(ROOT/'data/analysis/inputs/frozen_rows.csv');audit=json.loads((ROOT/'results/frozen/data_audit.json').read_text());mapping=[]
 # Recover all observations directly from byte-preserved raw files; manifest assigns frozen roles only.
 for meta in audit['raw_files']:
  path=ROOT/'data/raw'/meta['source_file'];assert sha(path)==meta['sha256'];raw=pd.read_csv(path);raw.TimeStamp=pd.to_datetime(raw.TimeStamp)
  dup=raw.duplicated(['TimeStamp']+SENSORS+['Equipment_state'],keep='first');ret=raw.loc[~dup].copy();f=d[d.source_file==meta['source_file']];assert (ret.index.to_numpy()+1==f.source_row.to_numpy()).all()
  assert np.array_equal(ret[SENSORS].to_numpy(),f[SENSORS].to_numpy());assert np.array_equal(ret.Equipment_state,f.label);assert np.array_equal(ret.TimeStamp,f.TimeStamp)
  d.loc[f.index,SENSORS]=ret[SENSORS].to_numpy();mapping.extend(dict(source_file=meta['source_file'],source_row=int(i+1),csv_line=int(i+2),removed_duplicate=bool(dup.iloc[i])) for i in range(len(raw)))
 pd.DataFrame(mapping).to_csv(out/'raw_mapping.csv',index=False);e=Engine(d);tt=d.index[d.split=='train'].to_numpy();ci=d.index[(d.split=='calibration')&(d.label==0)].to_numpy();assert d.loc[tt,'label'].eq(0).all()
 old=joblib.load(ROOT/'models/baseline.joblib');original_r5=joblib.load(ROOT/'models/R5.joblib');b0=dict(name='original',short_window=None,entries={},threshold=old['threshold'],postprocessing='none');fitrows=[]
 for w in [20,1]:
  ii=tt[e.starts_for(w)[tt]>=0];x=e.x(w,ii);b=fit_pca(x,2,FEATURES if w==20 else SENSORS);assert b['q_valid'];cn=ci[e.starts_for(w)[ci]>=0];ref=reference(score(b,e.x(w,cn))['Q']);b0['entries'][w]=dict(bundle=b,reference=ref)
  np.savez_compressed(out/f'PCA_W{w}_train.npz',x=x,row_ids=d.loc[ii,'row_id'].to_numpy(str),feature_names=np.array(FEATURES if w==20 else SENSORS))
  fitrows.extend(dict(model=f'PCA_W{w}',row_id=x,role='normal_T') for x in d.loc[ii,'row_id'])
 b0['threshold']=threshold(e.predict(b0,ci).score,.01);assert np.isclose(b0['threshold'],old['threshold'],atol=1e-10,rtol=1e-10)
 scaler=StandardScaler().fit(e.raw[tt]);valid,u,z=lagdata(e,scaler,tt,2);um=u.mean(0);zm=z.mean(0);uc=u-um;gram=uc.T@uc/len(u);lam=1e-3*np.trace(gram)/u.shape[1];coef=np.linalg.solve(gram+lam*np.eye(u.shape[1]),uc.T@(z-zm)/len(u));err=z-((u-um)@coef+zm);cov=covariance(err)
 r5=dict(kind='ridge',L=2,scaler=scaler,u_mean=um,z_mean=zm,coef=coef,lambda_=float(lam),covariance=cov,active=True,score_id='R5')
 assert np.allclose(coef,original_r5['coef'],atol=1e-12,rtol=1e-12);assert np.allclose(cov['cov'],original_r5['covariance']['cov'],atol=1e-12,rtol=1e-12)
 np.savez_compressed(out/'R5_train.npz',u=u,z=z,residual=err,row_ids=d.loc[tt[valid],'row_id'].to_numpy(str));fitrows.extend(dict(model='R5',row_id=x,role='normal_T') for x in d.loc[tt[valid],'row_id']);pd.DataFrame(fitrows).to_csv(out/'fit_rows.csv',index=False)
 s=aux_score(r5,e,b0,ci);theta=float(np.quantile(s[np.isfinite(s)],.999,method='higher'));expected=json.loads((ROOT/'configs/model_manifest.json').read_text());assert abs(theta-expected['G1_theta'])<=1e-10
 cal=e.predict(b0,ci);cal['R5_score']=s;cal.to_csv(out/'calibration_scores.csv',index=False)
 joblib.dump(b0,out/'baseline_regenerated.joblib');joblib.dump(r5,out/'R5_regenerated.joblib');checks=[]
 for split in ['selection','test']:
  ids=d.index[d.split==split].to_numpy();a=generate(e,old,original_r5,ids,expected['G1_theta'],'G1');b=generate(e,b0,r5,ids,theta,'G1');assert np.array_equal(a.prediction,b.prediction);assert np.allclose(a.integrated_score,b.integrated_score,atol=1e-9,rtol=1e-9)
  b.to_csv(out/f'{split}_regenerated_predictions.csv',index=False);checks.append(dict(split=split,rows=len(ids),exact_alarms=True,max_score_difference=float(np.max(abs(a.integrated_score-b.integrated_score)))))
 js(out/'training_reproduction.json',dict(passed=True,PCA_pipelines=2,PCA_fit_calls=4,R5_fit_calls=1,new_search=False,threshold=theta,B0_threshold=b0['threshold'],lambda_=lam,checks=checks,selected_models_replaced=False,numeric_tolerance='atol=rtol=1e-9 scores; 1e-12 R5 coefficients',regenerated_model_hashes={p.name:sha(p) for p in out.glob('*.joblib')}));print(out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)

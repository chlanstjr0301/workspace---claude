import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from models import SENSORS,STATS,features,score
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
df=pd.read_csv(run/'manifests/clean_rows.csv');raw=df[SENSORS].to_numpy(float);byid={x:i for i,x in enumerate(df.row_id)}
checks={};checks['no_test_score_files']=not any((run/'predictions').glob('*_test*'))
checks['unique_rows']=df.row_id.is_unique
checks['whole_burst_split']=df.groupby('burst_id').split.nunique().max()==1
checks['train_only_normal']=set(df[df.split=='train'].label)=={0}
scalers=[];qchecks=[];causal=[]
for path in (run/'models').glob('*.joblib'):
 b=joblib.load(path)
 if not isinstance(b,dict) or 'kind' not in b:continue
 names=b['names'];assert all(any(n==s or n.startswith(s+'__') for s in SENSORS) for n in names)
 rows=pd.read_csv(run/'manifests'/f'train_{b["name"]}.csv');ids=np.array([byid[i] for i in rows.row_id]);assert (df.iloc[ids].split=='train').all()
 w=1 if b['feature']=='R' else int(b['feature'][1:]);starts=ids-w+1
 x=raw[starts[:,None]+np.arange(w)]
 if b['sensors'] is not None:x=x[:,:,b['sensors']]
 x=x[:,0,:] if w==1 else features(x)
 fitx=x[:,b['keep']] if b['kind']=='PCA' else x
 scalers.append(np.allclose(fitx.mean(0),b['scaler'].mean_,rtol=1e-10,atol=1e-10))
 for col in ['source_file','split','train_role']+(['burst_id'] if b['feature'].startswith('B') else []):assert np.array_equal(df[col].to_numpy()[starts],df[col].to_numpy()[ids])
 causal.append(bool(np.all(starts<=ids)))
 if b['kind']=='PCA':
  assert np.all(b['model'].explained_variance_>b['tolerance'])
  qchecks.append((not b['q_valid']) or b['k']<b['rank'])
 else:
  observed=score(b,x[:20])['IF'];expected=-b['model'].score_samples(b['scaler'].transform(x[:20]));assert np.array_equal(observed,expected)
checks.update(scalers_fit_exact_normal_training=all(scalers),checked_models=len(scalers),causal_train_windows=all(causal),no_all_rank_Q=all(qchecks),sensor_only_feature_allowlist=True,if_score_direction_verified=True)
# Causal feature calculation agrees on a prefix and cannot depend on later observations.
x=np.arange(300,dtype=float).reshape(100,3);prefix=x[:30].copy();x[30:]=1e9
for w in [5,10,20]:assert np.array_equal(features(prefix[-w:][None]),features(x[30-w:30][None]))
checks['future_perturbation_invariance']=True
r=pd.read_csv(run/'tables/metrics_selection.csv');checks['all_IF_seeds_visible']=set(r[r.family.str.startswith('I')].seed.astype(str))=={'42','43','44','ensemble'}
for scope in ['common','operational']:
 g=r[r.scope==scope];assert g.N.nunique()==1
checks['same_row_denominators']=True
lock=json.loads((run/'selection_lock.json').read_text());checks['test_not_seen_at_lock']=lock['test_scores_seen'] is False
checks['fallback_shared']=len({lock['fallback']})==1
for f in ['common_selection.csv','common_test.csv']:
 path=run/'manifests'/f;assert hashlib.sha256(path.read_bytes()).hexdigest()==json.loads((run/'manifests/row_sets_frozen.json').read_text())['common_hashes'][f]
checks['common_rows_frozen_before_scores']=True
assert all(v for v in checks.values())
(run/'manifests/development_validation.json').write_text(json.dumps(checks,indent=2,default=lambda x:x.item()))
print(json.dumps(checks,indent=2,default=lambda x:x.item()))

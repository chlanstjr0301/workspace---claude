from core import *
from models import fit_pca,fit_if,score
R=Path(__file__).resolve().parents[1];d=loadrows(R/'manifests/review_rows.csv');blocks=json.loads((R/'manifests/blocks.json').read_text());lock=json.loads((R/'frozen_selection.json').read_text());e=Engine(d,R);checks={};extra=[]
# Fit membership, rank, numerical formula, label exclusions.
trainset=set(d.loc[d.split=='train','row_id']);calset=set(d.loc[(d.split=='calibration')&(d.label==0),'row_id']);audit=[]
for p in (R/'models').glob('final_*.joblib'):
 sys=joblib.load(p);assert set(sys['cal_ids'])<=calset
 for member in sys['members']:
  for w,m in member.items():
   b=m['bundle'];assert set(b['train_rows'])<=trainset;assert set(m['pool_ids'])<=calset
   assert all(any(n.startswith(s) for s in SENSORS) for n in b['names']);assert b['kind']=='IF' or b['k']<b['rank']
   ids=np.array(blocks['inner_late']['eval']);ids=ids[e.window(w,b['aware'])[ids]>=0][:50];x,_=e.x(ids,w,b['aware'],b['feat']);z=b['scaler'].transform(x[:,b['keep']])
   if b['kind']=='PCA':
    t=(z-b['model'].mean_)@b['model'].components_.T;r=z-(t@b['model'].components_+b['model'].mean_);raw=e.rawscore(b,ids);assert np.allclose((r*r).sum(1),raw['Q'],atol=1e-10);assert np.allclose((t*t/b['model'].explained_variance_).sum(1),raw['T2']);assert (b['model'].explained_variance_>b['tolerance']).all()
   audit.append(dict(system=p.name,route=w,kind=b['kind'],fit_n=len(b['train_rows']),cal_n=m['pool_n'],routed_cal_n=m['routed_n'],pool_policy=m['pool_policy'],rank=b['rank'],removed=sum(~b['keep']),thresholds=sys['thresholds']))
 checks[p.name]='fit/calibration membership and feature/rank/formula passed'
pd.DataFrame(audit).to_csv(R/'tables/fit_calibration_audit.csv',index=False)
# Future sensor perturbation and future length metadata cannot change prefix scores or routes.
for name in ['P1_Q',lock['selected']['candidate']]:
 sys=joblib.load(R/'models'/f'final_{name}.joblib');ids=np.array(blocks['inner_late']['eval']);before=e.predict(sys,ids)
 altered=d.copy();cutoffs={}
 for file,g in d.loc[ids].groupby('source_file'):
  cut=int(g.index[min(25,len(g)-1)]);cutoffs[file]=cut;future=(altered.source_file==file)&(altered.index>cut);altered.loc[future,SENSORS]=altered.loc[future,SENSORS]*-1000+12345
 altered['burst_length']=999999;altered['source_row']=999999;altered['label']=1-altered.label
 ee=Engine(altered,R);after=ee.predict(sys,ids)
 keep=np.array([i<=cutoffs[d.loc[i,'source_file']] for i in ids]);assert np.allclose(before.score.to_numpy()[keep],after.score.to_numpy()[keep],rtol=0,atol=0);assert np.array_equal(before.route.to_numpy()[keep],after.route.to_numpy()[keep]);checks[name+'_future_invariance']=int(keep.sum())
# No raw windows cross data/role boundaries; no original observation sharing across splits.
for aware in [False,True]:
 for w in [1,3,5,10,20]:
  st=e.window(w,aware);ids=np.flatnonzero(st>=0)
  for c in ['source_file','split','review_partition','train_role']+(['burst_id'] if aware else []):assert (d.loc[ids,c].to_numpy()==d.loc[st[ids],c].to_numpy()).all()
checks['window_boundary_and_unique_row_ids']=bool(d.row_id.is_unique)
# Controlled same-fit/same-cal/common-row PCA versus IF; P1/P2 inputs exactly coincide here.
train=np.flatnonzero((d.split=='train')&(e.window(20,True)>=0));xu,names=e.x(train,20,False,'full');xb,_=e.x(train,20,True,'full');assert np.array_equal(xu,xb);bm=fit_pca(xu,2,names);joblib.dump(bm,R/'models/controlled_pca_commonfit.joblib');ibs=[fit_if(xu,seed,names) for seed in [42,43,44]]
for i,b in enumerate(ibs):joblib.dump(b,R/'models'/f'controlled_if_commonfit_s{42+i}.joblib')
for block in ['inner_early','inner_late']:
 cal=np.array(blocks[block]['cal']);cal=cal[e.window(20,True)[cal]>=0];ev=np.array(blocks[block]['eval']);ev=ev[e.window(20,True)[ev]>=0];xc,_=e.x(cal,20,True,'full');xe,_=e.x(ev,20,True,'full');xeu,_=e.x(ev,20,False,'full');assert np.array_equal(xe,xeu)
 pc=score(bm,xc)['Q'];pe=score(bm,xe)['Q'];ic=[];ie=[]
 for b in ibs:
  a=score(b,xc)['IF'];ref=np.quantile(a,.95,method='higher');ic.append(a/ref);ie.append(score(b,xe)['IF']/ref)
 for name,sc,se in [('PCA_both_P1_P2_identical',pc,pe),('IF_ensemble_both_identical',np.mean(ic,axis=0),np.mean(ie,axis=0))]:
  for target in [.001,.005,.01]:h=threshold(sc,target);extra.append(dict(block=block,model=name,target=target,train_n=len(train),cal_n=len(cal),fallback=False,**metrics(d.loc[ev,'label'],se,se>h)))
pd.DataFrame(extra).to_csv(R/'tables/controlled_comparison.csv',index=False);checks['common_fit_P1_P2_feature_identity']=True
# Verify immutable original artifacts again.
inv=json.loads((R/'inventory.json').read_text());assert sha(inv['zip'])==inv['zip_sha256']
for a in json.loads((R/'manifests/data_audit.json').read_text()):assert sha(R.parents[1]/'data'/a['file'])==a['sha256']
checks['original_hashes_preserved']=True;checks['point_adjustment']='none';checks['two_inner_blocks_share_fit']='intentional, not independent folds';checks['earlier_eval_normal_in_later_calibration']='rolling temporal reuse, disjoint within each fold; disclosed';js(R/'validation.json',checks);print(json.dumps(checks,indent=2))

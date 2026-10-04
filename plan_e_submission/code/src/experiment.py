"""Locked-design PCA experiment. Test scoring requires the development lock."""
import argparse,hashlib,itertools,json,os,platform,shutil,sys,time
from pathlib import Path
from datetime import datetime,timezone
os.environ["MPLBACKEND"]="Agg"
import joblib,numpy as np,pandas as pd,scipy,sklearn,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss,log_loss,precision_recall_curve,roc_curve,ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits
from models import SENSORS,STATS,FEATURES,features,fit_pca,fit_if,score,reference,fit_sigmoid,probability
from evaluation import metrics,threshold,postprocess,alarm_summary
ROOT=Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x)))
def csv(p,records): pd.DataFrame(records).to_csv(p,index=False)
def now(): return datetime.now(timezone.utc).isoformat()

class Experiment:
    def __init__(self,out):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        for name in ['tables','models','predictions','figures','manifests','logs']: (self.out/name).mkdir(exist_ok=True)
        self.protocol=json.loads((ROOT/'configs/protocol.json').read_text())
        self.df=pd.read_csv(ROOT/'configs/first_experiment/frozen_rows.csv',keep_default_na=False)
        self.df.TimeStamp=pd.to_datetime(self.df.TimeStamp)
        self.raw=self.df[SENSORS].to_numpy(float);self.y=self.df.label.to_numpy(int);self.n=len(self.df)
        self.splits={s:self.df.index[self.df.split==s].to_numpy() for s in ['train','calibration','selection','test']}
        self.caln=self.splits['calibration'][self.y[self.splits['calibration']]==0]
        self.starts={};self.rawscores={};self.native={};self.op={};self.meta={};self.failures=[];self.bundles={};self.fallback=None
        self.fkeys=['R']+[f'{m}{w}' for w in [5,10,20] for m in ['U','B']]
        self.audit();self.windows()
    def save(self,folder,name,records): csv(self.out/folder/name,records)
    def audit(self):
        assert self.df.row_id.is_unique and set(self.y[self.splits['train']])=={0}
        assert self.df.groupby('burst_id').split.nunique().max()==1
        diagnostics=[];duplicates=[]
        for f in ['press_data_normal.csv','outlier_data.csv']:
            p=ROOT.parent/'data'/f;a=pd.read_csv(p);a['source_row']=np.arange(1,len(a)+1)
            assert not a.isna().any().any()
            for s in SENSORS: a[s]=pd.to_numeric(a[s],errors='raise')
            a.TimeStamp=pd.to_datetime(a.TimeStamp,errors='raise')
            key=['TimeStamp']+SENSORS+['Equipment_state'];dup=a.duplicated(key,keep='first');clean=a.loc[~dup].copy()
            for _,d in a[dup].iterrows():
                first=a[(a[key]==d[key]).all(axis=1)].iloc[0]
                duplicates.append(dict(source_file=f,removed_source_row=int(d.source_row),kept_source_row=int(first.source_row)))
            expected=self.df[self.df.source_file==f]
            assert clean.source_row.tolist()==expected.source_row.tolist()
            assert np.array_equal(clean[SENSORS].to_numpy(),expected[SENSORS].to_numpy())
            assert np.array_equal(clean.TimeStamp.to_numpy(),expected.TimeStamp.to_numpy())
            assert np.array_equal(clean.Equipment_state.to_numpy(),expected.label.to_numpy())
            gaps=clean.TimeStamp.diff().dt.total_seconds().dropna()
            assert (gaps>0).all()
            diagnostics.append(dict(file=f,sha256=sha(p),original_rows=len(a),clean_rows=len(clean),removed=int(dup.sum()),missing=int(a.isna().sum().sum()),numeric_failures=0,timestamp_failures=0,date=sorted(a.TimeStamp.dt.strftime('%Y-%m-%d').unique()),label_counts=a.Equipment_state.value_counts().to_dict(),gaps_over_05=int((gaps>.5).sum()),gap_max=float(gaps.max()),gap_median=float(gaps.median()),burst_count=int((gaps>.5).sum()+1)))
        original=json.loads((ROOT/'configs/first_experiment/frozen_audit.json').read_text())
        for a,b in zip(diagnostics,original['files']): assert a['sha256']==b['sha256']
        tr=self.df.loc[self.splits['train']];gaps=tr.TimeStamp.diff().dt.total_seconds().fillna(float('inf'))
        sensitivity=[]
        for t in [.2,.5,1.]: sensitivity.append(dict(gap=t,bursts=int((gaps>t).sum()),same_as_05=bool(np.array_equal(gaps>t,gaps>.5))))
        audit=dict(created=now(),files=diagnostics,prior_full_data_summary_seen=True,original_csv_unchanged=True,burst_sensitivity=sensitivity,split_origin='first experiment immutable artifacts',notes='No failure-onset, maintenance, pressure, quality, utilization, physical units or operating load histories supplied.')
        p=self.out/'manifests/audit.json'
        if not p.exists(): js(p,audit);self.save('manifests','duplicates.csv',duplicates)
        versions={m.__name__:m.__version__ for m in [np,pd,scipy,sklearn,matplotlib,joblib]}
        env=dict(python=sys.version,platform=platform.platform(),versions=versions,threads=2,device='CPU',protocol_sha256=sha(ROOT/'configs/protocol.json'),rows_sha256=sha(ROOT/'configs/first_experiment/frozen_rows.csv'),source_sha256={p.name:sha(p) for p in (ROOT/'src').glob('*.py')})
        p=self.out/'manifests/environment.json'
        if p.exists():
            old=json.loads(p.read_text());assert old['protocol_sha256']==env['protocol_sha256'] and old['rows_sha256']==env['rows_sha256']
        else:
            js(p,env);shutil.copytree(ROOT/'configs',self.out/'manifests/configs',ignore=shutil.ignore_patterns('colab_sessions.json'),dirs_exist_ok=True)
            shutil.copytree(ROOT/'src',self.out/'source',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
        self.df.to_csv(self.out/'manifests/clean_rows.csv',index=False)
    def windows(self):
        coverage=[];self.common={};self.same={}
        for key in self.fkeys:
            w=1 if key=='R' else int(key[1:]);starts=np.full(self.n,-1,int)
            group=['source_file','split','train_role']+(['burst_id'] if key.startswith('B') else [])
            for _,g in self.df.groupby(group,sort=False,dropna=False):
                idx=g.index.to_numpy()
                if len(idx)>=w: starts[idx[w-1:]]=idx[:len(idx)-w+1]
            ids=np.flatnonzero(starts>=0)
            assert np.all(ids-starts[ids]==w-1)
            for col in ['source_file','split','train_role']+(['burst_id'] if key.startswith('B') else []):
                assert np.all(self.df[col].to_numpy()[ids]==self.df[col].to_numpy()[starts[ids]])
            self.starts[key]=starts
            m=self.df[['row_id','source_file','source_row','label','split','burst_id','burst_pos','burst_length']].copy()
            m['available']=starts>=0;m['reason']=np.where(starts>=0,'available','insufficient past observations in partition or burst')
            m['window_rows']=w;m['start_row_id']='';m['elapsed_seconds']=np.nan;m['crosses_gap']=False
            m.loc[ids,'start_row_id']=self.df.row_id.to_numpy()[starts[ids]]
            m.loc[ids,'elapsed_seconds']=(self.df.TimeStamp.to_numpy()[ids]-self.df.TimeStamp.to_numpy()[starts[ids]])/np.timedelta64(1,'s')
            m.loc[ids,'crosses_gap']=self.df.burst_id.to_numpy()[ids]!=self.df.burst_id.to_numpy()[starts[ids]]
            m.to_csv(self.out/'manifests'/f'rows_{key}.csv',index=False)
            for (sp,y),g in m.groupby(['split','label']):
                for condition,mask in [('all',np.ones(len(g),bool)),('start',g.burst_pos<w),('short_burst',g.burst_length<w)]:
                    q=g.loc[np.asarray(mask)]
                    coverage.append(dict(feature=key,split=sp,label=y,condition=condition,total=len(q),available=int(q.available.sum()),excluded=int((~q.available).sum()),coverage=float(q.available.mean()) if len(q) else None,crossing_windows=int(q.crosses_gap.sum()),elapsed_max=q.elapsed_seconds.max()))
        for sp,ids in self.splits.items():
            self.common[sp]=ids[np.all(np.stack([v[ids]>=0 for v in self.starts.values()]),axis=0)]
            self.df.loc[self.common[sp],['row_id','label','split']].to_csv(self.out/'manifests'/f'common_{sp}.csv',index=False)
            for w in [5,10,20]:
                x=ids[self.starts[f'B{w}'][ids]>=0];self.same[sp,w]=x
                self.df.loc[x,['row_id','label','split']].to_csv(self.out/'manifests'/f'common_W{w}_{sp}.csv',index=False)
        self.save('tables','coverage.csv',coverage)
        js(self.out/'manifests/row_sets_frozen.json',dict(created=now(),scores_seen=False,common_hashes={p.name:sha(p) for p in (self.out/'manifests').glob('common*.csv')})) if not (self.out/'manifests/row_sets_frozen.json').exists() else None
    def ids(self,feature,split):
        a=self.splits[split];return a[self.starts[feature][a]>=0]
    def x(self,feature,ids,sensors=None):
        w=1 if feature=='R' else int(feature[1:]);st=self.starts[feature][ids];assert (st>=0).all()
        a=self.raw[st[:,None]+np.arange(w)]
        if sensors is not None: a=a[:,:,sensors]
        return a[:,0,:] if feature=='R' else features(a)
    def fit(self,name,feature,k=None,seed=None,sensors=None,trainids=None):
        p=self.out/'models'/f'{name}.joblib'
        if p.exists(): return joblib.load(p)
        idx=self.ids(feature,'train') if trainids is None else trainids
        assert (self.df.loc[idx,'split']=='train').all() and not self.y[idx].any()
        ss=SENSORS if sensors is None else [SENSORS[i] for i in sensors]
        names=ss if feature=='R' else [s+'__'+t for t in STATS for s in ss]
        tic=time.perf_counter();x=self.x(feature,idx,sensors)
        b=fit_pca(x,k,names) if k is not None else fit_if(x,seed,names)
        b.update(name=name,feature=feature,sensors=sensors,fit_seconds=time.perf_counter()-tic,training_windows=len(idx))
        joblib.dump(b,p,compress=3)
        self.df.loc[idx,['row_id','split','burst_id']].to_csv(self.out/'manifests'/f'train_{name}.csv',index=False)
        meta={key:b[key] for key in ['kind','name','feature','sensors','fit_seconds','training_windows','names']}
        if k is not None: meta.update({key:b[key] for key in ['k','rank','tolerance','removed','eigenvalues','cumulative_variance','q_valid']})
        js(self.out/'logs'/f'fit_{name}.json',meta)
        print('FIT',name,'n=',len(idx),'seconds=',round(b['fit_seconds'],3),flush=True)
        return b
    def scores(self,b,stage):
        p=self.out/'predictions'/f'raw_{b["name"]}_{stage}.npz'
        if p.exists(): return dict(np.load(p))
        if stage=='test': assert (self.out/'selection_lock.json').exists()
        ids=np.concatenate([self.ids(b['feature'],s) for s in (['calibration','selection'] if stage=='develop' else ['test'])])
        tic=time.perf_counter();values=score(b,self.x(b['feature'],ids,b['sensors']));result={}
        for name,value in values.items():
            if name=='contributions': continue
            a=np.full(self.n,np.nan);a[ids]=value;assert np.isfinite(a[ids]).all();result[name]=a
        np.savez_compressed(p,**result)
        js(self.out/'logs'/f'score_{b["name"]}_{stage}.json',dict(seconds=time.perf_counter()-tic,rows=len(ids)))
        return result
    def specifications(self):
        for f in self.fkeys:
            pm='P0' if f=='R' else 'P1' if f[0]=='U' else 'P2';im='I'+pm[1];w=1 if f=='R' else int(f[1:])
            for k in ([1,2] if f=='R' else [2,3,5,8]): yield f'{pm}_W{w}_K{k}',f,k,None
            for seed in [42,43,44]: yield f'{im}_W{w}_S{seed}',f,None,seed
    def develop(self):
        assert not (self.out/'selection_lock.json').exists(), 'Locked run cannot be retuned'
        for name,f,k,seed in self.specifications():
            try:
                b=self.fit(name,f,k,seed);self.scores(b,'develop')
            except ValueError as e: self.failures.append(dict(name=name,reason=str(e)))
        js(self.out/'logs/exclusions.json',self.failures)
        self.load('develop')
        p0=[k for k in self.native if k.startswith('P0')]
        rows=[self.measure(k,'selection','native',.01) for k in p0]
        chosen,status=self.choose(pd.DataFrame(rows));self.fallback=chosen['key']
        js(self.out/'fallback_lock.json',dict(key=self.fallback,status=status,selection_only=True))
        self.operations();self.evaluate('selection');self.evaluate('calibration')
        self.matched();self.sensor_study();self.choose_main()
    def load(self,stage):
        exclusions=json.loads((self.out/'logs/exclusions.json').read_text())
        excluded={a['name'] for a in exclusions}
        for name,f,k,seed in self.specifications():
            if name in excluded: continue
            b=self.fit(name,f,k,seed);self.bundles[name]=b;raw=self.scores(b,'develop')
            if stage=='test':
                te=self.scores(b,'test')
                for s in raw: raw[s][self.splits['test']]=te[s][self.splits['test']]
            self.rawscores[name]=raw
            normpath=self.out/'models'/f'norm_{name}.json'
            if normpath.exists(): norms=json.loads(normpath.read_text())
            else:
                norms={};cn=self.ids(f,'calibration');cn=cn[self.y[cn]==0]
                for s in (['Q','T2'] if k is not None else ['IF']):
                    if s=='Q' and not b['q_valid']: continue
                    try: norms[s]=reference(raw[s][cn])
                    except ValueError as e: self.failures.append(dict(name=name+'_'+s,reason=str(e)))
                js(normpath,norms)
            runtime=b['fit_seconds']+json.loads((self.out/'logs'/f'score_{name}_develop.json').read_text())['seconds']
            if k is not None:
                if 'Q' in norms and 'T2' in norms: raw['QT']=np.maximum(raw['Q']/norms['Q'],raw['T2']/norms['T2'])
                for s in list(norms)+(['QT'] if 'QT' in raw else []):
                    key=name+'_'+s;cn=self.ids(f,'calibration');cn=cn[self.y[cn]==0]
                    ref=norms[s] if s!='QT' else reference(raw[s][cn]);self.native[key]=raw[s]/ref
                    self.meta[key]=dict(base=name,feature=f,family=name[:2],window=1 if f=='R' else int(f[1:]),k=k,score=s,seed='deterministic',reference=ref,runtime=runtime)
            elif 'IF' in norms:
                self.native[name]=raw['IF']/norms['IF'];self.meta[name]=dict(base=name,feature=f,family=name[:2],window=1 if f=='R' else int(f[1:]),k=0,score='IF',seed=seed,reference=norms['IF'],runtime=runtime)
        for f in self.fkeys:
            m='I0' if f=='R' else 'I1' if f[0]=='U' else 'I2';w=1 if f=='R' else int(f[1:]);keys=[f'{m}_W{w}_S{s}' for s in [42,43,44]]
            key=f'{m}_W{w}_ensemble';self.native[key]=np.mean([self.native[k] for k in keys],axis=0)
            self.meta[key]={**self.meta[keys[0]],'base':key,'seed':'ensemble','runtime':sum(self.meta[k]['runtime'] for k in keys),'reference':None}
        if (self.out/'fallback_lock.json').exists(): self.fallback=json.loads((self.out/'fallback_lock.json').read_text())['key'];self.operations()
        js(self.out/'models/score_metadata.json',self.meta)
    def operations(self):
        for k,s in self.native.items():
            m=self.meta[k]
            if m['family'].startswith('P'): fallback=self.native[self.fallback]
            elif m['seed']=='ensemble':
                keys=[k.replace('ensemble',f'S{seed}') for seed in [42,43,44]]
                self.op[k]=np.mean([np.where(np.isfinite(self.native[a]),self.native[a],self.native[f'I0_W1_S{seed}']) for a,seed in zip(keys,[42,43,44])],axis=0);continue
            else: fallback=self.native[f'I0_W1_S{m["seed"]}']
            self.op[k]=np.where(np.isfinite(s),s,fallback)
    def measure(self,key,split,scope,target,ids=None):
        s=self.op[key] if scope=='operational' else self.native[key];allids=self.splits[split]
        ev=ids if ids is not None else allids if scope=='operational' else self.common[split] if scope=='common' else allids[np.isfinite(s[allids])]
        cn=self.caln[np.isfinite(s[self.caln])];th=threshold(s[cn],target)
        return dict(key=key,**self.meta[key],split=split,scope=scope,target_fpr=target,threshold=th,calibration_normal_n=len(cn),calibration_actual_fpr=float(np.mean(s[cn]>th)),**metrics(self.y[ev],s[ev],s[ev]>th))
    def evaluate(self,split):
        records=[]
        for k in self.native:
            for scope in ['native','common','operational']:
                for target in [.005,.01,.02,.05]: records.append(self.measure(k,split,scope,target))
        r=pd.DataFrame(records);r.to_csv(self.out/'tables'/f'metrics_{split}.csv',index=False)
        seeds=r[r.seed.astype(str).isin(['42','43','44'])]
        seeds.groupby(['family','window','scope','target_fpr'])[['Recall','Precision','F1','F2','FPR','FNR','AP','PR_AUC_trapezoid','ROC_AUC','Accuracy']].agg(['mean','std']).to_csv(self.out/'tables'/f'if_seed_summary_{split}.csv')
        pairs=[]
        for w in [5,10,20]:
            keys=[k for k in self.native if self.meta[k]['family'] in ['P1','I1'] and self.meta[k]['window']==w]
            for ka in keys:
                kb=ka.replace('P1','P2').replace('I1','I2')
                if kb not in self.native: continue
                ev=self.same[split,w];a=self.measure(ka,split,'pair',.01,ev);b=self.measure(kb,split,'pair',.01,ev)
                pair=dict(unaware=ka,aware=kb,window=w,k=self.meta[ka]['k'],score=self.meta[ka]['score'],seed=self.meta[ka]['seed'],N=len(ev),normal=int((self.y[ev]==0).sum()),anomaly=int(self.y[ev].sum()),positive_rate=float(self.y[ev].mean()))
                for metric in ['TP','FN','FP','TN','Recall','Precision','F1','F2','FPR','AP','PR_AUC_trapezoid','ROC_AUC']:
                    pair['unaware_'+metric]=a[metric];pair['aware_'+metric]=b[metric];pair['delta_'+metric]=b[metric]-a[metric]
                pairs.append(pair)
        self.save('tables',f'burst_pairs_{split}.csv',pairs)
        dummy=[]
        for scope,ev in [('operational',self.splits[split]),('common',self.common[split])]:
            for y in [0,1]: dummy.append(dict(scope=scope,prediction=y,**metrics(self.y[ev],np.full(len(ev),y),np.full(len(ev),y))))
        self.save('tables',f'dummy_{split}.csv',dummy)
        self.score_complements(split)
    def choose(self,r,alternative=False):
        if alternative: return r.sort_values(['F1','Recall','AP','runtime','key'],ascending=[False,False,False,True,True]).iloc[0].to_dict(),'F1 alternative'
        good=r[r.FPR<=.01]
        if len(good): return good.sort_values(['Recall','F1','AP','runtime','key'],ascending=[False,False,False,True,True]).iloc[0].to_dict(),'FPR target met in selection'
        return r.sort_values(['FPR','Recall','F1','AP','runtime','key'],ascending=[True,False,False,False,True,True]).iloc[0].to_dict(),'TARGET UNMET: descriptive minimum-FPR fallback'
    def score_complements(self,split):
        rows=[]
        for name,f,k,seed in self.specifications():
            keys=[name+'_'+s for s in ['Q','T2','QT']]
            if not all(key in self.native for key in keys):continue
            ids=self.common[split];ps=[]
            for key in keys:
                s=self.native[key];cn=self.caln[np.isfinite(s[self.caln])];ps.append(s[ids]>threshold(s[cn],.01))
            q,t,c=ps;y=self.y[ids]
            r=dict(base=name,scope='common',normal=int((y==0).sum()),anomaly=int(y.sum()))
            for label,lname in [(0,'normal'),(1,'anomaly')]:
                m=y==label
                r.update({f'{lname}_Q_only':int((m&q&~t).sum()),f'{lname}_T2_only':int((m&t&~q).sum()),f'{lname}_QT_added_vs_Q':int((m&c&~q).sum()),f'{lname}_QT_lost_vs_Q':int((m&q&~c).sum()),f'{lname}_QT_added_vs_T2':int((m&c&~t).sum()),f'{lname}_QT_lost_vs_T2':int((m&t&~c).sum())})
            rows.append(r)
        self.save('tables',f'score_complements_{split}.csv',rows)
    def matched(self):
        pairs=pd.read_csv(self.out/'tables/burst_pairs_selection.csv');chosen=pairs[(pairs.delta_Recall>0)|(pairs.delta_F1>0)]
        results=[];cache={}
        for _,row in chosen.iterrows():
            key=row.unaware;meta=self.meta[key];f=meta['feature'];aware='B'+f[1:];n=len(self.ids(aware,'train'))
            # Each PCA sample repetition is a real random training subset, not a PCA seed change.
            seeds=[42,43,44] if meta['family']=='P1' or str(meta['seed'])=='ensemble' else [int(meta['seed'])]
            if str(meta['seed'])=='ensemble':continue # constituent seeds are recorded; ensemble auxiliary is unnecessary.
            for seed in seeds:
                name=meta['base']+f'_matched{seed}'
                if name not in cache:
                    train=np.sort(np.random.default_rng(seed).choice(self.ids(f,'train'),n,replace=False))
                    b=self.fit(name,f,k=meta['k'] if meta['family']=='P1' else None,seed=seed,trainids=train)
                    raw=self.scores(b,'develop');cn=self.ids(f,'calibration');cn=cn[self.y[cn]==0]
                    if meta['family']=='P1':
                        cq=reference(raw['Q'][cn]);ct=reference(raw['T2'][cn]);raw['QT']=np.maximum(raw['Q']/cq,raw['T2']/ct)
                    cache[name]=raw
                s=cache[name][meta['score']];cn=self.caln[np.isfinite(s[self.caln])];th=threshold(s[cn],.01);ev=self.same['selection',meta['window']]
                results.append(dict(unaware=key,aware=row.aware,sampling_seed=seed,training_count=n,original_unaware_count=len(self.ids(f,'train')),target_fpr=.01,threshold=th,**metrics(self.y[ev],s[ev],s[ev]>th)))
        self.save('tables','matched_training_count_selection.csv',results)
        js(self.out/'logs/matched_budget.json',dict(question='Does burst advantage persist after equalizing train endpoint counts?',triggered_pairs=len(chosen),actual_unique_fits=len(cache),random_sampling_seeds=[42,43,44],not_used_for_selection=True))
    def sensor_study(self):
        rows=[];exclusions=[]
        for size in [1,2,3]:
            for subset in itertools.combinations(range(3),size):
                for f in ['R','B10']:
                    names=[SENSORS[i] for i in subset];train=self.ids(f,'train');x=self.x(f,train,subset)
                    featnames=names if f=='R' else [s+'__'+t for t in STATS for s in names]
                    probe=fit_pca(x,1,featnames);rank=probe['rank'];k=min(2,max(1,rank-1))
                    name=f'Ablation_{f}_'+''.join(map(str,subset));b=self.fit(name,f,k=k,sensors=list(subset));raw=self.scores(b,'develop')
                    cn=self.ids(f,'calibration');cn=cn[self.y[cn]==0];ev=self.same['selection',10] # identical rows across all subsets/raw/rolling
                    if rank>1:
                        raw['QT']=np.maximum(raw['Q']/reference(raw['Q'][cn]),raw['T2']/reference(raw['T2'][cn]));scores=['Q','T2','QT']
                    else:
                        scores=['T2'];exclusions.append(dict(name=name,reason='rank 1: Q and QT invalid; T2 only'))
                    for s in scores:
                        th=threshold(raw[s][cn],.01);rows.append(dict(model=name,sensors='+'.join(names),feature=f,k=k,rank=rank,score=s,scope='common_W10',threshold=th,**metrics(self.y[ev],raw[s][ev],raw[s][ev]>th)))
        self.save('tables','sensor_combinations_selection.csv',rows);js(self.out/'logs/sensor_exclusions.json',exclusions)
    def choose_main(self):
        results=pd.read_csv(self.out/'tables/metrics_selection.csv');op=results[results.scope=='operational'];p=op[op.family.str.startswith('P')];i=op[(op.family.str.startswith('I'))&(op.seed=='ensemble')]
        primary,status=self.choose(p[p.target_fpr==.01]);alt,astatus=self.choose(p,True);ibase,istatus=self.choose(i[i.target_fpr==.01])
        # Best within each family is chosen using development only, for a separate best-config table.
        best=[]
        for fam in ['P0','P1','P2','I0','I1','I2']:
            r=op[(op.family==fam)&(op.target_fpr==.01)]
            if fam.startswith('I'):r=r[r.seed=='ensemble']
            a,st=self.choose(r);best.append(dict(**a,selection_status=st))
        self.save('tables','best_by_family_selection.csv',best)
        auxiliary=self.post_analysis(primary['key'],'selection',primary['target_fpr'])
        good=auxiliary[auxiliary.FPR<=.01]
        pp='none' if not len(good) else good.sort_values(['Recall','F1','policy_priority'],ascending=[False,False,True]).iloc[0].policy
        for choice in [primary,alt,ibase]:self.error_analysis(choice['key'],'selection',choice['target_fpr'])
        prob=self.prob_analysis(primary['key'],'selection')
        frozen=dict(created=now(),primary=primary,primary_status=status,alternative=alt,if_primary=ibase,if_status=istatus,fallback=self.fallback,postprocess=pp,best_by_family=best,protocol_sha256=sha(ROOT/'configs/protocol.json'),rows_sha256=sha(ROOT/'configs/first_experiment/frozen_rows.csv'),test_scores_seen=False,probability=prob,all_candidates=[k for k in self.native],normalization='normal calibration Q95 ratio',test_policy='evaluate locked models and predeclared candidate tables once; no retuning')
        # Model/reference integrity is checked before any final score call.
        frozen['model_hashes']={p.name:sha(p) for p in (self.out/'models').glob('*')}
        js(self.out/'selection_lock.json',frozen)
        print('LOCKED',primary['key'],status,'selection Recall=',primary['Recall'],'FPR=',primary['FPR'],flush=True)
    def post_analysis(self,key,split,target):
        ids=self.splits[split];s=self.op[key];th=threshold(s[self.caln],target);raw=s[ids]>th;frame=self.df.loc[ids].reset_index(drop=True)
        records=[];bursts=[]
        for j,policy in enumerate(['none','consecutive2','two_of_three']):
            pred=postprocess(frame,raw,policy);b=alarm_summary(frame,pred);b['policy']=policy;b['model']=key;bursts.append(b)
            positive=b[b.label==1];det=positive[~positive.no_alarm]
            r=dict(key=key,split=split,policy=policy,policy_priority=j,threshold=th,anomaly_bursts=len(positive),missed_anomaly_bursts=int(positive.no_alarm.sum()),median_first_alarm_seconds=det.first_alarm_since_observed_burst_start_seconds.median(),new_FN_vs_none=int((raw&~pred.astype(bool)&(self.y[ids]==1)).sum()),additional_TP_vs_none=int((~raw&pred.astype(bool)&(self.y[ids]==1)).sum()),**metrics(self.y[ids],s[ids],pred))
            records.append(r)
        out=pd.DataFrame(records);out.to_csv(self.out/'tables'/f'postprocess_{split}.csv',index=False)
        pd.concat(bursts).to_csv(self.out/'tables'/f'burst_alarms_{split}.csv',index=False)
        # Per-burst delay, including missed bursts, permits paired delay calculations.
        return out
    def prob_analysis(self,key,split):
        s=self.op[key];cal=self.splits['calibration'];ev=self.splits[split];path=self.out/'models/sigmoid.joblib'
        if path.exists(): obj=joblib.load(path)
        else: obj=fit_sigmoid(s[cal],self.y[cal]);joblib.dump(obj,path);js(self.out/'models/sigmoid.json',obj)
        pred=probability(obj,s[ev]);prev=float(self.y[cal].mean());base=np.full(len(ev),prev)
        rows=[]
        for name,p in [('sigmoid',pred),('calibration_prevalence',base)]:
            rows.append(dict(method=name,key=key,split=split,N=len(ev),normal=int((self.y[ev]==0).sum()),anomaly=int(self.y[ev].sum()),positive_rate=float(self.y[ev].mean()),brier=float(brier_score_loss(self.y[ev],p)),log_loss=float(log_loss(self.y[ev],p,labels=[0,1])),calibration_positive_rate=prev))
        self.save('tables',f'probability_{split}.csv',rows)
        rel=[]
        for b in range(10):
            mask=(pred>=b/10)&((pred<(b+1)/10) if b<9 else pred<=1)
            rel.append(dict(bin=b,lower=b/10,upper=(b+1)/10,n=int(mask.sum()),mean_probability=float(pred[mask].mean()) if mask.any() else None,observed_fraction=float(self.y[ev][mask].mean()) if mask.any() else None))
        self.save('tables',f'reliability_{split}.csv',rel)
        fig,ax=plt.subplots();rr=pd.DataFrame(rel).dropna();ax.plot(rr.mean_probability,rr.observed_fraction,'o-');ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Mean predicted probability',ylabel='Observed anomaly fraction',title=f'{split}: sigmoid (bin counts in CSV)');fig.tight_layout();fig.savefig(self.out/'figures'/f'reliability_{split}.png',dpi=160);plt.close(fig)
        self.save('predictions',f'probability_{split}.csv',dict(row_id=self.df.row_id.iloc[ev],label=self.y[ev],score=s[ev],probability=pred))
        return rows[0]
    def error_analysis(self,key,split,target):
        ids=self.splits[split];m=self.meta[key];s=self.op[key];th=threshold(s[self.caln],target);pred=s[ids]>th
        frame=self.df.loc[ids].copy();frame['score']=s[ids];frame['threshold']=th;frame['prediction']=pred.astype(int)
        frame['error']=np.where((self.y[ids]==0)&pred,'FP',np.where((self.y[ids]==1)&~pred,'FN','correct'))
        feature=m['feature'];fallback=~np.isfinite(self.native[key][ids]);frame['fallback']=fallback
        fallbackkey=self.fallback if key.startswith('P') else 'I0_W1_ensemble' if m['seed']=='ensemble' else f'I0_W1_S{m["seed"]}'
        frame['used_model']=np.where(fallback,fallbackkey,key)
        start=self.starts[feature][ids].copy();start[fallback]=ids[fallback]
        frame['window_start']=self.df.TimeStamp.to_numpy()[start];frame['window_end']=frame.TimeStamp
        frame['window_rows']=ids-start+1;frame['elapsed_seconds']=(frame.TimeStamp.to_numpy()-frame.window_start.to_numpy())/np.timedelta64(1,'s')
        frame['observation_from_burst_start_seconds']=self.df.groupby('burst_id').TimeStamp.transform(lambda g:(g-g.iloc[0]).dt.total_seconds()).iloc[ids].to_numpy()
        for col in ['Q','T2','QT']:frame[col]=np.nan
        if key.startswith('P'):
            for usekey,mask in [(key,~fallback),(self.fallback,fallback)]:
                if not mask.any():continue
                base=self.meta[usekey]['base'];raw=self.rawscores[base];b=self.bundles[base];pos=ids[mask]
                for col in ['Q','T2','QT']:
                    if col in raw:frame.loc[pos,col]=raw[col][pos]
                values=score(b,self.x(b['feature'],pos,b['sensors']));contrib=values['contributions'];kept=np.asarray(b['names'])[b['keep']]
                assert np.allclose(contrib.sum(1),values['Q'])
                for sensor in SENSORS:
                    cols=[i for i,name in enumerate(kept) if name.startswith(sensor)]
                    frame.loc[pos,sensor+'__Q_contribution']=contrib[:,cols].sum(1)
                for i,name in enumerate(kept):frame.loc[pos,'Q_feature__'+name]=contrib[:,i]
        tag=f'{key}_target{target:g}'.replace('.', 'p')
        frame.to_csv(self.out/'predictions'/f'operational_{tag}_{split}.csv',index=False)
        frame[frame.error!='correct'].to_csv(self.out/'predictions'/f'errors_{tag}_{split}.csv',index=False)
        conditions=[];train=self.df.loc[self.splits['train']];w=m['window']
        for cond,mask in [('all',np.ones(len(frame),bool)),('burst_start',frame.burst_pos<w),('after_start',frame.burst_pos>=w),('short_burst',frame.burst_length<w),('long_burst',frame.burst_length>=w),('fallback',frame.fallback),('main_model',~frame.fallback)]:
            conditions.append((cond,np.asarray(mask)))
        definitions={}
        for sensor in SENSORS:
            cuts=np.unique(np.quantile(np.abs(train[sensor]),[.25,.5,.75]));definitions[sensor]=cuts.tolist();bins=np.searchsorted(cuts,np.abs(frame[sensor]),side='right')
            for j in range(len(cuts)+1):conditions.append((f'{sensor}_magnitude_bin{j}',bins==j))
        change=self.df.groupby(['source_file','split','burst_id'],sort=False).AI2_Current.diff().abs()
        cuts=np.unique(np.quantile(change.loc[self.splits['train']].dropna(),[.25,.5,.75]));definitions['absolute_current_change']=cuts.tolist()
        values=change.loc[ids].to_numpy();bins=np.searchsorted(cuts,values,side='right')
        conditions.append(('current_change_unavailable',~np.isfinite(values)))
        for j in range(len(cuts)+1):conditions.append((f'current_change_bin{j}',(bins==j)&np.isfinite(values)))
        groups=[]
        for name,mask in conditions:
            g=frame.loc[mask];groups.append(dict(key=key,condition=name,total=len(g),normal=int((g.label==0).sum()),anomaly=int((g.label==1).sum()),FP=int((g.error=='FP').sum()),FN=int((g.error=='FN').sum()),FPR=float((g.error=='FP').sum()/(g.label==0).sum()) if (g.label==0).any() else None,FNR=float((g.error=='FN').sum()/(g.label==1).sum()) if (g.label==1).any() else None))
        self.save('tables',f'error_conditions_{tag}_{split}.csv',groups);js(self.out/'manifests/error_bin_definitions.json',definitions)
        return frame
    def final(self):
        lock=json.loads((self.out/'selection_lock.json').read_text());assert not lock['test_scores_seen']
        assert lock['protocol_sha256']==sha(ROOT/'configs/protocol.json')
        assert lock['rows_sha256']==sha(ROOT/'configs/first_experiment/frozen_rows.csv')
        for name,h in lock['model_hashes'].items(): assert sha(self.out/'models'/name)==h
        gate=self.out/'manifests/final_evaluation_started.json'
        assert not gate.exists(),'Final evaluation already started: investigate explicitly instead of automatic reevaluation'
        js(gate,dict(started=now(),lock_sha256=sha(self.out/'selection_lock.json'),split_match=lock['rows_sha256']==sha(ROOT/'configs/first_experiment/frozen_rows.csv')))
        self.load('test');self.evaluate('test')
        choices={}
        for name in ['primary','alternative','if_primary']:
            c=lock[name];result=self.measure(c['key'],'test','operational',c['target_fpr']);choices[name]=result
            self.error_analysis(c['key'],'test',c['target_fpr']);self.plot(c['key'],'test',c['target_fpr'],name)
        self.post_analysis(lock['primary']['key'],'test',lock['primary']['target_fpr']);self.prob_analysis(lock['primary']['key'],'test')
        best=[]
        for c in lock['best_by_family']:best.append(dict(**self.measure(c['key'],'test','operational',.01),selection_status=c['selection_status']))
        self.save('tables','best_by_family_test.csv',best)
        # Persist continuous native/operational predictions for every candidate, including each IF seed.
        for split in ['calibration','selection','test']:
            ids=self.splits[split];a={k:v[ids] for k,v in self.native.items()};b={k:v[ids] for k,v in self.op.items()}
            pd.DataFrame({'row_id':self.df.row_id.iloc[ids].to_numpy(),**a}).to_csv(self.out/'predictions'/f'all_native_{split}.csv.gz',index=False)
            pd.DataFrame({'row_id':self.df.row_id.iloc[ids].to_numpy(),**b}).to_csv(self.out/'predictions'/f'all_operational_{split}.csv.gz',index=False)
        js(self.out/'final_summary.json',dict(completed=now(),choices=choices,locked_postprocess=lock['postprocess'],no_retuning=True))
        print('FINAL',json.dumps(choices,ensure_ascii=False),flush=True)
    def plot(self,key,split,target,tag):
        ids=self.splits[split];s=self.op[key][ids];y=self.y[ids];th=threshold(self.op[key][self.caln],target)
        precision,recall,_=precision_recall_curve(y,s);fpr,tpr,_=roc_curve(y,s)
        fig,axs=plt.subplots(1,3,figsize=(13,3.5));ConfusionMatrixDisplay.from_predictions(y,s>th,labels=[0,1],ax=axs[0],colorbar=False)
        axs[1].plot(recall,precision);axs[1].set(xlabel='Recall',ylabel='Precision',title='PR (continuous scores)')
        axs[2].plot(fpr,tpr);axs[2].set(xlabel='FPR',ylabel='TPR',title='ROC (continuous scores)')
        fig.suptitle(f'{key}: {split}');fig.tight_layout();fig.savefig(self.out/'figures'/f'{tag}_{split}.png',dpi=160);plt.close(fig)
        # Scatter only: never connect long observation gaps.
        fig,axs=plt.subplots(2,1,figsize=(11,5))
        for label,ax in enumerate(axs):
            mask=y==label;ax.scatter(self.df.TimeStamp.iloc[ids[mask]],s[mask],s=3);ax.axhline(th,color='red',linestyle='--');ax.set(ylabel='Q95-scaled score',title=f'label={label}')
        fig.tight_layout();fig.savefig(self.out/'figures'/f'{tag}_timeline_{split}.png',dpi=160);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--stage',choices=['prepare','develop','final','all'],default='all');args=parser.parse_args()
    with threadpool_limits(limits=2):
        e=Experiment(args.out)
        if args.stage in ['develop','all']:e.develop()
        if args.stage in ['final','all']:e.final()
if __name__=='__main__':main()

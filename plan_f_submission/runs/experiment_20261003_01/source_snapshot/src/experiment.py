"""Fixed-design CPU experiment; --stage develop never scores final evaluation rows."""
import argparse, hashlib, itertools, json, os, platform, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn, scipy, torch, matplotlib
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss
from models import SENSORS, FEATURES, features, train_if, train_ae, score, norm_fit, norm_apply, seed_all
from evaluation import metrics, threshold, postprocess, alarm_summary

ROOT=Path(__file__).resolve().parents[1]

def js(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

class Experiment:
    def __init__(self,out):
        seed_all(42)
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.protocol=json.loads((ROOT/'configs/protocol.json').read_text())
        for p in ['tables','predictions','models','figures','manifests','logs']: (self.out/p).mkdir(exist_ok=True)
        self.df=pd.read_csv(ROOT/'configs/frozen/rows.csv',keep_default_na=False)
        self.df.TimeStamp=pd.to_datetime(self.df.TimeStamp)
        self.raw=self.df[SENSORS].to_numpy(dtype=np.float64); self.y=self.df.label.to_numpy()
        assert set(self.df[self.df.split=='train'].label)=={0}
        assert self.df.row_id.is_unique
        assert self.df.groupby('burst_id').split.nunique().max()==1
        self.candidates=[('M2',1)]+[(m,w) for w in [5,10,20] for m in ['M0','M1','M3','M4']]
        self.windows={}; self.scores={}; self.bundles={}; self.timings=[]
        self.splits={s:self.df.index[self.df.split==s].to_numpy() for s in ['train','calibration','selection','test']}
        audit=json.loads((ROOT/'configs/frozen/audit.json').read_text())
        for a in audit['files']: assert sha(ROOT/'data'/a['file'])==a['sha256']
        manifest=dict(protocol_sha256=sha(ROOT/'configs/protocol.json'),rows_sha256=sha(ROOT/'configs/frozen/rows.csv'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').glob('*.py')},python=sys.version,platform=platform.platform(),versions={m.__name__:m.__version__ for m in [np,pd,sklearn,scipy,torch,matplotlib,joblib]},seeds=[42,43,44],device='CPU',threads=2)
        mp=self.out/'manifests/environment.json'
        if mp.exists():
            old=json.loads(mp.read_text()); assert old['rows_sha256']==manifest['rows_sha256'] and old['protocol_sha256']==manifest['protocol_sha256']
        else:
            js(mp,manifest); shutil.copytree(ROOT/'configs',self.out/'manifests/configs',dirs_exist_ok=True)
        self.build_windows()

    def key(self,m,w): return f'{m}_L{w}'

    def build_windows(self):
        coverage=[]
        for m,w in self.candidates:
            key=self.key(m,w); starts=np.full(len(self.df),-1,dtype=int); groups=['source_file','split','train_role']
            if m in ['M1','M4']: groups+=['burst_id']
            for _,g in self.df.groupby(groups,sort=False,dropna=False):
                ids=g.index.to_numpy()
                if len(ids)>=w: starts[ids[w-1:]]=ids[:len(ids)-w+1]
            end=np.flatnonzero(starts>=0)
            assert np.all(end-starts[end]==w-1)
            assert np.all(self.df.split.to_numpy()[end]==self.df.split.to_numpy()[starts[end]])
            assert np.all(self.df.source_file.to_numpy()[end]==self.df.source_file.to_numpy()[starts[end]])
            if m in ['M1','M4']: assert np.all(self.df.burst_id.to_numpy()[end]==self.df.burst_id.to_numpy()[starts[end]])
            self.windows[key]=starts
            meta=self.df[['row_id','source_file','source_row','TimeStamp','label','split','train_role','burst_id','burst_pos','burst_length']].copy()
            meta['eligible']=starts>=0; meta['window_rows']=w; meta['window_start_row_id']='';meta['elapsed_seconds']=np.nan;meta['crosses_burst']=False
            meta.loc[end,'window_start_row_id']=self.df.row_id.to_numpy()[starts[end]]
            meta.loc[end,'elapsed_seconds']=(self.df.TimeStamp.to_numpy()[end]-self.df.TimeStamp.to_numpy()[starts[end]])/np.timedelta64(1,'s')
            meta.loc[end,'crosses_burst']=self.df.burst_id.to_numpy()[end]!=self.df.burst_id.to_numpy()[starts[end]]
            meta['reason']=np.where(starts>=0,'available','insufficient past rows in partition/burst')
            meta.to_csv(self.out/'manifests'/f'rows_{key}.csv',index=False)
            for (split,label),g in meta.groupby(['split','label']):
                coverage.append(dict(key=key,split=split,label=label,total=len(g),available=int(g.eligible.sum()),excluded=int((~g.eligible).sum()),coverage=float(g.eligible.mean()),crossing_windows=int(g.crosses_burst.sum()),median_elapsed_seconds=float(g.elapsed_seconds.median()),max_elapsed_seconds=float(g.elapsed_seconds.max())))
        self.common={}
        for split,ids in self.splits.items():
            self.common[split]=ids[np.all(np.stack([self.windows[k][ids]>=0 for k in self.windows]),axis=0)]
            self.df.loc[self.common[split],['row_id','label','split']].to_csv(self.out/'manifests'/f'common_{split}.csv',index=False)
        pd.DataFrame(coverage).to_csv(self.out/'tables/coverage.csv',index=False)
        # Explicit same-length pair intersections for matched burst comparisons.
        for w in [5,10,20]:
            for split,ids in self.splits.items():
                ids=ids[self.windows[f'M4_L{w}'][ids]>=0]
                self.df.loc[ids,['row_id','label','split']].to_csv(self.out/'manifests'/f'common_L{w}_{split}.csv',index=False)

    def x(self,key,ids):
        m=key.split('_')[0]; w=int(key.split('_L')[1]); starts=self.windows[key][ids]
        assert np.all(starts>=0)
        a=self.raw[starts[:,None]+np.arange(w)]
        return a[:,0,:] if m=='M2' else features(a) if m in ['M3','M4'] else a

    def fit_one(self,key,seed,match_key=None):
        name=f'{key}_s{seed}'+('_matched' if match_key else '')
        p=self.out/'models'/f'{name}.joblib'; timing=self.out/'logs'/f'{name}.json'
        if p.exists() and timing.exists():
            self.timings.append(json.loads(timing.read_text())); return joblib.load(p)
        ids=self.splits['train']; ids=ids[self.windows[key][ids]>=0]
        isae=key.startswith(('M0','M1'))
        if isae:
            train=ids[self.df.loc[ids,'train_role'].to_numpy()=='fit']; early=ids[self.df.loc[ids,'train_role'].to_numpy()=='early_stop']
        else: train=ids; early=np.array([],dtype=int)
        if match_key:
            rng=np.random.default_rng(seed)
            train=np.sort(rng.choice(train,sum(self.windows[match_key][train]>=0),replace=False))
            if isae: early=np.sort(rng.choice(early,sum(self.windows[match_key][early]>=0),replace=False))
        seed_all(seed); tic=time.perf_counter()
        if isae:
            sr=self.raw[(self.df.split=='train')&(self.df.train_role=='fit')]
            bundle=train_ae(self.x(key,train),self.x(key,early),sr,seed,self.protocol['lstm'])
        else: bundle=train_if(self.x(key,train),seed)
        t=time.perf_counter()-tic
        bundle.update(key=key,seed=seed,feature_names=SENSORS if key.startswith(('M0','M1','M2')) else FEATURES)
        joblib.dump(bundle,p,compress=3)
        record=dict(key=key,seed=seed,matched=bool(match_key),fit_seconds=t,training_windows=len(train),early_stop_windows=len(early),epochs=len(bundle.get('history',[])))
        js(timing,record); self.timings.append(record)
        self.df.loc[train,['row_id','source_row','split','burst_id']].to_csv(self.out/'manifests'/f'train_{name}.csv',index=False)
        if isae:
            pd.DataFrame(bundle['history']).to_csv(self.out/'logs'/f'history_{name}.csv',index=False)
            self.df.loc[early,['row_id','source_row','split','burst_id']].to_csv(self.out/'manifests'/f'early_{name}.csv',index=False)
        print('FIT',name,record,flush=True)
        return bundle

    def raw_scores(self,key,seed,stage,bundle=None,suffix=''):
        p=self.out/'predictions'/f'raw_{key}_s{seed}_{stage}{suffix}.npz'
        if p.exists(): return np.load(p)['scores']
        if stage=='test': assert (self.out/'selection_lock.json').exists(), 'Final scores gated until selection is locked'
        splits=['calibration','selection'] if stage=='develop' else ['test']
        ids=np.concatenate([self.splits[s] for s in splits]); ids=ids[self.windows[key][ids]>=0]
        out=np.full(len(self.df),np.nan); bundle=bundle or self.fit_one(key,seed)
        tic=time.perf_counter(); out[ids]=score(bundle,self.x(key,ids)); duration=time.perf_counter()-tic
        assert np.isfinite(out[ids]).all()
        np.savez_compressed(p,scores=out)
        js(self.out/'logs'/f'score_{key}_s{seed}_{stage}{suffix}.json',dict(rows=len(ids),seconds=duration,seconds_per_row=duration/len(ids)))
        return out

    def develop_models(self):
        for m,w in self.candidates:
            key=self.key(m,w)
            for seed in [42,43,44]:
                bundle=self.fit_one(key,seed)
                self.raw_scores(key,seed,'develop',bundle)
        pd.DataFrame(self.timings).drop_duplicates(['key','seed','matched']).to_csv(self.out/'tables/training_budget.csv',index=False)

    def load_scores(self,stage):
        scores={}; norms={}
        normalcal=self.splits['calibration']; normalcal=normalcal[self.y[normalcal]==0]
        for m,w in self.candidates:
            key=self.key(m,w); scores[key]=[]; norms[key]=[]
            for seed in [42,43,44]:
                dev=self.raw_scores(key,seed,'develop')
                path=self.out/'models'/f'norm_{key}_s{seed}.joblib'
                if path.exists(): norm=joblib.load(path)
                else:
                    norm=norm_fit(dev[normalcal][np.isfinite(dev[normalcal])]); joblib.dump(norm,path)
                a=dev.copy()
                if stage=='test':
                    test=self.raw_scores(key,seed,'test'); a[self.splits['test']]=test[self.splits['test']]
                scores[key].append(a); norms[key].append(norm)
        self.raws=scores; self.norms=norms
        self.op={}; self.native={}
        for key,arrs in scores.items():
            ns=[]; ops=[]
            for j,a in enumerate(arrs):
                transformed=norm_apply(norms[key][j],a)
                fallback=norm_apply(norms['M2_L1'][j],scores['M2_L1'][j])
                ns.append(transformed); ops.append(np.where(np.isfinite(transformed),transformed,fallback))
            self.native[key]=ns+[np.mean(ns,axis=0)]
            self.op[key]=ops+[np.mean(ops,axis=0)]

    def evaluate(self,split):
        records=[]; ids=self.splits[split]; cal=self.splits['calibration']; caln=cal[self.y[cal]==0]
        for m,w in self.candidates:
            key=self.key(m,w)
            for j,seed in enumerate([42,43,44,'ensemble']):
                for scope in ['native','common','operational']:
                    s=self.op[key][j] if scope=='operational' else self.native[key][j]
                    ev=ids if scope=='operational' else self.common[split] if scope=='common' else ids[np.isfinite(s[ids])]
                    cn=caln[np.isfinite(s[caln])]
                    for target in [.005,.01,.02,.05]:
                        th=threshold(s[cn],target); p=s[ev]>th
                        records.append(dict(key=key,model=m,window=w,seed=seed,split=split,scope=scope,target_fpr=target,threshold=th,calibration_normal_n=len(cn),calibration_observed_fpr=float(np.mean(s[cn]>th)),**metrics(self.y[ev],s[ev],p)))
        results=pd.DataFrame(records); results.to_csv(self.out/'tables'/f'metrics_{split}.csv',index=False)
        for scope in ['common','operational','native']:
            sub=results[(results.scope==scope)&(results.seed!='ensemble')]
            sub.groupby(['key','target_fpr'])[['Recall','Precision','F1','F2','FPR','AP','PR_AUC_trapezoid','ROC_AUC','Accuracy']].agg(['mean','std']).to_csv(self.out/'tables'/f'seed_mean_std_{split}_{scope}.csv')
        # Matched endpoint set, matched L: do not rank different native coverages.
        pairs=[]
        for a,b in [('M0','M1'),('M3','M4')]:
            for w in [5,10,20]:
                ev=ids[self.windows[f'{b}_L{w}'][ids]>=0]
                for j,seed in enumerate([42,43,44,'ensemble']):
                    for target in [.005,.01,.02,.05]:
                        r=dict(pair=a+'/'+b,window=w,seed=seed,target_fpr=target,N=len(ev))
                        for m in [a,b]:
                            key=f'{m}_L{w}'; s=self.native[key][j]; cn=caln[np.isfinite(s[caln])]; th=threshold(s[cn],target)
                            r.update({m+'_'+k:v for k,v in metrics(self.y[ev],s[ev],s[ev]>th).items()})
                        for metric in ['Recall','F1','FPR','AP']: r['delta_'+metric]=r[b+'_'+metric]-r[a+'_'+metric]
                        pairs.append(r)
        pd.DataFrame(pairs).to_csv(self.out/'tables'/f'burst_pairs_{split}.csv',index=False)
        baselines=[]
        for scope,ev in [('operational',ids),('common',self.common[split])]:
            for p in [0,1]: baselines.append(dict(scope=scope,baseline='all_normal' if p==0 else 'all_anomaly',**metrics(self.y[ev],np.full(len(ev),p),np.full(len(ev),p))))
        pd.DataFrame(baselines).to_csv(self.out/'tables'/f'dummy_{split}.csv',index=False)
        return results

    def post_table(self,split):
        ids=self.splits[split]; cal=self.splits['calibration']; caln=cal[self.y[cal]==0]; records=[]
        for key in self.op:
            s=self.op[key][-1]
            for target in [.005,.01,.02,.05]:
                th=threshold(s[caln],target); raw=s[ids]>th
                for policy in ['none','consecutive2','two_of_three']:
                    p=postprocess(self.df.loc[ids],raw,policy); alarms=alarm_summary(self.df.loc[ids],p)
                    normal=alarms[alarms.label==0]; anomaly=alarms[alarms.label==1]
                    records.append(dict(key=key,target_fpr=target,threshold=th,policy=policy,**metrics(self.y[ids],s[ids],p),alarm_episodes=int(alarms.alarm_episodes.sum()),normal_alarm_episodes=int(normal.alarm_episodes.sum()),duplicate_episodes=int(alarms.duplicate_episodes.sum()),anomaly_bursts_without_alarm=int(anomaly.no_alarm.sum()),observed_start_to_first_alarm_median_seconds=anomaly.first_alarm_since_observed_burst_start_seconds.median()))
        result=pd.DataFrame(records); result.to_csv(self.out/'tables'/f'postprocessing_{split}.csv',index=False)
        return result

    def choose(self,post):
        runtime={}
        for key in self.op:
            runtime[key]=sum(json.loads((self.out/'logs'/f'{key}_s{s}.json').read_text())['fit_seconds'] for s in [42,43,44])
            runtime[key]+=sum(json.loads((self.out/'logs'/f'score_{key}_s{s}_develop.json').read_text())['seconds'] for s in [42,43,44])
        post=post.copy(); post['runtime']=post.key.map(runtime)
        eligible=post[post.FPR<=.01]; compliant=len(eligible)>0
        main=eligible.sort_values(['Recall','F1','AP','runtime','key','policy','target_fpr'],ascending=[False,False,False,True,True,True,True]).iloc[0] if compliant else post.sort_values(['FPR','Recall','F1','key'],ascending=[True,False,False,True]).iloc[0]
        alt=post.sort_values(['F1','Recall','AP','runtime','key','policy','target_fpr'],ascending=[False,False,False,True,True,True,True]).iloc[0]
        return dict(created_utc=datetime.now(timezone.utc).isoformat(),compliant=bool(compliant),main=main.to_dict(),f1_alternative=alt.to_dict(),test_used=False,protocol_sha256=sha(ROOT/'configs/protocol.json'),rows_sha256=sha(ROOT/'configs/frozen/rows.csv'))

    def auxiliary(self):
        """Bounded development-only explanatory experiments; excluded from final selection."""
        cal=self.splits['calibration']; ids=self.splits['selection']; records=[]; saved={}
        combos=[list(c) for n in [1,2,3] for c in itertools.combinations(range(3),n)]
        for cols in combos:
            key='M2_sensors_'+''.join(map(str,cols)); scores=[]
            for seed in [42,43,44]:
                path=self.out/'models'/f'{key}_s{seed}.joblib'
                if path.exists(): bundle=joblib.load(path)
                else: bundle=train_if(self.raw[self.splits['train']][:,cols],seed); joblib.dump(bundle,path,compress=3)
                s=np.full(len(self.df),np.nan); ev=np.r_[cal,ids]; s[ev]=score(bundle,self.raw[ev][:,cols]); scores.append(s)
                th=threshold(s[cal[self.y[cal]==0]],.01)
                records.append(dict(experiment='sensor_combination',key=key,seed=seed,removed_feature='',**metrics(self.y[ids],s[ids],s[ids]>th)))
            saved[key]=scores
        pd.DataFrame(records).to_csv(self.out/'tables/sensor_combinations.csv',index=False)
        # Matched per-seed detections where full combination finds anomalies missed by a singleton.
        examples=[]
        for j,seed in enumerate([42,43,44]):
            full=saved['M2_sensors_012'][j]; th=threshold(full[cal[self.y[cal]==0]],.01)
            for single in [0,1,2]:
                s=saved[f'M2_sensors_{single}'][j]; sth=threshold(s[cal[self.y[cal]==0]],.01)
                for i in ids[(self.y[ids]==1)&(full[ids]>th)&(s[ids]<=sth)]: examples.append(dict(row_id=self.df.row_id.iloc[i],seed=seed,single_sensor=single,full_score=full[i],full_threshold=th,single_score=s[i],single_threshold=sth))
        pd.DataFrame(examples,columns=['row_id','seed','single_sensor','full_score','full_threshold','single_score','single_threshold']).to_csv(self.out/'tables/sensor_combination_rescues.csv',index=False)
        records=[]; key='M4_L10'; train=self.splits['train']; train=train[self.windows[key][train]>=0]
        ev=ids[self.windows[key][ids]>=0]; cn=cal[(self.y[cal]==0)&(self.windows[key][cal]>=0)]
        xtrain=self.x(key,train); xev=self.x(key,ev); xcal=self.x(key,cn)
        for drop in range(-1,15):
            keep=[i for i in range(15) if i!=drop]
            for seed in [42,43,44]:
                path=self.out/'models'/f'ablation_M4_L10_drop{drop}_s{seed}.joblib'
                if path.exists(): bundle=joblib.load(path)
                else: bundle=train_if(xtrain[:,keep],seed); joblib.dump(bundle,path,compress=3)
                s=score(bundle,xev[:,keep]); sc=score(bundle,xcal[:,keep]); th=threshold(sc,.01)
                records.append(dict(removed_feature='none' if drop==-1 else FEATURES[drop],seed=seed,**metrics(self.y[ev],s,s>th)))
        ab=pd.DataFrame(records); base=ab[ab.removed_feature=='none'].set_index('seed')
        for metric in ['Recall','F1','FPR','AP']: ab['delta_'+metric]=ab[metric]-ab.seed.map(base[metric])
        ab.to_csv(self.out/'tables/feature_ablation.csv',index=False)
        # Trigger sample-count control only on development improvement, never final results.
        pairs=pd.read_csv(self.out/'tables/burst_pairs_selection.csv'); matched=[]
        for a,b in [('M0','M1'),('M3','M4')]:
            for w in [5,10,20]:
                check=pairs[(pairs.pair==a+'/'+b)&(pairs.window==w)&(pairs.seed=='ensemble')&(pairs.target_fpr==.01)].iloc[0]
                if not(check.delta_F1>0 or check.delta_Recall>0): continue
                key=f'{a}_L{w}'; aware=f'{b}_L{w}'; ev=ids[self.windows[aware][ids]>=0]
                for seed in [42,43,44]:
                    bundle=self.fit_one(key,seed,match_key=aware); s=self.raw_scores(key,seed,'develop',bundle,suffix='_matched')
                    cn=cal[(self.y[cal]==0)&np.isfinite(s[cal])]; th=threshold(s[cn],.01)
                    matched.append(dict(key=key,aware=aware,seed=seed,**metrics(self.y[ev],s[ev],s[ev]>th)))
        pd.DataFrame(matched,columns=['key','aware','seed']+list(metrics([0,1],[0,1],[0,1]))).to_csv(self.out/'tables/matched_training_count.csv',index=False)
        print('Development auxiliary experiments complete',flush=True)

    def probabilities(self,lock,split):
        cal=self.splits['calibration']; ids=self.splits[split]; records=[]
        for role in ['main','f1_alternative']:
            choice=lock[role]; key=choice['key']; s=self.op[key][-1]; path=self.out/'models'/f'probability_{role}.joblib'
            if path.exists(): obj=joblib.load(path)
            else:
                center=float(np.mean(s[cal])); scale=max(float(np.std(s[cal])),1e-8); x=(s[cal]-center)/scale; y=self.y[cal]
                def fun(ab):
                    z=ab[0]*x+ab[1]
                    return np.mean(np.logaddexp(0,z)-y*z)+0.5*ab[0]**2/len(x)
                fit=minimize(fun,[1.,-2.],method='L-BFGS-B',bounds=[(0,None),(None,None)])
                assert fit.success,fit.message
                obj=dict(a=float(fit.x[0]),b=float(fit.x[1]),center=center,scale=scale,normal_n=int(sum(y==0)),anomaly_n=int(sum(y==1)),calibration_prevalence=float(np.mean(y)),fit_success=bool(fit.success)); joblib.dump(obj,path)
            p=expit(obj['a']*(s[ids]-obj['center'])/obj['scale']+obj['b'])
            mapped=float(expit(obj['a']*(choice['threshold']-obj['center'])/obj['scale']+obj['b']))
            raw=s[ids]>choice['threshold']; mapped_p=p>mapped
            records.append(dict(role=role,key=key,split=split,normal_fit_n=obj['normal_n'],anomaly_fit_n=obj['anomaly_n'],slope=obj['a'],probability_threshold=mapped,judgement_disagreements=int(np.sum(raw!=mapped_p)),Brier=brier_score_loss(self.y[ids],p),log_loss=log_loss(self.y[ids],np.clip(p,1e-15,1-1e-15),labels=[0,1]),uncalibrated_sigmoid_proxy_Brier=brier_score_loss(self.y[ids],expit(s[ids])),uncalibrated_sigmoid_proxy_log_loss=log_loss(self.y[ids],expit(s[ids]),labels=[0,1]),constant_prevalence_Brier=brier_score_loss(self.y[ids],np.full(len(ids),obj['calibration_prevalence'])),constant_prevalence_log_loss=log_loss(self.y[ids],np.full(len(ids),obj['calibration_prevalence']),labels=[0,1])))
            pd.DataFrame(dict(row_id=self.df.row_id.iloc[ids],label=self.y[ids],score=s[ids],probability=p,raw_prediction=raw.astype(int),probability_prediction=mapped_p.astype(int))).to_csv(self.out/'predictions'/f'probability_{role}_{split}.csv',index=False)
        pd.DataFrame(records).to_csv(self.out/'tables'/f'probability_{split}.csv',index=False)

    def selected_predictions(self,lock,split):
        ids=self.splits[split]; records=[]
        for role in ['main','f1_alternative']:
            c=lock[role]; key=c['key']; w=int(key.split('_L')[1]); s=self.op[key][-1]
            pred=postprocess(self.df.loc[ids],s[ids]>c['threshold'],c['policy'])
            p=self.df.loc[ids].copy(); starts=self.windows[key][ids]; fallback=starts<0; starts=np.where(fallback,ids,starts)
            p['window_start_row_id']=self.df.row_id.to_numpy()[starts]; p['window_start_time']=self.df.TimeStamp.to_numpy()[starts]
            p['window_rows']=np.where(fallback,1,w); p['window_elapsed_seconds']=(p.TimeStamp.to_numpy()-p.window_start_time.to_numpy())/np.timedelta64(1,'s')
            p['actual_model']=np.where(fallback,'M2_L1',key); p['fallback']=fallback
            p['score']=s[ids]; p['threshold']=c['threshold']; p['raw_prediction']=(s[ids]>c['threshold']).astype(int); p['prediction']=pred; p['policy']=c['policy']
            p['error']=np.where((p.label==0)&(pred==1),'FP',np.where((p.label==1)&(pred==0),'FN','correct'))
            p.to_csv(self.out/'predictions'/f'{role}_{split}.csv',index=False)
            p[p.error!='correct'].to_csv(self.out/'predictions'/f'{role}_{split}_FP_FN.csv',index=False)
            alarm_summary(p,pred).to_csv(self.out/'tables'/f'alarms_{role}_{split}.csv',index=False)
            records.append(dict(role=role,key=key,policy=c['policy'],threshold=c['threshold'],**metrics(self.y[ids],s[ids],pred)))
        pd.DataFrame(records).to_csv(self.out/'tables'/f'selected_{split}.csv',index=False)
        # Every model/seed/row, all four fixed thresholds, with provenance and eligibility.
        for key in self.op:
            starts=self.windows[key][ids]; fallback=starts<0; effective=np.where(fallback,ids,starts)
            cal=self.splits['calibration']; caln=cal[self.y[cal]==0]
            for j,seed in enumerate([42,43,44,'ensemble']):
                p=self.df.loc[ids].copy(); p['native_available']=~fallback; p['actual_model']=np.where(fallback,'M2_L1',key)
                p['window_start_row_id']=self.df.row_id.to_numpy()[effective]; p['window_start_time']=self.df.TimeStamp.to_numpy()[effective]
                p['window_rows']=np.where(fallback,1,int(key.split('_L')[1])); p['window_elapsed_seconds']=(p.TimeStamp.to_numpy()-p.window_start_time.to_numpy())/np.timedelta64(1,'s')
                p['score']=self.op[key][j][ids]; p['native_score']=self.native[key][j][ids]
                for target in [.005,.01,.02,.05]:
                    th=threshold(self.op[key][j][caln],target); tag=str(target)
                    p['threshold_'+tag]=th; p['prediction_'+tag]=(p.score>th).astype(int)
                p.to_csv(self.out/'predictions'/f'all_{key}_s{seed}_{split}.csv.gz',index=False)

    def run(self,stage):
        if stage in ['develop','all']:
            self.develop_models(); self.load_scores('develop'); self.evaluate('selection'); post=self.post_table('selection')
            lockpath=self.out/'selection_lock.json'
            if not lockpath.exists():
                self.auxiliary(); lock=self.choose(post); js(lockpath,lock)
            else: lock=json.loads(lockpath.read_text())
            self.probabilities(lock,'selection'); self.selected_predictions(lock,'selection')
            print('LOCKED SELECTION',json.dumps(lock,ensure_ascii=False),flush=True)
        if stage in ['final','all']:
            lock=json.loads((self.out/'selection_lock.json').read_text())
            assert lock['protocol_sha256']==sha(ROOT/'configs/protocol.json')
            self.load_scores('test'); self.evaluate('test'); self.post_table('test'); self.probabilities(lock,'test'); self.selected_predictions(lock,'test')
            js(self.out/'final_complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),lock_sha256=sha(self.out/'selection_lock.json'),no_retuning=True))
            print('FINAL COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); parser.add_argument('--stage',choices=['develop','final','all'],default='all'); args=parser.parse_args()
    Experiment(args.out).run(args.stage)

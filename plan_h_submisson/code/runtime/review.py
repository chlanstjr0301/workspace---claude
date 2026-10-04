"""Frozen inference, post-hoc controls and observational event analysis. NO fit or selection."""
from v2 import *
from datetime import datetime,timezone
import argparse

def savecsv(d,p):
 p=R/p;p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False)

def reportmetrics(f):
 m=measure(f);m['normal_n']=int((f.label==0).sum());m['anomaly_n']=int((f.label==1).sum())
 if not m['anomaly_n']:
  for k in ['Recall','FNR','F1','F2','Precision']:m[k]=None
 return m

def slices(d,p):
 b={k:np.array(v) for k,v in p['blocks'].items()};b['S']=np.sort(np.r_[b['S1'],b['S2']]);b['H']=d.index[d.split=='test'].to_numpy();return b

def frozen_prediction(b,s,c):
 o=b.copy()
 if c['id']=='B0':return o
 ex,g,h,p1,p2=gate(b,s,c['theta'],c['rule'])
 o['B0_prediction']=b.prediction;o['R5_score']=s;o['R5_available']=np.isfinite(s);o['aux_threshold']=c['theta'];o['h']=h;o['h_previous1']=p1;o['h_previous2']=p2;o['aux_alarm']=ex;o['prediction']=(b.prediction.astype(bool)|ex).astype(int);o['integrated_score']=np.maximum(b.score/b.threshold,g);o['rule']=c['rule'];o['alarm_time']=o.TimeStamp
 o['error']=np.select([(o.label==1)&(o.prediction==0),(o.label==0)&(o.prediction==1)],['FN','FP'],default='correct')
 assert np.array_equal(o.prediction,o.integrated_score>1);assert np.array_equal(h,np.isfinite(s)&(s>c['theta']))
 return o

def run():
 start=time.perf_counter();p=json.loads((R/'validation_protocol.json').read_text());assert sha(R/'validation_protocol.json')==(R/'validation_protocol.sha256').read_text().split()[0]
 for n,h in p['core_sources'].items():assert sha(R/'src'/n)==h,n
 for n,h in json.loads((R/'model_manifest.json').read_text())['hashes'].items():assert sha(R/'models'/n)==h,n
 d,e,b0,r5=context();blocks=slices(d,p);proof=[];frames={};tables=[];pairs=[];reuse=[]
 # Compute once per original partition; reporting-block boundaries do NOT reset state.
 for partition in ['selection','test']:
  ids=d.index[d.split==partition].to_numpy();b=base_predictions(e,b0,ids);s=aux_score(r5,e,b0,ids,b.score)
  for c in p['controls']:
   allf=frozen_prediction(b,s,c)
   for block,ii in blocks.items():
    if len(ii) and d.loc[ii[0],'split']==partition:
     f=allf.loc[ii].copy();frames[(block,c['id'])]=f;savecsv(f,f'predictions/{block}_{c["id"]}.csv')
  proof.append(dict(partition=partition,rows=len(ids),B0_reloaded=True,R5_reloaded=True))
 for block in ['S1','S2','S','V','H']:
  base=frames[block,'B0'];bm=measure(base)
  for c in p['controls']:
   cid=c['id'];f=frames[block,cid];pa=pair_counts(base,f);assert pa['new_FN']==0 and pa['removed_FP']==0
   m=reportmetrics(f);row=dict(block=block,model=cid,**m,**pa,delta_F1=m['F1']-bm['F1'],delta_F2=m['F2']-bm['F2'],delta_FPR=m['FPR']-bm['FPR'],**judge(bm,measure(f),'effect' if block in ['S','H'] else 'maintenance'))
   tables.append(row)
   for ref in ['B0','R5_A0','B_same_threshold']:
    z=frames[block,ref];chg=f.prediction.to_numpy()!=z.prediction.to_numpy()
    if chg.any():
     q=f.loc[chg].copy();q['block']=block;q['model']=cid;q['reference']=ref;q['reference_prediction']=z.loc[chg,'prediction'];q['change']=np.select([(q.label==1)&(q.prediction==1),(q.label==1)&(q.prediction==0),(q.label==0)&(q.prediction==1)],['new_TP','new_FN','additional_FP'],default='removed_FP');pairs.append(q)
 # Verify and reuse original configurations / predictions; no duplicate configuration search.
 expected=[('S','B0','B0_S_pooled'),('S','G1','G1_Q999_S_pooled'),('S','R5_A0','C0_S_pooled'),('S','B_same_threshold','G0_Q999_S_pooled'),('V','B0','B0_V'),('V','G1','G1_Q999_V'),('H','B0','B0_H'),('H','G1','G1_Q999_H')]
 for block,cid,oldname in expected:
  path=R/'inputs/expected'/f'{oldname}.csv';old=pd.read_csv(path);f=frames[block,cid]
  assert old.row_id.tolist()==f.row_id.tolist();assert np.array_equal(old.label,f.label);assert np.array_equal(old.prediction,f.prediction)
  for col in ['score','integrated_score','R5_score']:
   if col in old and col in f:assert np.allclose(pd.to_numeric(old[col]),f[col],rtol=1e-10,atol=1e-10,equal_nan=True)
  reuse.append(dict(block=block,model=cid,old_prediction_sha256=sha(path),rows=len(f),prediction_match=True,base_score_max_error=float(np.max(abs(old.score.to_numpy()-f.score.to_numpy())))))
 savecsv(pd.DataFrame(tables),'metrics.csv');savecsv(pd.concat(pairs,ignore_index=True),'paired_changes.csv');js(R/'reproduction.json',dict(exact_prediction_matches=reuse,model_reloading=proof))
 # Normal-only calibration audit, never fit anything.
 ci=d.index[(d.split=='calibration')&(d.label==0)].to_numpy();cb=base_predictions(e,b0,ci);cs=aux_score(r5,e,b0,ci,cb.score);valid=cs[np.isfinite(cs)];theta=float(np.quantile(valid,.999,method='higher'));assert theta==p['controls'][2]['theta']
 js(R/'calibration_audit.json',dict(total_normal=len(ci),available=len(valid),unavailable=len(ci)-len(valid),q=.999,method='higher',theta=theta,rank_zero_based=int(np.ceil(.999*(len(valid)-1))),ties=int((valid==theta).sum()),strict_exceedances=int((valid>theta).sum()),start=str(d.loc[ci,'TimeStamp'].min()),end=str(d.loc[ci,'TimeStamp'].max()),resolution_available=1/len(valid),source_row_ids=d.loc[ci,'row_id'].tolist(),old_threshold=cal_threshold(cb.prediction,cs,0)[0]))
 event_analysis(frames);timing_audit(d,e,b0,r5,frames);js(R/'run_status.json',dict(status='completed',seconds=time.perf_counter()-start,fit_calls=0,new_selection=False,finished=datetime.now(timezone.utc).isoformat()))
 print(pd.DataFrame(tables)[['block','model','TP','FN','FP','F1','F2','passes','failed_rules']].to_string(index=False),flush=True)

def event_analysis(frames):
 events=[];norm=[];descriptive=[];eventrows=[]
 for block in ['S1','S2','S','V','H']:
  ref=frames[block,'B0'];g1=frames[block,'G1']
  # Stored burst IDs are gap proxies. Additional label runs split within a burst.
  for _,bg in ref.groupby(['source_file','split','burst_id'],sort=False):
   seg=(bg.label!=bg.label.shift()).cumsum()
   for _,v in bg.groupby(seg,sort=False):
    if v.label.iloc[0]!=1:continue
    event=f'{block}:{v.burst_id.iloc[0]}:{v.source_row.iloc[0]}';o=dict(block=block,event_id=event,source_file=v.source_file.iloc[0],burst_id=v.burst_id.iloc[0],start=str(v.TimeStamp.iloc[0]),end=str(v.TimeStamp.iloc[-1]),rows=len(v),duration_seconds=(v.TimeStamp.iloc[-1]-v.TimeStamp.iloc[0]).total_seconds(),physical_onset_unknown=True,left_censored=bool(v.index[0]==bg.index[0]))
    for cid in ['B0','G1']:
     z=frames[block,cid].loc[v.index];a=z[z.prediction==1];o[cid+'_detected']=bool(len(a));o[cid+'_TP']=int(z.prediction.sum());o[cid+'_FN']=int((z.prediction==0).sum());o[cid+'_first_alarm']=str(a.TimeStamp.iloc[0]) if len(a) else None;o[cid+'_observed_start_delay_seconds']=(a.TimeStamp.iloc[0]-v.TimeStamp.iloc[0]).total_seconds() if len(a) else None
    o['new_whole_event']=not o['B0_detected'] and o['G1_detected'];o['added_detected_rows']=o['G1_TP']-o['B0_TP'];o['coverage_extended_already_detected']=o['B0_detected'] and o['added_detected_rows']>0;events.append(o)
    vv=g1.loc[v.index].copy();vv['event_id']=event;eventrows.append(vv)
  for cid in ['B0','G1','B_same_threshold','D_old_threshold_G1','R5_A0']:
   f=frames[block,cid];seconds=0.;episodes=0;alarm_bursts=0
   for _,bg in f.groupby(['source_file','split','burst_id'],sort=False):
    y=bg.label.to_numpy();a=(bg.prediction.to_numpy()==1)&(y==0);dt=bg.TimeStamp.diff().dt.total_seconds().to_numpy();normal_pairs=(y==0)&np.r_[False,y[:-1]==0];seconds+=float(np.nansum(np.where(normal_pairs&(dt<=.5)&(dt>=0),dt,0)));episodes+=int((a&~np.r_[False,a[:-1]]).sum());alarm_bursts+=int(a.any())
   fp=int(((f.label==0)&(f.prediction==1)).sum());norm.append(dict(block=block,model=cid,normal_n=int((f.label==0).sum()),FP_rows=fp,FP_episodes=episodes,normal_observed_span_seconds=seconds,FP_rows_per_observed_hour=fp*3600/seconds if seconds else None,FP_episodes_per_observed_hour=episodes*3600/seconds if seconds else None,normal_groups_with_alarm=alarm_bursts,exposure_is_verified_runtime=False))
  for cid in ['B0','G1']:
   f=frames[block,cid]
   for group_type,cols in [('date',['source_file']),('observation_group',['source_file','burst_id'])]:
    for key,z in f.groupby(cols,sort=False):
     b=ref.loc[z.index];pa=pair_counts(b,z);descriptive.append(dict(block=block,model=cid,group_type=group_type,group=str(key),acquisition_date=str(z.TimeStamp.iloc[0].date()),**reportmetrics(z),**pa,new_eligibility_rule=False))
 savecsv(pd.DataFrame(events),'events.csv');savecsv(pd.concat(eventrows,ignore_index=True),'event_row_predictions.csv');savecsv(pd.DataFrame(norm),'normal_exposure.csv');savecsv(pd.DataFrame(descriptive),'temporal_descriptive.csv')
 hh=frames['H','G1'];savecsv(hh[hh.error.isin(['FN','FP'])],'remaining_H_errors.csv');savecsv(hh[(hh.label==1)&(hh.prediction==1)&(hh.B0_prediction==0)],'added_H_detections.csv')
 ev=pd.DataFrame(events);h=ev[ev.block=='H'];js(R/'event_summary.json',dict(H_events=len(h),B0_detected_events=int(h.B0_detected.sum()),G1_detected_events=int(h.G1_detected.sum()),new_whole_events=int(h.new_whole_event.sum()),extended_events=int(h.coverage_extended_already_detected.sum()),added_detected_rows=int(h.added_detected_rows.sum()),physical_failure_count='unknown',left_censored_events=int(h.left_censored.sum())))

def timing_audit(d,e,b0,r5,frames):
 records=[];preds=[];rawstarts=e.starts_for(3,True);bstarts=e.starts_for(20);zs=r5['scaler'].transform(e.raw)
 for block in ['S','V','H']:
  f=frames[block,'G1']
  for i,row in f.iterrows():
   available=rawstarts[i]>=0;start=bstarts[i] if bstarts[i]>=0 else i;last=i-1 if available else None
   rec=dict(block=block,row_id=row.row_id,source_row=row.source_row,label=int(row.label),burst_id=row.burst_id,burst_pos=row.burst_pos,B0_raw_first_id=d.row_id.iloc[start],B0_raw_last_id=row.row_id,B0_input_last_time=str(row.TimeStamp),R5_available=bool(available),R5_predictor_last_time=str(d.TimeStamp.iloc[last]) if available else None,R5_target_time=str(row.TimeStamp),R5_residual_last_observed_time=str(row.TimeStamp) if available else None,earliest_alarm_time=str(row.TimeStamp),forecast_horizon_seconds=(row.TimeStamp-d.TimeStamp.iloc[last]).total_seconds() if available else None,B0_prediction=int(row.B0_prediction),G1_prediction=int(row.prediction),decision_backdated=False,computation_latency_not_in_timestamp=True)
   if available:
    u=np.r_[zs[i-1],zs[i-2]];zpred=(u-r5['u_mean'])@r5['coef']+r5['z_mean'];rawpred=zpred*r5['scaler'].scale_+r5['scaler'].mean_
    for k,sensor in enumerate(SENSORS):rec[sensor+'_prediction']=rawpred[k];rec[sensor+'_raw_residual']=e.raw[i,k]-rawpred[k]
   records.append(rec)
 savecsv(pd.DataFrame(records),'row_timing_and_residuals.csv')
 # Full original selection rather than split report blocks => avoids artificial warm-up.
 js(R/'timing_summary.json',dict(horizon='t-2,t-1 predict t; residual only available after t received',G1_dependency='current residual and previous residual: raw t,t-1,t-2,t-3 within continuity group',P1_dependency='last20 rows inside original partition, gap crossing allowed',report_block_reset=False,physical_session_metadata_present=False,first_2_R5_unavailable=True,no_earlier_alarm_claim=True))

if __name__=='__main__':run()

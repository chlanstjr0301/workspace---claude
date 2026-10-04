from v2 import *
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def main():
 lock=json.loads((R/'selected_config.json').read_text());sel=lock['selected'];blocks=['S_pooled']+[b for b in ['V','H'] if (R/f'{b}_evaluation.json').exists()];episodes=[];conditions=[];coverage=[];paired=[];differences=[];confirmation=[]
 for block in blocks:
  candidates=['B0','C0']+[c['id'] for c in json.loads((R/'protocol_v2.json').read_text())['candidates']] if block=='S_pooled' else ['B0',sel]
  frames={cid:pd.read_csv(R/'predictions'/f'{cid}_{block}.csv') for cid in candidates}
  for cid,r in frames.items():
   r.TimeStamp=pd.to_datetime(r.TimeStamp);ep=alarm_summary(r,r.prediction);ep['candidate']=cid;ep['block']=block;episodes.append(ep)
   for y,g in r.groupby('label'):
    coverage.append(dict(block=block,candidate=cid,label=int(y),rows=len(g),native_B0_W20=int(g.native_W20.sum()),B0_fallback=int(g.fallback.sum()),R5_available=int(g.R5_available.sum()) if 'R5_available' in g else None,final_unavailable=0))
   for name,mask in [('all',np.ones(len(r),bool)),('burst_first2',r.burst_pos<=2),('burst_position3',r.burst_pos==3),('burst_first19',r.burst_pos<=19),('burst_later',r.burst_pos>19),('B0_fallback',r.fallback.astype(bool)),('B0_cross_gap',r.crosses_gap.astype(bool)),('short_burst',r.burst_length<20)]:
    z=r[mask];n0=int((z.label==0).sum());n1=int((z.label==1).sum());fp=int(((z.label==0)&(z.prediction==1)).sum());fn=int(((z.label==1)&(z.prediction==0)).sum());conditions.append(dict(candidate=cid,block=block,condition=name,normal=n0,anomaly=n1,FP=fp,FN=fn,FPR=fp/n0 if n0 else np.nan,FNR=fn/n1 if n1 else np.nan))
   r[r.error.isin(['FN','FP'])].to_csv(R/'tables'/f'errors_{cid}_{block}.csv',index=False)
   if cid==sel:
    for bid,g in r.groupby('burst_id',sort=False):
     first_h=np.flatnonzero(g.h.to_numpy(bool));first_a=np.flatnonzero(g.aux_alarm.to_numpy(bool));dt=float((g.TimeStamp.iloc[first_a[0]]-g.TimeStamp.iloc[first_h[0]]).total_seconds()) if len(first_h) and len(first_a) else np.nan
     confirmation.append(dict(block=block,burst_id=bid,label=int(g.label.iloc[0]),rows=len(g),had_unconfirmed_h=bool(len(first_h)),had_confirmed_aux=bool(len(first_a)),first_h_to_first_confirmed_aux_seconds=dt,never_confirmed_despite_h=bool(len(first_h) and not len(first_a))))
  if sel:
   for comp in ['B0']+(['C0'] if block=='S_pooled' else []):
    a=frames[comp];b=frames[sel];pc=pair_counts(a,b);changed=b[b.prediction!=a.prediction].copy();changed['block']=block;changed['comparator']=comp;changed['candidate']=sel;changed['comparator_prediction']=a.loc[changed.index,'prediction'];changed['change']=np.where(changed.label==1,np.where(changed.prediction==1,'new_TP','new_FN'),np.where(changed.prediction==1,'additional_FP','removed_FP'));paired.append(changed);differences.append(dict(block=block,comparator=comp,candidate=sel,**pc,new_TP_bursts=int(changed[(changed.label==1)&(changed.prediction==1)].burst_id.nunique()),new_FN_bursts=int(changed[(changed.label==1)&(changed.prediction==0)].burst_id.nunique()),independent_failure_events='unknown; burst counts are observation groups'))
  fig,ax=plt.subplots(figsize=(7,5))
  for cid in ['B0']+(['C0'] if block=='S_pooled' else [])+([sel] if sel else []):
   r=frames[cid];pr,rc,_=precision_recall_curve(r.label,r.integrated_score);m=measure(r);ax.plot(rc,pr,label=cid);ax.scatter(m['Recall'],m['Precision'],s=35)
  ax.set(xlabel='Recall',ylabel='Precision',title=f'{block}: continuous score curves, actual operating points');ax.legend();fig.tight_layout();fig.savefig(R/'figures'/f'PR_{block}.png',dpi=140);plt.close(fig)
 pd.DataFrame(conditions).to_csv(R/'tables/error_conditions.csv',index=False);pd.DataFrame(coverage).to_csv(R/'tables/coverage.csv',index=False);eps=pd.concat(episodes);eps.to_csv(R/'tables/alarm_episodes.csv',index=False);pd.DataFrame(confirmation).to_csv(R/'tables/confirmation_delay.csv',index=False);pd.DataFrame(differences).to_csv(R/'paired_counts.csv',index=False)
 if paired:pd.concat(paired).to_csv(R/'paired_selected_rows.csv',index=False)
 delays=[]
 for block in blocks:
  if not sel:continue
  s=eps[(eps.block==block)&(eps.candidate==sel)]
  for comp in ['B0']+(['C0'] if block=='S_pooled' else []):
   a=eps[(eps.block==block)&(eps.candidate==comp)];m=s.merge(a,on=['burst_id','label'],suffixes=('_selected','_comparator'))
   for z in m.itertuples():delays.append(dict(block=block,comparator=comp,burst_id=z.burst_id,label=z.label,rows=z.rows_selected,selected_no_alarm=z.no_alarm_selected,comparator_no_alarm=z.no_alarm_comparator,selected_first_alarm_seconds=z.first_alarm_since_observed_burst_start_seconds_selected,comparator_first_alarm_seconds=z.first_alarm_since_observed_burst_start_seconds_comparator,difference_seconds=z.first_alarm_since_observed_burst_start_seconds_selected-z.first_alarm_since_observed_burst_start_seconds_comparator))
 pd.DataFrame(delays).to_csv(R/'tables/first_observation_alarm_delays.csv',index=False)
 trials=pd.read_csv(R/'all_trials.csv');fig,ax=plt.subplots(figsize=(7,5));ax.scatter(trials.FP,trials.FN)
 for t in trials.itertuples():ax.annotate(t.candidate,(t.FP,t.FN),xytext=(5,5 if t.rule!='G2' else -15),textcoords='offset points',fontsize=8)
 ax.scatter([8,9],[3,1],marker='*',s=100);ax.annotate('B0',(8,3));ax.annotate('C0',(9,1));ax.set(xlabel='FP on 1255 normal observations',ylabel='FN on 21 anomaly observations',title='Selection tradeoff (duplicates disclosed)');fig.tight_layout();fig.savefig(R/'figures/selection_tradeoff.png',dpi=140);plt.close(fig)
 if sel:
  for block in blocks:
   fig,axes=plt.subplots(1,2,figsize=(8,3))
   for cid,ax in zip(['B0',sel],axes):
    r=pd.read_csv(R/'predictions'/f'{cid}_{block}.csv');m=measure(r);mat=np.array([[m['TN'],m['FP']],[m['FN'],m['TP']]]);ax.imshow(mat,cmap='Blues')
    for (i,j),v in np.ndenumerate(mat):ax.text(j,i,str(v),ha='center',va='center',color='red')
    ax.set(xticks=[0,1],yticks=[0,1],xlabel='Prediction',ylabel='Label',title=f'{cid}: {block}')
   fig.tight_layout();fig.savefig(R/'figures'/f'confusion_{block}.png',dpi=140);plt.close(fig)
 js(R/'analysis_summary.json',dict(paired=differences,independent_events='not identifiable',FP_per_hour='not reported: acquisition gaps are not observed normal operating time',point_adjustment=False));print(pd.DataFrame(differences).to_string(index=False))
if __name__=='__main__':main()

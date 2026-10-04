from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,ConfusionMatrixDisplay
lock=json.loads((R/'frozen_selection.json').read_text());t=pd.read_csv(R/'all_trials.csv');summary=pd.read_csv(R/'baseline_vs_candidates.csv');coverage=[];conditions=[];budgets=[];fullpairs=pd.read_csv(R/'paired_errors.csv');fullpairs[fullpairs.change!='unchanged'].to_csv(R/'tables/changed_rows_only.csv',index=False)
for stage in ['selection','historical']:
 rows={c['candidate']:loadrows(R/'predictions'/f'{stage}_frozen_{c["candidate"]}.csv') for c in lock['final_configs']};baseline=rows['baseline'];n=int((baseline.label==0).sum());fp=int(((baseline.label==0)&(baseline.prediction==1)).sum())
 for criterion,delta in [('conservative',0),('primary',.001),('auxiliary',.002)]:budgets.append(dict(evaluation=stage,normal_n=n,baseline_FP=fp,criterion=criterion,delta_FPR=delta,allowed_extra_FP=allowed_fp(n,delta),maximum_FP=min(fp+allowed_fp(n,delta),allowed_fp(n,.01)),absolute_FP_cap=allowed_fp(n,.01)))
 fig,ax=plt.subplots(figsize=(7,5));plotted=set()
 for cfg in lock['final_configs']:
  r=rows[cfg['candidate']];pr,rc,_=precision_recall_curve(r.label,r.score)
  if cfg['system'] not in plotted:ax.step(rc,pr,where='post',label=cfg['system']);plotted.add(cfg['system'])
  mm=metrics(r.label,r.score,r.prediction);ax.scatter(mm['Recall'],mm['Precision'],s=55,marker='o' if cfg['candidate']=='baseline' else 'x',label=cfg['candidate']+' operating point')
 ax.set(xlabel='Recall',ylabel='Precision',title=stage+' PR and locked operating points');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(R/'figures'/f'{stage}_PR.png',dpi=170);plt.close(fig)
 for name,r in rows.items():
  for y,g in r.groupby('label'):coverage.append(dict(evaluation=stage,candidate=name,label=y,n=len(g),W20_native=int(g.native_W20.sum()),short_complement=int(g.short_complement.sum()),P0_fallback=int(g.fallback.sum()),unavailable=0,native_coverage=float(g.native_W20.mean()),total_statistical_coverage=float((~g.fallback).mean())))
  for cond,mask in [('first19',r.burst_pos<=19),('after19',r.burst_pos>19),('fallback',r.fallback),('short',r.short_complement),('cross_gap',r.crosses_gap),('within_burst',~r.crosses_gap)]:
   g=r[mask];conditions.append(dict(evaluation=stage,candidate=name,condition=cond,normal=int((g.label==0).sum()),anomaly=int((g.label==1).sum()),FP=int((g.error=='FP').sum()),FN=int((g.error=='FN').sum())))
  fig,ax=plt.subplots(figsize=(4,4));ConfusionMatrixDisplay.from_predictions(r.label,r.prediction,ax=ax,colorbar=False);ax.set_title(stage+' '+name,fontsize=10);fig.tight_layout();fig.savefig(R/'figures'/f'{stage}_{name}_confusion.png',dpi=150);plt.close(fig)
  fig,axs=plt.subplots(2,1,figsize=(11,5))
  for y,ax in enumerate(axs):
   z=r[r.label==y]
   # Line segments only within a burst; no interpolation across observation gaps.
   for bid,g in z.groupby('burst_id',sort=False):ax.plot(g.TimeStamp,g.score,color='grey',lw=.45,alpha=.7)
   for err,color in [('FP','red'),('FN','orange')]:g=z[z.error==err];ax.scatter(g.TimeStamp,g.score,c=color,s=24,label=err)
   ax.axhline(float(z.threshold.iloc[0]),c='black',ls='--');ax.set_ylabel('normal' if y==0 else 'anomaly');ax.legend(loc='upper right')
  fig.suptitle(stage+' '+name+' — lines stop at burst gaps');fig.autofmt_xdate();fig.tight_layout();fig.savefig(R/'figures'/f'{stage}_{name}_error_timeline.png',dpi=150);plt.close(fig)
 # Current operating configs only on historical set; never optimize a historical threshold.
 fig,ax=plt.subplots(figsize=(6,4))
 if stage=='selection':
  for system,g in t.groupby('system'):ax.plot(g.FP,g.FN,'o-',ms=4,label=system)
 for name,r in rows.items():m=metrics(r.label,r.score,r.prediction);ax.scatter(m['FP'],m['FN'],marker='*',s=110,label=name)
 ax.set(xlabel='FP count',ylabel='FN count',title=stage+' FN–FP tradeoff');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(R/'figures'/f'{stage}_FN_FP.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for system,g in t.groupby('system'):
 g=g.sort_values('threshold')
 for metric,ax in zip(['F1','F2'],axes):ax.plot(g.threshold,g[metric],'-o',ms=3,label=system);ax.set(xlabel='Score threshold',ylabel=metric,title='Selection '+metric)
for ax in axes:ax.axvline(lock['baseline_threshold'],ls='--',color='grey',label='Original threshold');ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(R/'figures/selection_threshold_F1_F2.png',dpi=170);plt.close(fig)
# Criteria audit: every candidate's rejection reason remains accessible.
reasons=[]
for _,a in t.iterrows():
 for rule in ['conservative','primary','auxiliary']:
  checks=['recall_increased','F2_increased','F1_not_worse','incremental_FP_limit','absolute_FPR_limit'];reasons.append(dict(candidate=a.candidate,criterion=rule,eligible=a[rule+'_eligible'],failed_requirements=';'.join(k for k in checks if not a[rule+'_'+k])))
pd.DataFrame(reasons).to_csv(R/'tables/eligibility_audit.csv',index=False);pd.DataFrame(coverage).to_csv(R/'tables/coverage.csv',index=False);pd.DataFrame(conditions).to_csv(R/'tables/error_conditions.csv',index=False);pd.DataFrame(budgets).to_csv(R/'tables/FP_integer_budgets.csv',index=False)
# Show diagnostic block budgets separately; no block selection or re-fit.
bm=pd.read_csv(R/'block_metrics.csv');bm[bm.candidate=='baseline'][['block','normal','anomaly','normal_bursts','anomaly_bursts','baseline_FP','conservative_allowed_extra_FP','primary_allowed_extra_FP','auxiliary_allowed_extra_FP']].to_csv(R/'tables/block_FP_budgets.csv',index=False)
js(R/'operational_recommendation.json',dict(recommendation='retain original PCA',reason='development selected warmup W3 failed Recall/F1/F2 non-regression on exposed historical evaluation; no reselection',frozen_selection_preserved=True,primary_development_selected=lock['criteria_selections']['primary']['candidate'],new_independent_evaluation=False))
js(R/'checkpoint.json',dict(stage='analysis_complete',figures=len(list((R/'figures').glob('*.png')))));print('Figures and conditional tables complete')

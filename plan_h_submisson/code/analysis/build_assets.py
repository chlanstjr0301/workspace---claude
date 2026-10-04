"""Tables/figures from saved predictions only. No model loading or fitting."""
from pathlib import Path
import os,json,math,hashlib
os.environ['MPLBACKEND']='Agg'
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import average_precision_score,precision_recall_curve,auc,roc_auc_score
ROOT=Path(__file__).resolve().parents[2];F=ROOT/'results/frozen';T=ROOT/'paper/tables';G=ROOT/'paper/figures';T.mkdir(exist_ok=True);G.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def ratio(a,b):return a/b if b else np.nan
def metric(f):
 y=f.label.to_numpy();p=f.prediction.to_numpy();tp=int(((y==1)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum());tn=int(((y==0)&(p==0)).sum());n=len(y)
 z=dict(N=n,N0=tn+fp,N1=tp+fn,TP=tp,FN=fn,FP=fp,TN=tn,Precision=ratio(tp,tp+fp),Recall=ratio(tp,tp+fn),FPR=ratio(fp,fp+tn),F1=ratio(2*tp,2*tp+fn+fp),F2=ratio(5*tp,5*tp+4*fn+fp),FNR=ratio(fn,tp+fn),Specificity=ratio(tn,tn+fp),NPV=ratio(tn,tn+fn),Accuracy=ratio(tp+tn,n),Balanced_Accuracy=.5*(ratio(tp,tp+fn)+ratio(tn,tn+fp)),MCC=ratio(tp*tn-fp*fn,math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),F05=ratio(1.25*tp,1.25*tp+.25*fn+fp),FDR=ratio(fp,tp+fp),FOR=ratio(fn,tn+fn),FP_per_1000=ratio(1000*fp,tn+fp))
 s=f.integrated_score if 'integrated_score' in f else f.score
 if len(set(y))==2:
  pr,re,_=precision_recall_curve(y,s);z.update(AP=average_precision_score(y,s),PR_AUC_trapezoid=auc(re,pr),ROC_AUC=roc_auc_score(y,s))
 return z
rows=[];evidence=[]
for block in ['S1','S2','S','V','H']:
 for model in ['B0','G1','B_same_threshold','R5_A0','D_old_threshold_G1']:
  p=F/'predictions'/f'{block}_{model}.csv';f=pd.read_csv(p);r=dict(block=block,model=model,**metric(f));rows.append(r)
  prior=pd.read_csv(F/'metrics.csv');r0=prior[(prior.block==block)&(prior.model==model)].iloc[0]
  assert all(r[k]==r0[k] for k in ['TP','FN','FP','TN'])
  assert all(abs(r[k]-r0[k])<1e-12 for k in ['F1','F2','FPR','AP','PR_AUC_trapezoid','ROC_AUC'])
  evidence.append(dict(claim=f'{block}/{model} metrics; main results or controls table',original_run='pca_g1_frozen_validation_20261004_221914',rows='data/analysis/inputs/frozen_rows.csv; configs/validation_protocol.json',prediction=str(p.relative_to(ROOT)),calculation='code/analysis/build_assets.py:metric',configuration='configs/model_manifest.json',sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
m=pd.DataFrame(rows);m.to_csv(ROOT/'results/paper_metrics.csv',index=False)
def table(name,header,body,align=None):
 a=align or 'l'+'r'*(len(header)-1);txt='\\begin{tabular}{'+a+'}\n\\toprule\n'+' & '.join(header)+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(str(x) for x in r)+' \\\\' for r in body)+'\n\\bottomrule\n\\end{tabular}\n';(T/name).write_text(txt)
sel=m[m.model.isin(['B0','G1'])];body=[]
for r in sel.itertuples():body.append([r.block,r.model,r.N0,r.N1,r.TP,r.FN,r.FP,r.TN,f'{r.Precision:.4f}',f'{r.Recall:.4f}',f'{100*r.FPR:.4f}',f'{r.F1:.4f}',f'{r.F2:.4f}'])
table('main_metrics.tex',['Set','Model','$N_0$','$N_1$','TP','FN','FP','TN','Prec.','Recall','FPR(\\%)','$F_1$','$F_2$'],body,'llrrrrrrrrrrr')
for block in ['S','V','H']:
 x=m[m.block==block];body=[]
 for r in x.itertuples():body.append([{'B_same_threshold':'G0, new $\\theta$','D_old_threshold_G1':'G1, old $\\theta$','R5_A0':'G0, old $\\theta$'}.get(r.model,r.model),r.FN,r.FP,f'{r.F1:.6f}',f'{r.F2:.6f}'])
 table(f'controls_{block}.tex',['Rule','FN','FP','$F_1$','$F_2$'],body)
for block in ['S1','S2','S','V','H']:
 x=m[(m.block==block)&m.model.isin(['B0','G1'])];body=[]
 for r in x.to_dict('records'):body.append([r['model']]+[f'{r[k]:.6f}' for k in ['FNR','Specificity','NPV','Accuracy','Balanced_Accuracy','MCC','F05','FDR','FOR']])
 table(f'detailed_{block}.tex',['Model','FNR','Spec.','NPV','Acc.','Bal.Acc.','MCC','$F_{0.5}$','FDR','FOR'],body)
table('ranking.tex',['Set','Model','AP','PR-AUC','ROC-AUC'],[[v.block,v.model,f'{v.AP:.6f}',f'{v.PR_AUC_trapezoid:.6f}',f'{v.ROC_AUC:.6f}'] for v in m[m.block.isin(['S','V','H']) & m.model.isin(['B0','G1'])].itertuples()]);
ev=pd.read_csv(F/'events.csv');h=ev[ev.block=='H'];table('events.tex',['Burst','Rows','B0 TP/FN','G1 TP/FN','Added','B0 $d$','G1 $d$'],[[r.burst_id,r.rows,f'{r.B0_TP}/{r.B0_FN}',f'{r.G1_TP}/{r.G1_FN}',r.added_detected_rows,f'{r.B0_observed_start_delay_seconds:.1f}',f'{r.G1_observed_start_delay_seconds:.1f}'] for r in h.itertuples()])
tr=pd.read_csv(ROOT/'provenance/history/pca_followup_v2_20261004_214810/all_trials.csv');table('six_candidates.tex',['Rule','$q$','FN','FP','$F_1$','$F_2$','Decision'],[[r.rule,r.q,r.FN,r.FP,f'{r.F1:.4f}',f'{r.F2:.4f}','selected' if r.eligible else ('duplicate' if r.status=='duplicate_excluded' else 'ineligible')] for r in tr.itertuples()])
old=pd.read_csv(ROOT/'provenance/history/pca_literature_rescue_20261004_210504/all_trials.csv');old.to_csv(ROOT/'results/literature_18_candidates.csv',index=False)
# Keep verbose reasons in CSV and a readable condensed LaTeX summary.
cols=old.columns.tolist();print('literature columns',cols)
body=[]
for _,r in old.iterrows():
 body.append([str(r.get('candidate',r.get('id',''))).replace('_',r'\_'),int(r.FN),int(r.FP),f'{r.F1:.4f}',f'{r.F2:.4f}','not selected'])
table('literature_candidates.tex',['ID','FN','FP','$F_1$','$F_2$','Decision'],body)
for block in ['S','H']:
 x=m[m.block==block];fig,axs=plt.subplots(1,2,figsize=(6.5,2.65));names=['B0','G0/new','G1/new','G0/old','G1/old'];order=['B0','B_same_threshold','G1','R5_A0','D_old_threshold_G1'];x=x.set_index('model').loc[order]
 for ax,key,col in zip(axs,['FN','FP'],['#C65533','#28688B']):
  bars=ax.bar(names,x[key],color=col);ax.bar_label(bars,padding=2);ax.set(ylabel='Observation count',title=f'{block}: {key}',ylim=(0,float(x[key].max())*1.28+1));ax.tick_params(axis='x',labelrotation=25)
 fig.tight_layout();
 for ext in ['pdf','svg','png']:fig.savefig(G/f'fn_fp_{block}.{ext}',dpi=220)
 plt.close(fig)
# A fixed earliest affected observation group, not a best-looking example.
cases=h[h.added_detected_rows>0].sort_values('start');case=cases.iloc[0];f=pd.read_csv(F/'predictions/H_G1.csv');f.TimeStamp=pd.to_datetime(f.TimeStamp);z=f[f.burst_id==case.burst_id].copy();z.to_csv(ROOT/'results/case_earliest_affected_H.csv',index=False);t=(z.TimeStamp-z.TimeStamp.iloc[0]).dt.total_seconds()
fig,ax=plt.subplots(5,1,sharex=True,figsize=(6.6,7.0),gridspec_kw={'height_ratios':[1,1,1,1.2,.8]})
for a,s in zip(ax[:3],['AI0_Vibration','AI1_Vibration','AI2_Current']):a.plot(t,z[s],color='#28688B',marker='.',lw=1);a.set_ylabel(s.replace('AI0_Vibration','AI0').replace('AI1_Vibration','AI1').replace('AI2_Current','Current'))
ax[3].plot(t,z.score/z.threshold,label='B0 / threshold',c='#28688B');ax[3].plot(t,z.R5_score/z.aux_threshold,label='R5 / threshold',c='#C65533');ax[3].axhline(1,c='black',ls='--',lw=1);ax[3].set_yscale('symlog',linthresh=1);ax[3].set_ylim(bottom=0);ax[3].set_ylabel('Score ratio');ax[3].legend(fontsize=9,ncol=2)
ax[4].scatter(t,np.zeros(len(z)),c=np.where(z.B0_prediction==1,'#28688B','#bbb'),marker='s',s=28);ax[4].scatter(t,np.ones(len(z)),c=np.where(z.prediction==1,'#C65533','#bbb'),marker='s',s=28);ax[4].set_yticks([0,1],['B0','G1']);ax[4].set_ylim(-.6,1.6);ax[4].set_xlabel('Seconds since first recorded observation (not fault onset)');fig.align_ylabels(ax);fig.tight_layout()
for ext in ['pdf','svg','png']:fig.savefig(G/f'case.{ext}',dpi=220)
plt.close(fig)
# All affected groups shown separately, no lines across gaps.
fig,axs=plt.subplots(len(cases),1,figsize=(6.5,5),squeeze=False)
for ax,(_,c) in zip(axs[:,0],cases.iterrows()):
 z=f[f.burst_id==c.burst_id];tt=(z.TimeStamp-z.TimeStamp.iloc[0]).dt.total_seconds();ax.plot(tt,z.score/z.threshold,label='B0',c='#28688B');ax.plot(tt,z.R5_score/z.aux_threshold,label='R5',c='#C65533');ax.axhline(1,ls='--',c='black',lw=.7);ax.set_yscale('symlog',linthresh=1);ax.set_ylim(bottom=0);ax.set(title=f'Observation group {c.burst_id}; added rows {c.added_detected_rows}',ylabel='Ratio');ax.legend(fontsize=8,loc='upper right')
axs[-1,0].set_xlabel('Seconds from each group start');fig.tight_layout();fig.savefig(G/'all_affected.pdf');fig.savefig(G/'all_affected.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(6.5,2.6));ax.axis('off');ax.set_xlim(-.025,1.025);boxes=[(.00,.5,.25,.36,'Sensors at t\n(raw values)'),(.34,.72,.27,.23,'P1 W20 / P0\nB0 alarm'),(.34,.13,.27,.38,'Lag-2 ridge residual\nMahalanobis score\nh[t] and h[t-1]'),(.74,.42,.24,.32,'OR\nG1 alarm at t')]
for x,y,w,hh,label in boxes:ax.add_patch(FancyBboxPatch((x,y),w,hh,boxstyle='round,pad=.01',facecolor='#edf3f7',edgecolor='#28688B'));ax.text(x+w/2,y+hh/2,label,ha='center',va='center',fontsize=10)
for a,b in [((.25,.70),(.34,.83)),((.25,.60),(.34,.32)),((.61,.83),(.74,.65)),((.61,.32),(.74,.49))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->'})
fig.tight_layout();fig.savefig(G/'flow.pdf');fig.savefig(G/'flow.svg');plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(6.5,3.4));colors=['#809AB1','#A1C3AD','#E3BA77','#D78168','#B6A5C7'];roles=[('Normal: 2022-07-12',[12012,2011,1255,731,3990]),('Anomaly: 2022-07-17',[0,132,21,90,357])]
for ax,(label,counts) in zip(axs,roles):
 left=0
 for n,name,c in zip(counts,['T','C','S','V','H'],colors):
  if n:ax.barh(0,n,left=left,color=c,label=name);ax.text(left+n/2,0,f'{name}',ha='center',va='center',fontsize=9)
  left+=n
 ax.set(yticks=[],title=label,xlabel='Within-file order; '+', '.join(f'{k}={v}' for k,v in zip(['T','C','S','V','H'],counts)));ax.set_xlim(0,left)
fig.tight_layout();fig.savefig(G/'roles.pdf');fig.savefig(G/'roles.svg');plt.close(fig)
# Continuous operating score already defined by gate; this is not binary-output AP.
fig,ax=plt.subplots(figsize=(3.3,2.7))
for model,color in [('B0','#28688B'),('G1','#C65533')]:
 z=pd.read_csv(F/'predictions'/f'H_{model}.csv');s=z.integrated_score;pr,re,_=precision_recall_curve(z.label,s);ax.plot(re,pr,label=model,color=color);r=m[(m.block=='H')&(m.model==model)].iloc[0];ax.scatter(r.Recall,r.Precision,color=color,s=30)
ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1.01),ylim=(0,1.01));ax.legend();fig.tight_layout();fig.savefig(G/'pr_H.pdf');plt.close(fig)
for claim,file in [('event coverage and first-alarm time','events.csv'),('normal observed spans, not uptime','normal_exposure.csv'),('calibration threshold and resolution','calibration_audit.json'),('causal time availability','row_timing_and_residuals.csv')]:evidence.append(dict(claim=claim,original_run='pca_g1_frozen_validation_20261004_221914',rows='data/analysis/inputs/frozen_rows.csv',prediction='results/frozen/'+file,calculation='code/runtime/review.py',configuration='configs/validation_protocol.json',sha256=hashlib.sha256((F/file).read_bytes()).hexdigest()))
pd.DataFrame(evidence).to_csv(ROOT/'provenance/evidence_map.csv',index=False)
(ROOT/'verification/paper_table_check.json').write_text(json.dumps(dict(passed=True,source='stored predictions only',metric_rows=len(m),confusion_exact=True,metric_atol=1e-12,case_selection='earliest chronological H group with added detections; complete group; all affected groups in appendix',new_fit_calls=0),indent=2))
print('Saved',len(m),'metric records and figures')

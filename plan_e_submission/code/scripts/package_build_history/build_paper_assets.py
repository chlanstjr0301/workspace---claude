from pathlib import Path
import json,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
r=Path((ROOT/'reports/current_paper_package.txt').read_text());run=ROOT/'runs/pca_20261003_cpu';fig=r/'paper/figures';fig.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
c=pd.read_csv(run/'tables/coverage.csv');c=c[(c.split=='test')&(c.condition=='all')]
figu,axs=plt.subplots(1,2,figsize=(6.7,2.35))
for label,ax in enumerate(axs):
 for pref,name,color,marker in [('U','Without reset','#0072B2','o'),('B','Burst reset','#D55E00','s')]:
  vals=[float(c[(c.feature==f'{pref}{w}')&(c.label==label)].coverage.iloc[0])*100 for w in [5,10,20]]
  ax.plot([5,10,20],vals,label=name,color=color,marker=marker)
 ax.set(xticks=[5,10,20],ylim=(0,105),xlabel='Window length (rows)',ylabel='Native coverage (%)',title=f'{"Normal" if label==0 else "Anomaly"} test observations')
axs[0].legend(fontsize=8);figu.tight_layout();figu.savefig(fig/'coverage.pdf');plt.close(figu)
pairs=pd.read_csv(run/'tables/burst_pairs_test.csv');rows=pairs[pairs.unaware.isin(['P1_W10_K2_QT','P1_W20_K2_Q'])]
figu,axs=plt.subplots(1,2,figsize=(6.7,2.4))
for ax,(_,row) in zip(axs,rows.iterrows()):
 ax.scatter([row.unaware_FPR*100,row.aware_FPR*100],[row.unaware_Recall*100,row.aware_Recall*100],c=['#0072B2','#D55E00'],s=70)
 for x,y,txt in [(row.unaware_FPR*100,row.unaware_Recall*100,'Without reset'),(row.aware_FPR*100,row.aware_Recall*100,'Burst reset')]:ax.annotate(txt,(x,y),xytext=(3,5),textcoords='offset points',fontsize=8)
 ax.set(xlabel='FPR (%)',ylabel='Recall (%)',title=f'W={row.window}, k=2, {row.score}')
 ax.margins(x=.6,y=.6)
figu.tight_layout();figu.savefig(fig/'burst_pairs.pdf');plt.close(figu)
post=pd.read_csv(run/'tables/postprocess_test.csv');figu,ax=plt.subplots(figsize=(3.2,2.2));x=np.arange(3);ax.bar(x-.16,post.FN,.32,label='False negatives',color='#D55E00');ax.bar(x+.16,post.FP,.32,label='False positives',color='#0072B2');ax.set_xticks(x,['None','2 consecutive','2 of 3']);ax.set(ylabel='Number of observations');ax.legend(fontsize=7);figu.tight_layout();figu.savefig(fig/'postprocess.pdf');plt.close(figu)
# Publication curves from frozen continuous scores; this does not fit or tune.
from sklearn.metrics import precision_recall_curve,roc_curve
op=pd.read_csv(run/'predictions/all_operational_test.csv.gz');df=pd.read_csv(run/'manifests/clean_rows.csv').set_index('row_id');y=df.loc[op.row_id,'label'].to_numpy()
figu,axs=plt.subplots(1,2,figsize=(6.7,2.5))
for key,label,color in [('P1_W20_K2_Q','PCA W20 Q','#0072B2'),('I1_W20_ensemble','IF W20 (matched input)','#D55E00'),('I1_W5_ensemble','IF W5 (selected)','#009E73')]:
 pr,re,_=precision_recall_curve(y,op[key]);fpr,tpr,_=roc_curve(y,op[key]);axs[0].plot(re,pr,label=label,color=color);axs[1].plot(fpr,tpr,label=label,color=color)
axs[0].set(xlabel='Recall',ylabel='Precision',xlim=(0,1.01),ylim=(0,1.02));axs[1].set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1.01),ylim=(0,1.02));axs[0].legend(fontsize=7,loc='lower left');figu.tight_layout();figu.savefig(fig/'curves.pdf');plt.close(figu)
figu,ax=plt.subplots(figsize=(3.2,2.3))
for sp,color in [('selection','#0072B2'),('test','#D55E00')]:
 rel=pd.read_csv(run/'tables'/f'reliability_{sp}.csv').dropna();ax.plot(rel.mean_probability,rel.observed_fraction,'o-',label=sp,color=color)
ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Mean predicted probability',ylabel='Observed anomaly fraction');ax.legend(fontsize=8);figu.tight_layout();figu.savefig(fig/'reliability.pdf');plt.close(figu)
# Preserve machine-readable evidence for every numerical table.
for name in ['metrics_with_counts_test.csv','metrics_with_counts_selection.csv','burst_pairs_test.csv','coverage.csv','postprocess_test.csv','score_complements_test.csv','sensor_combinations_selection.csv','probability_selection.csv','probability_test.csv']:
 dest=r/'paper/table_sources';dest.mkdir(exist_ok=True);shutil.copy2(run/'tables'/name,dest/name)
print('Paper figures and table sources ready',fig)

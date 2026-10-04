from common import *
import shutil,zipfile,platform,sys
from importlib.metadata import distributions
P=R.parents[1];old=P/'runs/pca_20261003_cpu';prev=P/'runs/adversarial_pca_20261004_103236';pkg=P/'deliverables/pca_paper_20261003_185434'
for name in ['inputs/original/models','inputs/prior_review','data','models','predictions','figures','logs','manifests','references','tables']:(R/name).mkdir(parents=True,exist_ok=True)
files=[]
def copy(src,dst):
 dst=R/dst
 if dst.exists():assert sha(dst)==sha(src)
 else:shutil.copy2(src,dst)
 files.append(dict(source=str(src),copy=str(dst.relative_to(R)),sha256=sha(src)))
for name in ['press_data_normal.csv','outlier_data.csv']:copy(P/'data'/name,'data/'+name)
copy(P/'configs/first_experiment/frozen_rows.csv','inputs/frozen_rows.csv')
copy(old/'selection_lock.json','inputs/original/selection_lock.json')
for name in ['P1_W20_K2.joblib','P0_W1_K2.joblib','norm_P1_W20_K2.json','norm_P0_W1_K2.json','score_metadata.json','sigmoid.joblib']:copy(old/'models'/name,'inputs/original/models/'+name)
for split in ['test','selection']:copy(old/'predictions'/f'operational_P1_W20_K2_Q_target0p01_{split}.csv',f'inputs/original_expected_{split}.csv')
for name in ['review_report_ko.md','all_trials.csv','frozen_selection.json','evaluation_lock.json']:copy(prev/name,'inputs/prior_review/'+name)
copy(pkg/'paper/manuscript_ko.tex','references/original_manuscript_ko.tex')
archives=[]
for name in ['hydraulic_pca_paper_code_data_ko_20261003.zip','adversarial_pca_review_20261004_103236.zip']:
 p=P/'deliverables'/name
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None;names=z.namelist();js(R/'manifests'/f'{name}_members.json',names)
  # Verify the actual source files used against their archived copies.
  matches=[]
  for item in files:
   src=Path(item['source']);anchor=pkg if name.startswith('hydraulic') else prev
   if src.is_relative_to(anchor):suffix=str(src.relative_to(anchor))
   elif name.startswith('hydraulic') and src.is_relative_to(old):suffix='results/pca_20261003_cpu/'+str(src.relative_to(old))
   elif name.startswith('hydraulic') and src.parent==P/'data':suffix='data/'+src.name
   elif name.startswith('hydraulic') and src.name=='frozen_rows.csv':suffix='code/configs/first_experiment/frozen_rows.csv'
   else:continue
   hits=[n for n in names if n.endswith('/'+suffix) or n==suffix];assert len(hits)==1,(suffix,hits);assert hashlib.sha256(z.read(hits[0])).hexdigest()==item['sha256'];matches.append(suffix)
 archives.append(dict(path=str(p),sha256=sha(p),members=len(names),verified_used_members=matches))
js(R/'inventory.json',dict(files=files,archives=archives,python=sys.version,platform=platform.platform(),threads=2,existing_venv_used=True,other_agent_files_accessed=False))
(R/'requirements.lock.txt').write_text('\n'.join(sorted(f"{a.metadata['Name']}=={a.version}" for a in distributions()))+'\n')
d=loadrows(R/'inputs/frozen_rows.csv');assert d.row_id.is_unique;assert d.groupby('burst_id').split.nunique().max()==1;audit=[]
for name,g in d.groupby('source_file',sort=False):
 raw=pd.read_csv(R/'data'/name);raw['source_row']=np.arange(1,len(raw)+1);raw.TimeStamp=pd.to_datetime(raw.TimeStamp);keys=['TimeStamp']+SENSORS+['Equipment_state'];mask=raw.duplicated(keys);c=raw[~mask];assert c.source_row.tolist()==g.source_row.tolist();assert np.array_equal(c[SENSORS].to_numpy(),g[SENSORS].to_numpy());assert np.array_equal(c.TimeStamp.to_numpy(),g.TimeStamp.to_numpy());assert np.array_equal(c.Equipment_state,g.label);gap=c.TimeStamp.diff().dt.total_seconds();assert (gap.dropna()>0).all();assert not raw.isna().any().any();audit.append(dict(file=name,original=len(raw),clean=len(c),duplicates=int(mask.sum()),sha256=sha(R/'data'/name),gaps_gt05=int((gap>.5).sum()),max_gap=float(gap.max()),dates=c.TimeStamp.dt.strftime('%Y-%m-%d').unique().tolist()))
js(R/'manifests/data_audit.json',audit);d.groupby(['split','label']).agg(rows=('row_id','size'),bursts=('burst_id','nunique')).to_csv(R/'manifests/split_counts.csv')
# Diagnostic selection halves retain existing predictions and all pre-boundary history.
blocks={}
for label in [0,1]:
 g=d[(d.split=='selection')&(d.label==label)];sizes=g.groupby('burst_id',sort=False).size();j=len(sizes)//2 if label==1 else int(np.argmin(abs(sizes.cumsum().to_numpy()[:-1]-len(g)/2)))+1;early=set(sizes.index[:j]);
 for name,mask in [('early',g.burst_id.isin(early)),('late',~g.burst_id.isin(early))]:blocks.setdefault(name,[]).extend(g.index[mask].tolist())
blocks={name:sorted(ids) for name,ids in blocks.items()};js(R/'manifests/diagnostic_blocks.json',blocks)
bcounts=[]
for name,idx in blocks.items():
 for y,g in d.loc[idx].groupby('label'):bcounts.append(dict(block=name,label=y,rows=len(g),bursts=g.burst_id.nunique()))
pd.DataFrame(bcounts).to_csv(R/'manifests/diagnostic_block_counts.csv',index=False)
protocol=dict(created=pd.Timestamp.now(tz='UTC').isoformat(),baseline='original P1 W20 k2 Q; StandardScaler; full15 stats ddof0; unchanged P0 k2 Q fallback; no postprocess',target_fprs=[.001,.002,.003,.005,.0075,.01,.0125,.015,.02,.025,.03,.05],criteria=dict(conservative=0,primary=.001,auxiliary=.002),comparison='Recall strictly increases, F2 strictly increases, F1 nondecrease, FPR<=baselineFPR+delta and <=.01',integer_rule='max extra FP=floor(N_normal*delta); never allow a minimum of1; absolute max floor(N_normal*.01)',selection='pooled F2 descending, FPR ascending, Recall descending, F1 descending, complexity ascending, stable candidate ID',tolerance=1e-12,tolerance_use='metric equalities/ties only; FP integer budgets exact Decimal floor',quantile='higher',decision='score > threshold',ties='normal',normalization='original Q95 references; modified full system recalibrated normal-only',candidate_budget=36,experiment_A=12,experiment_B='conditional; at most2 hypotheses and24 configurations; no global feature replacement',B_possible_minimums=dict(normal_train_windows=100,normal_calibration_native_windows=100,routed_normal_calibration_rows=10),B_sparse_rule='retain original PCA/P0 system if any minimum fails; score evaluation forbidden before decision',budget_seconds=5400,threads=2,selection_scope='original selection all2097 rows; original train and calibration unchanged',blocks='two burst-preserving temporal diagnostic subsets of same selection, not new folds/holdout; windows not reset at diagnostic reporting boundary',fit='normal original train only; original train_role boundaries retained',historical='already exposed; only reproduced baseline before freeze, at most5 unique frozen configurations after',prior_exposure='both original development and historical evaluation have been seen; no new independent validation',sensor_inputs=SENSORS,forbidden_inputs=['Equipment_state','source_file','date','source_row','burst_id','future_burst_length'])
js(R/'protocol.json',protocol);(R/'hypotheses.md').write_text('''# 사전 가설\n\nA: 검출기·특징·정렬·fallback·후처리를 고정하고 정상 보정 목표 FPR 12개만 바꾸면, 기존보다 Recall/F2가 증가하면서 F1과 FP 예산을 만족할 수 있는가? 같은 selection 전체 행에서 비교한다. AP/PR-AUC/ROC-AUC는 임계값에 따라 변하지 않아야 한다. 실패 시 조건을 완화하지 않는다.\n\nB는 A 완료와 개발 오류조건 확인 이후에만 가설을 추가한다. 기존15특징 P1 W20이 주 모델이며 이전 median/IQR나 모든버스트 초기화 라우팅을 반복하지 않는다. 최대24개 추가 운영구성. 후보 평가 전 최소표본·부족시원복·보정규칙을 고정한다.\n\n보수 ΔFPR≤0, 주≤0.001, 보조≤0.002. 모든 기준에서 Recall 증가/F2 증가/F1 비악화/절대FPR≤1%를 유지한다. 조건/가중치는 프로젝트 자체 설계이고 실제 비용비가 아니다.\n''')
prior=pd.read_csv(R/'inputs/prior_review/all_trials.csv');x=prior[(prior.block=='pooled_inner')&(prior.scope=='operational')&prior.candidate.isin(['P1_Q','MEDIANIQR_Q'])];x.to_csv(R/'tables/prior_comparison.csv',index=False);pl=json.loads((R/'inputs/prior_review/frozen_selection.json').read_text());el=json.loads((R/'inputs/prior_review/evaluation_lock.json').read_text());js(R/'manifests/prior_history.json',dict(selected=pl['selected'],frozen_time=pl['time'],evaluation_lock_time=el['time'],outer_checked_after_selection=True,interpretation='median/IQR selected among FPR-eligible candidates; not Recall-superior to original P1 target1%; P1 target1% failed old inner FPR constraint; better than threshold-only0.1% by TP1/FP-2 on old inner set. Later degradation was post-selection. No broad repetition justified.'))
js(R/'checkpoint.json',dict(stage='prepared',protocol_sha256=sha(R/'protocol.json')));print('Prepared',R);print(pd.DataFrame(bcounts).to_string(index=False))

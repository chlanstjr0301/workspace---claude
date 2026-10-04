from common import *
import subprocess,sys,tempfile
started=time.monotonic();d=loadrows(R/'inputs/frozen_rows.csv');e=Engine(d);proto=json.loads((R/'protocol.json').read_text());old=R/'inputs/original';lock=json.loads((old/'selection_lock.json').read_text());meta=json.loads((old/'models/score_metadata.json').read_text());entries={}
for w,key in [(20,lock['primary']['key']),(1,lock['fallback'])]:
 m=meta[key];entries[w]=dict(bundle=joblib.load(old/'models'/f'{m["base"]}.joblib'),reference=m['reference'])
base=dict(name='original',short_window=None,entries=entries,threshold=lock['primary']['threshold'],postprocessing='none')
joblib.dump(base,R/'models/baseline.joblib',compress=3);cal=d.index[(d.split=='calibration')&(d.label==0)].to_numpy();se=d.index[d.split=='selection'].to_numpy();te=d.index[d.split=='test'].to_numpy();cr=e.predict(base,cal);sr=e.predict(base,se);br,bm=evaluated(sr,base['threshold']);assert abs(threshold(cr.score,.01)-base['threshold'])<1e-12
br.to_csv(R/'predictions/selection_baseline.csv',index=False);cr.to_csv(R/'predictions/calibration_baseline.csv',index=False)
# Separate process original saved inference: original test score comparison authorized solely for reproduction.
parts=[]
with tempfile.TemporaryDirectory(dir=R/'logs') as temp:
 for f,g in d.loc[te].groupby('source_file',sort=False):
  ip=Path(temp)/f;op=Path(temp)/(f+'.pred.csv');g[['TimeStamp']+SENSORS].to_csv(ip,index=False);pr=subprocess.run([sys.executable,str(R/'src/infer.py'),'--run',str(old),'--csv',str(ip),'--out',str(op)],capture_output=True,text=True);assert pr.returncode==0,pr.stderr;a=pd.read_csv(op);a['row_id']=g.row_id.to_numpy();parts.append(a)
saved=pd.concat(parts);expected=pd.read_csv(R/'inputs/original_expected_test.csv');assert saved.row_id.tolist()==expected.row_id.tolist();assert np.array_equal(saved.prediction,expected.prediction);assert np.allclose(saved.score,expected.score,rtol=1e-10,atol=1e-10)
# Independent normal-only refit for original k2 models, with references recomputed normal-only.
retrained=dict(base,entries={});fitrows=[]
for w in [1,20]:
 ids=d.index[(d.split=='train')&(e.starts_for(w)>=0)].to_numpy();assert not d.loc[ids,'label'].any();b=fit_pca(e.x(w,ids),2,SENSORS if w==1 else FEATURES);assert b['q_valid'];cn=cal[e.starts_for(w)[cal]>=0];ref=reference(score(b,e.x(w,cn))['Q']);retrained['entries'][w]=dict(bundle=b,reference=ref);fitrows.extend(dict(model=f'retrained_W{w}',row_id=d.loc[i,'row_id'],role='normal_train') for i in ids)
rr=e.predict(retrained,te);rh=threshold(e.predict(retrained,cal).score,.01);rr,rm=evaluated(rr,rh);assert np.array_equal(rr.prediction,expected.prediction);assert np.allclose(rr.score,expected.score,atol=1e-10,rtol=1e-10);rr.to_csv(R/'predictions/reproduced_original_test.csv',index=False);joblib.dump(retrained,R/'models/retrained_original.joblib',compress=3);pd.DataFrame(fitrows).to_csv(R/'manifests/refit_rows.csv',index=False)
js(R/'reproduction.json',dict(saved_model_separate_process=dict(rows=len(saved),max_score_delta=float(np.max(abs(saved.score.to_numpy()-expected.score.to_numpy()))),identical_decisions=True),normal_only_retraining=dict(metrics=rm,threshold=rh,max_score_delta=float(np.max(abs(rr.score.to_numpy()-expected.score.to_numpy())))),development_baseline=bm))
(R/'reproduction_report_ko.md').write_text(f'''# 기존 최고 PCA 재현\n\n저장 모델 별도 프로세스: 4347행의 원행 순서·점수·판정이 기존 파일과 일치했다.\n정상 전용 재학습: P1 W20 k2 Q와 P0 W1 k2 Q를 원분할/train_role 경계로 재학습하고 normal calibration Q95 정렬 및1%분위수를 다시 계산해도 TP{rm['TP']}/FN{rm['FN']}/FP{rm['FP']}/TN{rm['TN']}를 재현했다. 원임계값 {base['threshold']:.17g}.\n개발 selection: TP{bm['TP']}/FN{bm['FN']}/FP{bm['FP']}/TN{bm['TN']}.\n원본 통계 특징은 mean/std(ddof0)/RMS/min/max, StandardScaler와 fullSVD PCA는 정상 train에만 fit했다. 유효rank>2를 확인했다. Equipment_state·날짜·파일명·행번호는 모델 입력에 없다.\n직전 median/IQR는 기존P1 target1%보다 Recall이 낮았으며, 당시FPR제약을 만족하는 후보 중 선택됐다. P1 임계값0.1% 대조보다 내부TP1/FP-2 이득은 있었지만 개발후반 악화는 선택고정 후 확인됐다. 직전탐색 전체를 반복할 근거는 없다.\n''')
blocks=json.loads((R/'manifests/diagnostic_blocks.json').read_text());rows=[];brows=[];seen={}
for target in proto['target_fprs']:
 tic=time.perf_counter();h=threshold(cr.score,target);name=f'A_q{target:.4f}';r,m=evaluated(sr,h);dupe=seen.get(float(h).hex());seen.setdefault(float(h).hex(),name);row=dict(candidate=name,experiment='A',system='original',target=target,threshold=h,threshold_delta=h-base['threshold'],complexity=0,status='completed',duplicate_threshold_of=dupe,calibration_actual_FPR=float((cr.score>h).mean()),calibration_ties=int((cr.score==h).sum()),seconds=time.perf_counter()-tic,**m,**paired(br,r))
 for rule,delta in proto['criteria'].items():row.update({rule+'_'+k:v for k,v in compare(bm,m,delta).items()})
 rows.append(row);r['candidate']=name;r.to_csv(R/'predictions'/f'selection_{name}.csv',index=False)
 for block,idx in blocks.items():
  g=r.loc[idx];b=br.loc[idx];mm=metrics(g.label,g.score,g.prediction);bb=metrics(b.label,b.score,b.prediction);record=dict(candidate=name,stage='selection',block=block,normal=int((g.label==0).sum()),anomaly=int((g.label==1).sum()),normal_bursts=g[g.label==0].burst_id.nunique(),anomaly_bursts=g[g.label==1].burst_id.nunique(),baseline_Recall=bb['Recall'],baseline_FPR=bb['FPR'],baseline_FP=bb['FP'],**mm)
  for rule,delta in proto['criteria'].items():record[rule+'_allowed_extra_FP']=allowed_fp(record['normal'],delta);record[rule+'_FP_limit_met']=mm['FP']<=bb['FP']+record[rule+'_allowed_extra_FP']
  brows.append(record)
t=pd.DataFrame(rows);assert max(t.AP)-min(t.AP)<1e-14;assert max(t.PR_AUC_trapezoid)-min(t.PR_AUC_trapezoid)<1e-14;assert max(t.ROC_AUC)-min(t.ROC_AUC)<1e-14;t.to_csv(R/'threshold_sweep.csv',index=False);t.to_csv(R/'all_trials.csv',index=False);pd.DataFrame(brows).to_csv(R/'block_metrics.csv',index=False)
# Error-condition evidence, selection only. No historical error list inspected for B.
conditions=[]
for cond,mask in [('fallback',br.fallback),('P1_W20',br.native_W20),('first19',br.burst_pos<=19),('after19',br.burst_pos>19),('cross_gap',br.crosses_gap),('within_burst',~br.crosses_gap)]:
 g=br[mask];conditions.append(dict(condition=cond,normal=int((g.label==0).sum()),anomaly=int((g.label==1).sum()),FP=int((g.error=='FP').sum()),FN=int((g.error=='FN').sum())))
pd.DataFrame(conditions).to_csv(R/'tables/selection_error_conditions.csv',index=False);br[br.error!='correct'].to_csv(R/'predictions/selection_baseline_errors.csv',index=False)
js(R/'stage_a_summary.json',dict(baseline=bm,eligible={rule:t[t[rule+'_eligible']].candidate.tolist() for rule in proto['criteria']},wall_seconds=time.monotonic()-started,AP_PR_ROC_invariant=True,unique_thresholds=t.threshold.nunique()))
js(R/'checkpoint.json',dict(stage='A_complete',new_operating_configurations=12));print(t[['candidate','TP','FN','FP','F1','F2','primary_eligible','auxiliary_eligible']].to_string(index=False));print(pd.DataFrame(conditions).to_string(index=False))

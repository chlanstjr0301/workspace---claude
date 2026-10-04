from core import *
import subprocess,sys,tempfile
R=Path(__file__).resolve().parents[1];P=R.parents[1];old=P/'runs/pca_20261003_cpu';d=loadrows(R/'manifests/original_rows.csv');d['review_partition']=d.split+':'+d.train_role
lock=json.loads((old/'selection_lock.json').read_text());ev=d.index[d.split=='test'].to_numpy();cal=d.index[(d.split=='calibration')&(d.label==0)].to_numpy();key='P1_W20_K2_Q';expected=pd.read_csv(old/'predictions'/f'operational_{key}_target0p01_test.csv');results=[]
with tempfile.TemporaryDirectory(dir=R/'logs') as tmp:
 for file,g in d.loc[ev].groupby('source_file',sort=False):
  ip=Path(tmp)/file;op=Path(tmp)/(file+'.pred.csv');g[['TimeStamp']+SENSORS].to_csv(ip,index=False)
  proc=subprocess.run([sys.executable,str(R/'src/infer.py'),'--run',str(old),'--csv',str(ip),'--out',str(op)],capture_output=True,text=True);assert proc.returncode==0,proc.stderr
  a=pd.read_csv(op);a['row_id']=g.row_id.to_numpy();results.append(a)
a=pd.concat(results);assert a.row_id.tolist()==expected.row_id.tolist();assert np.array_equal(a.prediction,expected.prediction);delta=float(np.max(abs(a.score.to_numpy()-expected.score.to_numpy())))
a.to_csv(R/'predictions/original_saved_inference.csv',index=False)
e=Engine(d,R,partition='split');spec=dict(id='P1_original',mode='unaware',feat='full',scaler='standard',k=2,kind='PCA',score='Q',align='q95');system=e.fit_system(spec,cal);rr,m=evaluate(e.predict(system,ev),system['thresholds']['0.01']);rr.to_csv(R/'predictions/original_retrained.csv',index=False);joblib.dump(system,R/'models/original_retrained_system.joblib',compress=3)
assert np.array_equal(rr.prediction,expected.prediction);assert np.allclose(rr.score,expected.score,atol=1e-10,rtol=1e-10)
summary=dict(saved_inference=dict(rows=len(a),score_max_absolute_delta=delta,identical_predictions=True,metrics=metrics(expected.label,a.score,a.prediction)),retraining=dict(metrics=m,score_max_absolute_delta=float(np.max(abs(rr.score.to_numpy()-expected.score.to_numpy()))),threshold=system['thresholds']['0.01'],old_threshold=lock['primary']['threshold'],models_fit_normal_only=True),first19_FN=int(((rr.error=='FN')&(rr.burst_pos<=19)).sum()),first19_anomaly=int(((rr.label==1)&(rr.burst_pos<=19)).sum()),first19_fallback_FN=int(((rr.error=='FN')&rr.fallback).sum()))
# Development conditions must corroborate the hypothesis before improvement experiments.
se=d.index[d.split=='selection'];dev,dm=evaluate(e.predict(system,se),system['thresholds']['0.01']);dev.to_csv(R/'predictions/original_selection_retrained.csv',index=False)
summary['original_selection']=dm;summary['development_first19_FN']=int(((dev.error=='FN')&(dev.burst_pos<=19)).sum());summary['development_first19_FP']=int(((dev.error=='FP')&(dev.burst_pos<=19)).sum())
js(R/'reproduction.json',summary)
(R/'reproduction_report_ko.md').write_text(f'''# 원결과 재현\n\n저장 모델을 별도 Python 프로세스에서 추론한 {len(a)}행의 ID·판정이 기존 파일과 일치했다. 최대 점수 오차 {delta:.3g}.\n정상 학습에서 원설정을 재학습한 경우도 TP {m['TP']}, FN {m['FN']}, FP {m['FP']}, TN {m['TN']}으로 일치했다. 재현과 신규 실험은 분리했다.\n과거 FN 28 중 첫19행 FN {summary['first19_FN']}개, fallback FN {summary['first19_fallback_FN']}개. 개발 선택에서도 첫19행 FP {summary['development_first19_FP']}개, FN {summary['development_first19_FN']}개였다. 과거 오류는 이미 공개된 사실이며 시작부 가설의 착안에 사용했다.\n기존 코드의 Q는 표준화 공간 잔차 제곱합이고 T²는 중심화한 PCA transform 좌표/학습 고유값이다. 모든 유지 고유값이 허용오차보다 크며 k<유효 rank를 검사했다.\n근거: reproduction.json, predictions/original_*.csv, src/reproduce.py.\n''')
print(json.dumps(summary,indent=2))

from common import *
import sys,subprocess,tempfile

def reproduce():
 d=loadrows(R/'inputs/frozen_rows.csv');base=joblib.load(R/'models/baseline.joblib');expected=pd.read_csv(R/'inputs/original_expected_test.csv');parts=[]
 with tempfile.TemporaryDirectory(dir=R/'logs') as temp:
  for name,g in d[d.split=='test'].groupby('source_file',sort=False):
   ip=Path(temp)/name;op=Path(temp)/(name+'.pred');g[['TimeStamp']+SENSORS].to_csv(ip,index=False);z=subprocess.run([sys.executable,str(R/'src/infer_system.py'),'--model',str(R/'models/baseline.joblib'),'--threshold',str(base['threshold']),'--csv',str(ip),'--out',str(op)],capture_output=True,text=True);assert z.returncode==0,z.stderr;r=pd.read_csv(op);r['row_id']=g.row_id.to_numpy();parts.append(r)
 r=pd.concat(parts);assert r.row_id.tolist()==expected.row_id.tolist();assert np.array_equal(r.prediction,expected.prediction);assert np.allclose(r.score,expected.score,rtol=1e-10,atol=1e-10);m=metrics(expected.label,r.score,r.prediction);assert (m['TP'],m['FN'],m['FP'],m['TN'])==(329,28,1,3989)
 js(R/'reproduction.json',dict(kind='stored baseline separate-process inference, not original PCA refit',metrics=m,rows=len(r),score_max_delta=float(np.max(abs(r.score.to_numpy()-expected.score.to_numpy()))),baseline_threshold=base['threshold'],baseline_model_sha256=sha(R/'models/baseline.joblib')));(R/'reproduction_report_ko.md').write_text('# 저장 원모델 추론 재현\n\n독립 프로세스에서 원평가4347행의 점수/판정을 대조했다. TP329/FN28/FP1/TN3989. 재학습 재현과 구분한다. 이번 새 학습은18개 운영구성의 보조 모델에만 수행한다.\n');print('Original stored baseline reproduced')
if __name__=='__main__':reproduce()

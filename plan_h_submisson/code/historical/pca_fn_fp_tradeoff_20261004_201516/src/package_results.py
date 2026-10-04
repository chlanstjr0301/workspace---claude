from common import *
import re,zipfile
text=(R/'logs/end_to_end_replay.log').read_text();matches=re.findall(r'Completed: (.+)/review_report_ko.html',text);assert matches,'Wait for full replay completion';rr=Path(matches[-1]);la=json.loads((R/'frozen_selection.json').read_text());lb=json.loads((rr/'frozen_selection.json').read_text());assert [x['candidate'] for x in la['final_configs']]==[x['candidate'] for x in lb['final_configs']]
for c in la['final_configs']:
 f='historical_frozen_'+c['candidate']+'.csv';a=pd.read_csv(R/'predictions'/f);b=pd.read_csv(rr/'predictions'/f);assert a.row_id.tolist()==b.row_id.tolist();assert np.array_equal(a.prediction,b.prediction);assert np.allclose(a.score,b.score,atol=1e-10,rtol=1e-10)
assert json.loads((rr/'final_verification.json').read_text())['all_candidates_completed']==36
js(R/'end_to_end_replay_verification.json',dict(command='bash RUN_EXPERIMENT.sh',new_run=str(rr),same_frozen_configurations=True,same_final_scores_and_predictions=True,refit_included=True,not_independent_statistical_evidence=True))
# Detailed calibration-route rates at locked thresholds, for sample-size transparency.
from collections import defaultdict
d=loadrows(R/'inputs/frozen_rows.csv');e=Engine(d);cal=d.index[(d.split=='calibration')&(d.label==0)].to_numpy();rows=[]
for c in la['final_configs']:
 q,_=evaluated(e.predict(joblib.load(R/c['model_path']),cal),c['threshold'])
 for route,g in q.groupby('route'):rows.append(dict(candidate=c['candidate'],route=route,normal_n=len(g),FP=int(g.prediction.sum()),actual_calibration_route_FPR=float(g.prediction.mean()),not_field_guarantee=True))
pd.DataFrame(rows).to_csv(R/'tables/calibration_route_rates.csv',index=False)
for name in ['review_report_ko.md','review_report_ko.html']:
 p=R/name;note='RUN_EXPERIMENT.sh를 실제 새 폴더에서 끝까지 재실행했고, 선택 구성과 과거 평가 점수·판정이 일치했다. 이는 코드 재현 검사이며 새 독립 검증이 아니다. end_to_end_replay_verification.json 참조.'
 if name.endswith('.md'):p.write_text(p.read_text()+'\n\n## 전체 명령 재실행 확인\n\n'+note+'\n')
 else:p.write_text(p.read_text().replace('</html>','<section><h2>전체 명령 재실행 확인</h2><p>'+note+'</p></section></html>'))
archive=R.parents[1]/'deliverables'/f'{R.name}_results.zip';assert not archive.exists(),'Refuse archive overwrite';(R/'archive_location.txt').write_text(str(archive)+'\n')
files=[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.venv' not in p.parts and p.name!='MANIFEST_SHA256.json']
for p in files:assert not any(s in p.name.lower() for s in ['oauth','token','credential','colab_sessions'])
js(R/'MANIFEST_SHA256.json',{str(p.relative_to(R)):sha(p) for p in files});files.append(R/'MANIFEST_SHA256.json')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,arcname=str(Path(R.name)/p.relative_to(R)))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 manifest=json.loads(z.read(str(Path(R.name)/'MANIFEST_SHA256.json')))
 for name,h in manifest.items():assert hashlib.sha256(z.read(str(Path(R.name)/name))).hexdigest()==h
h=sha(archive);Path(str(archive)+'.sha256').write_text(h+'  '+archive.name+'\n');print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,files=len(files),sha256=h,verified=True),indent=2))

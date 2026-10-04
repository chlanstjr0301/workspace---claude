from core import *
import shutil,zipfile,re,subprocess
R=Path(__file__).resolve().parents[1];P=R.parents[1];log=(R/'logs/end_to_end_replay.log').read_text();matches=re.findall(r'Completed: (.+)/review_report_ko.html',log)
assert matches,'End-to-end replay not yet complete'
replay=Path(matches[-1]);a=json.loads((R/'frozen_selection.json').read_text());b=json.loads((replay/'frozen_selection.json').read_text());assert a['selected']['candidate']==b['selected']['candidate'];assert a['selected']['target']==b['selected']['target']
for file in ['historical_baseline.csv','historical_selected.csv','historical_IF_same_W20.csv']:
 x=pd.read_csv(R/'predictions'/file);y=pd.read_csv(replay/'predictions'/file);assert np.array_equal(x.prediction,y.prediction);assert np.allclose(x.score,y.score,atol=1e-10,rtol=1e-10)
assert json.loads((replay/'final_verification.json').read_text())['all_required_outputs_present']
js(R/'end_to_end_replay_verification.json',dict(replay_folder=str(replay),command='bash RUN_REVIEW.sh',fresh_training=True,same_selected_candidate=True,same_final_predictions=True,statistical_independence=False))
(R/'data').mkdir(exist_ok=True)
for f in ['press_data_normal.csv','outlier_data.csv']:
 dst=R/'data'/f
 if dst.exists():assert sha(dst)==sha(P/'data'/f)
 else:shutil.copy2(P/'data'/f,dst)
shutil.copy2(P/'deliverables/pca_paper_20261003_185434/paper/manuscript_ko.pdf',R/'references/original_manuscript_ko.pdf')
fit=[];cal=[]
for path in (R/'models').glob('*.joblib'):
 obj=joblib.load(path)
 if isinstance(obj,dict) and 'train_rows' in obj:
  fit.extend(dict(model=path.name,row_id=i,role='normal_fit') for i in obj['train_rows'])
 if isinstance(obj,dict) and 'members' in obj:
  for j,member in enumerate(obj['members']):
   for w,m in member.items():cal.extend(dict(system=path.name,member=j,route=w,row_id=i,role='normal_calibration',pool_policy=m['pool_policy']) for i in m['pool_ids'])
pd.DataFrame(fit).to_csv(R/'manifests/model_fit_rows.csv',index=False);pd.DataFrame(cal).to_csv(R/'manifests/model_calibration_rows.csv',index=False)
(R/'review_report_ko.md').write_text((R/'review_report_ko.md').read_text()+'\n\n## 한 번의 명령 재실행 확인\n\n`bash RUN_REVIEW.sh`를 실제 새 폴더에서 끝까지 실행했다. 선택한 후보·임계값 목표와 기준선/신규/IF의 과거 평가 판정이 모두 일치했다. 모델 재학습을 포함한 코드 재현 확인이며 독립적인 통계 반복을 뜻하지 않는다. 근거: end_to_end_replay_verification.json.\n')
# Sanitization: source/data/results only; no credentials, environments or bytecode.
files=[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.venv' not in p.parts and p.name!='MANIFEST_SHA256.json']
for p in files:assert not any(s in p.name.lower() for s in ['oauth','token','colab_sessions','credential'])
js(R/'MANIFEST_SHA256.json',{str(p.relative_to(R)):sha(p) for p in files});files.append(R/'MANIFEST_SHA256.json')
archive=P/'deliverables'/f'adversarial_pca_review_20261004_103236.zip';assert not archive.exists(),'Do not overwrite existing ZIP'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,arcname=str(Path(R.name)/p.relative_to(R)))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 manifest=json.loads(z.read(str(Path(R.name)/'MANIFEST_SHA256.json')))
 for name,wanted in manifest.items():assert hashlib.sha256(z.read(str(Path(R.name)/name))).hexdigest()==wanted
h=sha(archive);Path(str(archive)+'.sha256').write_text(h+'  '+archive.name+'\n');js(P/'reports/adversarial_review_archive.json',dict(path=str(archive),size_bytes=archive.stat().st_size,sha256=h,files=len(files),all_payload_hashes_verified=True,run=str(R)))
print(json.dumps(dict(zip=str(archive),bytes=archive.stat().st_size,sha256=h,files=len(files)),indent=2))

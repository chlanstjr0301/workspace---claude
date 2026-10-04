"""Verify archive inputs and frozen inference, without fitting or retuning."""
from pathlib import Path
import hashlib,json,sys,tempfile
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/src'))
from infer import infer
manifest=ROOT/'MANIFEST_SHA256.json'
if not manifest.exists():raise SystemExit('Missing package manifest')
expected=json.loads(manifest.read_text())
for name,wanted in expected.items():
 path=ROOT/name
 if not path.is_file():raise AssertionError(f'Missing {name}')
 got=hashlib.sha256(path.read_bytes()).hexdigest()
 if got!=wanted:raise AssertionError(f'Hash mismatch: {name}')
run=ROOT/'results/pca_20261003_cpu';audit=json.loads((run/'manifests/audit.json').read_text())
for f in audit['files']:assert hashlib.sha256((ROOT/'data'/f['file']).read_bytes()).hexdigest()==f['sha256']
rows=pd.read_csv(run/'manifests/clean_rows.csv');lock=json.loads((run/'selection_lock.json').read_text());key=lock['primary']['key']
expected_pred=pd.read_csv(run/'predictions'/f'operational_{key}_target0p01_test.csv')
count=0
with tempfile.TemporaryDirectory(prefix='pca_package_verify_') as temp:
 for file,df in rows[rows.split=='test'].groupby('source_file',sort=False):
  path=Path(temp)/file;df[['TimeStamp','AI0_Vibration','AI1_Vibration','AI2_Current']].to_csv(path,index=False)
  result=infer(run,path);expected_file=expected_pred[expected_pred.source_file==file]
  assert np.allclose(result.score,expected_file.score,rtol=1e-10,atol=1e-10)
  assert np.array_equal(result.prediction,expected_file.prediction)
  count+=len(result)
print(json.dumps({'verified_files':len(expected),'verified_final_prediction_rows':count,'raw_data_unchanged':True,'refitting_performed':False},indent=2))

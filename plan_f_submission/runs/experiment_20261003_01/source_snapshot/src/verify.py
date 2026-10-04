"""Verify data preservation, split/window leakage, metric consistency and deployment parity."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from infer import infer
from evaluation import metrics
ROOT=Path(__file__).resolve().parents[1]

def verify(out):
    out=Path(out); rows=pd.read_csv(ROOT/'configs/frozen/rows.csv');audit=json.loads((ROOT/'configs/frozen/audit.json').read_text())
    checks={}
    checks['source_csv_hashes_unchanged']=all(hashlib.sha256((ROOT/'data'/a['file']).read_bytes()).hexdigest()==a['sha256'] for a in audit['files'])
    checks['unique_analysis_rows']=bool(rows.row_id.is_unique)
    checks['whole_burst_split']=bool(rows.groupby('burst_id').split.nunique().max()==1)
    for path in (out/'manifests').glob('rows_M*.csv'):
        m=pd.read_csv(path);m=m[m.eligible];lookup=rows.set_index('row_id');start=lookup.loc[m.window_start_row_id].reset_index(drop=True);end=lookup.loc[m.row_id].reset_index(drop=True)
        assert (start.split==end.split).all() and (start.source_file==end.source_file).all()
        if path.name.startswith(('rows_M1','rows_M4')):assert (start.burst_id==end.burst_id).all()
    checks['all_window_boundaries_checked']=True
    p=pd.read_csv(out/'predictions/main_test.csv');sel=pd.read_csv(out/'tables/selected_test.csv').query("role=='main'").iloc[0]
    measured=metrics(p.label,p.score,p.prediction)
    checks['saved_confusion_matches_predictions']=all(int(sel[k])==measured[k] for k in ['TP','FN','FP','TN'])
    errors=[]
    for file,g in p.groupby('source_file',sort=False):
        result,timing=infer(g,out)
        maxdiff=float(np.max(np.abs(result.score.to_numpy()-g.score.to_numpy())))
        assert np.allclose(result.score,g.score,rtol=1e-5,atol=1e-6)
        assert np.array_equal(result.prediction,g.prediction)
        errors.append(dict(file=file,max_score_difference=maxdiff,**timing))
    checks['standalone_inference_matches_final_predictions']=True
    checks['final_lock_hash_matches']=json.loads((out/'final_complete.json').read_text())['lock_sha256']==hashlib.sha256((out/'selection_lock.json').read_bytes()).hexdigest()
    assert all(checks.values()),checks
    (out/'verification.json').write_text(json.dumps(dict(checks=checks,inference_roundtrip=errors),indent=2))
    print('VERIFIED',json.dumps(checks),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);verify(p.parse_args().out)

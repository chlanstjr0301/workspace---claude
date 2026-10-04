"""Portable facade of the already checked finite-input interface; no new policy."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code/runtime'))
import evaluate_new_data as api
api.R=ROOT
# API expects model_manifest at root; use a scoped in-memory path redirect through temporary workspace.
import argparse,shutil,tempfile
p=argparse.ArgumentParser();p.add_argument('--sensors',required=True);p.add_argument('--provenance',required=True);p.add_argument('--labels');p.add_argument('--output',required=True);a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='g1_fixed_api_') as tmp:
 r=Path(tmp);shutil.copytree(ROOT/'models',r/'models');shutil.copy2(ROOT/'configs/model_manifest.json',r/'model_manifest.json');api.R=r;api.evaluate(a)

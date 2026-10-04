from pathlib import Path
import subprocess,sys,shutil,json,hashlib,zipfile
root=Path('/content/hydraulic_ai');run=root/'runs/experiment_20261003_01'
def call(*args):subprocess.run([sys.executable,*args],cwd=root,check=True)
if not (run/'final_complete.json').exists():call('src/analyze.py','--out',str(run))
if not (run/'final_complete.json').exists():call('src/experiment.py','--out',str(run),'--stage','final')
call('src/analyze.py','--out',str(run))
call('src/report.py','--out',str(run))
call('src/verify.py','--out',str(run))
for folder in ['src','configs','scripts']:
 shutil.copytree(root/folder,run/'source_snapshot'/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for f in ['README.md','requirements.lock.txt']:shutil.copy2(root/f,run/'source_snapshot'/f)
manifest={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file() and p.name not in ['artifact_hashes.json','driver.log','process.json']}
(run/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile('/content/hydraulic_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in run.rglob('*'):
  if p.is_file():z.write(p,str(p.relative_to(root)))
print('ARTIFACTS_READY',Path('/content/hydraulic_results.zip').stat().st_size,flush=True)

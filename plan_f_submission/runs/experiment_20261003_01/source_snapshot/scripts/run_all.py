"""One-command reproducible run, unique output directory or explicit resume."""
import argparse,subprocess,sys,shutil,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--resume');args=p.parse_args()
out=Path(args.resume).resolve() if args.resume else root/'runs'/datetime.now(timezone.utc).strftime('experiment_%Y%m%dT%H%M%S_%fZ')
if not args.resume: out.mkdir(parents=True,exist_ok=False)
def call(*cmd):subprocess.run([sys.executable,*cmd],cwd=root,check=True)
if not (root/'configs/frozen').exists():call('src/prepare.py')
if not (out/'final_complete.json').exists():
    if not (out/'selection_lock.json').exists():call('src/experiment.py','--out',str(out),'--stage','develop')
    call('src/experiment.py','--out',str(out),'--stage','final')
call('src/analyze.py','--out',str(out));call('src/report.py','--out',str(out));call('src/verify.py','--out',str(out))
snapshot=out/'source_snapshot'
for folder in ['src','configs','scripts']:
    shutil.copytree(root/folder,snapshot/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for file in ['README.md','requirements.lock.txt']:
    if (root/file).exists():shutil.copy2(root/file,snapshot/file)
print('Completed:',out/'report.html')

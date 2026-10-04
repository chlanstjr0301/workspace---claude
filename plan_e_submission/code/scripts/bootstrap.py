"""Install only into this package's code/.venv; do not modify system Python."""
from pathlib import Path
import os,shutil,subprocess,sys
root=Path(__file__).resolve().parents[1];venv=root/'.venv';py=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
uv=shutil.which('uv')
if not uv:
 candidate=Path.home()/'.local/bin/uv'
 if candidate.is_file():uv=str(candidate)
if not py.exists():
 if uv:subprocess.run([uv,'venv',str(venv),'--python',sys.executable],check=True)
 else:subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
if uv:cmd=[uv,'pip','sync','--python',str(py),str(root/'requirements.lock.txt')]
else:cmd=[str(py),'-m','pip','install','-r',str(root/'requirements.lock.txt')]
subprocess.run(cmd,check=True)
print('Private interpreter:',py)

from pathlib import Path
import hashlib,json,sys
r=Path(__file__).resolve().parents[1];n=0;bad=[]
for line in (r/'SHA256SUMS').read_text().splitlines():
 h,name=line.split('  ',1);p=r/name;n+=1
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:bad.append(name)
print(json.dumps(dict(checked=n,passed=not bad,failures=bad),indent=2));sys.exit(bool(bad))

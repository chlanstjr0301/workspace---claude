"""Create and verify a new shareable archive without overwriting any archive."""
from common import *
import sys,zipfile

def package(dest):
 dest=Path(dest).resolve();dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():raise FileExistsError(dest)
 exclude_names={'MANIFEST_SHA256.json','package_verification.json'}
 def included(p):
  rel=p.relative_to(R)
  if not p.is_file() or rel.parts[0]=='replays' or '__pycache__' in rel.parts or p.name in exclude_names:return False
  if rel.parts[0]=='references' and p.suffix.lower() in ['.pdf','.txt','.png']:return False
  return True
 files=sorted(p for p in R.rglob('*') if included(p));manifest={str(p.relative_to(R)):sha(p) for p in files};js(R/'MANIFEST_SHA256.json',manifest);files.append(R/'MANIFEST_SHA256.json')
 with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,R.name+'/'+str(p.relative_to(R)))
 with zipfile.ZipFile(dest) as z:
  assert z.testzip() is None
  for path,h in manifest.items():assert hashlib.sha256(z.read(R.name+'/'+path)).hexdigest()==h
 verification=dict(path=str(dest),sha256=sha(dest),size_bytes=dest.stat().st_size,members=len(files),CRC_passed=True,all_member_SHA256_passed=True,omitted='local reference PDFs/full-text/screenshots and redundant replay copies; literature URLs/read notes/hashes included',created=pd.Timestamp.now(tz='UTC').isoformat());js(R/'package_verification.json',verification);js(dest.with_suffix('.verification.json'),verification);print(json.dumps(verification,ensure_ascii=False))
if __name__=='__main__':package(sys.argv[1])

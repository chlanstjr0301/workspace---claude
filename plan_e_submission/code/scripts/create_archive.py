"""Create a reproducible-content research archive, excluding secrets and environments."""
from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.package).resolve();out=Path(a.out).resolve()
if out.exists():raise SystemExit('Refuse to overwrite an existing archive')
forbidden={'__pycache__','.venv','.git','colab_sessions.json','tectonic.tar.gz'}
def include(f):
 rel=f.relative_to(root)
 return f.is_file() and not any(x in forbidden for x in rel.parts) and f!=out and not any(s in f.name.lower() for s in ['oauth','credentials','auth_token','colab_sessions.json']) and not f.name.startswith('preview_')
files=sorted(f for f in root.rglob('*') if include(f) and f.name!='MANIFEST_SHA256.json')
manifest={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
mp=root/'MANIFEST_SHA256.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));files.append(mp)
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in files:z.write(f,str(Path(root.name)/f.relative_to(root)))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 assert not any('/.venv/' in n or 'colab_sessions.json' in n or '/__pycache__/' in n for n in z.namelist())
digest=hashlib.sha256(out.read_bytes()).hexdigest();out.with_suffix('.zip.sha256').write_text(digest+'  '+out.name+'\n')
print(json.dumps({'zip':str(out),'bytes':out.stat().st_size,'files':len(files),'sha256':digest},indent=2))

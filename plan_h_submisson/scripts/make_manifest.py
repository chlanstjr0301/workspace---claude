"""Explicitly regenerate package hashes; excludes these two indexes to avoid recursion."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]
files=[]
for p in sorted(R.rglob('*')):
 rel=p.relative_to(R).as_posix()
 if p.is_symlink():raise RuntimeError('No symlinks permitted: '+rel)
 if not p.is_file() or rel in ['manifest.json','SHA256SUMS'] or any(x in p.relative_to(R).parts for x in ['.venv','__pycache__','work']):continue
 files.append(dict(path=rel,size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(R/'manifest.json').write_text(json.dumps(dict(package='PCA + R5 prediction-residual auxiliary alarm (G1)',scope='Curated user-custody publication and fixed-model reproduction archive',hash_exclusions=['manifest.json','SHA256SUMS','outputs created after extraction'],model_manifest='models/manifest.json',files=files),ensure_ascii=False,indent=2))
(R/'SHA256SUMS').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in files))
print(len(files),'files indexed')

from pathlib import Path
import zipfile,hashlib,json
root=Path('/content/hydraulic_ai_alt');out=Path('/content/hydraulic_alt_development.zip')
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for p in (root/'runs/pca_20261003_cpu').rglob('*'):
  if p.is_file():z.write(p,str(p.relative_to(root)))
 for p in root.glob('*.log'):z.write(p,'reports/remote_'+p.name)
lock=json.loads((root/'runs/pca_20261003_cpu/selection_lock.json').read_text())
print(json.dumps({'zip_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'primary':lock['primary'],'alternative':lock['alternative'],'if_primary':lock['if_primary'],'postprocess':lock['postprocess']}))

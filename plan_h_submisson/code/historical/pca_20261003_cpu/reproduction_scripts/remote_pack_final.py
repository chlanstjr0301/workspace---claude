from pathlib import Path
import zipfile,hashlib,json
root=Path('/content/hydraulic_ai_alt');run=root/'runs/pca_20261003_cpu'
assert (run/'final_summary.json').exists(),'final not complete'
files=[p for p in run.rglob('*') if p.is_file()]
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(run/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
files.append(run/'download_manifest.json')
out=Path('/content/hydraulic_alt_final.zip')
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,str(p.relative_to(root)))
 for p in root.glob('*.log'):z.write(p,'reports/final_remote_'+p.name)
print(json.dumps({'zip_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'files':len(files)}))

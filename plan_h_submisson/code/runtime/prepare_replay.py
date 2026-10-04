from common import *
import sys,shutil
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
for name in ['src','models','inputs','manifests','data']:
 shutil.copytree(R/name,out/name,ignore=shutil.ignore_patterns('__pycache__'))
for p in R.iterdir():
 if p.is_file() and p.suffix in ['.json','.sha256','.txt','.md','.sh']:shutil.copy2(p,out/p.name)
for sub in ['logs','figures','predictions']: (out/sub).mkdir(exist_ok=True)
# Packaged historical evidence records immutable hashes; absolute source paths are documentary only.
print(out)

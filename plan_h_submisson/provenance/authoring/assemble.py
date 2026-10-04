from pathlib import Path
import shutil,json,hashlib,csv,datetime,urllib.request,zipfile,platform,subprocess
P=Path('/home/lim/hydraulic_ai_alt');W=Path(__file__).resolve().parent;R=W/'pca_g1_paper_code_data';R.mkdir(exist_ok=False)
for n in ['paper/figures','paper/tables','paper/fonts','code/runtime','code/analysis','code/historical','configs','data/raw','data/analysis','models','results/frozen','results/history','provenance/history','provenance/references','environment/tools','scripts','verification','reviews']:(R/n).mkdir(parents=True,exist_ok=True)
inv=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(s,d,role):
 s=Path(s);d=R/d;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,d);inv.append(dict(source=str(s.relative_to(P)) if s.is_relative_to(P) else str(s),package=str(d.relative_to(R)),sha256=sha(d),role=role,modified=False))
def tree(src,dst,role,exclude=()):
 for p in src.rglob('*'):
  if p.is_file() and not any(x in p.relative_to(src).parts for x in ['__pycache__','replays',*exclude]) and p.suffix not in ['.pyc','.pdf','.png','.jpg','.zip']:
   cp(p,Path(dst)/p.relative_to(src),role)
F=P/'runs/pca_g1_frozen_validation_20261004_221914'
for p in (F/'src').glob('*.py'):cp(p,Path('code/runtime')/p.name,'unchanged frozen runtime, including later external input validation')
for sub,dest in [('inputs','data/analysis/inputs'),('manifests','configs/manifests'),('models','models'),('predictions','results/frozen/predictions')]:tree(F/sub,dest,'frozen selected artifacts')
for p in (F/'data').glob('*.csv'):cp(p,Path('data/raw')/p.name,'byte-preserved raw CSV')
for p in F.iterdir():
 if p.is_file() and p.suffix in ['.csv','.json','.md','.sha256','.txt']:
  cp(p,Path('results/frozen')/p.name,'frozen validation evidence')
for p in ['validation_protocol.json','validation_protocol.sha256','model_manifest.json']:cp(F/p,Path('configs')/p,'fixed execution protocol')
tree(F/'logs','provenance/history/frozen_logs','original logs')
for n in ['pca_literature_rescue_20261004_210504','pca_followup_v2_20261004_214810','pca_fn_fp_tradeoff_20261004_201516','adversarial_pca_20261004_103236','pca_20261003_cpu']:
 s=P/'runs'/n
 for sub in ['src','source','source_final','reproduction_source','reproduction_scripts']:
  if (s/sub).exists():tree(s/sub,f'code/historical/{n}/{sub}','historical original; not default entrypoint')
 for sub in ['predictions','tables','models','manifests','logs']:
  if (s/sub).exists():tree(s/sub,f'results/history/{n}/{sub}','related original research evidence')
 for p in s.iterdir():
  if p.is_file() and p.suffix in ['.csv','.json','.md','.txt','.sh','.sha256']:
   cp(p,Path('provenance/history')/n/p.name,'original registration, results or report')
 for sub in ['inputs','references']:
  if (s/sub).exists():tree(s/sub,f'provenance/history/{n}/{sub}','prior dependencies and notes',exclude=('models','predictions'))
for n in ['g1_experiment_usage_audit_20261004_225957','data_usage_audit_20261004_225053']:
 tree(P/'runs'/n,Path('provenance')/n,'read-only usage audit',exclude=('src',))
# Global source and preprocessing/split definitions, without unrelated virtual environments.
tree(P/'src','code/historical/initial_src','original preprocessing and experiment code')
tree(P/'scripts','code/historical/initial_scripts','original scripts')
tree(P/'configs','provenance/original_configs','original split and configuration')
# Replay logs only: do not mistake identical copies for independent datasets.
for s in (P/'runs').rglob('replays/*/logs'):
 tree(s,Path('provenance/replays')/s.relative_to(P/'runs'),'actual replay logs, not new candidates')
T=P/'deliverables/pca_paper_20261003_185434'
cp(T/'paper/manuscript_ko.tex','provenance/previous_manuscript_ko.tex','Korean format precedent, not G1 paper')
cp(T/'tools/tectonic','environment/tools/tectonic','existing compiler distribution')
cp(T/'tools/TECTONIC_LICENSE.txt','environment/tools/TECTONIC_LICENSE.txt','third-party license')
for p in (T/'paper/fonts').iterdir():cp(p,Path('paper/fonts')/p.name,'Korean font and license')
cp(F/'requirements.lock.txt','environment/requirements.lock.txt','verified original exact versions')
# Official style download and archival source metadata.
url='https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip'
b=urllib.request.urlopen(url,timeout=45).read();(R/'provenance/icml2026_official.zip').write_bytes(b)
with zipfile.ZipFile(R/'provenance/icml2026_official.zip') as z:
 for n in z.namelist():
  p=Path(n)
  if p.suffix in ['.sty','.bst']:(R/'paper'/p.name).write_bytes(z.read(n))
(R/'provenance/template_source.json').write_text(json.dumps(dict(url=url,year=2026,downloaded=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=hashlib.sha256(b).hexdigest(),english_option='default anonymous submission; no accepted/preprint',korean='extended research manuscript, inherited ICML preprint layout with kotex; not official submission'),indent=2))
(R/'provenance/source_inventory.csv').write_text('')
with (R/'provenance/source_inventory.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=inv[0].keys());w.writeheader();w.writerows(inv)
(R/'environment/build_host.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),threads=2),indent=2))
(W/'status.json').write_text(json.dumps(dict(stage='assembled immutable source evidence',package=str(R),files=len(inv)),indent=2))
print(R,len(inv))

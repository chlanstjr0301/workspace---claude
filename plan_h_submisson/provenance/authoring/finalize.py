from pathlib import Path
import json,csv,hashlib,shutil,zipfile,datetime
W=Path(__file__).resolve().parent;R=W/'pca_g1_paper_code_data';P=W.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=R/'scripts/build_papers.sh';s=p.read_text();s=s.replace('cd "$ROOT/paper"','if [[ ! -x "$COMPILER" ]]; then chmod u+x "$COMPILER"; fi\ncd "$ROOT/paper"');p.write_text(s)
p=W/'write_references.py';s=p.read_text().replace(" b=f'@{kind}"," venue=venue.replace('&',r'\\&')\n b=f'@{kind}");p.write_text(s)
p=R/'provenance/template_source.json';j=json.loads(p.read_text());j['korean']=j['korean'].replace('kotex','xeCJK / bundled Noto Serif CJK KR');p.write_text(json.dumps(j,ensure_ascii=False,indent=2))
(R/'code/build').mkdir(exist_ok=True)
for p in W.glob('*.py'):shutil.copy2(p,R/'code/build'/p.name)
for p in W.glob('*.log'):shutil.copy2(p,R/'verification'/p.name)
# Immutable original file verification, then explicitly classify new code.
p=R/'provenance/source_inventory.csv';rows=list(csv.DictReader(p.open()));bad=[]
for x in rows:
 q=R/x['package']
 if q.is_file() and sha(q)!=x['sha256']:bad.append(x['package'])
(R/'verification/source_preservation.json').write_text(json.dumps({'original_entries':len(rows),'byte_mismatches':bad,'passed':not bad},indent=2))
assert not bad,bad
seen={x['package'] for x in rows}
for base in ['scripts','code/analysis','code/build']:
 for q in (R/base).rglob('*'):
  if q.is_file() and q.suffix in ['.py','.sh'] and q.relative_to(R).as_posix() not in seen:
   rows.append({'source':'new publication packaging implementation; not historical original','package':q.relative_to(R).as_posix(),'sha256':sha(q),'role':'new portable entrypoint or documentation builder; no candidate search','modified':'new'})
with p.open('w') as f:
 w=csv.DictWriter(f,fieldnames=['source','package','sha256','role','modified']);w.writeheader();w.writerows(rows)
# Tie generated claims to exact manuscript locations without altering metrics.
p=R/'provenance/evidence_map.csv';rows=list(csv.DictReader(p.open()));keys=list(rows[0])+['manuscript_location']
for x in rows:x['manuscript_location']='KO/EN Table 2; controls Table 3 and Appendix tables; results/paper_metrics.csv' if 'metrics' in x['claim'] else 'KO/EN data/method/results sections and Appendices A–F'
for claim,evidence,location,code in [
 ('Raw dates, deduplication and roles','results/frozen/data_audit.json','KO/EN Section 3, Table 1, Figure 2','code/runtime/audit_data.py'),
 ('No unused unique observations and repeated exposure','provenance/g1_experiment_usage_audit_20261004_225957/count_summary.json','KO Section 5.3 / Appendix E; EN Section 5 / Appendix E','code/historical'),
 ('Causal pipeline and exact thresholds','configs/model_manifest.json','KO/EN Section 4 and Figure 1 / Appendix A','code/runtime/models.py; code/runtime/auxiliary.py; code/runtime/v2.py'),
 ('Fixed component training reproduction','verification/fixed_training/training_reproduction.json','KO/EN implementation section and Appendix F','scripts/train_fixed.py'),
 ('FN/FP paired controls plots','results/paper_metrics.csv','KO/EN results controls figure','code/analysis/build_assets.py'),
 ('Affected event case plots use saved measurements only','results/frozen/added_H_detections.csv','KO/EN case figure and Appendix D','code/analysis/build_assets.py')]:
 q=R/evidence;rows.append(dict(claim=claim,original_run='frozen validation and publication packaging; scope distinguished',rows='data/analysis/inputs/frozen_rows.csv',prediction=evidence,calculation=code,configuration='configs/model_manifest.json; configs/validation_protocol.json',sha256=sha(q) if q.is_file() else '',manuscript_location=location))
with p.open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
(R/'RUN_REVIEW.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
"$PY" scripts/verify_hashes.py
"$PY" scripts/reproduce.py --out "${1:-work/inference}" --check
"$PY" code/analysis/build_assets.py
bash scripts/build_papers.sh
''')
(R/'README.md').write_text((R/'README.md').read_text()+'\nA combined entrypoint after setup is `bash RUN_REVIEW.sh work/inference`. Python 3.14 must be installed; setup prefers `uv` and otherwise requires working `venv`/`ensurepip`. The independently installed validation environment is recorded under `verification/`; no virtual environment is shipped. Archived historical builders may retain source-location provenance; the documented runtime and build entrypoints resolve the extracted package root.\n')
with (R/'CHANGELOG.md').open('a') as f:f.write('\nPublication packaging corrections: R5 training pairs 11292→11296 after actual fixed reconstruction; predictor matrix orientation; escaped bibliography ampersand; xeCJK/font engine; official English T1 Times encoding; anonymous correspondence placeholder; indicator glyph; unclipped flow diagram and separated split labels. These are manuscript/build corrections, not model or prediction changes. Historical fit/test counts are left unchanged; this packaging round fits two selected PCA pipelines and one R5 per training reproduction, separately recorded.\n')
# Only remove generated bytecode/caches, never scientific artifacts.
for p in list(R.rglob('__pycache__')):shutil.rmtree(p)
print('Prepared final package',R)

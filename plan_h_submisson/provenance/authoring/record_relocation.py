from pathlib import Path
import json,hashlib,shutil,csv,subprocess
import pymupdf
W=Path(__file__).resolve().parent;R=W/'pca_g1_paper_code_data';E=Path('/tmp/pca_g1_publication_extract_20261005/pca_g1_paper_code_data');D=Path('/tmp/pca_g1_publication_work_20261005')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
# Original assembly/document-generator scripts are lineage evidence, not runtime dependencies.
A=R/'provenance/authoring';A.mkdir(exist_ok=True)
for p in (R/'code/build').glob('*.py'):
 shutil.copy2(p,A/p.name)
 if p.name not in ['write_review.py','write_references.py']:p.unlink()
 else:p.write_text(p.read_text().replace("R=Path(__file__).parent/'pca_g1_paper_code_data'","R=Path(__file__).resolve().parents[2]"))
p=R/'scripts/build_papers.sh';s=p.read_text().replace(' cd "$ROOT/reviews"',' "${PYTHON:-$ROOT/.venv/bin/python}" "$ROOT/code/build/write_review.py"\n cd "$ROOT/reviews"');p.write_text(s)
p=R/'provenance/template_source.json';p.write_text(p.read_text().replace('Noto Serif CJK KR','Noto Sans CJK KR'))
js(R/'data/provenance.example.json',dict(independently_collected=False,sensor_semantics_compatible=False,prior_usage='unknown',equipment_id='REPLACE_WITH_CONFIRMED_ID',acquisition_dates=[],units='REPLACE_WITH_CONFIRMED_UNITS',note='Metadata template only: not synthetic performance data. Set booleans only from acquisition/use evidence.'))
with (R/'README.md').open('a') as f:f.write('\n`data/provenance.example.json` is metadata format only; confirm rather than assume independence and unit compatibility. Paper build regenerates the self-review manifest from the rebuilt PDFs; use `PYTHON=/path/to/compatible/python bash scripts/build_papers.sh` if your environment is not `.venv`. The editable manuscripts are the authoritative final prose; `provenance/authoring/` preserves assembly-stage scripts only.\n')
# Save original actual command and results, excluding bulky regenerated copies.
V=R/'verification/relocation';V.mkdir(exist_ok=True)
for p in D.iterdir():
 if p.is_file():shutil.copy2(p,V/p.name)
for src,dst in [('training/training_reproduction.json','training_reproduction.json'),('inference/portable_inference_check.json','inference_check.json'),('inference/validation_report.json','validation_report.json'),('inference/run_status.json','run_status.json'),('api_replay/result.json','api_result.json'),('api_replay/predictions_lock.json','api_prediction_lock.json')]:shutil.copy2(D/src,V/dst)
for p in (D/'inference/logs').glob('*.log'):
 (V/'inference_logs').mkdir(exist_ok=True);shutil.copy2(p,V/'inference_logs'/p.name)
comparisons=[]
for rel in ['paper/paper_ko.pdf','paper/paper_en_icml.pdf','reviews/adversarial_review_ko.pdf']:
 a=pymupdf.open(R/rel);b=pymupdf.open(E/rel)
 texts=[p.get_text() for p in a];tb=[p.get_text() for p in b]
 assert texts==tb,rel
 comparisons.append(dict(path=rel,pages=len(a),identical_page_text=True,identical_bytes=sha(R/rel)==sha(E/rel)))
assert sha(R/'results/paper_metrics.csv')==sha(E/'results/paper_metrics.csv')
for rel in ['paper/tables/main.tex','paper/tables/controls.tex']:
 if (R/rel).exists():assert sha(R/rel)==sha(E/rel)
js(V/'document_comparison.json',dict(passed=True,documents=comparisons,metric_csv_byte_identical=True,all_25_confusion_metric_records_preserved=True,pdf_bytes_may_change_due_to_generation_metadata=True))
js(V/'summary.json',dict(passed=True,original_project_masked='/home/lim/hydraulic_ai_alt',method='bubblewrap read-only root; empty tmpfs mounted over original project; separate extracted archive and independent package environment',python='/tmp/pca_g1_publication_verify_env_20261005/bin/python',archive_sha256=sha(W/'relocation_candidate.zip'),steps=['1999 initial file hashes','25 stored-model prediction tables, exact row IDs and alarms','online/batch and finite-input validation','raw CSV fixed training; exact alarms and numeric tolerance','metric/table/figure generation from saved predictions','all three LaTeX PDFs built with identical page text','new-input schema check and fixed API on historical H smoke data'],new_tuning=False,independent_new_performance=False,environment_note='Read-only /dev/shm caused joblib serial fallback warning; numerical BLAS threads capped at 2; no parallel fit required.'))
# Update inventory paths for newly archived authoring code and append actual new portable sources.
p=R/'provenance/source_inventory.csv';rows=list(csv.DictReader(p.open()));original=[x for x in rows if x['modified']!='new'];seen={x['package'] for x in original}
for base in ['scripts','code/analysis','code/build','provenance/authoring']:
 for q in sorted((R/base).rglob('*')):
  if q.is_file() and q.suffix in ['.py','.sh'] and '__pycache__' not in q.parts and q.relative_to(R).as_posix() not in seen:
   original.append(dict(source='publication-round new source; not prior historical implementation',package=q.relative_to(R).as_posix(),sha256=sha(q),role='archived assembly provenance' if base.startswith('provenance') else 'portable package entrypoint / assets / document generation',modified='new'))
with p.open('w') as f:
 w=csv.DictWriter(f,fieldnames=['source','package','sha256','role','modified']);w.writeheader();w.writerows(original)
print('Recorded independent relocation; portable review rebuild added')
# Copy only changed executable paths to already extracted package for a bounded document rebuild check.
for rel in ['scripts/build_papers.sh','code/build/write_review.py','code/build/write_references.py']:shutil.copy2(R/rel,E/rel)

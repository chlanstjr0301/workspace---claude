from pathlib import Path
import json,csv,hashlib,shutil,subprocess,zipfile,datetime,re
import pymupdf
W=Path(__file__).resolve().parent;R=W/'pca_g1_paper_code_data';P=W.parents[1];E=Path('/tmp/pca_g1_publication_extract_20261005/pca_g1_paper_code_data');D=Path('/tmp/pca_g1_publication_work_20261005');V=R/'verification/relocation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
for name in ['setup_exact.log','setup_exact_retry.log','documents_with_manifest.log']:shutil.copy2(D/name,V/name)
# The new review builder refreshes target digests after compiling; validate its actual output.
audit=json.loads((E/'reviews/review_manifest.json').read_text())
assert all(sha(E/p)==h for p,h in audit['files'].items())
for name in ['paper_ko.pdf','paper_en_icml.pdf']:
 assert [p.get_text() for p in pymupdf.open(R/'paper'/name)]==[p.get_text() for p in pymupdf.open(E/'paper'/name)]
assert len(pymupdf.open(E/'reviews/adversarial_review_ko.pdf'))==7
js(V/'updated_build_check.json',dict(passed=True,setup_command='bash scripts/setup_env.sh',setup_status='passed on normal host into independently extracted folder',restricted_setup_attempt='failed uv libc discovery under read-only namespace; log preserved, no model result affected',build_command='bash scripts/build_papers.sh',build_status='passed with original project masked using extracted .venv; three PDFs',paper_page_text_identical=True,rebuilt_review_target_hashes_match=True,final_manuscript_pdf_identity='Archive retains visually reviewed final originals; regenerated PDF byte metadata can differ without text changes'))
# Verify final selected/source identity and exact reviewed manuscript match.
m=json.loads((R/'reviews/review_manifest.json').read_text());assert all(sha(R/p)==h for p,h in m['files'].items())
for f in ['baseline.joblib','R5.joblib']:assert sha(R/'models'/f)==json.loads((R/'models/manifest.json').read_text())['hashes'][f]
# Final runtime entrypoints equal those actually exercised in the extracted folder.
checked=[]
for base in ['code/runtime','code/analysis','scripts']:
 for f in sorted((R/base).rglob('*')):
  if f.is_file() and f.suffix in ['.py','.sh']:
   rel=f.relative_to(R)
   if (E/rel).is_file():
    assert sha(f)==sha(E/rel),rel
    checked.append(str(rel))
js(V/'final_executable_identity.json',dict(passed=True,checked=checked,scope='Final scripts/core/analysis identical to extracted executed version; make_manifest hashes itself only via package manifest exclusion scope',note='Not all archived precursor scripts were rerun; they are original evidence and not default execution dependencies'))
js(R/'verification/publication_round_execution.json',dict(scope='Additional package verification only; do not add to historical audit candidate count',new_candidates=0,new_selection_rounds=0,stored_model_reproduction_invocations=2,selected_fixed_training_invocations=2,PCA_pipeline_fits=4,PCA_internal_fit_calls=8,R5_fit_calls=2,distinct_hyperparameter_configurations_added=0,raw_input_and_selected_models_preserved=True,independent_new_acquisition_evaluations=0,comparison_basis='in-project minimal verification plus independently extracted project-masked reproduction',historical_audit_unchanged=True))
# Preserve packaging scripts as clearly labeled source evidence.
for f in W.glob('*.py'):shutil.copy2(f,R/'provenance/authoring'/f.name)
# Update new-code inventory after final assembly; originals stay byte-identical.
ip=R/'provenance/source_inventory.csv';rows=list(csv.DictReader(ip.open()));rows=[x for x in rows if x['modified']!='new']
for x in rows:assert sha(R/x['package'])==x['sha256'],x['package']
for base in ['scripts','code/analysis','code/build','provenance/authoring']:
 for f in sorted((R/base).rglob('*')):
  if f.is_file() and f.suffix in ['.py','.sh'] and '__pycache__' not in f.parts:
   rows.append(dict(source='publication-round source, not historic experiment',package=str(f.relative_to(R)),sha256=sha(f),role='archival authoring/assembly record' if base.startswith('provenance') else 'portable code',modified='new'))
with ip.open('w') as f:
 w=csv.DictWriter(f,fieldnames=['source','package','sha256','role','modified']);w.writeheader();w.writerows(rows)
for p in list(R.rglob('__pycache__')):shutil.rmtree(p)
# No unresolved citations, missing glyphs, or overfull boxes in the final builds.
issues=[]
for folder in ['paper','reviews']:
 for p in (R/folder).glob('*.log'):
  for line in p.read_text(errors='replace').splitlines():
   if any(t in line for t in ['Overfull \\hbox','Missing character:','Citation `','Reference `']) and ('undefined' in line or not ('Citation' in line or 'Reference' in line)):issues.append({'file':str(p.relative_to(R)),'line':line})
assert not issues,issues
js(R/'verification/final_consistency.json',dict(passed=True,review_model_and_manuscript_hashes_match=True,original_models_unchanged=True,source_inventory_originals_unchanged=True,final_tex_critical_warnings=issues,independent_new_data=False,scope='fixed implementation, document arithmetic and package reproducibility; not independent generalization'))
subprocess.run([str(P/'.venv/bin/python'),str(R/'scripts/make_manifest.py')],check=True)
subprocess.run([str(P/'.venv/bin/python'),str(R/'scripts/verify_hashes.py')],check=True,stdout=(W/'final_internal_hash_check.json').open('w'))
Z=P/'deliverables/pca_g1_paper_code_data_20261005_001500.zip'
assert not Z.exists()
with zipfile.ZipFile(Z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(R.rglob('*')):
  assert not p.is_symlink(),p
  if p.is_file():z.write(p,Path(R.name)/p.relative_to(R))
with zipfile.ZipFile(Z) as z:
 assert z.testzip() is None
 final_extract=Path('/tmp/pca_g1_final_archive_check_20261005');final_extract.mkdir(exist_ok=False);z.extractall(final_extract)
 q=final_extract/R.name
 check=subprocess.run([str(P/'.venv/bin/python'),str(q/'scripts/verify_hashes.py')],capture_output=True,text=True,check=True)
 r=json.loads((q/'reviews/review_manifest.json').read_text());assert all(sha(q/p)==h for p,h in r['files'].items())
 for rel in ['paper/paper_ko.pdf','paper/paper_en_icml.pdf','reviews/adversarial_review_ko.pdf']:assert (q/rel).stat().st_size>1000
 h=sha(Z);Z.with_suffix('.zip.sha256').write_text(h+'  '+Z.name+'\n')
 receipt=dict(zip=str(Z),bytes=Z.stat().st_size,MiB=round(Z.stat().st_size/1024**2,2),sha256=h,crc_pass=True,final_fresh_extraction=str(final_extract),hash_check=json.loads(check.stdout),review_target_hashes_match=True,files=len(z.namelist()),runtime_same_as_relocation_validated=True,papers={rel:dict(path=str(R/rel),pages=len(pymupdf.open(R/rel)),sha256=sha(R/rel)) for rel in ['paper/paper_ko.pdf','paper/paper_en_icml.pdf','reviews/adversarial_review_ko.pdf']})
 js(W/'delivery_receipt.json',receipt);js(Z.with_suffix('.verification.json'),receipt);js(W/'status.json',dict(stage='complete',receipt=str(W/'delivery_receipt.json')))
 print(json.dumps(receipt,ensure_ascii=False,indent=2))

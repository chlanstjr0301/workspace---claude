from core import *
R=Path(__file__).resolve().parents[1];P=R.parents[1];lock=json.loads((R/'frozen_selection.json').read_text());t=pd.read_csv(R/'all_trials.csv');p=t.query("block=='pooled_inner' and scope=='operational'");assert len(p)==54;assert len(p.candidate.unique())==18
expected=p[(p.kind=='PCA')&(p.FPR<=.01)].sort_values(['Recall','F2','F1','AP','candidate','target'],ascending=[False,False,False,False,True,True]).iloc[0];assert expected.candidate==lock['selected']['candidate'];assert expected.target==lock['selected']['target']
checks=[];summary=pd.read_csv(R/'baseline_vs_improved.csv')
for stage in ['outer_development','historical']:
 for alias in ['baseline','selected','IF_same_W20','selected_consecutive2','selected_two_of_three']:
  r=loadrows(R/'predictions'/f'{stage}_{alias}.csv');m=metrics(r.label,r.score,r.prediction);s=summary[(summary.evaluation==stage)&(summary.alias==alias)&(summary.scope=='operational')].iloc[0]
  for k,v in m.items():assert np.isclose(v,s[k],equal_nan=True), (stage,alias,k)
  assert r.row_id.is_unique;assert len(r)==(4347 if stage=='historical' else 1095);checks.append(dict(stage=stage,alias=alias,metrics_match=True,n=len(r)))
for file,h in lock['code_sha256'].items():assert sha(R/'src'/file)==h
assert sha(R/'frozen_selection.json')==json.loads((R/'evaluation_lock.json').read_text())['selection_sha256']
for item in json.loads((R/'manifests/data_audit.json').read_text()):assert sha(P/'data'/item['file'])==item['sha256']
assert not json.loads((R/'logs/failures.json').read_text())
check=json.loads((R/'saved_inference_validation.json').read_text());assert sum(x['n'] for x in check)==13041
# Coverage sums include the adaptive short-window routes as native, not only full W20.
c=pd.read_csv(R/'tables/native_coverage.csv');q=c[c.candidate!='P0_Q'];assert ((q.native_normal+q.fallback_normal)==q.normal).all();assert ((q.native_anomaly+q.fallback_anomaly)==q.anomaly).all()
# Paired errors counts sum to the observed TP/FP change.
a=pd.read_csv(R/'paired_errors.csv');a=a[(a.evaluation=='historical')&(a.comparison=='controlled_baseline')];assert (a.change=='new_TP').sum()-(a.change=='lost_TP').sum()==313-329;assert (a.change=='new_FP').sum()-(a.change=='removed_FP').sum()==3-1
required=['inventory.json','reproduction_report_ko.md','critique_matrix.csv','hypotheses.md','protocol.json','all_trials.csv','frozen_selection.json','baseline_vs_improved.csv','paired_errors.csv','requirements.lock.txt','RUN_REVIEW.sh','review_report_ko.md','next_data_requirements_ko.md','rubric_evidence.csv']
for p in required:assert (R/p).is_file()
js(R/'final_verification.json',dict(checks=checks,candidates=54,additional_policies=2,all_required_outputs_present=True,selection_and_source_unchanged=True,original_data_preserved=True,saved_model_inference_rows=13041))
js(R/'checkpoint.json',dict(stage='complete',verifications_passed=True,new_candidate_improved=False,selected=lock['selected']['candidate'],decision='retain historical baseline pending new data; no reselection of locked candidate'))
print('Verified all required outputs and numerical comparisons')

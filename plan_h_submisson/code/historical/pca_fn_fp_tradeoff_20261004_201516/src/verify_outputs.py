from common import *
lock=json.loads((R/'frozen_selection.json').read_text());t=pd.read_csv(R/'all_trials.csv');s=pd.read_csv(R/'baseline_vs_candidates.csv');assert len(t)==36;assert (t.status=='completed').all();assert lock['final_unique_count']==3
assert lock['A_primary'] is None;assert lock['A_descriptive_best']['candidate']=='A_q0.0020'
for k in ['conservative','primary','auxiliary']:
 assert lock['criteria_selections'][k]['candidate']=='B_W3_q0.0020';assert t[k+'_eligible'].sum()==14
validation=json.loads((R/'validation_report.json').read_text());assert validation['A_rank_metrics_invariant'];assert sum(v['n'] for v in validation['separate_inference'])==13041;assert validation['frozen_hash']==sha(R/'frozen_selection.json')
for candidate,tp,fn,fp in [('baseline',329,28,1),('A_q0.0020',324,33,0),('B_W3_q0.0020',323,34,0)]:
 x=s[(s.evaluation=='historical')&(s.scope=='operational')&(s.candidate==candidate)].iloc[0];assert (x.TP,x.FN,x.FP)==(tp,fn,fp)
# These checks verify the recorded run claims; they are never used to fit or select a candidate.
for stage in ['selection','historical']:
 a=loadrows(R/'predictions'/f'{stage}_frozen_A_q0.0020.csv');b=loadrows(R/'predictions'/f'{stage}_frozen_B_W3_q0.0020.csv');mask=a.native_W20.to_numpy();assert np.array_equal(a.score.to_numpy()[mask],b.score.to_numpy()[mask]);assert np.array_equal(a.prediction.to_numpy()[mask],b.prediction.to_numpy()[mask])
required=['inventory.json','protocol.json','hypotheses.md','reproduction_report_ko.md','all_trials.csv','threshold_sweep.csv','frozen_selection.json','baseline_vs_candidates.csv','block_metrics.csv','paired_errors.csv','requirements.lock.txt','RUN_EXPERIMENT.sh','validation_report.json','review_report_ko.md','review_report_ko.html']
for f in required:assert (R/f).is_file()
js(R/'final_verification.json',dict(required_outputs_present=True,all_candidates_completed=36,historical_evaluated=3,saved_inference_rows=13041,A_and_B_common_W20_identical=True,threshold_and_selection_not_retuned=True,baseline_preserved=True))
js(R/'checkpoint.json',dict(stage='complete',conclusion='no sustained improvement; retain original PCA',frozen_selection_sha256=sha(R/'frozen_selection.json')));print('All final output checks passed')

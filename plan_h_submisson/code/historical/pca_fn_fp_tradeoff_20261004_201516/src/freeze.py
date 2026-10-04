from common import *
d=loadrows(R/'inputs/frozen_rows.csv');e=Engine(d);t=pd.read_csv(R/'all_trials.csv');proto=json.loads((R/'protocol.json').read_text());base=joblib.load(R/'models/baseline.joblib');base_m=json.loads((R/'stage_a_summary.json').read_text())['baseline'];eligible=t[t.status=='completed'];selections={}
for rule in proto['criteria']:selections[rule]=pick(eligible[eligible[rule+'_eligible']].to_dict('records'))
a=eligible[eligible.experiment=='A'];a_main=pick(a[a.primary_eligible].to_dict('records'));a_descriptive=pick(a.to_dict('records'))
configs=[dict(candidate='baseline',system='original',threshold=base['threshold'],target=.01,reason='original exact operating point')]
if a_main:configs.append(dict(candidate=a_main['candidate'],system=a_main['system'],threshold=a_main['threshold'],target=a_main['target'],reason='A primary eligible'))
else:configs.append(dict(candidate=a_descriptive['candidate'],system=a_descriptive['system'],threshold=a_descriptive['threshold'],target=a_descriptive['target'],reason='A descriptive F2-best control; NOT eligible because Recall not increased'))
for rule,x in selections.items():
 if x and x['candidate'] not in [q['candidate'] for q in configs]:configs.append(dict(candidate=x['candidate'],system=x['system'],threshold=x['threshold'],target=x['target'],reason=rule+' selection'))
# Deduplicate systems at exact same threshold; original target .01 duplicate is not new.
unique=[]
for x in configs:
 if not any(y['system']==x['system'] and y['threshold']==x['threshold'] for y in unique):unique.append(x)
assert len(unique)<=5
for x in unique:
 p=R/'models'/('baseline.joblib' if x['system']=='original' else x['system']+'.joblib');x['model_path']=str(p.relative_to(R));x['model_sha256']=sha(p)
for sp in ['selection','test']:d[d.split==sp][['row_id','source_file','source_row','label','burst_id']].to_csv(R/'manifests'/f'evaluation_rows_{sp}.csv',index=False)
js(R/'frozen_selection.json',dict(created=pd.Timestamp.now(tz='UTC').isoformat(),baseline=base_m,baseline_threshold=base['threshold'],A_primary=a_main,A_descriptive_best=a_descriptive,criteria_selections=selections,final_configs=unique,final_unique_count=len(unique),decision='primary candidate eligible' if selections['primary'] else 'no improvement candidate; retain baseline',postprocessing='none',normal_training_only=True,development_labels_used_for_selection=True,threshold_values_fit_on_normal_calibration_only=True,models={p.name:sha(p) for p in (R/'models').glob('*.joblib')},code_sha256={p.name:sha(p) for p in (R/'src').glob('*.py')},protocol_sha256=sha(R/'protocol.json'),B_hypothesis_sha256=sha(R/'B_hypothesis_lock.json'),evaluation_rows_sha256={s:sha(R/'manifests'/f'evaluation_rows_{s}.csv') for s in ['selection','test']},test_seen_for_baseline_reproduction_only=True,prior_test_already_public=True))
print('Frozen',[(x['candidate'],x['threshold']) for x in unique]);print('criteria', {k:v['candidate'] if v else None for k,v in selections.items()})

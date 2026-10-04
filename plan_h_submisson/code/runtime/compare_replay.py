from common import *
import sys
source=Path(sys.argv[1]);checks=[]
for name in ['metrics.csv','events.csv','normal_exposure.csv','paired_changes.csv','row_timing_and_residuals.csv','mandatory_conditions.csv','control_effects.csv']:
 a=pd.read_csv(source/name);b=pd.read_csv(R/name);pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-10,atol=1e-10);checks.append(dict(file=name,same_values=True,exact_bytes=sha(source/name)==sha(R/name)))
for a in (source/'predictions').glob('*.csv'):
 b=R/'predictions'/a.name;check=sha(a)==sha(b);assert check,a.name
checks.append(dict(all_prediction_csv_exact_bytes=True))
js(R/'portable_replay.json',dict(passed=True,replay_root=str(R),original_project_not_read_by_execution=True,checks=checks))
print('All fixed predictions and analysis tables reproduced')

from v2 import *

def main():
 d,e,b0,r5=context();records=[];assert d.row_id.is_unique
 for file,g in d.groupby('source_file',sort=False):
  raw=pd.read_csv(R/'data'/file);raw.TimeStamp=pd.to_datetime(raw.TimeStamp);sub=['TimeStamp']+SENSORS+['Equipment_state'];dup=raw.duplicated(subset=sub,keep='first');clean=raw[~dup]
  assert g.source_row.tolist()==(clean.index+1).tolist();assert np.array_equal(g.TimeStamp.to_numpy(),clean.TimeStamp.to_numpy());assert np.allclose(g[SENSORS].to_numpy(),clean[SENSORS].to_numpy(),atol=0,rtol=0);assert np.array_equal(g.label,clean.Equipment_state)
  dt=clean.TimeStamp.diff().dt.total_seconds();records.append(dict(source_file=file,sha256=sha(R/'data'/file),original_rows=len(raw),retained_rows=len(clean),duplicates_removed=int(dup.sum()),duplicate_source_rows=(raw.index[dup]+1).tolist(),missing_count=int(raw[sub].isna().sum().sum()),negative_time_steps=int((dt<0).sum()),gaps_over_05=int((dt>.5).sum()),maximum_gap_seconds=float(dt.max()),median_step_seconds=float(dt.median()),date_min=str(clean.TimeStamp.min()),date_max=str(clean.TimeStamp.max()),labels=clean.Equipment_state.unique().tolist()))
 t=d.index[d.split=='train'].to_numpy();c=d.index[(d.split=='calibration')&(d.label==0)].to_numpy();fits=[]
 for w,entry in b0['entries'].items():
  ii=t[e.starts_for(w)[t]>=0];x=e.x(w,ii);bu=entry['bundle'];x=x[:,bu['keep']];sc=bu['scaler'];pc=bu['model'];z=sc.transform(x);cov=np.cov(z,rowvar=False);ev=pc.explained_variance_
  a=float(np.max(abs(sc.mean_-x.mean(0))));b=float(np.max(abs(sc.var_-x.var(0))));eig=float(np.max(abs(pc.components_@cov-ev[:,None]*pc.components_)));assert np.allclose(sc.mean_,x.mean(0),rtol=1e-10,atol=1e-10);assert np.allclose(sc.var_,x.var(0),rtol=1e-10,atol=1e-10);assert eig<1e-8;assert pc.n_components_<bu['rank']
  cc=c[e.starts_for(w)[c]>=0];qq=score(bu,e.x(w,cc))['Q'];ref=float(np.quantile(qq,.95,method='higher'));assert np.isclose(ref,entry['reference'],rtol=1e-12,atol=1e-12)
  fits.append(dict(window=w,normal_train_rows=len(ii),mean_max_error=a,variance_max_error=b,eigenpair_max_error=eig,calibration_Q95=ref,stored_reference=entry['reference'],no_refit=True));d.loc[ii,['row_id','source_file','source_row','split','train_role']].to_csv(R/'manifests'/f'B0_fit_W{w}.csv',index=False)
 cp=base_predictions(e,b0,c);h=float(np.quantile(cp.score,.99,method='higher'));assert h==b0['threshold']
 roles=[]
 for (split,label),g in d.groupby(['split','label'],sort=False):roles.append(dict(original_split=split,label=int(label),rows=len(g),bursts=g.burst_id.nunique(),start=str(g.TimeStamp.min()),end=str(g.TimeStamp.max())))
 pd.DataFrame(roles).to_csv(R/'data_roles.csv',index=False);js(R/'data_audit.json',dict(raw_files=records,B0_train_moment_eigenpair_checks=fits,B0_calibration_threshold=h,all_raw_rows_assigned_except_duplicate=True,split_row_ids_disjoint=True,normal_anomaly_dates_confounded=True,physical_units_and_session_ids_not_documented=True,calibration_anomalies_not_used_for_frozen_threshold=True))
 print('Raw rows, duplicate mapping, train moments/eigenpairs and calibration parameters verified without fitting')
if __name__=='__main__':main()

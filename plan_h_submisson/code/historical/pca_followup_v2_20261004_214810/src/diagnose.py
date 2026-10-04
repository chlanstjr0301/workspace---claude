from auxiliary import *
import subprocess,sys,tempfile
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def main():
 d=loadrows(R/'inputs/frozen_rows.csv');e=Engine(d);b0=joblib.load(R/'models/baseline.joblib');r5=joblib.load(R/'models/R5.joblib');old=json.loads((R/'inputs/previous/selected_config.json').read_text());theta=old['configs']['R5_A0']['threshold'];blocks=json.loads((R/'manifests/blocks.json').read_text());ids=np.array(sorted(blocks['S1']+blocks['S2']));b,bm=evaluated(e.predict(b0,ids),b0['threshold']);s=aux_score(r5,e,b0,ids,b.score);c,cm=combine(b,s,theta,'R5',b0['threshold'])
 expected_b=pd.read_csv(R/'inputs/previous/baseline_S_pooled.csv');expected_c=pd.read_csv(R/'inputs/previous/R5_A0_S_pooled.csv');expected_a1=pd.read_csv(R/'inputs/previous/R5_A1_S_pooled.csv')
 assert b.row_id.tolist()==expected_b.row_id.tolist()==expected_c.row_id.tolist();assert np.allclose(b.score,expected_b.score,rtol=1e-10,atol=1e-10);assert np.allclose(c.aux_score,expected_c.aux_score,equal_nan=True,rtol=1e-10,atol=1e-10);assert np.array_equal(c.prediction,expected_c.prediction);assert np.array_equal(c.prediction,expected_a1.prediction)
 # Independently load the baseline in another interpreter, preserving original test partition resets.
 parts=[]
 with tempfile.TemporaryDirectory(dir=R/'logs') as temp:
  for f,g in d[d.split=='test'].groupby('source_file',sort=False):
   ip=Path(temp)/f;op=Path(temp)/(f+'.out');g[['TimeStamp']+SENSORS].to_csv(ip,index=False);z=subprocess.run([sys.executable,str(R/'src/infer_system.py'),'--csv',str(ip),'--out',str(op),'--model',str(R/'models/baseline.joblib'),'--threshold',str(b0['threshold'])],capture_output=True,text=True);assert z.returncode==0,z.stderr;a=pd.read_csv(op);a['row_id']=g.row_id.to_numpy();parts.append(a)
 h=pd.concat(parts);expected_h=pd.read_csv(R/'inputs/original_expected_test.csv');assert h.row_id.tolist()==expected_h.row_id.tolist();assert np.array_equal(h.prediction,expected_h.prediction);assert np.allclose(h.score,expected_h.score,atol=1e-10,rtol=1e-10)
 # Baseline identity is recovered, not replaced: artifact name is 'original', output ID 'baseline'.
 reproduction=dict(baseline_original_id='baseline',bundle_name=b0.get('name'),notation='B0 is this round alias; A0 previously meant the original alarm function, not a competing model ID',baseline= bm,old_R5=cm,old_A0_A1_same_selection_predictions=True,old_A0_A1_identical_config=old['configs']['R5_A0']['threshold']==old['configs']['R5_A1']['threshold'],historical=metrics(expected_h.label,h.score,h.prediction),fresh_process_baseline_match=True,stored_R5_match=True,R5_V_previously_evaluated=False)
 js(R/'reproduction.json',reproduction);b.to_csv(R/'predictions/reproduced_B0_S.csv',index=False);c.to_csv(R/'predictions/reproduced_C0_S.csv',index=False)
 # Recover detailed causal sensor prediction and window provenance on development rows only.
 def details(ii):
  base,_=evaluated(e.predict(b0,ii),b0['threshold']);a=base.copy();ss=aux_score(r5,e,b0,ii,base.score);a['R5_score']=ss;a['R5_threshold']=theta;a['R5_h']=np.isfinite(ss)&(ss>theta);a['C0_prediction']=a.prediction.astype(bool)|a.R5_h;a['block']=['S1' if k in blocks['S1'] else 'S2' if k in blocks['S2'] else str(d.loc[k,'split']) for k in ii];valid,u,z=lagdata(e,r5['scaler'],ii,2);pred=(u-r5['u_mean'])@r5['coef']+r5['z_mean'];predraw=r5['scaler'].inverse_transform(pred);err=z-pred
  for j,sensor in enumerate(SENSORS):
   a[sensor+'__predicted']=np.nan;a.loc[a.index[valid],sensor+'__predicted']=predraw[:,j];a[sensor+'__standardized_residual']=np.nan;a.loc[a.index[valid],sensor+'__standardized_residual']=err[:,j]
  a['R5_target_time']=a.TimeStamp;a['R5_alarm_available_time']=a.TimeStamp;a['R5_forecast_issue_time']=pd.NaT;a['R5_input_start_time']=pd.NaT;a['R5_input_rows']='';vv=np.asarray(ii)[valid];a.loc[vv,'R5_forecast_issue_time']=d.loc[vv-1,'TimeStamp'].to_numpy();a.loc[vv,'R5_input_start_time']=d.loc[vv-2,'TimeStamp'].to_numpy();a.loc[vv,'R5_input_rows']=[json.dumps(d.loc[[i-2,i-1],'row_id'].tolist()) for i in vv];a['R5_available']=valid;a['warmup']=~valid
  a['B0_input_rows']=[json.dumps(d.loc[e.starts_for(int(w))[i]:i,'row_id'].tolist()) for i,w in zip(ii,a.route)];a['session_boundary']=a.burst_pos==1;a['B0_feature_json']=[json.dumps(dict(zip(b0['entries'][int(w)]['bundle']['names'],e.x(int(w),np.array([i]))[0]))) for i,w in zip(ii,a.route)]
  return a
 detail=details(ids);detail['B0_prediction']=detail.prediction;changed=detail[detail.C0_prediction.astype(int)!=detail.prediction];changed=changed.copy();changed['change']=np.where(changed.label==1,'new_TP','new_FP');changed.to_csv(R/'tables/diagnostic_changed_rows.csv',index=False)
 # Pre-fixed diagnostic window = original W20 past19 + R5 lag2 future2, constrained to original file/split.
 surroundings=[];summary=[]
 for idx,row in changed.iterrows():
  g=d[(d.source_file==row.source_file)&(d.split==row.split)&(d.train_role==row.train_role)];ii=g.index.to_numpy();pos=int(np.where(ii==idx)[0][0]);ni=ii[max(0,pos-19):pos+3];q=details(ni);q['case_row_id']=row.row_id;q['relative_row']=ni-idx;surroundings.append(q)
  session=detail[detail.burst_id==row.burst_id];sp=session.index.to_numpy();at=int(np.where(sp==idx)[0][0]);prev=[bool(session.R5_h.iloc[at-j]) if at>=j else False for j in [1,2]];nex=bool(session.R5_h.iloc[at+1]) if at+1<len(session) else False
  summary.append(dict(row_id=row.row_id,change=row.change,block=row.block,burst_id=row.burst_id,burst_pos=int(row.burst_pos),B0_score=float(row.score),R5_score=float(row.R5_score),threshold=theta,previous_h1=prev[0],previous_h2=prev[1],next_h_diagnostic_only=nex,isolated_both_neighbors=not prev[0] and not nex))
  fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True)
  for j,sensor in enumerate(SENSORS):
   for _,seg in q.groupby('burst_id',sort=False):axes[j].plot(seg.TimeStamp,seg[sensor],'.-',label='observed',lw=.8);axes[j].plot(seg.TimeStamp,seg[sensor+'__predicted'],'x--',label='R5 predicted',lw=.7)
   axes[j].set_ylabel(sensor);axes[j].axvline(row.TimeStamp,color='red');axes[j].axvspan(row.TimeStamp,q.TimeStamp.max(),alpha=.08,color='orange')
  for _,seg in q.groupby('burst_id',sort=False):axes[3].plot(seg.TimeStamp,seg.R5_score/theta,'.-',label='R5/theta');axes[3].plot(seg.TimeStamp,seg.score/b0['threshold'],'.-',label='B0/threshold')
  axes[3].axhline(1,color='black',ls='--');axes[3].axvline(row.TimeStamp,color='red');axes[3].set_yscale('symlog');axes[3].legend();fig.suptitle(f'{row.change}: {row.row_id}, {row.block}; future orange = diagnosis only');fig.autofmt_xdate();fig.tight_layout();fig.savefig(R/'figures'/f'diagnosis_{row.row_id.replace(":","_")}.png',dpi=130);plt.close(fig)
 pd.concat(surroundings).to_csv(R/'tables/diagnostic_neighborhoods.csv',index=False);pd.DataFrame(summary).to_csv(R/'tables/diagnostic_h_history.csv',index=False)
 # Similar sensor contexts elsewhere in normal T/C; never a model fitting/selection rule.
 normal=d.index[(d.label==0)&d.split.isin(['train','calibration'])].to_numpy();normal=normal[e.starts_for(3,True)[normal]>=0];sc=r5['scaler'].transform(e.raw);contexts=np.concatenate([sc[normal-2],sc[normal-1],sc[normal]],axis=1);near=[]
 for idx,row in changed.iterrows():
  v=np.r_[sc[idx-2],sc[idx-1],sc[idx]];dist=np.sqrt(np.sum((contexts-v)**2,axis=1));used={row.burst_id};count=0
  for j in np.argsort(dist):
   k=normal[j];bid=d.loc[k,'burst_id']
   if bid in used:continue
   used.add(bid);count+=1;z=details(np.array([k]));z['case_row_id']=row.row_id;z['context_distance']=dist[j];z['neighbor_rank']=count;near.append(z)
   if count==3:break
 pd.concat(near).to_csv(R/'tables/similar_normal_contexts.csv',index=False)
 # FPR feasibility from stored baseline only; do not evaluate R5 on V.
 oldbm=pd.read_csv(R/'inputs/previous/block_metrics.csv');oldV=pd.read_csv(R/'inputs/previous/baseline_V.csv');vm=metrics(oldV.label,oldV.score,oldV.prediction);budget=[]
 for name,mm in [(n,oldbm[(oldbm.candidate=='baseline')&(oldbm.block==n)].iloc[0].to_dict()) for n in ['S1','S2','S_pooled']]+[('V',vm)]:
  n=int(mm['TN']+mm['FP']);budget.append(dict(block=name,N0=n,N1=int(mm['TP']+mm['FN']),FN=int(mm['FN']),FP=int(mm['FP']),FPR=mm['FPR'],FP_absolute_cap=allowed_fp(n,.01),extra_FP_budget=allowed_fp(n,.001),baseline_feasible=int(mm['FP'])<=allowed_fp(n,.01)))
 pd.DataFrame(budget).to_csv(R/'tables/baseline_feasibility.csv',index=False)
 text=f'''# S2 추가 FP의 사전 진단

기존 실제 출력 ID는 baseline, 저장 bundle name은 {b0.get('name')}이다. 이번 B0는 그 모델의 별칭이다. 이전 수식 A0는 원경보 함수였다. 원P1 W20 k2 Q 및 P0 fallback을 다른 최고 점수 모델로 교체하지 않았다.

저장 모델로 S1+S2의 B0(18TP/3FN/8FP), C0=기존R5_A0(20TP/1FN/9FP)를 재현했다. 별도 프로세스 B0 과거평가329TP/28FN/1FP도 일치했다. R5_A1은 S 판정이 A0와 같지만 임계값은 달라 동일 설정은 아니다. 이번에는 C0 하나만 사용한다. R5의 V 점수는 이전에 없었으며 이번 진단에서도 계산하지 않았다.

진단 표시 범위는 점수 확인 전에 diagnostic_plan.json에 과거19행·미래2행으로 정했다. 미래2행은 그림 전후 맥락에만 쓰고 예측·판정에는 포함하지 않는다. 공백을 선으로 잇지 않았다.

바뀐 행은 정상 추가경보1개와 이상 추가탐지2개다. 표의 원행·센서값·PCA 입력범위/특징·원점수·예측센서값·잔차·시간·warm-up을 diagnostic_changed_rows.csv와 diagnostic_neighborhoods.csv에 저장했다.

{pd.DataFrame(summary).to_string(index=False)}

R5는 t-2,t-1 센서로 t의 센서를 예측한다. 예측을 만들 수 있는 마지막 입력시각은 t-1이고, 잔차 경보는 실제 t 관측을 받은 뒤 t에서만 가능하다. 미래 관측이 필요하지 않으며 과거로 경보를 소급하지 않는다. 파일/split/train_role와 gap>0.5초에서 예측 이력을 초기화한다. B0의 P1 통계는 원래처럼 같은 원분할 안의 공백을 넘을 수 있다.

추가 FP의 직전 h와 새 TP의 직전 h는 위 표와 같이 확인했다. 현재 하나의 정상 사례에서 나온 패턴을 일반적인 운전조건의 원인으로 단정하지 않는다. 정상 T/C에서 학습 scaler 기준 [t-2,t-1,t]가 가까운 별도 burst3개씩을 찾아 similar_normal_contexts.csv에 기록했다. 이것은 진단이며 이웃에 따라 예외 경보 규칙을 만들지 않는다.

새로 탐지한 이상2행의 burst 수는 {changed[changed.label==1].burst_id.nunique()}개다. 같은 이상 관측 구간의 두 행이지 독립 고장2건이라는 증거가 없다. 고장 시작/정비 이력이 없어 실제 고장 탐지 지연을 산출할 수 없다.

현재 원점수·행 정렬·strict>·t 예측타깃과 경보시각 재현에서는 구현 결함이 발견되지 않았다. 전류 부호/극단값·라벨을 수정하지 않았다. 확인된 문제는 실제 정상 라벨 행에서 보조 점수가 임계값을 넘은 것이다. 경보 지속조건으로 이를 제거할 가능성과 동시에 짧은 이상의 추가 탐지를 잃을 위험을 지정한 G0/G1/G2에서만 검증한다.

기존 B0의 S1/S2/V 절대FPR는 모두1% 이하다. S2 정상616행의 추가FP예산은0이고 이 조건은 변경하지 않는다. V에서0FN/1.0F2에 엄격 개선을 요구했던 것은 평가 정책의 문제다. 이를 유지 판정으로 바꾸어도 C0의 S2 FP6→7 및 F1/F2 악화는 여전히 탈락이다. S/V/H가 이미 노출됐으며 새 프로토콜도 노출을 없애지 않는다.
'''
 (R/'diagnosis_ko.md').write_text(text);js(R/'diagnostic_summary.json',dict(cases=summary,baseline_feasible=all(x['baseline_feasible'] for x in budget),extra_anomaly_bursts=int(changed[changed.label==1].burst_id.nunique()),implementation_defect_found=False,new_candidate_scores_seen=False));print(pd.DataFrame(summary).to_string(index=False));print('Diagnosis saved before candidate experiment')
if __name__=='__main__':main()

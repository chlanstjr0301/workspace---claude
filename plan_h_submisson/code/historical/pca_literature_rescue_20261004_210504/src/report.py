from common import *
from sklearn.metrics import precision_recall_curve,roc_curve
import matplotlib.pyplot as plt
import html,re

def mdtable(df,cols=None):
 if cols:df=df[cols]
 def val(x):return f'{x:.6f}' if isinstance(x,float) and np.isfinite(x) else '' if pd.isna(x) else str(x)
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(val(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

def report():
 trials=pd.read_csv(R/'all_trials.csv');bm=pd.read_csv(R/'block_metrics.csv');ev=pd.read_csv(R/'baseline_vs_candidates.csv');diag=json.loads((R/'model_diagnostics.json').read_text());lock=json.loads((R/'selected_config.json').read_text());counts=pd.read_csv(R/'manifests/block_counts.csv');thresholds=pd.read_csv(R/'thresholds.csv');df=loadrows(R/'inputs/frozen_rows.csv');base=bm[(bm.candidate=='baseline')&(bm.block=='S_pooled')].iloc[0];hist=ev[(ev.candidate=='baseline')&(ev.block=='historical')].iloc[0]
 conditions=[];burst_tables=[];coverage=[];pairs=[]
 for f in sorted((R/'predictions').glob('*.csv')):
  if not any(f.stem.endswith('_'+b) for b in ['S_pooled','V','historical']):continue
  r=pd.read_csv(f);block='historical' if f.stem.endswith('_historical') else 'V' if f.stem.endswith('_V') else 'S_pooled';cid=f.stem.removesuffix('_'+block);r['candidate']=cid;r['evaluation_block']=block
  conditionset={'all':np.ones(len(r),bool),'first19':r.burst_pos<=19,'later':r.burst_pos>19,'fallback':r.fallback.astype(bool),'main_W20':~r.fallback.astype(bool),'cross_gap':r.crosses_gap.astype(bool),'no_cross_gap':~r.crosses_gap.astype(bool),'short_burst_lt20':r.burst_length<20,'long_burst_ge20':r.burst_length>=20}
  if 'aux_available' in r:conditionset.update(aux_available=r.aux_available.astype(bool),aux_unavailable=~r.aux_available.astype(bool))
  for name,mask in conditionset.items():
   g=r[mask];n0=int((g.label==0).sum());n1=int((g.label==1).sum());fp=int(((g.label==0)&(g.prediction==1)).sum());fn=int(((g.label==1)&(g.prediction==0)).sum());conditions.append(dict(candidate=cid,block=block,condition=name,normal=n0,anomaly=n1,FP=fp,FN=fn,FPR=fp/n0 if n0 else np.nan,FNR=fn/n1 if n1 else np.nan))
  for bid,g in r.groupby('burst_id',sort=False):burst_tables.append(dict(candidate=cid,block=block,burst_id=bid,label=int(g.label.iloc[0]),rows=len(g),FN=int(((g.label==1)&(g.prediction==0)).sum()),FP=int(((g.label==0)&(g.prediction==1)).sum()),aux_available=int(g.aux_available.sum()) if 'aux_available' in g else 0,native_W20=int(g.native_W20.sum())))
  for y,g in r.groupby('label'):
   coverage.append(dict(candidate=cid,block=block,label=int(y),rows=len(g),native_W20=int(g.native_W20.sum()),native_fraction=float(g.native_W20.mean()),fallback=int(g.fallback.sum()),aux_available=int(g.aux_available.sum()) if 'aux_available' in g else None,unavailable_final=0))
  if 'base_prediction' in r:
   z=r[r.prediction!=r.base_prediction].copy();z['change']=np.where(z.label==1,'new_TP','new_FP');pairs.append(z)
  r[r.error.isin(['FN','FP'])].to_csv(R/'tables'/f'errors_{cid}_{block}.csv',index=False)
 pd.DataFrame(conditions).to_csv(R/'tables/error_conditions.csv',index=False);pd.DataFrame(burst_tables).to_csv(R/'tables/burst_errors.csv',index=False);pd.DataFrame(coverage).to_csv(R/'tables/coverage.csv',index=False);pd.concat(pairs).to_csv(R/'paired_errors.csv',index=False)
 # All registered candidates are shown; no failed candidate has V/historical auxiliary scores.
 fig,ax=plt.subplots(figsize=(8,5))
 for key,g in trials.groupby('score_id',sort=False):ax.plot(g.FP,g.FN,'o-',label=key)
 ax.scatter([base.FP],[base.FN],marker='*',s=150,c='black',label='Original');ax.set(xlabel='False positives (N0=1255)',ylabel='False negatives (N1=21)',title='Selection S1+S2: all 18 preregistered configurations');ax.legend(ncol=2);fig.tight_layout();fig.savefig(R/'figures/fn_fp_tradeoff.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for key,g in trials.groupby('score_id',sort=False):
  a=[0,.0005,.001]
  for metric,ax in zip(['F1','F2'],axes):ax.plot(a,g[metric],'o-',label=key);ax.axhline(base[metric],color='black',ls='--',alpha=.2);ax.set(xlabel='Additional calibration FP budget (fraction)',ylabel=metric)
 axes[1].legend(ncol=2);fig.tight_layout();fig.savefig(R/'figures/f1_f2_budgets.png',dpi=150);plt.close(fig)
 # Representative budget0 for each family, a fixed display choice rather than new selection.
 fig,axes=plt.subplots(1,2,figsize=(12,5))
 for cid in ['baseline']+[f'R{i}_A0' for i in range(6)]:
  r=pd.read_csv(R/'predictions'/f'{cid}_S_pooled.csv');s=r.integrated_score if 'integrated_score' in r else r.score/base.get('threshold',1);pr,rc,_=precision_recall_curve(r.label,s);fpr,tpr,_=roc_curve(r.label,s);axes[0].plot(rc,pr,label=cid);axes[1].plot(fpr,tpr,label=cid);m=metrics(r.label,s,r.prediction);axes[0].scatter([m['Recall']],[m['Precision']],s=20)
 axes[0].set(xlabel='Recall',ylabel='Precision',title='S1+S2 continuous scores; dots = frozen operating points');axes[1].set(xlabel='FPR',ylabel='Recall',title='ROC (auxiliary view)');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(R/'figures/selection_pr_roc.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(11,3))
 for cid,block,ax in [('baseline','S_pooled',axes[0]),('R5_A0','S_pooled',axes[1]),('baseline','historical',axes[2])]:
  r=pd.read_csv(R/'predictions'/f'{cid}_{block}.csv');m=metrics(r.label,r.score,r.prediction);mat=np.array([[m['TN'],m['FP']],[m['FN'],m['TP']]]);ax.imshow(mat,cmap='Blues')
  for (i,j),v in np.ndenumerate(mat):ax.text(j,i,str(v),ha='center',va='center',color='red')
  ax.set(xticks=[0,1],yticks=[0,1],xlabel='Prediction',ylabel='Label',title=f'{cid}: {block}')
 fig.tight_layout();fig.savefig(R/'figures/confusion_matrices.png',dpi=150);plt.close(fig)
 # Separate acquisition dates and do not connect across gaps.
 fig,axes=plt.subplots(2,1,figsize=(12,6))
 r=pd.read_csv(R/'predictions/R5_A0_S_pooled.csv');r.TimeStamp=pd.to_datetime(r.TimeStamp)
 for y,ax in enumerate(axes):
  g=r[r.label==y]
  for _,z in g.groupby('burst_id',sort=False):ax.plot(z.TimeStamp,z.integrated_score,color='steelblue',lw=.6)
  z=g[g.error.isin(['FN','FP'])];ax.scatter(z.TimeStamp,z.integrated_score,color='red',s=20,label='FP/FN');ax.axhline(1,c='black',ls='--');ax.set_yscale('symlog');ax.set(title=f'Selection only, R5_A0, class={y}, {g.TimeStamp.iloc[0].date()}',ylabel='Integrated score');ax.legend()
 fig.autofmt_xdate();fig.tight_layout();fig.savefig(R/'figures/error_timeline_separate_dates.png',dpi=150);plt.close(fig)
 budgets=[]
 for block in ['S1','S2','S_pooled','V','historical']:
  z=(bm if block.startswith('S') else ev);b=z[(z.block==block)&(z.candidate=='baseline')].iloc[0];n=int(b.TN+b.FP)
  budgets.append(dict(block=block,normal=n,anomaly=int(b.TP+b.FN),baseline_FP=int(b.FP),extra_FP_main=allowed_fp(n,.001),extra_FP_strict=0,extra_FP_exploratory=allowed_fp(n,.002),absolute_FP_cap=allowed_fp(n,.01)))
 pd.DataFrame(budgets).to_csv(R/'tables/fp_integer_budgets.csv',index=False)
 summary=trials[['candidate','threshold','TP','FN','FP','new_TP','new_FP','F1','F2','FPR','eligible','failures']]
 selectedtext='선정 후보 없음. 기존 PCA 유지.' if lock['selected'] is None else str(lock['selected'])
 text=f'''# 기존 최우수 PCA의 미탐 감소를 위한 문헌 기반 제한 실험

작성일 2026-10-04 · 실행 {R.name} · 로컬 CPU 2 threads

## 결론

**{selectedtext}** 6개 보조 점수 × 3개 보정 예산, 총18개 운영 구성을 실행했다. 원모델의 경보를 보존하는 OR만 허용했다. 선택에서 모든 조건을 통과한 후보는0개이며 후반/과거 결과로 탈락 후보를 다시 선정하지 않았다.

기존 원모델은 이미 노출된 과거 평가의 **실제 이상357개 중329개 탐지,28개 미탐; 정상3990개 중1개 오경보**로 재현됐다. 정상1000관측당 오경보 {hist.FP_per_1000:.6f}개다. Recall {hist.Recall:.6f}, Precision {hist.Precision:.6f}, F1 {hist.F1:.10f}, F2 {hist.F2:.10f}, FPR {hist.FPR:.8f}, AP {hist.AP:.8f}. 배포하거나 원모델을 덮어쓰지 않았다.

선택 pooled만 보면 R5_A0/R5_A1(2시점 예측잔차)이 FN3→1, FP8→9, F1 0.765957→0.800000, F2 0.818182→0.884956이었다. **이는 적격 개선이 아니다.** S2의 정상616개 중 FP6→7, FPR0.974026%→1.136364%로 바뀌어, 추가FP예산0과 절대1%를 모두 위반했다. S2의 F1 0.500000→0.461538, F2 0.714286→0.681818도 악화했다. R4의 추가TP1 역시 같은 문제로 탈락했다. 이를 단순히 F2 개선 성공이라고 보고하면 시간 구간 검증 조건을 빠뜨리게 된다.

| 목표 | 판정 | 근거 |
| --- | --- | --- |
| (a) 같은 행에서 FN 감소 | 선택 pooled 일부 후보에서 관찰 | R5 FN3→1, R4 FN3→2; 기존 TP 손실0 |
| (b) 사전 추가FP예산 | 주 개선 후보 없음 | R4/R5 S2 추가FP1 > 허용0; 절대FPR1%도 초과 |
| (c) F2 증가와 F1 비하락 | pooled 일부 통과, 필수 블록 조건 실패 | S2 F1/F2 하락 |
| (d) 선택하지 않은 후반 시간 검증 | 통과 후보 없음 / 개선 확인 불가 | 선정 후보가 없고 V 원모델 FN=0. R0 대조군도 개선0 |

## 원모델 재현과 반복 개발 이력

P1은 같은 파일·split·train_role 안에서 긴 공백을 넘어 과거20행을 잇는다. 센서별 mean/std(ddof=0)/RMS/min/max, 총15특징을 정상 학습으로 표준화하고 full-SVD PCA의2개 성분을 유지한다. Q는 표준화 공간 복원오차의 제곱합이다. 부족한 창은 원래 P0 W1 k2 Q로 보완한다. 각각 정상calibration Q95로 나눈 점수를 연결하고 원 임계값1.3078792257682357보다 클 때 경보, 후처리 없음이다. 날짜·라벨·파일명·행번호·burst_id는 센서 특징에 포함하지 않았다.

이번에는 저장 모델을 별도 프로세스에서 라벨 없는 입력으로 추론해4347행의 ID·점수·판정을 대조했다. 원설정 재학습은 직전 회차에서 수행한 기록과 구분한다. 이번 새 학습은 보조 모델에 한정했다. 근거는 reproduction.json, reproduction_report_ko.md, models/baseline.joblib, inputs/original_expected_test.csv다.

직전 median/IQR은 원모델보다 Recall이 높아 선정된 것이 아니었다. 당시 내부99이상에서83TP/8FP였고 원15특징 target1%는85TP/31FP로 FPR 제약 미달이었다. 당시 적격 후보군에서는 선정됐지만, 고정 뒤 개발 후반 및 과거 평가에서 악화했다. 이후 3행 보완은 개발109TP/2FN/0FP였으나 과거323TP/34FN/0FP로 원모델보다 미탐이 늘었다. 두 변경은 이번에 반복하지 않았다. inputs/previous_*_report.md와 all_trials 사본, archive_audit.json에 근거를 남겼다.

원본 정상20000행에서 TimeStamp·센서·Equipment_state가 같은 추가중복1행만 제거해19999행, 이상600행이다. 행 번호를 보존했다. 정상/이상 날짜는2022-07-12/2022-07-17로 달라 날짜와 상태가 얽혀 있다. 주 간격은0.1초지만 >0.5초 공백은 정상598·이상20개이고 최대16.643/8.572초다. 큰 값과 음의 전류는 제거하지 않았다. 단위·AI0/AI1의 상하 대응·운전 부하는 확인되지 않았다.

## 데이터 역할과 사전 등록

정상 train12012행/360버스트에서만 원 전처리와 새 보조 모델을 적합했다. 정상calibration2011행/63버스트는 임계값에만 썼다. 이상calibration132행은 새 학습/보정에 사용하지 않았다. 원 selection 정상1986행/58버스트·이상111행/4버스트를 각 클래스의 시간순 burst 개수로 나누었다. 서로 다른 두 날짜를 동시에 관측한 정상/이상처럼 묶지 않았다.

{mdtable(counts)}

S1+S2는 정상1255·이상21, 양성비율 {21/1276:.6%}이다. V는 정상731·이상90이다. 정상train/calibration/selection/test 원분할은 유지했다. S/V는 원selection 안의 보고·선택 역할 분리다. 원모델의 기존 윈도 상태는 reporting block에서 임의 초기화하지 않는다. 새 예측잔차는 파일·split·train_role·gap>.5초마다 초기화한다. S2/V 첫 버스트의 과거행을 갖는 원 P1의 인과적 상태 이월은 원정책 보존이며 fit 또는 임계값 재적합이 아니다. 이 평가를 독립 iid fold 또는 새 독립 테스트라고 부르지 않는다.

preregistration.json은 보조 점수 계산 전 기록했고 SHA-256은 `{sha(R/'preregistration.json')}`이다. 구현 해시는 fit 전에 implementation_lock.json으로, 선택은 V 개봉 전에 selected_config.json으로 고정했다. 과거 연구에서 개발·과거평가·FN 조건을 이미 보았고 이번도 반복 개발이다. 검출기 학습은 정상 전용이지만 운영 구성 선택은 개발 이상 라벨을 사용했다.

## 방법과 예산

원경보 A0 OR (aux_available AND aux_score > tau_aux)만 사용했다. 원 TP를 잃지 않지만 원 FP도 줄일 수 없다. 사용 불가능한 보조 경로는 원경보로 돌아가며 모든 원행이 평가에 남는다. 후처리·억제·다수결·point adjustment는 없다.

{mdtable(pd.DataFrame(budgets))}

주 예산은 ΔFPR≤0.001(0.1%p), 절대FPR≤1%다. 정수로 내림하며 작은 블록에 최소1개를 허용하지 않았다. S1/S2 각각 예산·F1/F2 비하락을 요구하고 pooled FN/F2 엄격 개선·F1 비하락을 요구했다. 동률1e-12, pooled F2→추가FP→FN→사전 계산량 순서→등록ID다. Δ0은 엄격 보조, Δ.002는 탈락 후보의 탐색적 표시에 불과하다. 현재18개는 엄격/탐색 예산에서도 완전 적격이 없다.

정상calibration 분모2011을 유지한 추가FP 예산0/0.0005/0.001은 정수0/1/2개다. A0가0이고 보조 점수가 있는 정상값을 내림차순 정렬해 k번째(0시작) 값을 임계값으로 삼아 strict > 판정했다. 동점은 경보하지 않는다. 없는 점수는 분모에 유지한다. 실제 threshold·동점 수는 thresholds.csv다. 이번 임계값은 모두 양수여서 통합 연속점수 max(base/h0,aux/tau)>1과 OR 판정이 일치한다. R0은 score/min(h0,tau)로 순위가 동일하다. 점수는 확률이 아니다.

- R0: 기존 점수의 경보 추가. 세 후보의 AP/사다리꼴PR-AUC/ROC-AUC가 원점수와 모두 동일함을 검사했다.
- R1: 원15차원 잔차 r=(I−PPᵀ)(z−mu), 정상train11974개에 LedoitWolf. 잔차 rank13, shrinkage {diag['R1']['shrinkage']:.8g}, 최소고유값 {diag['R1']['min_eigenvalue']:.8g}, 조건수 {diag['R1']['condition']:.4f}, epsilon {diag['R1']['epsilon']:.4g}. Cholesky 선형해법으로 거리 계산. Q와 Spearman {diag['R1']['spearman_Q']:.6f}, 순위는 달랐으나 선택에서 추가TP는 없었다. PCA 고유벡터 재학습의 효과라고 주장하지 않는다.
- R2/R3: 정상train에서 모든 eligible burst를 덮는 결정적4096행을 먼저 고정했다. 잔차 축을 추가 표준화하지 않았고 중앙 쌍거리²는 {diag['R2']['s0_squared']:.9f}; 두 대역폭만 썼다. C=0.0244140625, ν=.01. libsvm 계수 β를 νN으로 나눠 α의 합1/상한C를 확인했다. 중심거리 1−2Kα+αᵀKα를 직접 계산했다. 128정상행의 직접 제약QP와 목적함수 차이는 {diag['R2']['dual_check']['objective_difference']:.3g}/{diag['R3']['dual_check']['objective_difference']:.3g}, 거리 최대차이는 {diag['R2']['dual_check']['distance_max_difference']:.3g}/{diag['R3']['dual_check']['distance_max_difference']:.3g}. default predict()를 사용하지 않았다. PCA+SVDD 하이브리드다.
- R4/R5: 원센서3개를 정상train으로 표준화하고 동일burst 내부 lag1/2로 현재 센서를 예측한다. 중심화 ridge와 비정규화 절편, lambda=1e-3 trace(UᵀU/n)/dimU. 학습행11652/11296, lambda {diag['R4']['lambda_']:.8g}/{diag['R5']['lambda_']:.8g}. 예측오차의 정상 LW 거리를 사용한다. 현재까지 L개가 연속할 때만 활성화한다. CVA의 Rs/Rr가 아닌 PCA+인과적 예측잔차 하이브리드다.

## 선택 결과: 실패 후보도 모두 공개

다음은 동일1276행에서 계산한 결과다. 임계값 단위는 보조 점수별로 다르므로 크기를 서로 비교하지 않는다. 모델 학습·계산 시간과 disabled/중단 여부는 all_trials.csv/model_diagnostics.json/logs에 있다.18개 모두 실행 완료했으며 실패/중단한 fit은 없었다. 선택 단계 경과시간은 {lock['seconds']:.3f}초로90분 예산보다 짧았다. PCA seed를 바꾼 가짜 반복을 하지 않았다.

{mdtable(summary)}

{mdtable(bm[bm.candidate.isin(['baseline','R4_A0','R5_A0'])],['candidate','block','TP','FN','FP','TN','Recall','Precision','F1','F2','FPR'])}

![FN FP 교환](figures/fn_fp_tradeoff.png)
![F1 F2](figures/f1_f2_budgets.png)
![PR ROC](figures/selection_pr_roc.png)

R1/R2/R3은 원 W20 잔차가 있을 때만 활성화한다. 개발의 원FN3개는 P0 fallback에 있어 이 세 보조 모델은 해당 미탐에 점수를 낼 수 없었다. 차원을 달리하는 P0 잔차를 임의 혼합하지 않았다. R4/R5는 일부 초기 미탐을 포착했지만 정상구간에 추가 경보를 만들었다. 조건별 분모와 FN/FP는 tables/error_conditions.csv, burst별 분포와 가용성은 tables/burst_errors.csv 및 coverage.csv다. 미래의 burst_length는 진단 표를 만드는 데만 썼다.

![공백을 끊은 오류 시간축](figures/error_timeline_separate_dates.png)

## V와 제한된 과거 평가

선정 후보가 없어 R1–R5를 V/과거test에 추가 적용하지 않았다. 선택 자료에서 미리 고른 R0_A0은 임계값이 원모델과 정확히 같았다. 두 이름으로 대조 저장했지만 실제 서로 다른 운영 구성은1개다. 이 중복을 새 개선 또는 독립 반복으로 세지 않는다.

{mdtable(ev,['candidate','block','TP','FN','FP','TN','Recall','Precision','F1','F2','FPR','AP','PR_AUC_trapezoid','ROC_AUC'])}

V 원모델 FN0은 FN의 엄격 감소가 불가능한 천장 상태다. 새로 수집한 미노출 독립 검증 통과가 아니며, 개선 확인 불가로 기록한다. 탈락 R5를 과거test에서 평가하지 않았으므로 그 과거 성능 수치는 없다. 과거test 결과를 보고 후보나 임계값을 재선택하지 않았다.

![혼동행렬](figures/confusion_matrices.png)

## 검증과 한계

validation_report.json에서 원본/모델/코드 해시, 분할·창 원시행 경계, 정상전용 fit, 별도 프로세스 라벨 없는 추론, 원행 정렬, OR 불변조건, 미래 센서값 변경 시 과거 점수·판정 불변, 라벨 변경 시 점수 불변, gap 초기화, strict > 동점 처리와 비가용 분모를 검사했다. S1+S2의 모든18후보와 제한된 V/과거 예측을 저장했다. 원TP손실0·기존FP제거0이라는 OR 제약도 전부 맞았다.

추가TP/FP의 원본행은 paired_errors.csv다. 수집 공백을 정상 노출 시간에 넣지 않았으므로 FP/hour는 산출하지 않았다. alarm_episodes.csv는 행FP와 연속 경보 episode를 구분하고 버스트 경계에서 끊는다. 첫 관측부터 첫 경보까지의 실제 timestamp 차이를 기록하며 무경보 burst도 남긴다. 이 값은 실제 고장 시작이나 고장 전 예측 시간과 다르다.

이상selection은4개 burst뿐이고 두 클래스가 다른 날짜다. S2 이상3행은 한 burst로, 반복개발에서 매우 작은 평가 분모가 결정을 좌우한다. 이번에는 iid 행 bootstrap·p-value·PCA seed 반복을 하지 않았다. burst도 독립 고장 사건이 아니므로 미래 일반화를 보장하는 신뢰구간을 제시하지 않는다. 중첩창은 iid가 아니고 LW 원문의 가정을 그대로 충족하지 않는다. 10Hz만으로 앨리어싱 발생을 확정하거나 PCA가 면역이라고 주장하지 않는다.

(a) 일부 개발FN 감소와 (c) pooled F1/F2 이득은 관찰됐다. (b) 시간블록 오경보 예산은 실패했고 (d) 후반 개선은 확인되지 않았다. **오경보 증가 없는 적격 개선, 임계값만의 적격 개선, 최종 유지할 신규 후보 모두 없다. 기존 PCA를 유지한다.** 압력·불량·비가동·정비 이력이 없으므로 직접 예측하거나 감소시켰다고 할 수 없다. 다양한 날짜·부하·설비의 정상과 이상, 실제 상태 전환 및 정비 시각을 새로 모으기 전에는 현장 개선을 입증했다고 할 수 없다.

## 재현 및 파일 안내

이 폴더에서 `bash RUN_EXPERIMENT.sh`를 실행하면 별도의 새 재현 디렉터리에 원모델 추론 확인→보조 모델 재학습→선택 고정→V/과거 제한 평가→검증→보고서를 작성한다. 원본 결과를 덮어쓰지 않는다. Python3.14.4와 requirements.lock.txt의 버전이 실제 환경이다. 다른 환경에서는 프로젝트 전용 가상환경을 만들고 잠금 패키지를 설치한 뒤 `PYTHON_BIN=/path/to/python bash RUN_EXPERIMENT.sh`로 지정한다. 데이터·split·모델·코드가 함께 제공되며 인증값은 포함하지 않는다.

처음의 prepare.py는 기존 프로젝트 자료를 모아 등록한 출처 기록용이다. 전달 패키지의 재현은 replay.py가 동봉 입력·사전등록을 사용하므로 원래 과거 run 경로를 요구하지 않는다. source/model/data 해시는 inventory.json, implementation_lock.json, selected_config.json, 최종 MANIFEST_SHA256.json에 있다. 원문 재배포 대신 아래 확인기록과 URL을 제공한다.

'''
 text+='## 전체 실행 재현 확인\n\n완료 후 end_to_end_replay.json에 새 폴더의 재학습·후보 선택·행별 점수/판정 대조 결과를 저장한다. 단순 저장 모델 재추론과 보조 모델 전체 재학습 재현을 구분하며, 재실행은 독립 통계 반복이 아니다. 최종 전달 패키지의 해당 JSON에서 실제 통과 여부를 확인할 수 있다.\n\n';text+=(R/'references/literature_notes_ko.md').read_text();(R/'review_report_ko.md').write_text(text)
 # Lightweight safe Markdown renderer for this report; no external JS, fonts or CDN.
 lines=text.splitlines();out=[];i=0
 def inline(s):
  s=html.escape(s);s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s);s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s);s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s);return s
 while i<len(lines):
  line=lines[i]
  if line.startswith('|'):
   out.append('<div class="scroll"><table>');header=True
   while i<len(lines) and lines[i].startswith('|'):
    cells=[s.strip() for s in lines[i].strip('|').split('|')]
    if not all(re.fullmatch(r'[- :]+',s) for s in cells):tag='th' if header else 'td';out.append('<tr>'+''.join('<'+tag+'>'+inline(s)+'</'+tag+'>' for s in cells)+'</tr>');header=False
    i+=1
   out.append('</table></div>');continue
  elif line.startswith('!['):
   m=re.match(r'!\[(.*?)\]\((.*?)\)',line);out.append(f'<figure><img src="{html.escape(m[2])}" alt="{html.escape(m[1])}"><figcaption>{html.escape(m[1])}</figcaption></figure>')
  elif line.startswith('#'):
   level=len(line)-len(line.lstrip('#'));out.append(f'<h{level}>'+inline(line[level:].strip())+f'</h{level}>')
  elif line.strip():out.append('<p>'+inline(line)+'</p>')
  i+=1
 (R/'review_report_ko.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>PCA 문헌 기반 제한 실험</title><style>body{max-width:1120px;margin:40px auto;padding:0 22px;color:#182333;background:#f8fafc;font:16px/1.85 sans-serif}h1{font-size:30px}h2{border-bottom:2px solid #ccd7e5;padding-top:28px}p{word-break:keep-all}table{border-collapse:collapse;min-width:650px;font-size:13px;background:white}td,th{border:1px solid #d5dce6;padding:8px;text-align:left}th{background:#e9eef6}.scroll{overflow:auto;margin:20px 0}img{max-width:100%;background:white}code{background:#e6ebf2;padding:2px 5px}a{color:#135a9e}strong{color:#803019}</style><body>'+''.join(out)+'</body></html>')
 (R/'next_data_requirements_ko.md').write_text('# 다음 자료 요구\n\n동일 날짜에서 정상·이상과 운전변화가 함께 관측되는 여러 세션, 새로운 날짜와 설비, 부하/공정/압력 및 실제 고장 전환·정비 이력이 필요하다. 정상 충분한 관측시간과 독립 세션을 확보해0.1%p 예산을 정수FP로 검증하고 실제 현장 비용·경보 대응 기준을 정해야 한다. 이번 탈락 후보를 현재 V/test에 맞춰 튜닝하지 않는다.\n')
 print('Reports and figures written')
if __name__=='__main__':report()

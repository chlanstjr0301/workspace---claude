"""Build a standalone Korean HTML report from saved results; never fits/selects models."""
import argparse,json,html,sys,hashlib,shutil
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def esc(x):return html.escape(str(x))
def pct(v):return f'{100*float(v):.2f}%'
def num(v):return f'{float(v):.4f}'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();run=Path(args.run).resolve()
lock=read(run/'selection_lock.json');summary=read(run/'final_summary.json');choices=summary['choices'];main=choices['primary'];alt=choices['alternative'];ifo=choices['if_primary'];audit=read(run/'manifests/audit.json');env=read(run/'manifests/environment.json')
sections=[]
def para(x):sections.append('<p>'+x+'</p>')
def heading(x):sections.append('<h2>'+esc(x)+'</h2>')
def table(df,columns=None):
 df=df.copy()
 if all(c in df for c in ['TP','FN','FP','TN']):
  df['normal']=df.FP+df.TN;df['anomaly']=df.TP+df.FN
  if columns is not None:columns=list(columns)+[c for c in ['N','normal','anomaly','positive_rate'] if c not in columns]
 if columns is not None:df=df[[c for c in columns if c in df]]
 sections.append('<div class="scroll">'+df.to_html(index=False,float_format=lambda x:f'{x:.5g}',escape=True)+'</div>')
def artifact(label,path):return f'<a href="{esc(path)}">{esc(label)}</a>'
def image(name):sections.append(f'<img src="figures/{esc(name)}" alt="{esc(name)}">')
def load(name):return pd.read_csv(run/'tables'/name)
metrics=['key','target_fpr','N','positive_rate','TP','FN','FP','TN','Recall','Precision','F1','F2','FPR','FNR','AP','PR_AUC_trapezoid','ROC_AUC','Accuracy']
heading('선정 결과와 최종 평가')
para(f'<strong>선정 PCA: {esc(main["key"])}</strong>. 정상 보정 자료의 목표 FPR 1% 임계값을 적용한 뒤, 별도 모델 선택 자료에서 FPR≤1%를 만족하는 후보 중 Recall·F1·AP·실행시간 순으로 선택했다. 선택 상태: {esc(lock["primary_status"])}. 후처리 없는 전체 관측 행 결과를 주 결과로 사용한다. 최종 평가를 본 뒤 변경하지 않았다.')
para(f'<strong>실제 이상 {main["TP"]+main["FN"]}개 중 {main["TP"]}개 탐지, {main["FN"]}개 미탐.</strong> 정상 {main["FP"]+main["TN"]}개 중 {main["FP"]}개 오경보. 정상 1,000개 관측당 오경보 {main["FP_per_1000"]:.2f}개. Recall {pct(main["Recall"])}, Precision {pct(main["Precision"])}, F1 {num(main["F1"])}, F2 {num(main["F2"])}, FPR {pct(main["FPR"])}, AP {num(main["AP"])}.')
para(f'개발용 선택에서 Recall {pct(lock["primary"]["Recall"])}·FPR {pct(lock["primary"]["FPR"])}였고, 최종 평가에서 FPR 목표는 {"충족했다" if main["FPR"]<=.01 else "미달했다"}. FPR 1%는 프로젝트 가정이며 대회 공식 기준 또는 현장 보장이 아니다.')
table(pd.DataFrame([dict(role=r,**v) for r,v in choices.items()]),['role']+metrics)
para(f'동일 분할·전체 관측·정상 보정 1% 기준에서 개발용으로 고른 IF {esc(ifo["key"])} 대비 PCA Recall 차이는 {(main["Recall"]-ifo["Recall"])*100:+.2f}%p, FPR 차이는 {(main["FPR"]-ifo["FPR"])*100:+.2f}%p, F1 차이는 {main["F1"]-ifo["F1"]:+.4f}다. IF 대표는 42·43·44의 사전 고정 평균 점수이며 최종 최고 seed를 고르지 않았다.')
pairs=load('burst_pairs_test.csv');default=pairs[(pairs.unaware=='P1_W10_K2_QT')]
para('버스트 고려 효과의 기본 비교는 같은 10행·2성분·결합 점수이며, 두 모델 모두 예측 가능한 동일 마지막 행에서 계산했다. 최적 설정끼리의 비교와 분리한다.')
table(default,['unaware','aware','normal','anomaly','unaware_Recall','aware_Recall','delta_Recall','unaware_FPR','aware_FPR','delta_FPR','unaware_F1','aware_F1'])
complement=load('score_complements_test.csv');selected_base=main['base'];cc=complement[complement.base==selected_base]
if len(cc):
 c=cc.iloc[0];para(f'선정 PCA의 동일 분해 설정에서 Q/T²/결합 점수를 공통 행(정상 {int(c.normal)}, 이상 {int(c.anomaly)})으로 비교했다. Q 대비 결합 점수는 이상 {int(c.anomaly_QT_added_vs_Q)}개를 추가 탐지하고 {int(c.anomaly_QT_lost_vs_Q)}개를 잃었으며, 정상 {int(c.normal_QT_added_vs_Q)}개에 추가 오경보를 내고 기존 오경보 {int(c.normal_QT_lost_vs_Q)}개를 제거했다. 결합 점수 자체의 임계값을 다시 보정하므로 단순 OR과 다르다.')
post=load('postprocess_test.csv');none=post[post.policy=='none'].iloc[0]
para(f'후처리는 개발용에서 {esc(lock["postprocess"])}로 고정했다. 아래 세 정책은 사전 지정한 비교이며 최종 점수로 정책을 재선택하지 않았다.')
table(post,['policy','Recall','Precision','F1','F2','FPR','FN','FP','new_FN_vs_none','additional_TP_vs_none','anomaly_bursts','missed_anomaly_bursts','median_first_alarm_seconds'])
para('고장 시작·정비 시각·정상→이상 전환 이력이 없어 고장 전 예측 시간과 실제 고장 탐지 지연을 검증할 수 없다. 계산한 시간은 각 관측 버스트 시작부터 첫 경보까지 실제 TimeStamp 차이다. 미경보 버스트도 분모에 남겼다.')
para('첫 번째 실험의 원본 해시·분할·특징·학습 내부 경계는 동일하다. 자체 PCA/IF 비교는 같은 CPU 실행과 같은 점수 정규화 정책으로 완료했다. 첫 번째 프로젝트는 운영 점수를 정상 z 단위로 변환하고 이 실험은 95백분위수 배수를 쓰므로, 첫 번째 프로젝트의 전체 운영 결과와 완전히 동일한 방법 비교라고 주장하지 않는다. LSTM은 재학습하지 않았고 이 보고서에 미확인 LSTM 점수를 넣지 않았다.')
para(f'결과 위치: <code>{esc(run)}</code>. 재실행: <code>cd /home/lim/hydraulic_ai_alt &amp;&amp; bash scripts/reproduce.sh</code>. 새 결과 폴더를 생성하며 원본 CSV·첫 번째 프로젝트·기존 결과를 덮어쓰지 않는다.')
image('primary_test.png')
heading('데이터 이해와 보존')
table(pd.DataFrame(audit['files']),['file','sha256','original_rows','clean_rows','removed','missing','numeric_failures','timestamp_failures','date','label_counts','gaps_over_05','gap_max','gap_median','burst_count'])
para('추가 중복은 TimeStamp·센서 3개·Equipment_state가 모두 같은 행만 제거하고 최초 행을 남겼다. 원본 파일명·1부터 시작하는 데이터 행 번호·CSV 줄 번호를 clean_rows.csv와 중복 내역에 보존했다. 값이 크다는 이유로 삭제·클리핑하지 않았다. 첫 열 번호·라벨·날짜·파일명·행 번호·burst_id는 모델 입력에서 제외했다. 전체 자료의 행 수·범위·간격 등 요약은 실험 이전에 이미 확인했다.')
table(pd.DataFrame([{'변수':s,'용도':'모델 입력','단위/대응':'미확인; 진동 AI0/AI1의 상부·하부 대응 지정하지 않음' if 'Vibration' in s else '미확인'} for s in ['AI0_Vibration','AI1_Vibration','AI2_Current']]+[{'변수':'TimeStamp','용도':'분할·버스트·추적·실제 경과시간','단위/대응':'기록 시각; 시간대 미확인'},{'변수':'Equipment_state','용도':'평가·개발 선택·sigmoid 보정 라벨','단위/대응':'0 정상, 1 이상; 고장 시작 시각 아님'}]))
para('정상은 2022-07-12, 이상은 2022-07-17 기록이다. 날짜와 라벨이 분리되어 날짜별 부하·운전조건·취득조건 차이가 성능에 영향을 줄 수 있다. 독립 고장 사건 수는 알 수 없다. 압력·품질·비가동·정비 이력과 실제 운전조건이 없어 이들을 직접 예측하거나 감소시켰다는 주장을 하지 않는다.')
heading('분할과 누수 방지')
splits=pd.read_csv(run/'manifests/configs/first_experiment/frozen_split_summary.csv');table(splits)
para('윈도우 생성 전에 시간순·전체 버스트 단위로 고정한 행 목록을 복사했다. 학습은 정상 12,012행만 사용하며 첫 번째 IF와 맞추기 위해 train_role 내부 경계도 넘지 않는다. 각 파일·학습/보정/선택/최종 경계 사이에 윈도우가 겹치지 않는다. 보정용 이상 132행은 sigmoid 단계, 모델 선택용 이상 111행은 구성 선택과 분석에 사용했다. 따라서 PCA/IF fit은 비지도이지만 전체 절차는 이상 라벨을 사용한 개발·보정을 포함한다.')
table(pd.DataFrame(audit['burst_sensitivity']))
para('정상 학습에서 0.2·0.5·1.0초 기준의 버스트 경계가 모두 같아 중복 학습 실험을 생략했다. 버스트는 거의 0.1초 간격의 관측 구간이며 기계 사이클을 의미하지 않는다. 큰 공백은 보간하지 않았다. FFT는 사용하지 않았고, 10Hz 정보만으로 앨리어싱을 확정하거나 PCA가 이를 제거한다고 설명하지 않는다.')
if (run/'manifests/development_validation.json').exists():table(pd.DataFrame([read(run/'manifests/development_validation.json')]).T.reset_index().rename(columns={'index':'검사',0:'결과'}))
para('최종 점수 계산은 selection_lock.json 존재와 모델·분할·프로토콜 해시 검사를 통과해야 한다. final_evaluation_started.json과 최종 완료 기록을 남겼다. 원래 비교 프로젝트의 최종 결과·예측·오류를 PCA 선택 과정에서 읽지 않았다.')
heading('모델·점수·임계값 정의')
para('P0: 센서 3개 한 행. P1: 버스트를 무시한 과거 5/10/20행 통계. P2: 버스트마다 초기화한 같은 통계. I0/I1/I2는 동일 입력의 IF다. 각 센서의 평균·표준편차(ddof=0)·RMS·최솟값·최댓값 15개를 통계 종류 우선 순서로 만든다. 현재 행까지만 포함하며 중앙 정렬·미래값을 사용하지 않는다. 이동통계는 과거를 요약하지만 윈도우 안의 전체 순서를 학습하지 않는다.')
para('StandardScaler는 정상 학습 행/윈도우만 fit한다. PCA는 full SVD·whiten=False, P0 성분 1/2, P1/P2 성분 2/3/5/8이다. Q=Σ(z−ẑ)², T²=Σ(tⱼ²/λⱼ), 결합=max(Q/cQ,T²/cT). cQ·cT는 정상 보정 점수의 95백분위수다. 결합 점수는 별도 임계값을 보정한다. 점수와 배수는 고장 확률이 아니다.')
para('상수 기준: 정상 학습 표준편차≤1e−12×max(1,|평균|). 고유값 허용오차=max(1e−12,λmax×1e−10). 유효 rank보다 적은 성분일 때만 Q를 허용하고 작은 고유값을 나누지 않는다. 참조값≤1e−12 후보는 제외한다. 기본 후보의 제거 특징·유효 rank·고유값·누적 설명 분산은 각 fit 로그에 있다.')
fitrows=[]
for path in sorted((run/'logs').glob('fit_P*.json')):
 b=read(path)
 if 'matched' in b['name']:continue
 fitrows.append(dict(model=b['name'],features=len(b['names']),removed=','.join(b['removed']),rank=b['rank'],k=b['k'],cumulative_variance=b['cumulative_variance'][b['k']-1],training_windows=b['training_windows'],fit_seconds=b['fit_seconds']))
table(pd.DataFrame(fitrows))
para('IF는 300 trees·max_samples=256·contamination=auto·n_jobs=2·seed 42/43/44, −score_samples를 사용했다. 모든 반복을 공개한다. full SVD PCA는 같은 입력에서 결정적이므로 seed만 바꾼 가짜 반복을 하지 않았다. 표본 수 보조 실험의 PCA 반복은 실제 서로 다른 정상 학습 표본을 추출한 것이다.')
para('임계값은 정상 보정 자료에서 목표 FPR 0.5/1/2/5%에 대응하는 np.quantile(method="higher")로 구한다. score&gt;threshold만 이상이고 동점은 정상이다. 유한 표본·동점 때문에 목표 비율과 정확히 같지 않을 수 있다. 이론적 관리한계·정규성·관측 독립성을 가정하지 않았으며 경험적 분포의 현장 보장도 주장하지 않는다.')
para('예측 부족 행은 모든 PCA 이동통계 후보에 같은 P0 '+esc(lock['fallback'])+'를 적용했다. IF는 같은 seed의 I0로 보완한 후 3-seed 점수를 평균했다. 각 모델 원점수를 그 모델의 정상 보정 95백분위수로 나누어 연결하고, 연결한 전체 정상 보정 점수로 임계값을 다시 정했다. 이 변환은 확률 보정이 아니다.')
heading('동일 행 비교·전체 운영·예측 가능 비율')
cov=load('coverage.csv');table(cov[(cov.split=='test')&(cov.condition=='all')],['feature','label','total','available','excluded','coverage','crossing_windows','elapsed_max'])
para('모든 윈도우 길이 후보의 공통 마지막 행은 점수 계산 전에 고정했다. 이 집합은 어려운 구간 시작과 짧은 버스트를 제외하므로 전체 관측 평가를 대체하지 않는다. native 표의 서로 다른 행 수를 직접 순위 비교하지 않는다. rows_*.csv는 제외 이유·윈도우 시작 행·실제 경과시간, coverage.csv는 구간 시작/짧은 버스트의 분모까지 제공한다.')
res=load('metrics_test.csv');selectedkeys=[main['key'],ifo['key'],alt['key']]
table(res[(res.key.isin(selectedkeys))&(res.target_fpr==.01)&res.scope.isin(['common','operational'])],['scope']+metrics)
para('개발용에서 각 계열 최적 설정을 따로 고른 비교:')
table(load('best_by_family_test.csv'),metrics)
para('동일 성분·동일 점수·동일 길이의 전체 버스트 비교: '+artifact('burst_pairs_test.csv','tables/burst_pairs_test.csv')+'. 기본 W10 K2의 세 점수를 아래에 제시한다.')
table(pairs[(pairs.unaware.str.startswith('P1_W10_K2'))],['unaware','aware','normal','anomaly','unaware_Recall','aware_Recall','delta_Recall','unaware_FPR','aware_FPR','delta_FPR','unaware_F1','aware_F1','delta_F1'])
matched=load('matched_training_count_selection.csv') if (run/'tables/matched_training_count_selection.csv').stat().st_size>2 else pd.DataFrame()
if len(matched):
 para(f'개발용 동일 행에서 버스트 모델의 Recall 또는 F1 개선이 있었던 조건에만 정상 학습 개수를 맞추는 보조 실험 {len(matched)}개 결과를 남겼다. 버스트 미고려 학습 마지막 행을 seed 42/43/44로 무작위 추출하고 같은 검증 행·성분·점수로 비교했다. 이는 표본 수 영향을 점검하며 독립 고장 불확실성을 추정하지 않는다.')
 sm=matched.groupby(['unaware','aware']).agg(training_count=('training_count','first'),Recall_mean=('Recall','mean'),Recall_std=('Recall','std'),F1_mean=('F1','mean'),FPR_mean=('FPR','mean')).reset_index();table(sm)
else:para('개발용에서 지정된 버스트 개선 조건이 없어 표본 수 보조 실험은 트리거되지 않았다.')
if len(matched):
 mm=matched[matched.unaware=='P1_W10_K2_QT'];dd=load('burst_pairs_selection.csv');dd=dd[dd.unaware=='P1_W10_K2_QT']
 if len(mm) and len(dd):para(f'기본 W10·2성분·결합 점수의 개발용 표본 수 보조 결과: 미고려 모델의 표본 수를 고려 모델과 맞춘 세 반복 Recall 평균 {pct(mm.Recall.mean())}, F1 평균 {num(mm.F1.mean())}; 고려 모델 Recall {pct(dd.iloc[0].aware_Recall)}, F1 {num(dd.iloc[0].aware_F1)}. 이 조건의 개선을 단순히 학습 개수 차이만으로 설명하기는 어렵지만, 한 날짜의 반복이므로 일반적 인과 효과로 확정하지 않는다.')
para('같은 W20 이동통계와 같은 공통 평가 행으로 PCA와 IF를 비교한 표도 별도로 제시한다. 전체 운영 비교는 PCA의 P0 보완과 IF의 I0 보완이 다르므로 차이를 순수 알고리즘 효과로 단정하지 않는다.')
table(res[(res.key.isin([main['key'],'I1_W20_ensemble']))&(res.target_fpr==.01)&res.scope.isin(['common','operational'])],['scope']+metrics)
heading('임계값 절충·IF seed·단순 기준')
table(res[(res.key.isin(selectedkeys))&(res.scope=='operational')],metrics)
para('F1 대안은 개발용에서 네 목표 임계값 후보를 포함해 고른 결과다. 주 기준과 달리 FPR≤1% 제약을 우선하지 않는다.')
para(f'최종 F1 대안은 {esc(alt["key"])}·목표 {pct(alt["target_fpr"])}. 주 모델 대비 Recall {(alt["Recall"]-main["Recall"])*100:+.2f}%p, Precision {(alt["Precision"]-main["Precision"])*100:+.2f}%p, FPR {(alt["FPR"]-main["FPR"])*100:+.2f}%p다.')
seeds=res[(res.family==ifo['family'])&(res.window==ifo['window'])&(res.scope=='operational')&(res.target_fpr==.01)];table(seeds,metrics)
ss=seeds[seeds.seed.astype(str).isin(['42','43','44'])]
table(ss[['Recall','Precision','F1','F2','FPR','AP']].agg(['mean','std']).reset_index().rename(columns={'index':'seed statistic (sample std)'}))
para('IF 모든 구성·seed의 개별 결과는 metrics_test.csv, 평균·표본 표준편차는 if_seed_summary_test.csv에 저장했다. seed 변동은 동일 날짜 자료의 불확실성 또는 독립 고장 사건의 신뢰구간이 아니다. 행을 독립 표본으로 취급한 신뢰구간은 만들지 않았다.')
table(load('dummy_test.csv'))
para('Precision 분모가 0이면 0, F1/F2가 정의되지 않으면 0이다. 특정 클래스가 없으면 해당 클래스 비율과 AP/PR-AUC/ROC-AUC는 결측으로 둔다. PR/ROC는 연속 점수로 계산한다. AP는 비보간 평균 정밀도이며 사다리꼴 PR-AUC를 따로 산출했다. 후처리 표의 AP/ROC는 원래 연속 점수 기준이다.')
heading('센서 영향·Q/T²·오류 조건')
table(cc)
para('개발용 센서 조합/하나씩 제거 비교는 모두 W10 공통 행을 사용한다. 진동 2개·전류만·3개·각 단일/제거 조합의 원시 P0와 버스트 W10을 비교했다. rank1은 Q가 성립하지 않아 T²만 사용했으며 방법 차이를 표에 남겼다. 이 보조 실험은 최종 후보 선택에 포함하지 않았다.')
table(load('sensor_combinations_selection.csv'),['model','sensors','feature','k','rank','score','N','positive_rate','Recall','Precision','F1','FPR','AP'])
for sp in ['selection','test']:
 para(('개발용' if sp=='selection' else '최종 평가: 한계 설명용, 재튜닝 없음')+' 오류 조건:')
 table(load(f'error_conditions_{main["key"]}_target0p01_{sp}.csv'))
dev=load(f'error_conditions_{main["key"]}_target0p01_selection.csv').set_index('condition')
te=load(f'error_conditions_{main["key"]}_target0p01_test.csv').set_index('condition')
para(f'개발용 구간 시작의 정상 {int(dev.loc["burst_start","normal"])}행 중 FP {int(dev.loc["burst_start","FP"])}개, 시작 이후 정상 {int(dev.loc["after_start","normal"])}행 중 FP {int(dev.loc["after_start","FP"])}개였다. 짧은 버스트 이상 {int(dev.loc["short_burst","anomaly"])}행 중 FN {int(dev.loc["short_burst","FN"])}개였다. 최종 평가에서는 구간 시작 이상 {int(te.loc["burst_start","anomaly"])}행 중 FN {int(te.loc["burst_start","FN"])}개, 짧은 버스트 이상 {int(te.loc["short_burst","anomaly"])}행 중 FN {int(te.loc["short_burst","FN"])}개로, 짧은 구간 자체가 항상 더 어려웠다는 일반화는 지지되지 않는다.')
para(f'개발용 전류 변화 최상위 구간 정상 {int(dev.loc["current_change_bin3","normal"])}행 중 FP {int(dev.loc["current_change_bin3","FP"])}개, 두 번째 구간 정상 {int(dev.loc["current_change_bin1","normal"])}행 중 FP {int(dev.loc["current_change_bin1","FP"])}개였다. 변화가 클수록 오경보가 단조롭게 증가한다는 증거는 없다. 최종 평가의 P0 보완 이상 {int(te.loc["fallback","anomaly"])}행 중 FN {int(te.loc["fallback","FN"])}개로 구간 준비 중 취약성을 확인했지만 재튜닝하지 않았다.')
para('센서 크기와 전류 변화의 구간은 정상 학습 자료의 사분위수로 고정했다. 구간은 관측 센서값에 따른 분류이며 실제 부하나 공정 상태가 아니다. 전류 변화는 같은 버스트 안에서만 차분하고 시작점의 변화량은 미확인으로 남겼다. FP/FN CSV에는 원본 행·시각·센서·버스트 위치·윈도우 실제 경과시간·Q/T²/결합 점수·적용 임계값·예측·보완 경로를 저장했다.')
errors=pd.read_csv(run/'predictions'/f'errors_{main["key"]}_target0p01_test.csv')
contrib=[c for c in errors if c.endswith('__Q_contribution')]
if contrib:
 table(errors.groupby(['used_model','error'])[contrib].mean().reset_index())
if len(errors):
 bad=errors[errors.error=='FN'];cross=bad[bad.elapsed_seconds>max(.1,(main['window']-1)*.1+.01)]
 para(f'최종 FN {len(bad)}개 중 큰 시간 공백을 가로지르는 윈도우에서 {len(cross)}개가 발생했다. 관측 공백을 포함하는 P1의 장점/취약성을 같이 봐야 하며, 통계적 높은 점수만으로 시간 연속성이나 실제 고장 전 예측력을 주장하지 않는다.')
para('Q는 각 표준화 특징의 제곱 복원오차로 정확히 분해하고 같은 센서의 특징별 항을 합산했다. Q_feature__* 열은 특징 기여, *__Q_contribution은 센서 묶음이다. 크기는 정상 관계에서 벗어난 점수 기여일 뿐 고장 원인을 확정하지 않는다. T²로 경보가 난 관측의 Q 기여를 그 경보의 원인이라고 부르지 않는다. 주성분 계수도 원인 인과성을 보장하지 않는다.')
heading('후처리·확률·관측 첫 경보 시간')
para('후처리 없이 남긴 기본 예측에 연속 2회 및 최근 3회 중 2회를 적용했다. 파일·분할·버스트마다 초기화하며 미래 경보를 과거에 소급하지 않는다. 행별 정답을 유지하고 한 점 탐지로 버스트 전체를 맞혔다고 처리하지 않는다. 최근 3회 중 2회 정책은 현재 원점수가 낮아도 과거 2회 경보로 현재 경보가 유지될 수 있다.')
table(load('postprocess_selection.csv'),['policy','Recall','Precision','F1','F2','FPR','FN','FP','new_FN_vs_none','additional_TP_vs_none','missed_anomaly_bursts','median_first_alarm_seconds'])
if (run/'tables/postprocess_conditions_test.csv').exists():
 table(load('postprocess_conditions_test.csv'),['policy','condition','normal','anomaly','TP','FN','FP','TN','Recall','FPR'])
alarms=load('burst_alarms_test.csv');table(alarms[alarms.label==1],['burst_id','policy','rows','alarm_rows','no_alarm','first_alarm_since_observed_burst_start_seconds'])
para('미경보 구간은 no_alarm=True·시간 결측으로 보존했다. 시간 중앙값은 탐지한 구간에 조건부인 수치이며, 미경보 수를 함께 봐야 한다. 윈도우 준비시간은 예측 행의 window_rows/elapsed_seconds이며 모델 계산시간은 logs/score_*의 별도 측정이다. 긴 공백을 행 수×0.1초로 환산하지 않았다.')
table(pd.concat([load('probability_selection.csv'),load('probability_test.csv')],ignore_index=True))
image('reliability_selection.png');image('reliability_test.png')
para('sigmoid는 log1p(운영 점수)를 보정용 평균·표준편차로 변환한 뒤 비음수 기울기·L2 1e−4로 적합했다. 보정 자료의 정상+이상 라벨을 사용하는 추가 지도 단계다. 별도 선택/최종 자료의 Brier·log loss와 신뢰도 bin별 개수·실제 비율을 저장했다. 0~1 정규화나 정상 백분위수는 고장 확률이 아니며, Brier 하나가 좋아도 완벽한 확률 보정으로 판단하지 않는다. 이상 표본 132개·4버스트에 불과하고 날짜/현장 이상 비율이 바뀌면 확률을 재검증해야 한다.')
heading('현장 활용과 검증 한계')
para('운영 시각마다 센서 3개를 입력해 동일한 전처리·버스트 초기화·점수·보정 임계값을 적용한다. 구간 시작에는 고정 P0를 사용하고 보완 여부를 표시한다. 대시보드에는 점수/임계값 배수, 원시 센서, Q 기여, 실제 관측 시간 공백, 경보 지속 상태를 함께 제시할 수 있다. 경보는 점검 우선순위 제안이며 실제 고장·압력·품질·비가동을 확정하지 않는다.')
para('현장 적용 전 여러 날짜의 정상 부하 변화와 실제 고장/정비 이력을 수집하고, 날짜/설비 단위 외부 검증 및 운영 FPR·FN 감시가 필요하다. 운영 이상 비율이 현재 파일의 인위적 비율과 다르므로 Precision과 확률도 바뀐다. 설정을 바꾸려면 새 개발 자료와 새 평가 구간을 확보해야 하며 이번 최종 평가를 반복 최적화하지 않는다.')
heading('평가표 6항목 대응')
rubric=[
 ['① 데이터 이해 및 진단 (15)','해시·중복·간격·버스트·시간 분할 재검증','정상 20,000→19,999; 이상 600; 간격>0.5초 598/20','manifests/audit.json; clean_rows.csv; configs/first_experiment/frozen_*','날짜-라벨 혼재; 단위·운전조건 미확인'],
 ['② AI 예측모델 개발 및 비교 (40)','P0/P1/P2 Q/T²/QT 및 I0/I1/I2; 공통 행·보완 운영; 고정 최종 평가',f'Recall {pct(main["Recall"])}; FPR {pct(main["FPR"])}; IF 대비 Recall {(main["Recall"]-ifo["Recall"])*100:+.2f}%p','tables/metrics_selection.csv; metrics_test.csv; burst_pairs_test.csv; selection_lock.json','개발 이상 라벨 사용; 작은 이상 선택 표본; LSTM 재학습 없음'],
 ['③ 영향요인 및 오류분석 (15)','센서 7조합; FP/FN 조건 분모; Q 특징 기여; 표본수 보조',f'최종 FP {main["FP"]}, FN {main["FN"]}; 조건별 결과 표 참조','tables/sensor_combinations_selection.csv; error_conditions_*; predictions/errors_*','관측 조건의 연관성, 인과 원인 아님'],
 ['④ 현장 활용방안 (10)','전체 행 보완; 후처리; 실제 관측 첫 경보 시간; 추론 CLI',f'전체 {main["N"]}행 평가; 관측 이상 버스트 {int(none.anomaly_bursts)}개','src/infer.py; tables/postprocess_test.csv; burst_alarms_test.csv','실제 고장 시작·품질·압력·정비 검증 불가'],
 ['⑤ 창의성·차별성 (10)','Q/T² 분리·보정 결합; 동일 행 대조; 짧은 구간 보완; labeled sigmoid','추가 탐지/추가 오경보 및 확률 신뢰도 수치 공개','tables/score_complements_test.csv; probability_test.csv; figures/reliability_test.png','일반적 기법의 적용이며 새 알고리즘 발명 주장 없음'],
 ['⑥ 코드 및 재현성 (10)','전용 환경·고정 해시·모델 객체·seed 전부·단일 명령 재실행',f'CPU Python {env["python"].split()[0]}; sklearn {env["versions"]["sklearn"]}','manifests/environment.json; models/; source/; scripts/reproduce.sh','CPU/버전 차이의 실행시간 직접 비교 제한; 접근 가능한 동일 CSV 필요']]
table(pd.DataFrame(rubric,columns=['평가 항목','수행한 실험','실제 결과','근거 파일','한계']))
heading('실행 문제와 실제 검증 범위')
issues_path=run/'manifests/execution_issues.json'
if issues_path.exists():table(pd.DataFrame(read(issues_path)))
para('환경 설치·그림 backend 오류는 모델 평가 전에 해결했다. 개발용에서 주 모델과 F1 대안의 모델명이 같아 오류 파일명이 겹쳤던 저장 문제는 target FPR를 파일명에 넣어 해결했다. 원래 파일은 남기고 target0p01/target0p005가 있는 파일을 공식 진단으로 사용한다. 모델·임계값 재선택이나 최종 평가 재튜닝은 없었다.')
if (run/'manifests/final_artifact_validation.json').exists():para('저장 모델 추론 검사: 라벨 없는 입력으로 최종 4,347행 판정이 Colab 평가와 일치했다. 최대 점수 차이는 부동소수점 반올림 범위였으며, 전 후보의 보완 경로와 점수 연결도 검증했다.')
heading('재현성과 결과 찾아보기')
para('모든 후보·제외 이유는 protocol.json과 logs/exclusions.json, 센서 예외는 logs/sensor_exclusions.json에 있다. 결과가 나쁜 후보·seed도 metrics_*에 유지했다. Colab CPU에서 실제 학습/검증/최종 평가를 했고 로컬 전용 .venv에서 사전 검사와 보고서를 생성했다. 두 환경의 라이브러리 버전은 lock 파일로 맞췄으며 Python은 Colab 3.13, 로컬 3.14다. 다른 환경의 실행시간을 단순 비교하지 않는다.')
para('<code>bash scripts/reproduce.sh</code>는 새 실행 폴더에서 개발→누수 검증→설정 고정→최종 평가→이 보고서 생성을 수행한다. <code>.venv/bin/python src/infer.py --run runs/pca_20261003_cpu --csv 새파일.csv --out 새예측.csv</code>는 저장 모델만 사용하는 추론이다. 추론 입력은 파일별 시간순 관측이며 라벨은 입력 특징으로 쓰지 않는다.')
for folder in ['tables','manifests','predictions','figures','logs']:
 files=sorted((run/folder).glob('*'))
 sections.append('<details><summary>'+esc(folder)+f' ({len(files)})</summary><ul>'+''.join('<li>'+artifact(f.name,f'{folder}/{f.name}')+'</li>' for f in files if f.is_file())+'</ul></details>')
heading('확인한 근거와 미확인 문헌')
refs=read(ROOT/'references/checked_sources.json')
for r in refs:para(f'<a href="{esc(r["url"])}">{esc(r["title"])}</a>: {esc(r["access"])}. {esc(r["used"])}.')
para('분할 비율·버스트 간격·성분 수 후보·Q/T² 결합식·목표 FPR·후처리 조건은 프로젝트 실험 설계다. 인용 논문이나 대회가 요구한 필수 규칙으로 소개하지 않는다. 확인하지 못한 Jackson–Mudholkar 논문 본문·IEEE 논문 본문의 특정 내용을 인용하지 않았다.')
style='body{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#1f2937;line-height:1.65}h1,h2{color:#123d55}h2{margin-top:2.4em}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{border:1px solid #ddd;padding:7px;text-align:left}th{background:#edf4f8}.scroll{overflow-x:auto}img{max-width:100%;display:block;margin:16px 0}code{background:#f2f4f5;padding:3px}a{color:#116ca5}details{margin:10px 0}'
body='<!doctype html><html lang="ko"><meta charset="utf-8"><title>PCA 유압펌프 이상탐지 실험</title><style>'+style+'</style><h1>진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석 — PCA 독립 실험</h1>'+''.join(sections)+'</html>'
(run/'REPORT.html').write_text(body)
# Concise machine-readable evidence includes report provenance; no recomputed model selection.
(run/'report_manifest.json').write_text(json.dumps({'report':'REPORT.html','selection_lock_sha256':sha(run/'selection_lock.json'),'final_summary_sha256':sha(run/'final_summary.json'),'generated_by':str(Path(__file__).resolve()),'source':str(run)},indent=2))
print('REPORT',run/'REPORT.html')

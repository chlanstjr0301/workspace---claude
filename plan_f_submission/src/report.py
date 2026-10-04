"""Korean evidence-linked report generated from actual saved outputs."""
import argparse,html,json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation import metrics,postprocess

ROOT=Path(__file__).resolve().parents[1]

def generate(out):
    out=Path(out); rd=lambda name:pd.read_csv(out/'tables'/name)
    lock=json.loads((out/'selection_lock.json').read_text()); choice=lock['main']; main=rd('selected_test.csv').query("role=='main'").iloc[0]
    selected=rd('selected_test.csv'); dev=rd('selected_selection.csv'); post=rd('postprocessing_test.csv'); pairs=rd('burst_pairs_test.csv'); cov=rd('coverage.csv')
    same=post[(post.key==choice['key'])&(post.target_fpr==choice['target_fpr'])]
    raw=same[same.policy=='none'].iloc[0]; chosen=same[same.policy==choice['policy']].iloc[0]
    sections=[]
    def p(s):sections.append('<p>'+s+'</p>')
    def h(s):sections.append('<h2>'+s+'</h2>')
    def tab(df):sections.append(df.to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0,escape=True,na_rep='미정의'))
    def link(path,text=None):return f'<a href="{path}">{html.escape(text or path)}</a>'
    def fig(name):sections.append(f'<a href="figures/{name}"><img src="figures/{name}" alt="{name}" loading="lazy"></a>')
    def pct(x):return f'{100*x:.2f}%'
    p('<b>선정 모델: '+html.escape(choice['key'])+'의 seed 42·43·44 점수 평균, 윈도우 부족 시 M2 보완, 후처리 '+html.escape(choice['policy'])+'.</b> 정상 보정 자료에서 목표 FPR '+pct(choice['target_fpr'])+'로 정한 임계값 '+f"{choice['threshold']:.6f}"+'를 고정했다. 별도 선택 검증에서 FPR 1% 이하 후보 중 Recall, F1, AP, 실행시간 순으로 선정했다. '+('검증 기준을 충족했다.' if lock['compliant'] else '<b>충족 후보가 없어 최소 FPR 후보를 참고용으로 기록했다.</b>'))
    p(f"최종 평가에서 실제 이상 {int(main.TP+main.FN)}개 중 <b>{int(main.TP)}개 탐지, {int(main.FN)}개 미탐</b>. 정상 {int(main.TN+main.FP)}개 중 <b>{int(main.FP)}개 오경보</b>. 정상 1,000개 관측당 오경보 {main.FP_per_1000:.2f}개.")
    tab(selected[['role','key','policy','TP','FN','FP','TN','Recall','Precision','F1','F2','FPR','AP','PR_AUC_trapezoid','ROC_AUC']])
    p(f"최종 평가 정상 {int(main.TN+main.FP)}행, 이상 {int(main.TP+main.FN)}행, 이상 비율 {pct(main.positive_rate)}. 이 표의 점수는 행 단위다. 버스트 한 점의 탐지를 전체 행 정답으로 확장하지 않았다. 날짜와 상태가 완전히 겹쳐 있으므로 고장 특성과 날짜별 운전조건을 구분한 성능은 아니다.")
    p(f"선정 임계값에서 후처리 전→후: FP {int(raw.FP)}→{int(chosen.FP)}, FN {int(raw.FN)}→{int(chosen.FN)}, Recall {pct(raw.Recall)}→{pct(chosen.Recall)}. 실제 고장 시작 시각이 없어 고장 사전 예측 시간은 검증하지 못했다. 현재 검증한 것은 주어진 상태 라벨의 탐지와 관측 구간 안에서의 경보 동작이다.")
    pair20=pairs[(pairs.pair=='M3/M4')&(pairs.window==20)&(pairs.seed=='ensemble')&(pairs.target_fpr==.01)].iloc[0]
    p(f"같은 길이20의 공통 행에서 버스트 고려 M4−미고려 M3 차이는 Recall {pair20.delta_Recall*100:+.2f}%p, F1 {pair20.delta_F1:+.4f}, FPR {pair20.delta_FPR*100:+.2f}%p였다(정상 보정 목표FPR1%, 후처리 없음). 이는 위의 최종 운영 방식과 조건이 다른 순수 쌍별 비교다.")
    p('현장 적용 전에는 정상·이상 전환을 포함한 여러 날짜·설비의 기록, 고장/정비 확정 시각, 부하·회전수·제품·작업단계, 센서 단위와 취득 필터, 현장 이상 비율 및 오경보 비용이 필요하다. '+link('selection_lock.json','선정 시점과 고정 규칙'))
    h('데이터 이해와 진단')
    p('분석 단위는 센서의 한 시각 관측 행이다. 버스트는 거의 0.1초 간격으로 이어진 관측 구간이며 실제 기계 사이클이나 독립 고장 사건이 아니다. 가이드북 PDF 9·11쪽에서 AI0=상부 진동, AI1=하부 진동, AI2=전류를 확인했다. 센서 단위, 개별 생산품, 설비 ID, 부하 조건, 실제 고장 종류는 미확인이다. 배경의 기어 마모·유압 저하·불량·비가동은 이 CSV에서 검증한 측정 결과가 아니다.')
    audit=json.loads((ROOT/'configs/frozen/audit.json').read_text())
    tab(pd.DataFrame([dict(file=a['file'],original_rows=a['original_rows'],analysis_rows=a['analysis_rows'],duplicates_removed=a['removed_duplicates'],missing=sum(a['missing'].values()),numeric_failure=sum(a['numeric_failures'].values()),timestamp_failure=a['timestamp_failures'],bursts=a['bursts']) for a in audit['files']]))
    p('이전 점검과 원본 행 수·결측·변환 실패·중복·큰 간격은 일치했다. 정상의 11,565·11,566번째 관측 중 뒤 행만 분석 목록에서 제외했다. TimeStamp·3센서·Equipment_state가 모두 같은 추가 중복만 제거했고, 큰 센서값은 자르거나 삭제하지 않았다. 원본 CSV는 해시로 보존을 확인했다. 행 번호·파일명·날짜·burst_id·Equipment_state는 검출기 입력에 넣지 않았다.')
    p('원본 전체 20,600행에서 모두 정상 예측 Accuracy는 97.0874%다. 이 때문에 Accuracy는 보조 지표로만 사용했다. 전체 요약 통계를 분할 전에 이미 살펴본 사실을 명시한다. 완전히 보지 않은 신규 날짜 외부 검증은 아니다.')
    tab(rd('sensor_descriptives_deduplicated.csv').head(8))
    p(link('manifests/configs/frozen/audit.json','전체 진단')+' · '+link('tables/largest_time_jumps.csv','가장 큰 시간 점프')+' · '+link('tables/sensor_descriptives_deduplicated.csv','중복 제거 후 센서 통계 (표준편차 ddof=1)'))
    h('성능 확인 전에 고정한 분할과 버스트')
    tab(pd.read_csv(ROOT/'configs/frozen/split_summary.csv'))
    p('정상 약60/20/20%, 이상 약40/60%를 시간 순서로 나누고, 개발 구간을 다시 앞쪽 보정·뒤쪽 선택 검증으로 나눴다. 정확한 비율보다 동일 버스트 보존을 우선했다. 정상 학습 내부 약85%는 LSTM 학습, 나머지는 조기 종료에 썼고 양쪽 경계를 넘는 윈도우도 금지했다. IF는 정상 학습 전체를 사용했다. 정규화는 정상 학습 자료에서만 적합했다.')
    p('0.5초 초과 간격은 새 버스트로 정했다. 정상 학습 자료의 0.2/0.5/1초 기준은 모두 360개 버스트로, 행별 구분도 같아 중복 실험을 생략했다. 이는 이번 실험의 가정이다. 큰 공백은 미관측 상태이며 보간하지 않았다.')
    tab(rd('burst_summary.csv').query('window==10'))
    fig('burst_gap_distribution.png');fig('splits_sensors_label0.png');fig('splits_sensors_label1.png')
    h('모델·학습 예산·가이드북과의 차이')
    tab(pd.DataFrame([['M0','LSTM-AE','버스트 미고려'],['M1','동일 LSTM-AE','버스트 내부'],['M2','3센서 한 행 Isolation Forest','시간 윈도우 없음'],['M3','15개 시간영역 특징 Isolation Forest','버스트 미고려'],['M4','M3와 동일 특징·설정','버스트 내부']],columns=['모델','검출기','시간 처리']))
    p('LSTM: 3센서 절댓값→MinMaxScaler, encoder64→32, repeat, decoder32→64, 선형3 출력. 학습은 전체 시퀀스 MSE, 점수는 마지막 관측 3센서 MSE. Adam .001, batch128, 최대20epoch, 조기종료 patience4/min_delta1e-5, LR factor.7/patience2, CPU 2 threads. 검출기는 정상 자료로만 학습했지만 선택·확률 보정에는 개발 이상 라벨을 썼으므로 전체 과정을 완전 비지도라고 부르지 않는다.')
    p('IF: 300 trees, max_samples256, seeds42/43/44, contamination=auto. 모델 내 자동 판정은 사용하지 않고 -score_samples로 클수록 이상하게 통일했다. M3/M4는 부호를 보존한 센서별 평균·모집단 표준편차(ddof=0)·RMS(sqrt(mean(x²)))·최소·최대의 15특징이다. 길이5/10/20, stride1, 후방 윈도우이며 마지막 행에 점수를 대응했다. FFT는 쓰지 않았다.')
    p('가이드북 정확 재현 미완료: 원문은 길이20, 최대800epoch, early patience120, 정상15000행 학습, 윈도우를 만든 뒤 검증 구분, 100시점 뒤 라벨 및 검증 P=R 임계값이다. 여기서는 PyTorch 구조 기반 공통 분할 비교로 바꿨다. 프레임워크 초기화·학습 예산 등도 다르다. 원문의 참고 F1 74.76%, Accuracy97.51%(TN3922 FP78 TP154 FN26)는 이번 평가 수치와 직접 비교하지 않는다. 원문 미래 라벨 방식만으로 10초 사전 예측이 입증되는 것은 아니다.')
    p('10Hz라는 사실만으로 앨리어싱 발생 여부를 확정할 수 없다. 원 신호 대역폭·센서 취득 주파수·안티앨리어싱 필터 확인이 필요하다. 시간영역 모델도 이미 발생한 앨리어싱을 제거하지 못한다.')
    tab(rd('training_budget.csv'));fig('lstm_learning_curves.png')
    budget=rd('training_budget.csv');ae=budget[budget.key.str.startswith(('M0','M1'))]
    p(f"기본 LSTM {len(ae)}개 학습 중 {int(sum(ae.epochs==20))}개가 최대20epoch에 도달했다. 따라서 이번 계산 예산에서의 비교이며, 더 오래 학습한 LSTM의 잠재 성능이나 두 모델 계열의 일반적인 우열을 입증하지 않는다.")
    h('공정한 공통 행 비교와 전체 행 운영')
    p('모든 모델·윈도우가 판단할 수 있는 동일 마지막 행의 교집합을 점수 계산 전에 저장했다. 아래 공통 행 비교는 구간 시작과 짧은 버스트의 어려운 행을 제외한다. native 범위는 모델별 진단용이며 서로 다른 행에서 계산한 성능으로 순위를 정하지 않았다. 동일 길이 쌍 비교는 해당 길이의 공통 행을 따로 사용한다.')
    test=rd('metrics_test.csv');cols=['key','N','TP','FN','FP','TN','Recall','Precision','F1','F2','FPR','AP']
    tab(test[(test.seed=='ensemble')&(test.scope=='common')&(test.target_fpr==.01)][cols])
    p('전체 행에서는 완전 윈도우가 없는 행을 M2로 보완했다. 원시 복원오차와 IF점수를 직접 연결하지 않았다. 각 모델의 정상 보정 점수 분위수를 표준정규 분위수에 대응시키고 양 끝을 선형 외삽한 뒤, seed별 M2 보완과 3seed 평균을 적용했다. 전체 정상 보정 분포에서 임계값을 다시 구했다. 이 공통 점수는 고장 확률이 아니다. 같은 보정 정상 자료를 변환과 임계값에 썼으므로 성능은 별도 선택·최종 구간에서 확인했다.')
    tab(cov[(cov.split=='test')&cov.key.isin(['M0_L10','M1_L10','M2_L1','M3_L10','M4_L10'])][['key','label','total','available','excluded','coverage','crossing_windows','median_elapsed_seconds','max_elapsed_seconds']])
    tab(test[(test.seed=='ensemble')&(test.scope=='operational')&(test.target_fpr==.01)][cols])
    h('버스트 처리의 효과와 학습 표본 수 차이')
    pp=pairs[(pairs.seed=='ensemble')&(pairs.target_fpr==.01)]
    tab(pp[['pair','window','N','delta_Recall','delta_F1','delta_FPR','delta_AP']])
    p('delta는 버스트 고려−미고려이며 0.01은 1%p 차이다. 위 표는 같은 길이·같은 행 비교다. 학습 윈도우 수는 training_budget.csv, 포함/제외 대상은 manifests/rows_*.csv에 저장했다. 개선이 관측된 개발 비교만 미고려 학습 윈도우를 고려 모델 수에 맞춰 추가 학습했다. 이 표본 수 보조 실험은 개발 자료에서만 해석한다.')
    match=rd('matched_training_count.csv');tab(match)
    # Fix per-family optimal baseline using development common rows; never pick using final scores.
    ds=rd('metrics_selection.csv');opts=[]
    for model in ['M0','M1','M2','M3','M4']:
        a=ds[(ds.model==model)&(ds.seed=='ensemble')&(ds.scope=='common')];e=a[a.FPR<=.01]
        best=(e if len(e) else a).sort_values(['Recall','F1','AP','key'],ascending=[False,False,False,True]).iloc[0]
        b=test[(test.key==best.key)&(test.target_fpr==best.target_fpr)&(test.seed=='ensemble')&(test.scope=='common')].iloc[0]
        opts.append(dict(model=model,dev_selected_key=best.key,target_fpr=best.target_fpr,dev_FPR=best.FPR,dev_Recall=best.Recall,test_FPR=b.FPR,test_Recall=b.Recall,test_F1=b.F1))
    opt=pd.DataFrame(opts);opt.to_csv(out/'tables/per_family_dev_optimal.csv',index=False);tab(opt)
    p('각 버전의 개발 최적 설정 비교는 길이·임계값이 달라 순수한 버스트 효과가 아니다. 위 선정은 설명용이며 이미 고정된 주 운영모델을 변경하지 않는다.')
    h('선정 규칙·임계값·seed 변동')
    p('정상 보정 자료의 상위0.5/1/2/5% 기준을 quantile(method=higher)로 정하고 score>threshold일 때만 이상으로 판정했다. 동점은 정상이다. 실제 보정 FPR은 metrics 표에 함께 기록했다. 주 선정은 전체 관측 행 운영 후보의 선택 검증 FPR≤1%, Recall→F1→AP→실행시간 순이다. 선택 검증의 표본이 적어 안정성을 보장하지 않는다. 최종 평가 곡선으로 임계값을 다시 정하지 않았다.')
    tab(dev);tab(selected)
    p('F1 우선 대안도 동일 개발 후보에서 고정했다. 두 선택이 같으면 두 기준이 이번 제한된 후보군에서 같은 결론을 낸 것이다. 최종 평가나 현장에서 FPR1%를 보장하지 않는다. Precision/AP는 이상 비율에 의존하므로 다른 현장 비율로 그대로 이전할 수 없다.')
    seedrows=[]
    for split in ['selection','test']:
        for seed in [42,43,44]:
            a=pd.read_csv(out/'predictions'/f"all_{choice['key']}_s{seed}_{split}.csv.gz")
            pred=postprocess(a,a['prediction_'+str(choice['target_fpr'])],choice['policy'])
            seedrows.append(dict(split=split,seed=seed,**metrics(a.label,a.score,pred)))
    ss=pd.DataFrame(seedrows);ss.to_csv(out/'tables/selected_seed_metrics.csv',index=False);tab(ss)
    agg=ss.groupby('split')[['Recall','Precision','F1','F2','FPR','AP']].agg(['mean','std']);agg.to_csv(out/'tables/selected_seed_mean_std.csv');sections.append(agg.to_html(float_format=lambda x:f'{x:.5f}'))
    p('표준편차는 세 seed 사이의 표본 표준편차(ddof=1)이며 데이터 불확실성의 신뢰구간이 아니다. 같은 날짜의 버스트를 독립 고장으로 취급하는 신뢰구간은 계산하지 않았다. 가장 좋은 seed를 선택하지 않고 고정된 3seed 평균을 운영 점수로 사용했다.')
    tab(rd('dummy_test.csv'));fig('curves_selection.png');fig('curves_test.png');fig('confusion_test.png')
    p('AP는 recall 증가량으로 precision을 가중합하는 지표, PR-AUC는 PR곡선을 사다리꼴 적분한 지표로 별도 보고했다. PR/ROC는 연속 점수로 계산한다. 후처리 표의 AP/AUC는 후처리 이전의 연속 점수 평가이며 이진 경보의 AUC가 아니다. 예측 양성이 없으면 Precision=0, F1/F2 분모가 0이면0, 실제 양성/음성 분모가 없거나 단일 클래스 AUC는 미정의(NaN)다.')
    h('센서 영향·상호작용·FP와 FN 조건')
    combo=rd('sensor_combinations.csv');tab(combo.groupby('key')[['Recall','F1','FPR','AP']].agg(['mean','std']).reset_index())
    cm=combo.groupby('key')[['Recall','F1','FPR']].mean();full=cm.loc['M2_sensors_012'];no0=cm.loc['M2_sensors_12']
    p(f"한 행 모델의 센서3개 조합에서 평균 Recall은 {pct(full.Recall)}였고, AI0 상부진동을 빼면 {pct(no0.Recall)}로 변했다. F1 변화는 {no0.F1-full.F1:+.4f}, FPR 변화는 {(no0.FPR-full.FPR)*100:+.2f}%p다. 이는 관측된 개발 데이터에서의 기여이며 고장의 원인을 뜻하지 않는다.")
    ab=rd('feature_ablation.csv');tab(ab.groupby('removed_feature')[['delta_Recall','delta_F1','delta_FPR','delta_AP']].mean().reset_index())
    p('센서 조합 비교는 M2의 7개 부분집합, 특징 제거는 M4 길이10의 15개 특징을 하나씩 제거한 개발 실험이다. 모두 세 seed, 정상 보정 목표FPR1%, 동일 선택 행에서 비교했다. 최종 모델 선택 후보를 늘리는 데 사용하지 않았다. 단일 센서가 놓치고 조합이 탐지한 행은 '+link('tables/sensor_combination_rescues.csv')+'에 저장했다. 제거 효과는 예측상의 기여이며 물리적 고장 원인 증명이 아니다.')
    tab(rd('error_conditions_selection.csv'));fig('sensor_interactions_selection.png')
    ec=rd('error_conditions_selection.csv'); near=ec[(ec.condition=='burst_start_first_9_rows')&(ec.group=='start')].iloc[0];later=ec[(ec.condition=='burst_start_first_9_rows')&(ec.group=='later')].iloc[0]
    p(f"개발 검증의 버스트 첫9행에서는 정상 {int(near.normal_n)}개 중 FP {int(near.FP)}개, 이상 {int(near.anomaly_n)}개 중 FN {int(near.FN)}개였다. 이후 행에서는 정상 {int(later.normal_n)}개 중 FP {int(later.FP)}개, 이상 {int(later.anomaly_n)}개 중 FN {int(later.FN)}개였다. 행 수와 시계열 의존성이 달라 오류 수만으로 집중도를 단정하지 않는다.")
    p('조건 경계는 정상 학습 센서 절댓값·버스트 내부 전류 변화의 사분위수다. 실제 부하/작업 조건이 없어 관측값 기반 분류다. 버스트 시작 전류 차분은 알 수 없음으로 남겼다. 오류 수뿐 아니라 조건별 정상·이상 분모와 FPR/FNR를 함께 저장했다. 최종 오류 조건 분석은 한계 설명용이며 재튜닝하지 않았다. '+link('tables/error_conditions_test.csv')+' · '+link('predictions/main_selection_FP_FN.csv')+' · '+link('predictions/main_test_FP_FN.csv'))
    h('후처리·경보 지연·조기탐지 검증 범위')
    tab(same[['policy','TP','FN','FP','TN','Recall','Precision','F1','F2','FPR','normal_alarm_episodes','duplicate_episodes','anomaly_bursts_without_alarm','observed_start_to_first_alarm_median_seconds']])
    changes=rd('postprocessing_row_changes_test.csv');tab(changes)
    p('새로 놓친 이상과 새로 탐지한 이상을 별도로 셌다. 최근3회 중2회 규칙은 직전 양성의 영향으로 현재 원시 판단이 음성인 행에서도 경보할 수 있어, 미탐의 순증가와 새로 놓친 행 수가 다를 수 있다. 후처리의 최종 F1이 원시 판단보다 낮아졌더라도 최종 결과를 보고 선택을 바꾸지 않았다.')
    p('연속2회와 최근3회 중2회를 비교했고 버스트 경계에서 상태를 초기화했다. 최근3회 규칙은 현재까지 최대3개 관측 중 2개가 양성이면 경보한다. 미래 관측을 사용하지 않고 과거로 소급하지 않는다. 중복 경보는 동일 버스트 내 첫 경보 episode 뒤 추가로 시작한 episode 수다. 경보 episode는 0→1 전이로 센다. 독립 고장 수가 아니다.')
    p('실제 고장 시작 시각·정상→고장 전환·정비 확정 시각이 없으며 라벨 생성 방식도 순간별 판정인지 파일 단위 부여인지 미확인이다. 파일 첫 행 또는 버스트 시작을 고장 시작으로 가정하지 않았다. 관측 시작→첫 경보의 중앙값은 경보가 있는 버스트에 한정되며 무경보 버스트 수를 함께 제시했다. 사전 예측 시간이나 실제 고장 탐지 지연으로 해석할 수 없다.')
    p('길이5/10/20 윈도우 준비시간은 연속 0.1초 관측에서 각각0.4/0.9/1.9초다. 실제 TimeStamp로 계산한 준비시간과 윈도우 전체 경과시간을 별도 저장했다. 계산시간은 CPU 배치 추론+모델 로딩에 걸린 실측값이며 준비시간과 다르다. 단일 행 실시간 배포 지연 보장은 아니다. 시간당 경보율은 미관측 공백의 분모 혼동을 피하려고 주 지표로 사용하지 않았다.')
    p(link('tables/window_warmup.csv')+' · '+link('tables/computation_time.csv')+' · '+link('tables/postprocessing_delays_test.csv')+' · '+link('tables/alarms_main_test.csv'))
    fig('timeline_selection_label0.png');fig('timeline_selection_label1.png');fig('timeline_test_label0.png');fig('timeline_test_label1.png')
    h('확률 보정은 별도의 라벨 사용 단계')
    tab(rd('probability_selection.csv'));tab(rd('probability_test.csv'));fig('reliability_selection.png');fig('reliability_test.png')
    tab(rd('probability_classification_test.csv'))
    p('sigmoid는 정상 보정2011행·이상 보정132행의 라벨을 사용했다. 양의 점수 방향을 유지하도록 slope≥0 제약과 L2 규제를 주었다. 같은 자료에서 보정 성능을 주장하지 않고 뒤쪽 선택 검증과 최종 평가에서 Brier/log loss·신뢰도를 확인했다. 보정 전 원시 점수는 확률이 아니므로 Brier를 직접 계산하지 않았다. 비교용 sigmoid(score)는 명시적으로 미보정 가짜 확률 기준이며, 상수 보정집합 이상비율도 함께 비교했다.')
    p('판정 임계값을 sigmoid로 대응시켜 전후 판정 차이를 저장했다. 확률 보정이 좋아도 탐지율 자체가 자동 개선되는 것은 아니다. 132개 이상 표본이 네 관측 버스트·한 날짜에 몰려 있어 신뢰 가능한 현장 고장 확률로 볼 수 없다. 현장 이상 비율과 운전조건이 달라지면 재검증이 필요하다.')
    h('평가표 요구사항 → 방법 → 실제 결과 → 근거 → 한계')
    mapping=[
      ['① 데이터 이해·진단 15','변수 근거, 중복/시간 점검, 사전 분할',f"정상19999·이상600, 버스트599·21, 보정/선택/최종 분리",'manifests/configs/frozen/audit.json; split_summary.csv; tables/burst_summary.csv','제품·설비ID·단위·운전조건 미확인; 추가 메타데이터 필요'],
      ['② AI 모델 40','M0~M4, 5/10/20행, 3seed, 공통행·전체행',f"{choice['key']}, 최종 Recall {main.Recall:.4f}, F1 {main.F1:.4f}, FPR {main.FPR:.4f}",'tables/metrics_test.csv; selected_test.csv; selection_lock.json','하나의 정상 날짜와 이상 날짜; 정확한 가이드북 원문 재현 아님'],
      ['③ 영향·오류 15','7센서조합, 15특징 제거, 조건별 오류 분모',f"선정 모델 FP{int(main.FP)}, FN{int(main.FN)}; 제거 효과 전부 공개",'tables/sensor_combinations.csv; feature_ablation.csv; error_conditions_selection.csv; predictions/main_test_FP_FN.csv','센서 기여는 물리 원인 아님; 운전 메타데이터 필요'],
      ['④ 현장활용 10','점검 우선순위·근거 시각·공백표시·지속경보 정책','행별 점수/경보/담당모델 및 사례 CSV 생성','predictions/main_test.csv; figures/timeline_test_label1.png; src/infer.py','비용·불량·비가동 감소 미검증; 현장 파일럿 필요'],
      ['⑤ 차별성 10','버스트 쌍 비교, M2 보완, 후처리, 확률 보정',f"후처리 FP {int(raw.FP)}→{int(chosen.FP)}, FN {int(raw.FN)}→{int(chosen.FN)}; 개선/악화 모두 저장",'tables/burst_pairs_test.csv; matched_training_count.csv; postprocessing_test.csv; probability_test.csv','보완은 결합모델 효과; 공통행은 시작 구간 제외; 확률 외부검증 필요'],
      ['⑥ 코드·재현성 10','고정해시, 환경버전, seed, 체크포인트, 단일명령','전처리~최종보고 코드와 모델·전처리 객체 저장','manifests/environment.json; models/; ../../README.md; ../../src/','라이브러리/CPU 차이로 실수 오차·실행시간 차이 가능']]
    tab(pd.DataFrame(mapping,columns=['평가 요구사항','실험 방법','실제 결과','근거 파일','한계와 추가 정보']))
    h('현장 활용 제안과 검증되지 않은 기대효과')
    p('점수가 높은 관측과 지속 경보를 점검·품질검사 우선순위로 제공한다. 작업자에게 시각, 상·하부 진동과 전류, 원시 판단과 지속 경보, M2 보완 여부, 데이터 공백을 함께 보여준다. 순간 변화와 지속 이상을 분리하되 짧은 이상 손실을 감수할 수 있는지 현장에서 결정해야 한다. 이 출력은 자동 설비 정지 명령이 아니다.')
    p('압력 안정화, 불량 감소, 비가동 및 비용 절감은 기대 활용 효과다. 압력·품질·생산·정비 기록과 연결한 전향적 평가가 없어 검증 성과로 주장하지 않는다. 추가 검증은 같은 운전조건에서 정상과 고장을 모두 포함하고 날짜·설비를 통째로 분리한 외부 평가가 필요하다.')
    h('재실행·근거 자료')
    p('새 실행: <code>python3 scripts/run_all.py</code>. 기존 실행 이어가기: <code>python3 scripts/run_all.py --resume runs/실행폴더</code>. 추론: <code>python3 src/infer.py --run runs/실행폴더 --input 한_시계열.csv --output 새_예측.csv</code>. 자세한 설치와 재실행 절차는 프로젝트 README.md에 있다. 실행별 폴더를 쓰며 원본·기존 실행을 덮어쓰지 않는다.')
    p('데이터 출처: 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), 소성가공 예지보전 AI 데이터셋, 스마트제조혁신추진단(㈜인터엑스), 2022.12.23., www.kamp-ai.kr. 가이드북은 프로젝트 references/guidebook.pdf, 확인한 쪽과 차이는 references/evidence.txt.')
    p('방법 참고: <a href="https://doi.org/10.1109/ICDM.2008.17">Liu·Ting·Zhou (2008)</a>, <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432">Saito·Rehmsmeier (2015)</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html">Isolation Forest 점수 정의</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html">AP 정의</a>, <a href="https://scikit-learn.org/stable/modules/calibration.html">확률 보정</a>, <a href="https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data">시간 순서 검증</a>. 이 문헌은 방법 근거이며 이번 분할·FPR·버스트 기준의 공식 출처가 아니다.')
    files=sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file());pd.DataFrame({'artifact':files}).to_csv(out/'artifact_index.csv',index=False)
    p(link('artifact_index.csv','전체 산출물 목록'))
    doc='<!doctype html><html lang="ko"><meta charset="utf-8"><title>유압펌프 이상탐지 실험 보고서</title><style>body{font-family:system-ui,sans-serif;max-width:1280px;margin:40px auto;padding:0 24px;line-height:1.7;color:#17212e}h1,h2{color:#103d63}h2{margin-top:48px}table{display:block;overflow:auto;border-collapse:collapse;font-size:13px;margin:20px 0}td,th{border:1px solid #ced8e1;padding:7px 11px;text-align:left}th{background:#eaf0f5}img{max-width:100%;border:1px solid #ddd;margin:14px 0}code{background:#f1f3f5;padding:3px}a{color:#16629b}</style><h1>진동·전류 시계열 기반 프레스 유압펌프 이상 탐지 및 오경보 분석</h1>'+''.join(sections)+'</html>'
    (out/'report.html').write_text(doc)
    summary=dict(selected=choice,final=main.to_dict(),postprocessing_before=raw.to_dict(),postprocessing_after=chosen.to_dict(),limitation='Only observed anomaly detection; no verified failure onset, date-label confounding',result_directory=str(out))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
    print(json.dumps(summary,ensure_ascii=False,default=str),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);generate(p.parse_args().out)

from common import *
import html,re
lock=json.loads((R/'frozen_selection.json').read_text());proto=json.loads((R/'protocol.json').read_text());trials=pd.read_csv(R/'all_trials.csv');s=pd.read_csv(R/'baseline_vs_candidates.csv');ops=s[s.scope=='operational'];dev=ops[ops.evaluation=='selection'];hist=ops[ops.evaluation=='historical'];blocks=pd.read_csv(R/'block_metrics.csv');bud=pd.read_csv(R/'tables/FP_integer_budgets.csv');sel=lock['criteria_selections']['primary'];val=json.loads((R/'validation_report.json').read_text());parts=[];web=[]
def mdtable(df):
 def v(x):return '' if pd.isna(x) else f'{x:.6f}' if isinstance(x,float) else str(x)
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(v(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
def add(title,body='',df=None):
 parts.append('## '+title+'\n\n'+body+'\n');web.append('<section><h2>'+html.escape(title)+'</h2>')
 for p in body.split('\n\n'):
  web.append('<p>'+re.sub(r'`([^`]+)`',r'<code>\1</code>',html.escape(p)).replace('\n','<br>')+'</p>')
 if df is not None:parts.append(mdtable(df)+'\n');web.append('<div class="table">'+df.to_html(index=False,float_format=lambda x:f'{x:.6f}',na_rep='—')+'</div>')
 web.append('</section>')
cols=['candidate','TP','FN','FP','TN','Recall','Precision','F1','F2','FPR','FNR','AP','PR_AUC_trapezoid','ROC_AUC','Accuracy','FP_per_1000']
add('결론: 기존 최고 PCA 유지', '''36개 운영 구성을 실제로 실행했지만, 개발에서 확인한 개선은 이미 노출된 과거 평가에서 유지되지 않았다. 기존 P1 W20·2성분·Q·StandardScaler·15특징·P0 보완·후처리 없음 구성을 유지한다. 선정 후보의 설정은 바꾸지 않았고 다른 후보를 과거 평가에 추가해 재선택하지 않았다.

기존 모델은 과거 이상357개 중329개 탐지,28개 미탐, 정상3,990개 중1개 오경보였다. 개발에서 고정한 3행 보완 후보는 이상323개 탐지·34개 미탐, 정상 오경보0개였다. FP1개를 줄이는 대신 FN6개가 늘고 F1 0.957787→0.950000, F2 0.935722→0.922330으로 낮아졌다. 주 개선 조건을 만족하지 않는다.

개발 selection에서는 임계값만 조정해 TP108/FN3/FP0을 얻었고, 짧은 창 보완이 TP109/FN2/FP0으로 이상1개를 더 탐지했다. 이 변화는 새 독립 자료의 성능 보장이 아니다.''')
add('1. 기준선 재현과 직전 실패의 의미', '''원 ZIP과 직전 검토 ZIP의 목록·CRC를 확인하고 이번에 사용한 파일을 압축본의 같은 파일과 해시 대조했다. 데이터는 정상20,000행에서 추가중복1개를 제거한19,999행과 이상600행이다. 원본 및 기존 모델·결과·논문은 수정하지 않았다.

기존 저장 모델을 별도 프로세스에서 추론해4,347행의 점수·판정을 대조했다. P1과P0를 정상 train에서 원설정으로 재학습하고 정상 calibration에서 참조값과 임계값을 다시 계산해도329TP/28FN/1FP/3989TN이 재현됐다. 저장 추론 재현과 재학습 재현은 reproduction.json에서 구분했다.

직전 median/IQR 후보는 원P1 target1%보다 Recall이 높아서 선택된 것이 아니다. 당시 내부99개 이상에서 median/IQR은83TP, 원P1 target1%는85TP였지만, 원P1의FPR1.5712%가 당시 제약을 넘었다. median/IQR은 적격 후보들 중 선택됐다. 원P1을 target0.1%로 조정한 대조군(82TP/10FP)보다83TP/8FP의 작은 이득은 있었으나 개발 후반의 악화는 선택 고정 이후 확인됐다. 직전의 넓은 탐색이나 중앙값/IQR 교체를 이번에 반복하지 않았다.

근거: inventory.json, reproduction_report_ko.md, inputs/prior_review/, tables/prior_comparison.csv, manifests/prior_history.json.''')
add('2. 사전 고정한 선택 규칙과 FP 정수 예산', '''이상=1. F1=2TP/(2TP+FN+FP), F2=5TP/(5TP+4FN+FP), FPR=FP/(FP+TN)으로 계산했다.

모든 적격 후보는 같은 selection 전체 행에서 기준선보다 Recall과F2가 엄격하게 증가하고, F1이 낮아지지 않으며, 절대FPR≤1%여야 한다. 여기에 보수 ΔFPR≤0, 주≤0.001, 보조≤0.002를 각각 적용했다. 0.001은0.1%p이며 기존FPR의0.1% 증가가 아니다.

추가FP 허용량은 Decimal로 N정상×ΔFPR을 계산해 내림했다. 표본이 작아도 최소1개를 임의로 허용하지 않았다. 절대FPR 한도도 FP≤floor(N정상×0.01)로 검사했다. 적격 후보 중F2→낮은FPR→Recall→F1→단순한 구성→고정candidate ID 순으로 선택했다. 수치 동률 허용오차는1e-12, 정수FP 예산에는 허용오차를 넣지 않았다.

이 선택 규칙과 F2 가중치는 이번 프로젝트의 자체 설계이며 대회 공식 기준이나 실제 현장 비용비를 의미하지 않는다.''',bud)
add('3. 분할과 시간 블록의 역할', '''원래 정상train12,012행/360버스트, 정상calibration2,011행/63버스트, selection 정상1,986행/58버스트·이상111행/4버스트, 과거test 정상3,990행/118버스트·이상357행/13버스트를 유지했다. 검출기·전처리는 정상train에만 fit했고 임계값은 정상calibration 점수로 계산했다. 후보 선택에는 selection 라벨을 사용했다.

selection을 버스트를 보존하는 앞/뒤 진단 블록으로만 나누었다. 새로운fold·holdout·교차검증이 아니며 블록 경계에서 예측을 다시 시작하거나 모델을 재학습하지 않았다. 동일한 selection 예측을 블록별로 집계한 것이다. 앞블록 정상981행은 주 기준 추가FP가0개, 뒤블록 정상1,005행은1개다.

이전 연구와 검토에서 개발자료·과거평가가 이미 공개됐으므로 이번 연구도 반복 개발이다. 정상과 이상은 각각2022-07-12/2022-07-17의 다른 날짜이고, 날짜/운전조건 교락을 해결하지 못했다.''',pd.read_csv(R/'tables/block_FP_budgets.csv'))
add('4. 실험A: 임계값만 변경', '''모델·특징·Q 점수·원Q95 배율·fallback·창처리·후처리를 모두 고정했다. 정상calibration 목표FPR 후보는0.1/0.2/0.3/0.5/0.75/1/1.25/1.5/2/2.5/3/5%의12개다. 기존 분위수higher와 score>threshold 판정을 유지했다. 동점은 정상이다.

정확한 기준 임계값은1.3078792257682357이며 목표1% 후보와 같다. 12개 수치 임계값에 중복은 없었다. 일부 임계값은 selection 판정이 같지만 같은 임계값이라는 뜻은 아니다. 모든 A 후보의 AP·사다리꼴PR-AUC·ROC-AUC는 동일함을 검사했다.

12개 모두 Recall 증가 조건을 통과하지 못했다. F2가 가장 높은 A_q0.0020은 임계값1.7186836831529793에서TP108/FN3/FP0으로 기존의FP8개를 제거했지만 FN은 그대로였다. 이는 오경보 감소의 기술적 대조군이며 주 개선 적격 후보가 아니다. frozen_selection.json의A_primary는null이고 별도A_descriptive_best로 표시했다.''',pd.read_csv(R/'threshold_sweep.csv')[['candidate','target','threshold','threshold_delta','TP','FN','FP','F1','F2','FPR','primary_eligible']])
add('5. 실험B: W20 준비 전 구간만 짧은 PCA로 보완', '''A 결과를 본 뒤 selection 오류조건만으로 한 개의 추가 가설을 고정했다. 기준선의 개발FN3개가 모두 P0가 사용된19개 이상행에 있고, W20이 사용된92개 이상행에는FN이 없었다. 이에 W20을 바꾸는 대신 W20이 부족하고 짧은 창이 준비된 행만3행 또는5행 PCA로 보완했다. 첫2행 또는4행에는 원P0를 유지했다.

과거 평가의 행번호·날짜·FN 목록으로 규칙을 만들지 않았다. 행 수는 분할 시작부터 현재까지의 가용 관측 수이고, 미래 버스트 길이는 쓰지 않았다. 원P1처럼 분할 안에서 공백을 넘어 과거 행을 연결하며 버스트 초기화 규칙을 변경하지 않았다. 따라서 이전에 실패한 모든버스트 초기화1→3→5→10→20 라우팅과 다르다.

정상train 최소100창, 동일길이 정상calibration 전체 최소100창, 실제 전환 경로 정상calibration 최소10행을 사전 고정했다. 미달이면 기존 시스템으로 되돌린다. 실제3행/5행은 정상학습12,008/12,004창, 정상보정2,009/2,007창, 전환경로17/15행이었다. 같은15특징·StandardScaler·2성분·Q로 학습했다. 짧은 모델의 Q95는 같은 길이의 가용 정상보정 전체에서 계산하고, 연결된 전체 운영점수에서 임계값을 다시 정했다. 적은 route표본으로 꼬리FPR가 안정적이라고 주장하지 않는다.

두구성×12임계값=24개로, A와 합계36개다. 후처리·센서교체·추가ensemble은 넣지 않았다. 3행/5행 보완은 동일한 선택지표를 기록했고, 단순성도 동률이어서 고정 ID 순으로3행을 골랐다. 최종 보수·주·보조 기준은 모두B_W3_q0.0020을 선택했다. 각 기준의 적격 후보 수는14개씩이었다.

선정B의 임계값은 A 대조와 정확히 같다. 개발에서 A 대비TP1 증가·FP변화0·F2 증가가 있어 짧은 보완의 추가 기여를 분리할 수 있었다. W20이 가용한 공통행의 점수는 A와B가 동일하다. 전체행에서만 나타난 개발 이득이며 어려운 시작행을 제외하지 않았다.''',pd.read_csv(R/'tables/route_calibration_counts.csv'))
add('6. 개발 전체와 진단 블록 결과','개발 전체 분모는 정상1,986·이상111(양성비율5.2933%)이다. 임계값 목표FPR와 실제selection FPR를 구분한다.',dev[cols])
add('블록별 결과','동일한 selection을 나눈 진단이며 별도 독립 검증이 아니다. 뒤블록은 기준선도 이상90개를 모두 탐지했으므로 그 블록에서 추가 Recall 이득은 없다.',blocks[blocks.candidate.isin([x['candidate'] for x in lock['final_configs']])][['candidate','block','normal','anomaly','normal_bursts','anomaly_bursts','TP','FN','FP','Recall','FPR','primary_allowed_extra_FP']])
add('7. 선택 고정 후 과거 평가', '''기준선, A 임계값 전용 대조, B 선정 보완의 중복 없는3개 구성만 실행했다. B_W5나 나머지 임계값을 과거 평가에서 실행해 다시 고르지 않았다. 코드·모델·평가 행·임계값·선택 이유를 frozen_selection.json에 먼저 고정했다.

아래는 이미 노출된 과거 평가 자료에서의 결과이며 새로운 독립 검증이 아니다. 분모는 정상3,990·이상357(양성비율8.2126%)이다. A는 정상1,000관측당FP0, B도FP0이지만, 기준선의1,000관측당FP0.2506을 줄이는 대신FN이 각각5개/6개 늘었다. 0FP가 현장 오경보0을 보장하지 않는다.

B는 AP가0.987098→0.989308로 올랐어도 고정 운영점의F1/F2와Recall은 나빠졌다. 점수 순위 개선을 목표 운영점의 성공으로 대신하지 않는다. B는 A보다도 TP1개가 적고FP는같아, 추가 보완의 이득이 과거 평가에서 유지되지 않았다.''',hist[cols])
add('8. 새로 잡은 이상과 새 오류를 분리', '''개발의 B는 새TP1·새미탐0, 기존FP8개 제거·새FP0개였다. 과거 평가의 B는 새TP0·새미탐6, 기존FP1개 제거·새FP0이었다. A의 과거 평가는 새미탐5개다. 따라서 과거 결과에서 FP 증가를 감수한 탐지 개선도 없고, FP 증가 없는 탐지 개선도 확인되지 않았다.

paired_errors.csv는 모든 원본행을 기준선과 연결하고, tables/changed_rows_only.csv는 네 종류의 변화가 발생한 행만 담았다. 파일·원래행번호·TimeStamp·센서·버스트위치·윈도우시각·사용모델·Q·threshold·정답·판정·기여값을 추적할 수 있다.''',ops[['evaluation','candidate','new_TP','lost_TP','removed_FP','new_FP','delta_TP','delta_FP']])
add('9. 판정 범위·경보 episode·시간', '''모든 관측에 점수가 있으며 unavailable은0이다. P1 W20 가용비율, 짧은 보완 비율, P0 비율을 coverage.csv에 따로 저장했다. 과거평가 기준선의P0는 정상19·이상19행,3행보완후P0는각2행이고짧은PCA는각17행이다. W20가용행은변하지않는다. 평가를공통행으로줄여성능을높였다고해석하지않는다.

경보episode는 같은버스트에서 이어진양성판정을묶고공백>0.5초에서끊었다. 정상FP는개발기준선8행이4episode,과거기준선1행이1episode였고두대조후보는0episode다. 이상관측버스트에는모두적어도한경보가있었지만이를고장사건전체를탐지한것으로바꾸지않았다. 무경보버스트수도 tables/alarm_episodes.csv에있다.

시간은실제TimeStamp차이로계산했다. 첫관측부터첫경보시간은고장시작이나고장전예측시간이아니다. 실제고장시작·정비정답이없어고장몇초전탐지라는값은계산하지않았다. 그림의선은버스트마다끊었고관측공백을연속측정으로보간하지않았다. 모델계산시간과창의관측경과시간을별도로저장했다.''')
add('10. 검증과 재현', f'''원본/사본 SHA-256, 원행정렬, 분할/train_role 경계를 넘지 않는 창, 신규모델 정상fit/calibration ID, 유효rank와작은고유값을검사했다. AP와사다리꼴PR-AUC를다르게계산하고 ROC-AUC에도연속점수를사용했다. 분모0의Precision/F1/F2는0, 클래스비율이나ranking metric이불가능한경우는결측으로남긴다.

최종3개구성을각각별도프로세스에서라벨없는CSV로추론해총13,041개판정과점수를대조했다. 각구성에서개발두파일앞부분26행에대해미래센서값을크게바꾸고라벨·행번호·미래버스트길이를바꿔도과거점수와라우트가변하지않는것을확인했다. 저장추론과원설정재학습은별도기록이다. PCA의seed만바꾼반복은없었다.

모든36개후보가완료됐고 실패/중단후보는0개다. CPU2스레드·기존전용환경을사용했고Colab이나유료자원은쓰지않았다. 추가학습은짧은PCA2개뿐이며90분예산을채우기위한탐색은하지않았다. 후보별시간은all_trials.csv, 구성별fit시간은fit_system_seconds에있다. 반복행의fit시간을합산해총시간으로오인하면안된다.

실행: `bash RUN_EXPERIMENT.sh`. 새로운형제실행폴더에실험과보고서를만들어기존결과를보존한다. `requirements.lock.txt`, `src/`, `models/`, `validation_report.json`이재현근거다. 기존프로젝트안에서실행할때원패키지대조검사를포함한다. 단일CSV추론은 `python src/infer_system.py --model models/warmup_W3.joblib --threshold 1.7186836831529793 --csv INPUT.csv --out NEW_OUTPUT.csv`다.''')
add('11. 요청한 일곱 가지 결론 구분', '''① 임계값만 조정한 개선: 개발에서FN3을유지하며FP8→0, F1/F2상승. Recall은동일하여주적격아님.
② 추가변경의별도기여: 개발에서A대비짧은보완으로TP1개추가·FP동일. 과거평가에서는TP1개감소.
③ 오경보증가없는개선: 개발B는해당하지만과거평가에서는미확인.
④ 오경보증가를감수한개선: 이번고정후보에서는해당없음. 전체36개절충점은숨기지않고공개.
⑤ 개발에서만좋은결과: B_W3_q0.0020은세기준모두통과했으나과거평가에서Recall/F1/F2가낮아짐.
⑥ 개선미확인및기존유지: 원P1 W20Q를유지한다. 고정선택파일을사후변경하지않고별도운영권고파일에남겼다.
⑦ 새자료없이판단불가: 독립날짜·설비·고장에대한일반화, 현장경보비용, 실제고장조기예측, 날짜/상태교락해소. 정상·이상이같은날짜와설비에서수집된새봉인자료가필요하다.''')
add('12. 실제 확인한 방법론 근거', '''Saito & Rehmsmeier (2015), The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. DOI10.1371/journal.pone.0118432. PLOS원문초록/지표정의와불균형평가설명을확인했다. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432

Davis & Goadrich (2006), The Relationship Between Precision-Recall and ROC Curves. DOI10.1145/1143844.1143874. 저자기관의ICML원문PDF에서초록·혼동행렬정의·PR/ROC관계및보간차이를확인했다. https://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf

PR관점과ROC를함께보되 AP와단순사다리꼴PR면적을동일시하지않는다. 사다리꼴면적은요청한수치적요약이며달성가능한PR보간의정확한면적이라고주장하지않는다. 목표FPR·FP예산·F2선택·라우팅·최소표본은논문이정한규칙이아닌자체설계다. DOI직접접근은오류가있어PLOS실제본문URL과저자기관PDF를사용했고접근제한을우회하지않았다.''')
header='# 기존 최고 PCA의 FN–FP 절충 실험\n\n작성일2026-10-04. 실행 폴더: '+str(R)+'\n\n'
(R/'review_report_ko.md').write_text(header+'\n'.join(parts))
figs=['selection_PR.png','selection_FN_FP.png','selection_threshold_F1_F2.png','historical_PR.png','historical_FN_FP.png','historical_B_W3_q0.0020_error_timeline.png']
web.append('<h2>핵심 그림</h2>'+''.join('<figure><img src="figures/'+f+'"><figcaption>'+html.escape(f)+'</figcaption></figure>' for f in figs))
(R/'review_report_ko.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PCA FN–FP 절충 실험</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.85 sans-serif;color:#172a3a}h1,h2{line-height:1.4}h2{margin-top:36px}p{word-break:keep-all}.table{overflow-x:auto}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd5dd;padding:6px}th{background:#eaf0f5}img{max-width:100%}code{background:#f1f4f6;padding:2px}figure{margin:24px 0}</style><h1>기존 최고 PCA의 FN–FP 절충 실험</h1><p><strong>개발 이득이 과거 평가에서 유지되지 않아 기존 PCA를 유지합니다.</strong></p>'+''.join(web)+'</html>')
js(R/'references/checked_sources.json',[dict(title='The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets',authors='Saito & Rehmsmeier',year=2015,doi='10.1371/journal.pone.0118432',url='https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432',scope='abstract, definitions, imbalance interpretation'),dict(title='The Relationship Between Precision-Recall and ROC Curves',authors='Davis & Goadrich',year=2006,doi='10.1145/1143844.1143874',url='https://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf',scope='abstract, metrics, relationship and interpolation sections')])
print('Reports written')

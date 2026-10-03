# Plan C 문헌조사

조사일: 2026-10-04  
범위: 시계열 이상탐지 평가, 공정 모니터링, 재구성 모델, 분포 이동, conformal calibration, 온라인 오경보 제어  
출처 정책: 학회·저널·PMLR·저자 원문 등 1차 자료를 우선했다. 블로그와 벤더 홍보자료는 사용하지 않았다.

---

## 1. 핵심 결론

1. **Point-adjusted F1을 쓰면 안 된다.** 한 점 검출을 전체 이상구간 검출로 바꾸어 무작위 점수도 강한 모델처럼 보일 수 있다.
2. **Window F1 하나로 모델을 고르면 안 된다.** range·alarm·block 수준 평가가 필요하다.
3. **단순 통계모델은 반드시 강한 기준선이어야 한다.** 최신 대규모 벤치마크에서도 통계·단순 모델이 복잡한 신경망보다 강한 경우가 반복된다.
4. **정상성은 고정되지 않는다.** 학습에 없던 정상 운전상태가 오경보를 만들 수 있다.
5. **Conformal p-value의 가정을 공개해야 한다.** 중첩 window는 교환가능하지 않으므로 burst/block audit가 필요하다.
6. **PCA contribution은 탐지 원인 후보이지 실제 고장 원인의 증명이 아니다.** smearing을 보완할 교차검증이 필요하다.

---

## 2. 논문별 적용 매트릭스

### P01. Kim et al. (2022), Towards a Rigorous Evaluation of Time-Series Anomaly Detection

- 출처: AAAI 2022, DOI 10.1609/aaai.v36i7.20680
- 원문: https://ojs.aaai.org/index.php/AAAI/article/view/20680
- 주요 결과: point adjustment가 탐지 성능을 크게 과대평가하며, 무작위 anomaly score도 높은 성능을 얻을 수 있음을 이론·실험으로 보였다.
- Plan C 반영:
  - point-adjusted F1 금지
  - raw window F1, AP, alarm range 지표 병기
  - 학습 없는 B0를 모든 모델이 넘어야 하는 기준선으로 사용

### P02. Tatbul et al. (2018), Precision and Recall for Time Series

- 출처: NeurIPS 2018
- 원문: https://proceedings.neurips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html
- 주요 결과: 시간범위 이상은 point precision/recall만으로 존재 여부, 중첩 범위, 검출 위치, 중복 검출을 표현할 수 없다.
- Plan C 반영:
  - window와 alarm range를 분리
  - 조기 검출을 강조할 경우 front-end bias를 민감도 분석으로만 사용
  - 하나의 실제 이상 range를 여러 burst로 잘라 사건 수를 부풀리지 않음

### P03. Liu and Paparrizos (2024), The Elephant in the Room

- 출처: NeurIPS 2024 Datasets and Benchmarks Track
- 원문: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html
- 주요 결과: 데이터 결함, 편향된 평가, 불일치한 벤치마킹이 TSAD의 핵심 문제이며 VUS-PR을 신뢰도 높은 지표로 제안한다. 단순 통계모델이 고급 신경망보다 우수한 경우도 확인한다.
- Plan C 반영:
  - Dynamic PCA/MSPC를 주력 후보로 격상
  - 가능하면 VUS-PR을 보조 지표로 구현
  - 모든 후보에 동일 window, threshold, metric protocol 사용

### P04. Ku, Storer, and Georgakis (1995), Disturbance Detection and Isolation by Dynamic PCA

- 출처: Chemometrics and Intelligent Laboratory Systems 30(1), 179–196
- DOI: <https://doi.org/10.1016/0169-7439(95)00076-3>
- 주요 결과: lagged measurement를 확장 데이터 행렬에 넣어 공정 동학을 PCA에 포함하고 T²와 Q/SPE로 disturbance를 감시한다.
- Plan C 반영:
  - 3채널의 lag 1/2/4 Dynamic PCA 구현
  - T²와 SPE를 각각 보고
  - raw LSTM-AE와 동일 조건 비교

### P05. MacGregor and Kourti (1995), Process Analysis, Monitoring and Diagnosis Using Multivariate Projection Methods

- 출처: Chemometrics and Intelligent Laboratory Systems 28(1), 3–21
- DOI: <https://doi.org/10.1016/0169-7439(95)80036-9>
- 주요 결과: PCA/PLS 기반 다변량 공정 모니터링은 여러 공정변수의 상관구조를 사용해 이상 감지와 진단을 연결한다.
- Plan C 반영:
  - 제조현장 표준 baseline으로 PCA-MSPC 사용
  - 단변량 threshold와 다변량 T²/SPE를 함께 비교

### P06. Liu, Ting, and Zhou (2008), Isolation Forest

- 출처: IEEE ICDM 2008
- 원문: https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
- 주요 결과: 이상치는 random partition에서 더 짧은 경로로 격리되며, iForest는 낮은 메모리와 거의 선형 계산비용으로 동작한다.
- Plan C 반영:
  - 특징 기반 비선형 비교군
  - `n_estimators`, `max_samples`만 제한적으로 탐색
  - contamination으로 test prevalence를 주입하지 않고 정상 calibration으로 threshold 설정

### P07. Malhotra et al. (2016), LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection

- 출처: arXiv:1607.00148, 원저자 원문
- 원문: https://arxiv.org/abs/1607.00148
- 주요 결과: 정상 multi-sensor sequence를 재구성하도록 LSTM encoder-decoder를 학습하고 reconstruction error로 이상을 탐지한다. 외부 부하·미관측 요인으로 예측이 어려운 경우 재구성 접근이 유용할 수 있다.
- Plan C 반영:
  - 공식 가이드북 LSTM-AE의 문헌적 근거
  - 정상-only training 유지
  - 전체 window/channel reconstruction error 사용
  - 통계모델보다 worst-block FAR가 나쁘면 최종모델로 선택하지 않음

### P08. Kim, Park, and Choo (2024), When Model Meets New Normals

- 출처: AAAI 2024, DOI 10.1609/aaai.v38i12.29210
- 원문: https://ojs.aaai.org/index.php/AAAI/article/view/29210
- 주요 결과: 정상분포가 시간에 따라 변하는 new normal 문제가 비지도 TSAD의 성능을 저하시킨다. test-time adaptation으로 새 정상에 적응하는 방식을 제안한다.
- Plan C 반영:
  - 정상 16,000~17,999행 저부하 상태를 핵심 FP 조건으로 분석
  - 자동 적응 대신 Yellow-S quarantine으로 보수적 운용
  - leave-one-block-out으로 unseen normal robustness 측정

### P09. Kim et al. (2026), CANDI

- 출처: AAAI 2026, DOI 10.1609/aaai.v40i17.38524
- 원문: https://ojs.aaai.org/index.php/AAAI/article/view/38524
- 주요 결과: 분포 이동 아래에서 잠재적 false positive만 선별해 적응하는 curated test-time adaptation을 제안한다.
- Plan C 반영:
  - 새 정상 후보를 무조건 재학습하지 않고 격리·검토 후 정상 registry에 편입
  - 현재 데이터와 마감에서는 CANDI 자체를 구현하지 않고 향후 운영안으로만 인용

### P10. Ishimtsev et al. (2017), Conformal k-NN Anomaly Detector for Univariate Data Streams

- 출처: PMLR 60, 213–227
- 원문: https://proceedings.mlr.press/v60/ishimtsev17a.html
- 주요 결과: conformal paradigm으로 anomaly score를 확률적 abnormality score로 변환하고 비정상성에 적응하는 경량 detector를 제안한다.
- Plan C 반영:
  - 모델별 raw score를 정상성 p-value로 변환
  - 모델 점수 scale 차이를 p-value 공간에서 비교

### P11. Zaffran et al. (2022), Adaptive Conformal Predictions for Time Series

- 출처: ICML 2022, PMLR 162
- 원문: https://proceedings.mlr.press/v162/zaffran22a.html
- 주요 결과: 일반 conformal의 exchangeability 가정은 시계열에 맞지 않으며, adaptive conformal을 종속·분포이동 시계열에 적용하는 방법을 연구한다.
- Plan C 반영:
  - window-level p-value에 무조건적 보장 문구 금지
  - block/burst calibration coverage audit
  - 향후 온라인 운영에서 adaptive alpha 업데이트 검토

### P12. Xu and Xie (2023), Sequential Predictive Conformal Inference for Time Series

- 출처: ICML 2023, PMLR 202
- 원문: https://proceedings.mlr.press/v202/xu23r.html
- 주요 결과: 시계열의 비교환성을 명시적으로 다루고 nonconformity score의 조건부 quantile을 순차적으로 재추정한다.
- Plan C 반영:
  - static conformal과 block-conditioned empirical calibration 비교
  - 현재 데이터에서는 복잡한 SPCI를 핵심모델로 구현하지 않음

### P13. Rebjock et al. (2021), Online False Discovery Rate Control for Anomaly Detection in Time Series

- 출처: NeurIPS 2021
- 원문: https://proceedings.neurips.cc/paper/2021/hash/def130d0b67eb38b7a8f4e7121ed432c-Abstract.html
- 주요 결과: serial dependence와 희귀 anomaly가 있는 online stream에서 FDR을 제어해 목표 precision을 관리하는 규칙을 제시한다.
- Plan C 반영:
  - 교대당 오경보 예산의 학술 근거
  - 현재 짧은 기록에서는 구현보다 운영 확장안으로 제시

### P14. Alcala and Qin (2009), Reconstruction-based Contribution for Process Monitoring

- 출처: Automatica
- 원문: https://www.sciencedirect.com/science/article/pii/S0005109809001277
- 주요 결과: 전통 contribution plot은 fault smearing과 오진 위험이 있으며 reconstruction-based contribution으로 진단을 개선한다.
- Plan C 반영:
  - 단순 squared contribution만으로 원인 선언 금지
  - counterfactual channel repair를 설명 audit에 포함

### P15. Van den Kerkhof et al. (2013), Analysis of Smearing-out in Contribution Plot Based Fault Isolation

- 출처: Chemical Engineering Science
- 원문: https://www.sciencedirect.com/science/article/pii/S0009250913005502
- 주요 결과: 상관된 변수에서 contribution smearing이 비고장 변수까지 퍼질 수 있으며 복합 fault의 정확한 isolation을 보장하지 못한다.
- Plan C 반영:
  - contribution, group ablation, channel repair의 3중 확인
  - “기여 채널”과 “물리적 고장 원인”을 구분

### P16. Rebjock et al. 및 range-aware 연구에서 가져오지 않는 것

- 현재 정상 관측은 약 0.56시간에 불과해 장기 온라인 FDR 성능을 검증할 수 없다.
- 실제 고장 시작 전 기록이 없어 조기예측 benchmark를 만들 수 없다.
- 따라서 online FDR과 adaptive conformal은 향후 확장으로만 두고 현재 성능표를 장식하지 않는다.

---

## 3. 문헌과 이 데이터가 충돌하는 지점

| 문헌 가정 | 현재 데이터 | 처리 |
|---|---|---|
| 여러 독립 이상 사건 | 독립 fault record 1건 | 사건 일반화 주장 금지 |
| 연속 시계열 | 최대 5초 burst + 긴 공백 | burst-safe window |
| 교환가능 calibration sample | 90% 중첩 window와 regime shift | burst/block audit |
| 고장 원인 라벨 | 설비 상태 0/1만 있음 | 기여 특징까지만 설명 |
| 충분한 온라인 기간 | 정상 실측 약 0.56시간 | FAR 상한과 추가수집 제안 |
| test-time adaptation 검증 가능 | 고장/새 정상 구별 라벨 부족 | 자동 적응 금지, quarantine |

---

## 4. 보고서 인용 원칙

- 논문이 보인 결과와 우리 데이터의 결과를 한 문장에 섞지 않는다.
- 논문은 방법 선택과 위험 통제의 근거로 사용한다.
- “문헌상 우수”가 아니라 “우리 동일 조건 실험에서 통과”한 모델만 최종 선택한다.
- arXiv 논문은 arXiv라고 표기하고 peer-reviewed 논문과 구분한다.
- DOI 또는 학회 공식 원문 링크를 참고문헌에 적는다.
- 논문에서 직접 검증하지 않은 제조 원인을 논문 권위로 확정하지 않는다.

---

## 5. 가장 강한 차별화 문장

> 최신 TSAD 연구가 경고한 point adjustment, new normal, non-exchangeable calibration 문제를 제공 데이터에서 직접 확인하고, 이를 burst-safe 평가, shift/fault 이중 점수, block/burst calibration audit로 구현했다.

이 문장은 다음 네 결과가 모두 있을 때만 사용한다.

1. point-adjustment 없는 모델표
2. 정상 block별 오경보
3. sensor stress 결과
4. C0/C1/C2 calibration coverage 비교

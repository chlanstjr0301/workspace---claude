# 전체 원자료 사용 이력 감사

결론: 확인된 미사용 고유 관측은 **0행**, 따라서 남은 미사용 연속 구간도 **0개**다. 원본20600행 중 고유 관측20599행(정상19999·이상600)은 실제 적합 기록 또는 저장된 점수/예측으로 사용을 확인했다. 정상 추가중복1행은 따로 분류했으며 새 검증 자료로 인정하지 않는다.

## 무엇을 대조했는가

현재 PCA 프로젝트에 남아 있는 12개 실행/재실행/사전준비 폴더, 886개 사용 근거 파일, 6개 ZIP의 실제 분할 관련 멤버를 조사했다. 최초 PCA 실험의 학습 행 명세와 실제 모델·fit 로그를 연결하고, 저장 NPZ의 유한 점수가 있는 행만 계산된 행으로 인정했다. split에 배정됐다는 사실만으로 사용으로 간주하지 않았다.

다른 에이전트 프로젝트는 공유된 고정 분할표의 해시만 읽기 전용으로 대조했다. 그 에이전트의 새 실행 폴더·모델 선택·오류 사례는 읽지 않았다. 해당 프로젝트의 모든 실험별 상세 이력을 검증했다는 뜻은 아니다. **최초 PCA 실행 하나만으로도 모든 고유 원관측의 실제 사용이 이미 증명되므로**, 다른 실험의 미확인 이력이 남아도 이 원자료에 미사용 고유행이 생기지는 않는다.

## 역할별 원래 분할

역할별 전체 행·원행 범위·수집 시각은 original_role_ranges.csv에 있다. 정상 학습12012행, 정상calibration2011행, 정상selection1986행, 정상과거평가3990행이다. 이상은 calibration132행·selection111행·과거평가357행이다. 정상 학습 원행 범위1–12013 안의11566행은 중복 제거됐으므로 행 범위 길이와 실제 행 개수는 다르다.

특히 이상 calibration132행은 최신 R5 정상 임계값 보정에는 쓰이지 않았지만, 최초 PCA에서 점수 계산과 **라벨을 사용하는 sigmoid 보정**에 사용됐다. 최신 모델만 보면서 이132행을 미사용으로 재분류하면 안 된다. 근거: runs/pca_20261003_cpu/source_final/experiment.py:325–327, models/sigmoid.joblib, models/sigmoid.json, predictions/all_operational_calibration.csv.gz, predictions/probability_selection.csv.

정상 학습은 manifests/train_P0_W1_K2.csv의12012행과 logs/fit_P0_W1_K2.json·실제 저장 모델로 확인했다. calibration/selection/H는 원본 clean_rows.csv에 정렬된 raw_P0_W1_K2_develop.npz 및 raw_P0_W1_K2_test.npz의 실제 유한 점수로 확인했다. P0는 단일 행 센서 입력이라 윈도 warm-up 제외가 없다.

## 겹치는 입력 창도 사용으로 추적

저장된 start_row_id–끝행, 실제 window_start 및 R5 lag/확인 상태의 의존 범위를 원행 ID로 펼쳤다. 334945개의 중복 제거한 입력 범위 기록이 input_window_footprints.csv.gz에 있다. 사용한 원행 ID 전체를 범위별로 기록하여, 중간의 삭제된 중복11566행을 숫자 범위만 보고 잘못 포함하지 않았다.

최초 PCA의 1/5/10/20행·버스트 고려/미고려 창은 저장된 feature manifest와 실제 학습/점수 파일을 연결했다. 이후 실행은 저장 window_start와 R5의 t−2…t, G1의 최대 t−3…t를 같은 파일·원분할·gap 경계 안에서 추적했다. 과거 행을 썼으면 현재 평가 끝행이 아니어도 사용이다. S1/S2/V 같은 보고 경계를 넘는 과거 입력 역시 사용 이력에 포함한다. 이번 자료는 P0 적합/점수 근거만으로도 모든 고유행의 직접 사용이 확인되므로, 입력 이력을 추가한다고 미사용행이 새로 생기거나 숨겨지지 않는다.

## 중복1행의 정확한 처리

press_data_normal.csv 원행11566(CSV 헤더 포함11567줄), 2022-07-12 00:43:33.196, 정상1행. 원행11565(CSV11566줄)와 TimeStamp·센서3개·Equipment_state가 모두 동일하다. 보존된 파이프라인에서는 추가중복을 제거하고11565를 사용했다. 삭제된11566 자체를 모델 입력에 썼다고 주장하지 않는다. **이미 사용한 관측의 중복**으로 분류하며 미사용 독립 관측이나 새로운 시간 구간이 아니다. 번호 열의 차이는 모델 입력이 아니다.

## 확인 불가를 처리하는 원칙

사용 근거가 없는 행은 unused가 아닌 usage_unverifiable(확인 불가)로 남기도록 구현했다. 이번 원자료의 고유행에서는 그 수가 0행이다. 실험별 특정 모델의 전체 적합·선택 이력까지 완전하다는 의미는 아니다. 부분적인 모델 계보·ID 불일치 등은 unresolved_artifact_lineage.json에 별도로 남겼다. 사전준비 폴더의 분할표만으로는 실제 실험 사용으로 세지 않았다. 저장되지 않은 외부 실험의 부재를 증명하지 않으며, 그것이 이미 확인된 사용을 취소하지도 않는다.

## 산출물 및 금지한 실행

- row_usage_ledger.csv: 원본20600행의 사용 상태·역할·증거 경로·중복 대응
- contiguous_usage_ranges.csv: 원파일 원행 연속 범위와 정상/이상 개수·수집 시각
- unused_or_uncertain_or_duplicate_ranges.csv: 미사용/확인 불가/중복 별도 목록. 현재는 중복1행만 존재
- evidence_files.csv: 사용 근거 파일의 SHA-256 및 확인 행 수
- split_manifest_inventory.csv, experiment_coverage.csv: 실험별 분할표·사용 기록 대조
- input_window_footprints.csv.gz: 실제 겹친 입력 창의 원행 ID 목록
- archive_inventory.json, shared_split_check.json: ZIP 내부와 공통 분할 해시

모델을 불러오거나 적합·추론·임계값 선택·성능 실험을 실행하지 않았다. CSV/NPZ/로그/소스/ZIP 읽기와 이력 표 작성만 수행했다. 원본과 기존 실험 결과는 변경하지 않았다.

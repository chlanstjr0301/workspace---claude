# PCA + R5 예측잔차 보조 경보(G1): 고정 검증 패키지

`review_report_ko.html`을 먼저 여세요. 데이터·모델·실행 코드·예측·그림·과거 근거 사본이 포함돼 있습니다. 기존 프로젝트 경로는 출처 기록일 뿐 재실행 때 읽지 않습니다. 원래 모델을 다시 학습하거나 선정하지 않습니다.

## 고정 검증 재실행

Python 3.14 환경에서 압축을 풀고:

```bash
bash RUN_REVIEW.sh /원하는/새로운_결과폴더
```

기존 출력 폴더를 덮어쓰지 않습니다. 패키지 안 `.venv`가 없으면 `python3.14 -m venv` 후 `requirements.lock.txt`를 설치합니다. Python 3.14가 없다면 별도로 준비해야 합니다. 시스템 Python을 교체하지 않습니다.

이미 동일 라이브러리 환경이 있다면:

```bash
PCA_PYTHON=/해당/가상환경/bin/python bash RUN_REVIEW.sh /새로운/결과폴더
```

CPU 라이브러리 스레드2개를 사용합니다. 네트워크가 필요한 부분은 패키지 설치뿐입니다. 원본으로 되돌아가는 절대경로·모델 재학습은 없습니다. `src/register_validation.py`는 최초 등록의 출처이며 재실행에서 호출하지 않습니다. 과거 검증151개는 동일 해시 결과를 재사용하고 새 고정 대조군·외부 입력 계약을 검증합니다.

## 새 자료 평가: 재학습·재보정 금지

센서 CSV 필수 열:

- `row_id`: 파일 전체에서 유일한 원행 ID
- `session_id`: 실제 획득 세션 식별자. 보고용 블록을 세션으로 만들지 않습니다. 같은 세션 행은 연속 배치합니다.
- `TimeStamp`: 세션 내부 엄격 증가 타임스탬프
- `AI0_Vibration`, `AI1_Vibration`, `AI2_Current`: 센서 숫자. 이름·순서를 유지하며 물리 의미/단위와 기존 장비 호환성을 별도로 확인합니다.

누락·무한대·중복 타임스탬프는 오류로 중단합니다. 원행 삭제나 보간을 하지 않습니다. 음의 전류와 큰 센서값은 유지합니다. 센서 파일에 `label`/`Equipment_state`를 넣지 마세요. 라벨 파일은 별도로 `row_id,label` 두 열, 이상=1·정상=0이며 센서 CSV와 원행이 정확히 같아야 합니다.

원자료 출처 JSON 예시(값을 실제로 확인하여 입력):

```json
{
  "independently_collected": false,
  "sensor_semantics_compatible": false,
  "prior_usage": "unknown",
  "equipment_id": "확인할 실제 설비 ID",
  "acquisition_dates": ["실제 수집일"],
  "units": "각 센서의 확인된 단위와 원 입력과의 호환성 근거",
  "note": "이 값들은 예시이며 실제 확인 전 true로 바꾸지 않음"
}
```

기존 원 자료의 물리 단위 자체가 미확인입니다. 새 자료 호환성은 자동 입증되지 않으며 제공자의 출처 확인이 필요합니다. 독립 여부를 JSON에 적었다는 이유만으로 과학적으로 증명된 것은 아닙니다. 재사용 H smoke test에는 명시적으로 `independently_collected=false`를 기록했습니다.

```bash
/가상환경/bin/python src/evaluate_new_data.py \
  --sensors /새자료/sensors.csv \
  --provenance /새자료/provenance.json \
  --labels /새자료/labels.csv \
  --output /존재하지_않는/평가폴더
```

라벨이 없으면 `--labels`를 생략하세요. 예측과 해시를 먼저 저장한 뒤에만 라벨을 엽니다. 라벨 없이는 FN/FP 검증을 완료하지 않습니다. 추가 시간 블록의 필수 조건은 자료를 보기 전에 별도 계획으로 고정해야 하며, 단순 파일 실행만으로 미등록 시간 일반화 조건을 통과했다고 판정하지 않습니다. 정상만 있으면 정상 오경보만 평가합니다. FN0→0은 유지이지 개선 재현이 아닙니다.

FP 제한은 추가FP≤floor(0.001×정상행수), 절대FP≤floor(0.01×정상행수). 작은 표본에 최소1개를 허용하지 않습니다. 원 B0 경보는 삭제하지 않습니다. 결과가 나빠도 파라미터를 바꾸지 않습니다.

## 주요 근거

- `validation_protocol.json`, `model_manifest.json`: 정확한 임계값·판정·행 집합·모델 해시
- `metrics.csv`, `mandatory_conditions.csv`: pooled/블록 판정과 원래 FP 예산
- `control_effects.csv`: 임계값 변경과 연속 확인 효과
- `predictions/`, `paired_changes.csv`: 전체 원행 판정과 바뀐 원행
- `events.csv`, `row_timing_and_residuals.csv`: 임시 관측 사건·입력 시각·경보 가능 시각
- `normal_exposure.csv`: 공백을 뺀 관측 시간과 오경보 episode
- `validation_report.json`, `prior_validation_reuse.json`: 새 검증과 기존 검증 재사용 구분
- `inputs/previous/`: 과거 설정·실행 코드·검증 기록 사본
- `data/`: 원 CSV의 변경 없는 사본. 데이터 재배포 권한은 이 분석으로 새로 부여되지 않습니다.

새로운 고장15건을 발견했다는 결과가 아닙니다. 독립 새 자료 성능 검증은 미실시입니다.

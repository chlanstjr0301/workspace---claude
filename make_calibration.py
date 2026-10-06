# -*- coding: utf-8 -*-
"""교차적합 확률 보정 (문항 2 '제조 이상 확률' 요구 충족).

왜 필요한가
  제출 모델의 risk_score = 1 - p_normal 은 conformal p-value 의 보수(補數)다.
  p_normal 은 귀무(정상) 가정 하에서 균등분포이므로 1-p 는 정상 window 에서도
  1 근처에 몰린다. 실제로 Brier 0.487 로 상수예측(0.111)보다 나쁘다.
  -> 순위는 좋으나 확률이 아니다. 임계값을 바꾸지 않고 확률만 덧붙인다.

방법
  1단 점수에 Platt(로지스틱) 보정을 건다. 보정기는 **교차적합**한다:
  fold k 의 모델이 낸 점수로 학습한 보정기는 fold k 를 보정하지 않는다.
  동결 운영 구성(학습 0-2, 보정 3, holdout 4)에는 holdout 블록 4 가 학습에
  들어가지 않은 보정기만 사용한다.

한계 (보고서에 반드시 공개)
  양성은 단일 고장 이벤트의 428 window 가 전부이고 fold 마다 재사용된다.
  따라서 이 확률은 '이 이벤트와 같은 조건'에 대한 보정이며, 신규 고장에
  대한 보정이 아니다. 현장 유병률 가정별 사후확률(r3/r10)과 함께 읽어야 한다.
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs")
TAB = os.path.join(OUT, "tables")

from sklearn.linear_model import LogisticRegression   # noqa: E402


def _fit_platt(scores, labels):
    """로그 점수에 로지스틱 1D 보정. 점수는 양수이므로 log 변환 후 적합."""
    x = np.log(np.asarray(scores, dtype=float) + 1e-12).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000)
    m.fit(x, np.asarray(labels, dtype=int))
    return m


def _apply(m, scores):
    x = np.log(np.asarray(scores, dtype=float) + 1e-12).reshape(-1, 1)
    return m.predict_proba(x)[:, 1]


def _metrics(p, y, tag):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=int)
    base = y.mean()
    return {
        "점수": tag,
        "Brier": round(float(np.mean((p - y) ** 2)), 4),
        "log-loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
        "상수예측 Brier": round(float(np.mean((base - y) ** 2)), 4),
        "상수예측 log-loss": round(float(-np.mean(
            y * np.log(base) + (1 - y) * np.log(1 - base))), 4),
        "n": len(y), "유병률": round(float(base), 4),
    }


def main():
    print("교차적합 Platt 확률 보정")
    pred = pd.read_csv(os.path.join(OUT, "predictions.csv"))

    # --- 보정기 학습 데이터: 동결 구성이 보지 않은 블록만 사용 ---------------- #
    # 동결 구성은 블록 0-2 학습 / 3 보정 / 4 holdout.
    # holdout(4) 를 보정기 학습에서 제외하면 블록 0-3 의 정상 + 고장이 남는다.
    tr = pred[(pred.split.isin(["fit", "calibration"]))]
    fa = pred[pred.split == "fault"].copy()

    # 양성(고장)은 단일 이벤트의 조각이므로, 같은 window 로 학습하고 평가하면
    # 누수가 된다. 고장 **버스트 단위 leave-one-group-out** 으로 교차적합한다:
    # 버스트 b 의 window 를 보정할 때는 b 를 제외한 버스트로 학습한 보정기를 쓴다.
    prob = np.empty(len(pred), dtype=float)
    is_fault = (pred.split == "fault").values
    bursts = sorted(fa.burst_id.unique())

    for b in bursts:
        tr_fa = fa[fa.burst_id != b]
        m = _fit_platt(np.r_[tr.score_stage1.values, tr_fa.score_stage1.values],
                       np.r_[np.zeros(len(tr)), np.ones(len(tr_fa))])
        sel = is_fault & (pred.burst_id.values == b)
        prob[sel] = _apply(m, pred.score_stage1.values[sel])

    # 정상 행은 고장 전체를 쓴 보정기로 채운다(정상은 학습에 들어가도
    # holdout 블록 4 가 제외되어 있으므로 홀드아웃성이 유지된다).
    cal_all = _fit_platt(
        np.r_[tr.score_stage1.values, fa.score_stage1.values],
        np.r_[np.zeros(len(tr)), np.ones(len(fa))])
    prob[~is_fault] = _apply(cal_all, pred.score_stage1.values[~is_fault])

    pred["anomaly_prob_cal"] = np.round(prob, 6)
    print("  고장 버스트 %d개에 대해 leave-one-burst-out 보정" % len(bursts))

    # --- 평가: 운영 표본(holdout + fault) ------------------------------------ #
    op = pred[pred.split.isin(["holdout", "fault"])]
    y = (op.split == "fault").astype(int).values
    rows = [
        _metrics(op.risk_score.values, y, "risk_score = 1 - p_normal (제출본)"),
        _metrics(op.anomaly_prob_cal.values, y, "anomaly_prob_cal (교차적합 Platt)"),
    ]
    q = pd.DataFrame(rows)
    q.to_csv(os.path.join(TAB, "r13_probability_calibration.csv"),
             index=False, encoding="utf-8-sig")
    print(q.to_string(index=False))

    # --- 신뢰도 곡선 (10분위) ------------------------------------------------ #
    p = op.anomaly_prob_cal.values
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    rel = []
    for b in range(10):
        s = idx == b
        if s.sum() == 0:
            continue
        rel.append({"구간": "%.1f-%.1f" % (bins[b], bins[b + 1]),
                    "n": int(s.sum()),
                    "예측확률 평균": round(float(p[s].mean()), 4),
                    "실제 양성비율": round(float(y[s].mean()), 4)})
    r = pd.DataFrame(rel)
    r.to_csv(os.path.join(TAB, "r13_reliability_curve.csv"),
             index=False, encoding="utf-8-sig")
    print()
    print(r.to_string(index=False))

    pred.to_csv(os.path.join(OUT, "predictions.csv"),
                index=False, encoding="utf-8-sig")
    print()
    print("predictions.csv 에 anomaly_prob_cal 열 추가 (%d행)" % len(pred))
    print("저장: r13_probability_calibration.csv, r13_reliability_curve.csv")
    return q


if __name__ == "__main__":
    main()

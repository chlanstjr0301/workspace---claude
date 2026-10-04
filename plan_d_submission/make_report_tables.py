# -*- coding: utf-8 -*-
"""보고서 보충표 생성 (모델·임계값 불변, 기존 산출물만 사용).

감사 지적 대응:
  D-P1-01 -> 원 게이트 vs 개정 게이트 선정 결과 비교
  D-P1-05 -> 운영 성능표 (CV 평균 vs 예측파일 holdout)
  D-P1-06 -> 사후확률 민감도표
  D-P2-04 -> 빨강 기준 FN 분해
  D-P1-03 -> 1단 기여특징의 채널 비중
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
TAB = os.path.join(HERE, "outputs", "tables")

from src import calibration as cal   # noqa: E402


def save(df, name):
    p = os.path.join(TAB, name)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print("  -> %s (%d행)" % (name, len(df)))


# --------------------------------------------------------------------------- #
# R1. 원 게이트 vs 개정 게이트  (D-P1-01)
# --------------------------------------------------------------------------- #
def r1_gate_comparison():
    g = pd.read_csv(os.path.join(TAB, "e2_selection_gates.csv"))
    bl0_fp = float(g.loc[g.model == "BL0", "fp_rate_worst"].iloc[0])

    rows = []
    for _, r in g.iterrows():
        # 원안: gate2 = 이상 Recall 하락만, gate3 = 엄격(허용오차 0), gate5 없음
        o2 = (not np.isfinite(r["worst_recall_drop_pp"])) or \
             r["worst_recall_drop_pp"] <= 20.0
        o3 = (r["model"] == "BL0") or (r["fp_rate_worst"] <= bl0_fp)
        o4 = r["bursts_detected_min"] >= 1
        orig = bool(o2 and o3 and o4)
        rows.append({
            "모델": r["model"],
            "F1(CV평균)": round(r["f1_mean"], 4),
            "최악블록 FP율": round(r["fp_rate_worst"], 4),
            "섭동 Recall하락(pp)": r["worst_recall_drop_pp"],
            "섭동 오경보증가(pp)": r["worst_shift_fp_increase_pp"],
            "교정오차(a=0.10)": r["coverage_err_at_0.10"],
            "원 게이트 통과": orig,
            "개정 게이트 통과": bool(r["passed"]),
        })
    df = pd.DataFrame(rows)
    save(df, "r1_gate_before_after.csv")
    print("     원 게이트 통과:", list(df[df["원 게이트 통과"]]["모델"]))
    print("     개정 게이트 통과:", list(df[df["개정 게이트 통과"]]["모델"]))
    return df


# --------------------------------------------------------------------------- #
# R2. 운영 성능표  (D-P1-05)
# --------------------------------------------------------------------------- #
def _metrics(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=p, recall=r,
                f1=2 * p * r / max(p + r, 1e-12), fpr=fp / max(fp + tn, 1))


def r2_operational():
    pr = pd.read_csv(os.path.join(TAB, "..", "predictions.csv"))
    op = pr[pr.split.isin(["holdout", "fault"])]
    y = (op.split == "fault").astype(int).values
    rows = []
    for name, pred in [
        ("1단 경보(노랑+빨강)", (op.alarm_level != "green").astype(int).values),
        ("최종 빨강", (op.alarm_level == "red").astype(int).values)]:
        m = _metrics(y, pred)
        rows.append({"기준": name, "단위": "window",
                     "표본": "holdout 2,929 + fault 428",
                     "유병률": round(float(y.mean()), 4), **m})
    e2 = pd.read_csv(os.path.join(TAB, "e2_model_comparison.csv")).set_index("model")
    for m in ["BL0", "M3", "BL1"]:
        if m not in e2.index:          # BL-1 은 full 실행 또는 동봉 CSV 가 있을 때만
            print("     (%s 결과 없음 - 행 생략)" % m)
            continue
        rows.append({"기준": "%s (CV 5-fold 평균)" % m, "단위": "window",
                     "표본": "fold별 정상 holdout + fault 428",
                     "유병률": 0.1203,
                     "TP": np.nan, "FP": np.nan, "FN": np.nan, "TN": np.nan,
                     "precision": round(e2.loc[m, "precision_mean"], 4),
                     "recall": round(e2.loc[m, "recall_mean"], 4),
                     "f1": round(e2.loc[m, "f1_mean"], 4),
                     "fpr": round(e2.loc[m, "fp_rate_mean"], 4)})
    df = pd.DataFrame(rows)
    save(df, "r2_operational_performance.csv")
    return df


# --------------------------------------------------------------------------- #
# R3. 사후확률 민감도  (D-P1-06)
# --------------------------------------------------------------------------- #
def r3_posterior():
    pr = pd.read_csv(os.path.join(TAB, "..", "predictions.csv"))
    op = pr[pr.split.isin(["holdout", "fault"])]
    y = (op.split == "fault").astype(int).values
    red = (op.alarm_level == "red").astype(int).values
    m = _metrics(y, red)
    rows = []
    for r in cal.posterior_sensitivity(m["recall"], m["fpr"],
                                       (0.01, 0.001, 0.0001)):
        rows.append({"가정 고장유병률": r["prevalence"],
                     "빨강 TPR": round(r["tpr"], 4),
                     "빨강 FPR": round(r["fpr"], 6),
                     "P(고장|빨강)": round(r["posterior"], 4)})
    df = pd.DataFrame(rows)
    save(df, "r3_posterior_sensitivity.csv")

    # risk_score 보정 품질 (공개용)
    risk = op.risk_score.values
    cons = y.mean()
    q = pd.DataFrame([{
        "지표": "Brier", "risk_score": round(float(np.mean((risk - y) ** 2)), 4),
        "상수예측(기저율)": round(float(np.mean((cons - y) ** 2)), 4)},
        {"지표": "log-loss",
         "risk_score": round(float(-np.mean(
             y * np.log(risk + 1e-12) + (1 - y) * np.log(1 - risk + 1e-12))), 3),
         "상수예측(기저율)": round(float(-np.mean(
             y * np.log(cons) + (1 - y) * np.log(1 - cons))), 3)}])
    save(q, "r3_risk_score_calibration.csv")
    return df, q


# --------------------------------------------------------------------------- #
# R4. 빨강 기준 FN 분해  (D-P2-04)
# --------------------------------------------------------------------------- #
def r4_red_fn():
    pr = pd.read_csv(os.path.join(TAB, "..", "predictions.csv"))
    fa = pr[pr.split == "fault"].reset_index(drop=True)
    fa["pos_in_burst"] = fa.groupby("burst_id").cumcount()
    fn = fa[fa.alarm_level != "red"]
    by_pos = (fn.assign(위치=np.where(fn.pos_in_burst < 2, "버스트 첫 2 window",
                                     "그 이후"))
              .groupby("위치").size().reset_index(name="미탐 window"))
    by_pos["전체 미탐 대비"] = (by_pos["미탐 window"] / len(fn)).round(4)
    save(by_pos, "r4_red_fn_by_position.csv")

    by_burst = (fn.groupby("burst_id").size().reset_index(name="미탐 window")
                .merge(fa.groupby("burst_id").size().reset_index(name="burst window수"),
                       on="burst_id"))
    by_burst["미탐률"] = (by_burst["미탐 window"] /
                        by_burst["burst window수"]).round(3)
    save(by_burst.sort_values("미탐 window", ascending=False),
         "r4_red_fn_by_burst.csv")
    print("     빨강 미탐 %d개 중 버스트 첫 2 window: %d개"
          % (len(fn), int(by_pos.loc[by_pos["위치"] == "버스트 첫 2 window",
                                     "미탐 window"].sum())))
    return by_pos, by_burst


# --------------------------------------------------------------------------- #
# R5. 1단 기여특징 채널 비중  (D-P1-03)
# --------------------------------------------------------------------------- #
def r5_channel_share():
    pr = pd.read_csv(os.path.join(TAB, "..", "predictions.csv"))
    rows = []
    for name, sel in (("고장 기록", pr.split == "fault"),
                      ("정상 holdout", pr.split == "holdout")):
        s = pr[sel]
        cur = s.top_reason_1.str.contains("CUR", na=False).sum()
        rows.append({"대상": name, "window 수": len(s),
                     "1순위 기여가 전류 채널": int(cur),
                     "비중": round(cur / max(len(s), 1), 4)})
    df = pd.DataFrame(rows)
    save(df, "r5_stage1_channel_share.csv")
    return df


def main():
    print("보고서 보충표 생성")
    r1_gate_comparison()
    r2_operational()
    r3_posterior()
    r4_red_fn()
    r5_channel_share()
    print("완료")


if __name__ == "__main__":
    main()

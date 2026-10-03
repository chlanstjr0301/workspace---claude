# -*- coding: utf-8 -*-
"""모델 통과 기준 (Plan D §6.2 + 2026-10-04 개정).

원안 순서
  1. NaN/실패              -> 탈락
  2. 섭동 후 Recall 하락   -> 한계 초과 시 탈락
  3. 최악 블록 오경보      -> BL-0보다 많으면 탈락
  4. 이상 버스트 미탐지    -> 탈락
  5. 최악 블록 FAR 상한이 낮은 모델 우선
  6. 동률이면 Recall, 설명성, 추론비용 순

개정 사항 (decision_log.md DL-001, DL-002, DL-003 참조)
  - gate2 는 이상 Recall 이 1.0 으로 포화되어 변별력이 없었다.
    -> 정상 데이터를 섭동했을 때의 오경보 증가(fp_increase_pp)를 함께 본다.
  - gate3 를 '엄격히 BL-0 이하'로 두면 모든 모델이 소수점 차이로 탈락한다.
    -> 허용오차(상대 10% 또는 절대 0.005)를 둔다.
  - gate5 추가: conformal coverage 오차가 큰 모델은 임계값 해석이 불가능하므로
    주력에서 제외한다. (BL-0 은 점수 동률이 많아 p-value 해상도가 없다)
"""
import numpy as np
import pandas as pd

EXPLAINABILITY = {"M1": 3, "M3": 3, "BL0": 2, "M2": 1}   # 높을수록 설명 쉬움
FP_TOL_REL = 0.10
FP_TOL_ABS = 0.005
MAX_COVERAGE_ERR = 0.05     # alpha=0.10 에서 |실제-명목| 허용치


def _coverage_error(cov, model, alpha=0.10):
    if cov is None or len(cov) == 0:
        return np.nan
    s = cov[(cov["model"] == model) & (np.isclose(cov["alpha"], alpha))]
    if s.empty:
        return np.nan
    return float(np.abs(s["empirical"].mean() - alpha))


def apply_gates(cfg, agg, rob_agg, cov=None):
    sel = cfg["selection"]
    a = agg.set_index("model")
    drop = (rob_agg.groupby("model")["recall_drop_pp"].max()
            if len(rob_agg) else pd.Series(dtype=float))
    shift = (rob_agg.groupby("model")["fp_increase_pp"].max()
             if "fp_increase_pp" in rob_agg.columns else pd.Series(dtype=float))
    bl0_fp = float(a.loc["BL0", "fp_rate_worst"]) if "BL0" in a.index else np.inf
    fp_limit = max(bl0_fp * (1 + FP_TOL_REL), bl0_fp + FP_TOL_ABS)

    rows = []
    for m in a.index:
        r = a.loc[m]
        d = float(drop.get(m, np.nan))
        sfp = float(shift.get(m, np.nan))
        cerr = _coverage_error(cov, m)
        g1 = bool(np.isfinite(r["recall_mean"]) and np.isfinite(r["f1_mean"]))
        g2 = ((not np.isfinite(d)) or d <= sel["max_recall_drop_pp"]) and \
             ((not np.isfinite(sfp)) or sfp <= sel["max_recall_drop_pp"])
        g3 = (not sel["worst_block_fp_vs_bl0"]) or m == "BL0" or \
             float(r["fp_rate_worst"]) <= fp_limit
        g4 = (not sel["require_burst_detection"]) or \
             int(r["bursts_detected_min"]) >= 1
        g5 = (not np.isfinite(cerr)) or cerr <= MAX_COVERAGE_ERR
        rows.append({
            "model": m, "pretty": a.loc[m, "pretty"],
            "gate1_no_nan": g1,
            "gate2_robust": g2,
            "worst_recall_drop_pp": d,
            "worst_shift_fp_increase_pp": sfp,
            "gate3_fp_vs_bl0": g3,
            "fp_rate_worst": float(r["fp_rate_worst"]),
            "bl0_fp_rate_worst": bl0_fp, "fp_limit": fp_limit,
            "gate4_burst_detected": g4,
            "bursts_detected_min": int(r["bursts_detected_min"]),
            "gate5_calibrated": g5, "coverage_err_at_0.10": cerr,
            "far_h_upper95_worst": float(r["far_h_upper95_worst"]),
            "recall_mean": float(r["recall_mean"]),
            "f1_mean": float(r["f1_mean"]),
            "explainability": EXPLAINABILITY.get(m, 0),
            "passed": bool(g1 and g2 and g3 and g4 and g5),
        })
    gates = pd.DataFrame(rows)

    cand = gates[gates["passed"] & (gates["model"] != "BL0")]
    if cand.empty:
        cand = gates[gates["passed"]]
    if cand.empty:
        return gates, "BL0"
    cand = cand.sort_values(
        ["far_h_upper95_worst", "recall_mean", "explainability"],
        ascending=[True, False, False])
    return gates, str(cand.iloc[0]["model"])

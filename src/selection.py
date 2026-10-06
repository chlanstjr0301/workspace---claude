# -*- coding: utf-8 -*-
"""모델 통과 기준 (분석 프로토콜 §6.2 + 개정 이력).

원안(v0) 순서
  1. NaN/실패              -> 탈락
  2. 섭동 후 Recall 하락   -> 한계 초과 시 탈락
  3. 최악 블록 오경보      -> BL-0보다 많으면 탈락 (허용오차 0)
  4. 이상 버스트 미탐지    -> 탈락
  5. 최악 블록 FAR 상한이 낮은 모델 우선
  6. 동률이면 Recall, 설명성 순

개정(v1, 제출본, DL-001~003)
  - gate2 에 정상측 오경보 증가(fp_increase_pp) 추가
  - gate3 에 허용오차(상대 10% 또는 절대 0.005)
  - gate5 신설: conformal coverage 오차

정정(v2, DL-016) — 내부 검증에서 드러난 v1 의 결함을 바로잡는다
  (a) v1 의 _coverage_error 는 fold 별 오차를 '먼저 평균'해 양방향 오차가
      상쇄되었다. M3 의 fold 오차는 -8.5pp~+9.9pp 로 흩어지는데 부호평균은
      0.0133 이 되어 한도 0.05 를 통과했다. -> fold 별 절댓값 평균으로 바꾼다.
  (b) v1 은 운영에 쓰지 않는 alpha=0.10 에서만 쟀다. 시스템의 실제 임계값은
      alpha=0.01 이므로 두 수준을 모두 계산하고, 판정은 운영 alpha 로 한다.
  (c) v1 은 BL-0 를 '다른 모델이 하나라도 통과하면' 후보에서 제외하는 규칙을
      코드에만 두고 보고서에 적지 않았다. -> 제거한다. BL-0 도 동등하게 경쟁한다.
  (d) 1순위 정렬키(far_h_upper95_worst)를 반환값에 함께 실어 공개한다.

apply_gates 는 세 버전의 선정 결과를 모두 반환하므로, 보고서는 어느 규칙이
어느 모델을 뽑는지 나란히 공개할 수 있다.
"""
import numpy as np
import pandas as pd

EXPLAINABILITY = {"M1": 3, "M3": 3, "BL0": 2, "M2": 1}   # 높을수록 설명 쉬움
FP_TOL_REL = 0.10
FP_TOL_ABS = 0.005
MAX_COVERAGE_ERR = 0.05
SORT_KEYS = ["far_h_upper95_worst", "recall_mean", "explainability"]


def _coverage_error(cov, model, alpha, how="abs_mean"):
    """fold 별 |실제-명목| 를 집계한다.

    how='abs_mean' : fold 별 절댓값의 평균 (v2 기본, 상쇄 없음)
    how='signed'   : 평균을 먼저 낸 뒤 절댓값 (v1 제출본. 상쇄가 일어난다)
    how='max'      : fold 별 절댓값의 최댓값 (가장 보수적)
    """
    if cov is None or len(cov) == 0:
        return np.nan
    s = cov[(cov["model"] == model) & (np.isclose(cov["alpha"], alpha))]
    if s.empty:
        return np.nan
    e = s["empirical"].values
    if how == "signed":
        return float(abs(e.mean() - alpha))
    if how == "max":
        return float(np.abs(e - alpha).max())
    return float(np.abs(e - alpha).mean())


def _pick(gates, flag, exclude_bl0):
    cand = gates[gates[flag]]
    if exclude_bl0:
        c2 = cand[cand["model"] != "BL0"]
        cand = c2 if not c2.empty else cand
    if cand.empty:
        return "BL0"
    return str(cand.sort_values(SORT_KEYS,
                                ascending=[True, False, False]).iloc[0]["model"])


def apply_gates(cfg, agg, rob_agg, cov=None, operating_alpha=None):
    sel = cfg["selection"]
    if operating_alpha is None:
        operating_alpha = cfg["calibration"]["yellow_p"]
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
        c_sign10 = _coverage_error(cov, m, 0.10, "signed")
        c_abs10 = _coverage_error(cov, m, 0.10, "abs_mean")
        c_max10 = _coverage_error(cov, m, 0.10, "max")
        c_op = _coverage_error(cov, m, operating_alpha, "abs_mean")

        g1 = bool(np.isfinite(r["recall_mean"]) and np.isfinite(r["f1_mean"]))
        # v0: 이상 Recall 하락만
        g2_v0 = (not np.isfinite(d)) or d <= sel["max_recall_drop_pp"]
        # v1/v2: 정상측 오경보 증가 병행
        g2 = g2_v0 and ((not np.isfinite(sfp)) or
                        sfp <= sel["max_recall_drop_pp"])
        g3_v0 = (m == "BL0") or (float(r["fp_rate_worst"]) <= bl0_fp)
        g3 = (not sel["worst_block_fp_vs_bl0"]) or m == "BL0" or \
             float(r["fp_rate_worst"]) <= fp_limit
        g4 = (not sel["require_burst_detection"]) or \
             int(r["bursts_detected_min"]) >= 1
        g5_v1 = (not np.isfinite(c_sign10)) or c_sign10 <= MAX_COVERAGE_ERR
        g5_v2 = (not np.isfinite(c_op)) or c_op <= MAX_COVERAGE_ERR

        rows.append({
            "model": m, "pretty": a.loc[m, "pretty"],
            "gate1_no_nan": g1,
            "gate2_robust": g2, "gate2_v0_recall_only": g2_v0,
            "worst_recall_drop_pp": d,
            "worst_shift_fp_increase_pp": sfp,
            "gate3_fp_vs_bl0": g3, "gate3_v0_strict": g3_v0,
            "fp_rate_worst": float(r["fp_rate_worst"]),
            "bl0_fp_rate_worst": bl0_fp, "fp_limit": fp_limit,
            "gate4_burst_detected": g4,
            "bursts_detected_min": int(r["bursts_detected_min"]),
            "gate5_v1_signed_a0.10": g5_v1,
            "gate5_v2_abs_operating": g5_v2,
            "cov_err_signed_a0.10": c_sign10,
            "cov_err_absmean_a0.10": c_abs10,
            "cov_err_max_a0.10": c_max10,
            "cov_err_absmean_operating": c_op,
            "operating_alpha": operating_alpha,
            "far_h_upper95_worst": float(r["far_h_upper95_worst"]),
            "recall_mean": float(r["recall_mean"]),
            "f1_mean": float(r["f1_mean"]),
            "explainability": EXPLAINABILITY.get(m, 0),
        })
    gates = pd.DataFrame(rows)

    gates["passed_v0"] = (gates.gate1_no_nan & gates.gate2_v0_recall_only &
                          gates.gate3_v0_strict & gates.gate4_burst_detected)
    gates["passed_v1"] = (gates.gate1_no_nan & gates.gate2_robust &
                          gates.gate3_fp_vs_bl0 & gates.gate4_burst_detected &
                          gates["gate5_v1_signed_a0.10"])
    gates["passed_v2"] = (gates.gate1_no_nan & gates.gate2_robust &
                          gates.gate3_fp_vs_bl0 & gates.gate4_burst_detected &
                          gates.gate5_v2_abs_operating)
    # 제출본 호환 열
    gates["passed"] = gates["passed_v2"]
    gates["coverage_err_at_0.10"] = gates["cov_err_signed_a0.10"]

    picks = {
        "v0_original": _pick(gates, "passed_v0", exclude_bl0=False),
        "v1_submitted": _pick(gates, "passed_v1", exclude_bl0=True),
        "v2_corrected": _pick(gates, "passed_v2", exclude_bl0=False),
    }
    return gates, picks["v2_corrected"], picks

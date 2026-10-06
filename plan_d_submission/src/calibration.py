# -*- coding: utf-8 -*-
"""정상성 conformal p-value (분석 프로토콜 §5.1).

p_normal(x) = (1 + #{s_i >= s(x)}) / (n + 1)
risk_score  = 1 - p_normal          (순위 기반 이상위험 점수. 사후확률 아님)
"""
import numpy as np


def conformal_p(cal_scores, new_scores):
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    n = len(cal)
    # #{s_i >= s} = n - searchsorted(cal, s, 'left')
    ge = n - np.searchsorted(cal, np.asarray(new_scores, dtype=float), side="left")
    return (1.0 + ge) / (n + 1.0)


def risk(cal_scores, new_scores):
    return 1.0 - conformal_p(cal_scores, new_scores)


def posterior_sensitivity(tpr, fpr, prevalences=(0.01, 0.001, 0.0001)):
    """현장 고장률 가정별 사후확률 (§5.1). p(고장|경보)."""
    out = []
    for pi in prevalences:
        denom = tpr * pi + fpr * (1 - pi)
        out.append({"prevalence": pi, "tpr": tpr, "fpr": fpr,
                    "posterior": (tpr * pi / denom) if denom > 0 else float("nan")})
    return out


def coverage(p_values, alphas=(0.01, 0.05, 0.10)):
    """정상 holdout 에서 p<=alpha 비율이 alpha 근처인지 (E8)."""
    p = np.asarray(p_values, dtype=float)
    return [{"alpha": a, "empirical": float((p <= a).mean()), "n": len(p)}
            for a in alphas]

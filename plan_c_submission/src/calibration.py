from __future__ import annotations

import numpy as np


def empirical_p_values(calibration_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    calibration_scores = np.asarray(calibration_scores, dtype=float)
    scores = np.asarray(scores, dtype=float)
    ordered = np.sort(calibration_scores)
    left = np.searchsorted(ordered, scores, side="left")
    greater_equal = len(ordered) - left
    return (greater_equal + 1.0) / (len(ordered) + 1.0)


def threshold_from_quantile(calibration_scores: np.ndarray, quantile: float) -> float:
    return float(np.quantile(calibration_scores, quantile, method="higher"))


def aggregate_by_burst(scores: np.ndarray, burst_ids: np.ndarray, mode: str = "max") -> np.ndarray:
    out = []
    for bid in np.unique(burst_ids):
        s = scores[burst_ids == bid]
        if mode == "max":
            out.append(float(np.max(s)))
        elif mode == "q95":
            out.append(float(np.quantile(s, 0.95)))
        else:
            raise ValueError(mode)
    return np.asarray(out)


def burst_p_values(
    calibration_scores: np.ndarray,
    calibration_bursts: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    burst_reference = aggregate_by_burst(calibration_scores, calibration_bursts, mode="max")
    return empirical_p_values(burst_reference, scores)

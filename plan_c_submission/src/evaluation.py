from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def binary_metrics(normal_scores: np.ndarray, anomaly_scores: np.ndarray, threshold: float) -> dict:
    y = np.r_[np.zeros(len(normal_scores), dtype=int), np.ones(len(anomaly_scores), dtype=int)]
    scores = np.r_[normal_scores, anomaly_scores]
    pred = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    normal_pred = pred[: len(normal_scores)]
    anomaly_pred = pred[len(normal_scores) :]
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auroc": float(roc_auc_score(y, scores)),
        "ap": float(average_precision_score(y, scores)),
        "fp_windows": int(normal_pred.sum()),
        "tn_windows": int(len(normal_pred) - normal_pred.sum()),
        "tp_windows": int(anomaly_pred.sum()),
        "fn_windows": int(len(anomaly_pred) - anomaly_pred.sum()),
        "fpr": float(normal_pred.mean()) if len(normal_pred) else 0.0,
        "threshold": float(threshold),
    }


def alarm_ids(alarms: np.ndarray, burst_ids: np.ndarray, persistence: int) -> np.ndarray:
    result = np.full(len(alarms), -1, dtype=int)
    next_id = 0
    for bid in np.unique(burst_ids):
        idx = np.flatnonzero(burst_ids == bid)
        a = alarms[idx]
        streak = 0
        active = False
        current_id = -1
        below = 0
        for local, flag in enumerate(a):
            if flag:
                streak += 1
                below = 0
                if not active and streak >= persistence:
                    active = True
                    current_id = next_id
                    next_id += 1
                    start = max(0, local - persistence + 1)
                    result[idx[start : local + 1]] = current_id
                elif active:
                    result[idx[local]] = current_id
            else:
                streak = 0
                if active:
                    below += 1
                    if below >= persistence:
                        active = False
                        current_id = -1
                        below = 0
                if active:
                    result[idx[local]] = current_id
    return result


def alarm_summary(
    scores: np.ndarray,
    threshold: float,
    burst_ids: np.ndarray,
    persistence: int,
    observed_hours: float,
) -> dict:
    alarms = scores > threshold
    ids = alarm_ids(alarms, burst_ids, persistence)
    event_count = int(len(np.unique(ids[ids >= 0])))
    detected_bursts = int(sum(np.any(ids[burst_ids == bid] >= 0) for bid in np.unique(burst_ids)))
    return {
        "alarm_events": event_count,
        "detected_bursts": detected_bursts,
        "total_bursts": int(len(np.unique(burst_ids))),
        "burst_detection_rate": float(detected_bursts / max(len(np.unique(burst_ids)), 1)),
        "far_per_observed_hour": float(event_count / observed_hours) if observed_hours > 0 else float("nan"),
        "alarm_id": ids,
    }


def poisson_upper_95(events: int, hours: float) -> float:
    from scipy.stats import chi2

    if hours <= 0:
        return float("nan")
    upper_count = chi2.ppf(0.975, 2 * (events + 1)) / 2
    return float(upper_count / hours)

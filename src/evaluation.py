# -*- coding: utf-8 -*-
"""평가 지표 (Plan D §6.1).

표기 규칙
 - window 지표는 중첩 window 라 독립 표본이 아니다.
 - 양성 독립 단위는 fault record 1건이다. 신뢰구간은 각주로만 쓴다.
 - 오경보 0건은 FAR=0 의 증거가 아니므로 Poisson 95% 상한을 병기한다.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             roc_auc_score)


def window_metrics(y_true, y_pred, score=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "precision": prec, "recall": rec, "f1": f1,
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred))
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else 0.0,
        "prevalence": float(y_true.mean()),
        "n": len(y_true),
    }
    if score is not None and len(np.unique(y_true)) > 1:
        out["AP"] = float(average_precision_score(y_true, score))
        out["AUROC"] = float(roc_auc_score(y_true, score))
    else:
        out["AP"] = float("nan")
        out["AUROC"] = float("nan")
    return out


def merge_alarms(flags, burst_ids, min_consecutive=1):
    """연속 window 경보를 하나의 '경보 이벤트'로 병합.
    버스트가 바뀌면 끊는다. 반환: 이벤트 수, 이벤트별 (start,end) 리스트."""
    flags = np.asarray(flags).astype(bool)
    burst_ids = np.asarray(burst_ids)
    events, start = [], None
    for i in range(len(flags)):
        new_burst = i > 0 and burst_ids[i] != burst_ids[i - 1]
        if flags[i] and (start is None or new_burst):
            if start is not None and i - start >= min_consecutive:
                events.append((start, i - 1))
            start = i
        elif not flags[i] and start is not None:
            if i - start >= min_consecutive:
                events.append((start, i - 1))
            start = None
    if start is not None and len(flags) - start >= min_consecutive:
        events.append((start, len(flags) - 1))
    return len(events), events


def poisson_upper95(k, exposure_hours):
    """관측 k건에 대한 Poisson 95% 상한 / 시간.  k=0 -> 3.0/T."""
    # chi2(0.95, 2k+2)/2 의 표값 (k=0..10)
    tbl = [3.00, 4.74, 6.30, 7.75, 9.15, 10.51, 11.84, 13.15, 14.43, 15.71, 16.96]
    lam = tbl[k] if k < len(tbl) else k + 1.645 * np.sqrt(k)
    return lam / max(exposure_hours, 1e-9)


def false_alarm_profile(flags, meta, min_consecutive=1):
    """정상 holdout 의 오경보 프로파일."""
    n_ev, _ = merge_alarms(flags, meta["burst_id"].values, min_consecutive)
    # 노출시간: window 가 덮는 실제 수집시간 (버스트 길이 합)
    bt = meta.groupby("burst_id").agg(
        t0=("time_start", "min"), t1=("time_end", "max"))
    hours = float((bt["t1"] - bt["t0"]).dt.total_seconds().sum()) / 3600.0
    fp = int(np.sum(flags))
    return {
        "fp_windows": fp,
        "n_windows": len(flags),
        "fp_rate": fp / max(len(flags), 1),
        "alarm_events": n_ev,
        "exposure_hours": hours,
        "far_per_hour": n_ev / max(hours, 1e-9),
        "far_per_hour_upper95": poisson_upper95(n_ev, hours),
    }


def detection_delay(flags, meta):
    """버스트 진입점 기준 탐지지연 민감도 (§6.1).
    주의: 실제 고장 전 리드타임이 아니다. 수집 버스트 진입 후 경과시간일 뿐."""
    flags = np.asarray(flags).astype(bool)
    bid = meta["burst_id"].values
    pos_all = meta["pos_in_burst_sec"].values
    rows = []
    for b in np.unique(bid):
        sel = bid == b
        f = flags[sel]
        pos = pos_all[sel]
        hit = np.where(f)[0]
        rows.append({
            "burst_id": int(b), "n_windows": int(sel.sum()),
            "detected": bool(len(hit) > 0),
            "delay_sec": float(pos[hit[0]]) if len(hit) else float("nan"),
        })
    return rows

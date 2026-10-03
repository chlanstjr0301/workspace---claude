# -*- coding: utf-8 -*-
"""데이터 로딩과 품질 감사 (Plan D §2.2 / E0)."""
import hashlib
import os

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path, channels):
    """원본 인덱스 열을 original_row 로 보존하고 TimeStamp 를 시간형으로 변환."""
    df = pd.read_csv(path)
    first = df.columns[0]
    if first.strip() == "" or first.lower().startswith("unnamed"):
        df = df.rename(columns={first: "original_row"})
    else:
        df.insert(0, "original_row", np.arange(len(df)))
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise ValueError("채널 누락: %s" % missing)
    return df


def audit(df, name, channels, gap_sec):
    """§2.2 품질 점검. 보고서 1장 표의 원본."""
    dt = df["TimeStamp"].diff().dt.total_seconds()
    span = (df["TimeStamp"].max() - df["TimeStamp"].min()).total_seconds()
    rec = {
        "source": name,
        "rows": len(df),
        "columns": df.shape[1],
        "t_start": str(df["TimeStamp"].min()),
        "t_end": str(df["TimeStamp"].max()),
        "span_sec": round(span, 1),
        "span_if_continuous_sec": round(len(df) / 10.0, 1),
        "dt_median_sec": round(float(dt.median()), 4),
        "dt_min_sec": round(float(dt.min()), 4),
        "dt_max_sec": round(float(dt.max()), 4),
        "n_gaps_gt_threshold": int((dt > gap_sec).sum()),
        "n_missing": int(df[channels].isna().sum().sum()),
        "n_dup_timestamp": int(df["TimeStamp"].duplicated().sum()),
        "n_non_monotonic": int((dt < 0).sum()),
        "equipment_state_values": sorted(df["Equipment_state"].unique().tolist()),
    }
    return rec


def channel_stats(df, name, channels):
    out = []
    for c in channels:
        x = df[c].values.astype(float)
        m, s = x.mean(), x.std()
        out.append({
            "source": name, "channel": c, "n": len(x),
            "mean": m, "std": s, "min": x.min(), "max": x.max(),
            "skew": float(((x - m) ** 3).mean() / (s ** 3 + 1e-12)),
            "kurtosis": float(((x - m) ** 4).mean() / (s ** 4 + 1e-12)),
        })
    return out


def load_all(cfg, root):
    ch = cfg["data"]["channels"]
    pn = os.path.join(root, cfg["data"]["normal"])
    po = os.path.join(root, cfg["data"]["outlier"])
    normal = load_raw(pn, ch)
    outlier = load_raw(po, ch)
    hashes = {"press_data_normal.csv": sha256(pn),
              "press_data_outlier.csv": sha256(po)}
    return normal, outlier, hashes

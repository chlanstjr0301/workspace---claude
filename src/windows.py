# -*- coding: utf-8 -*-
"""버스트 분할과 gap-safe windowing (분석 프로토콜 §2.1-§2.2).

불변식: 어떤 window 도 버스트 경계를 가로지르지 않는다.
       -> tests/test_windows.py 가 이를 검증한다.
"""
import numpy as np
import pandas as pd


def segment_bursts(df, gap_sec):
    """인접 시각 차이가 gap_sec 보다 크면 새 버스트. burst_id 배열 반환."""
    dt = df["TimeStamp"].diff().dt.total_seconds().values
    new = np.zeros(len(df), dtype=bool)
    new[1:] = dt[1:] > gap_sec
    return np.cumsum(new)


def burst_table(df, gap_sec):
    bid = segment_bursts(df, gap_sec)
    g = pd.DataFrame({"burst_id": bid, "t": df["TimeStamp"].values})
    agg = g.groupby("burst_id")["t"].agg(["count", "min", "max"])
    agg["dur_sec"] = (agg["max"] - agg["min"]).dt.total_seconds()
    agg = agg.rename(columns={"count": "n_samples", "min": "t_start", "max": "t_end"})
    return agg.reset_index()


def make_windows(df, channels, seq, gap_sec, stride=1, source="normal"):
    """버스트 내부에서만 window 생성.

    반환: dict(X=(n,seq,ch), meta=DataFrame)
    meta 열: source, burst_id, idx_start, idx_end, original_row_start,
             original_row_end, time_start, time_end, pos_in_burst_sec,
             dt_std, label
    """
    X = df[channels].values.astype(float)
    ts = df["TimeStamp"].values
    rows = df["original_row"].values
    label = df["Equipment_state"].values
    bid = segment_bursts(df, gap_sec)
    dt_all = df["TimeStamp"].diff().dt.total_seconds().values

    chunks, meta = [], []
    for b in np.unique(bid):
        idx = np.where(bid == b)[0]
        if len(idx) < seq:
            continue
        s0 = idx[0]
        t0 = ts[s0]
        for i in range(0, len(idx) - seq + 1, stride):
            a, z = idx[i], idx[i + seq - 1]
            chunks.append(X[a:z + 1])
            dts = dt_all[a + 1:z + 1]
            meta.append({
                "source": source,
                "burst_id": int(b),
                "idx_start": int(a), "idx_end": int(z),
                "original_row_start": int(rows[a]),
                "original_row_end": int(rows[z]),
                "time_start": ts[a], "time_end": ts[z],
                "pos_in_burst_sec": float(
                    (ts[a] - t0) / np.timedelta64(1, "s")),
                "dt_std": float(np.std(dts)) if len(dts) else 0.0,
                "label": int(label[a:z + 1].max()),
            })
    if not chunks:
        return np.empty((0, seq, len(channels))), pd.DataFrame(columns=[
            "source", "burst_id", "idx_start", "idx_end",
            "original_row_start", "original_row_end", "time_start",
            "time_end", "pos_in_burst_sec", "dt_std", "label"])
    return np.stack(chunks), pd.DataFrame(meta)


def block_split(meta, n_blocks):
    """정상 데이터를 시간순 n_blocks 로 나눈다.

    경계 버스트가 두 블록에 걸치지 않도록 '버스트 단위'로 배정한다 (§2.3).
    """
    bursts = np.sort(meta["burst_id"].unique())
    parts = np.array_split(bursts, n_blocks)
    block_of = {}
    for k, part in enumerate(parts):
        for b in part:
            block_of[b] = k
    return meta["burst_id"].map(block_of).values


def cv_folds(n_blocks):
    """fold 당 (train blocks, calibration block, holdout block).

    모든 블록이 정확히 한 번 holdout 이 되도록 순환한다.
    """
    folds = []
    for k in range(n_blocks):
        hold = k
        cal = (k + 1) % n_blocks
        train = [b for b in range(n_blocks) if b not in (hold, cal)]
        folds.append({"fold": k, "train": train, "cal": cal, "holdout": hold})
    return folds

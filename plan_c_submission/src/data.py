from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]


@dataclass(frozen=True)
class WindowSet:
    values: np.ndarray
    row_start: np.ndarray
    row_end: np.ndarray
    time_start: np.ndarray
    time_end: np.ndarray
    burst_id: np.ndarray
    burst_position: np.ndarray
    burst_length: np.ndarray
    source: str

    def take(self, mask: np.ndarray) -> "WindowSet":
        return WindowSet(
            values=self.values[mask],
            row_start=self.row_start[mask],
            row_end=self.row_end[mask],
            time_start=self.time_start[mask],
            time_end=self.time_end[mask],
            burst_id=self.burst_id[mask],
            burst_position=self.burst_position[mask],
            burst_length=self.burst_length[mask],
            source=self.source,
        )


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed:") or c == ""]
    df = df.drop(columns=unnamed, errors="ignore")
    required = {"TimeStamp", "Equipment_state", *SIGNAL_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df = df.copy()
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="raise")
    df["original_row"] = np.arange(len(df), dtype=int)
    return df


def assign_bursts(df: pd.DataFrame, gap_seconds: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    dt = out["TimeStamp"].diff().dt.total_seconds()
    boundary = dt.isna() | (dt > gap_seconds) | (dt < 0)
    out["burst_id"] = boundary.cumsum().astype(int) - 1
    out["dt_seconds"] = dt
    out["burst_position"] = out.groupby("burst_id").cumcount()
    out["burst_length"] = out.groupby("burst_id")["burst_id"].transform("size")
    return out


def make_windows(df: pd.DataFrame, window_size: int, source: str) -> WindowSet:
    values: list[np.ndarray] = []
    starts: list[int] = []
    ends: list[int] = []
    time_starts: list[np.datetime64] = []
    time_ends: list[np.datetime64] = []
    burst_ids: list[int] = []
    positions: list[int] = []
    lengths: list[int] = []

    for bid, group in df.groupby("burst_id", sort=True):
        group = group.sort_values("TimeStamp")
        x = group[SIGNAL_COLUMNS].to_numpy(dtype=np.float64)
        if len(group) < window_size:
            continue
        rows = group["original_row"].to_numpy(dtype=int)
        times = group["TimeStamp"].to_numpy(dtype="datetime64[ns]")
        for i in range(len(group) - window_size + 1):
            values.append(x[i : i + window_size])
            starts.append(int(rows[i]))
            ends.append(int(rows[i + window_size - 1]))
            time_starts.append(times[i])
            time_ends.append(times[i + window_size - 1])
            burst_ids.append(int(bid))
            positions.append(i)
            lengths.append(len(group))

    if not values:
        shape = (0, window_size, len(SIGNAL_COLUMNS))
        arr = np.empty(shape, dtype=np.float64)
    else:
        arr = np.stack(values)
    return WindowSet(
        values=arr,
        row_start=np.asarray(starts, dtype=int),
        row_end=np.asarray(ends, dtype=int),
        time_start=np.asarray(time_starts, dtype="datetime64[ns]"),
        time_end=np.asarray(time_ends, dtype="datetime64[ns]"),
        burst_id=np.asarray(burst_ids, dtype=int),
        burst_position=np.asarray(positions, dtype=int),
        burst_length=np.asarray(lengths, dtype=int),
        source=source,
    )


def audit_dataframe(df: pd.DataFrame, gap_seconds: float) -> dict:
    dt = df["TimeStamp"].diff().dt.total_seconds()
    internal = dt[(dt >= 0) & (dt <= gap_seconds)]
    duration = (df["TimeStamp"].iloc[-1] - df["TimeStamp"].iloc[0]).total_seconds()
    return {
        "rows": int(len(df)),
        "state_values": ",".join(map(str, sorted(df["Equipment_state"].unique()))),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_timestamps": int(df["TimeStamp"].duplicated().sum()),
        "negative_time_steps": int((dt < 0).sum()),
        "gaps_over_threshold": int((dt > gap_seconds).sum()),
        "bursts": int(df["burst_id"].nunique()),
        "median_internal_dt": float(internal.median()),
        "wall_duration_sec": float(duration),
        "observed_duration_sec": float(len(df) * 0.1),
    }


def temporal_burst_blocks(df: pd.DataFrame, n_blocks: int = 5) -> dict[int, np.ndarray]:
    burst_ids = np.asarray(sorted(df["burst_id"].unique()), dtype=int)
    chunks = np.array_split(burst_ids, n_blocks)
    return {i: chunk for i, chunk in enumerate(chunks)}


def fold_masks(windows: WindowSet, blocks: dict[int, np.ndarray], fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_blocks = len(blocks)
    test_block = fold
    calibration_block = (fold + 1) % n_blocks
    train_blocks = [i for i in range(n_blocks) if i not in {test_block, calibration_block}]
    train_bursts = np.concatenate([blocks[i] for i in train_blocks])
    train_mask = np.isin(windows.burst_id, train_bursts)
    calibration_mask = np.isin(windows.burst_id, blocks[calibration_block])
    test_mask = np.isin(windows.burst_id, blocks[test_block])
    return train_mask, calibration_mask, test_mask

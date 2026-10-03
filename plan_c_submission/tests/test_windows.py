from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import assign_bursts, load_csv, make_windows


ROOT = Path(__file__).resolve().parents[2]


def test_windows_never_cross_gap() -> None:
    df = assign_bursts(load_csv(ROOT / "data" / "raw" / "press_data_normal.csv"), 0.5)
    windows = make_windows(df, 10, "normal")
    for start, end, bid in zip(windows.row_start, windows.row_end, windows.burst_id):
        rows = df.iloc[start : end + 1]
        assert rows["burst_id"].nunique() == 1
        assert int(rows["burst_id"].iloc[0]) == int(bid)
        assert np.all(rows["TimeStamp"].diff().dt.total_seconds().dropna() <= 0.5)


def test_expected_window_count() -> None:
    df = assign_bursts(load_csv(ROOT / "data" / "raw" / "press_data_normal.csv"), 0.5)
    assert len(make_windows(df, 10, "normal").values) == 14871

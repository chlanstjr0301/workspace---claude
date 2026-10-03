from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import assign_bursts, fold_masks, load_csv, make_windows, temporal_burst_blocks


ROOT = Path(__file__).resolve().parents[2]


def test_fold_bursts_are_disjoint() -> None:
    df = assign_bursts(load_csv(ROOT / "data" / "raw" / "press_data_normal.csv"), 0.5)
    windows = make_windows(df, 10, "normal")
    blocks = temporal_burst_blocks(df, 5)
    for fold in range(5):
        train, calibration, test = fold_masks(windows, blocks, fold)
        train_ids = set(windows.burst_id[train])
        cal_ids = set(windows.burst_id[calibration])
        test_ids = set(windows.burst_id[test])
        assert train_ids.isdisjoint(cal_ids)
        assert train_ids.isdisjoint(test_ids)
        assert cal_ids.isdisjoint(test_ids)
        assert not np.any(train & calibration)
        assert not np.any(train & test)
        assert not np.any(calibration & test)

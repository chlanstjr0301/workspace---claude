# -*- coding: utf-8 -*-
"""불변식: window 는 시간 공백을 가로지르지 않는다 (Plan D 즉시 에스컬레이션 항목)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml                               # noqa: E402
from src import data as D                 # noqa: E402
from src import windows as WD             # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
CH = CFG["data"]["channels"]
GAP = CFG["windows"]["gap_sec"]


@pytest.fixture(scope="module")
def normal():
    return D.load_raw(os.path.join(ROOT, CFG["data"]["normal"]), CH)


def test_no_window_crosses_gap(normal):
    seq = CFG["windows"]["length"]
    X, meta = WD.make_windows(normal, CH, seq, GAP, source="normal")
    assert len(X) > 0
    span = (meta["time_end"] - meta["time_start"]).dt.total_seconds().values
    # 같은 버스트 안이면 최대 (seq-1)*0.1초 + 허용오차
    assert span.max() <= (seq - 1) * GAP, "window 가 공백을 가로질렀다"
    assert (meta["original_row_end"] - meta["original_row_start"] == seq - 1).all()


def test_window_count_matches_bursts(normal):
    seq = CFG["windows"]["length"]
    bt = WD.burst_table(normal, GAP)
    expected = int(np.maximum(bt["n_samples"] - seq + 1, 0).sum())
    X, _ = WD.make_windows(normal, CH, seq, GAP, source="normal")
    assert len(X) == expected


def test_gap_aware_is_strict_subset_of_naive(normal):
    seq = CFG["windows"]["length"]
    X, _ = WD.make_windows(normal, CH, seq, GAP, source="normal")
    naive = len(normal) - seq + 1
    assert len(X) < naive, "gap-aware 가 naive 보다 적어야 한다"


def test_blocks_do_not_split_bursts(normal):
    seq = CFG["windows"]["length"]
    _, meta = WD.make_windows(normal, CH, seq, GAP, source="normal")
    meta["block"] = WD.block_split(meta, CFG["cv"]["n_blocks"])
    per = meta.groupby("burst_id")["block"].nunique()
    assert (per == 1).all(), "하나의 버스트가 두 블록에 걸쳤다"


def test_cv_folds_cover_all_blocks():
    n = CFG["cv"]["n_blocks"]
    folds = WD.cv_folds(n)
    assert len(folds) == n
    assert sorted(f["holdout"] for f in folds) == list(range(n))
    for f in folds:
        assert f["cal"] != f["holdout"]
        assert f["cal"] not in f["train"] and f["holdout"] not in f["train"]

# -*- coding: utf-8 -*-
"""실시간 재생 판정이 일괄 예측과 같은지 확인한다 (고장 기록, 428 window)."""
import os
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import stream_replay as SR  # noqa: E402


def test_stream_replay_matches_batch_on_fault_record():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    normal, outlier, mg, vib, m1, cal1, m2, cal2, _ = SR.fit_frozen(cfg)
    rp, _ = SR.replay(outlier, SR.StreamJudge(cfg, mg, vib, m1, cal1, m2, cal2))
    pred = pd.read_csv(os.path.join(ROOT, "outputs", "predictions.csv"))
    ref = pred[pred.source != "normal"].merge(rp, on="original_row_start", suffixes=("", "_s"))
    assert len(ref) == len(rp) == 428
    assert (ref.alarm_level == ref.alarm_level_s).all()
    assert np.allclose(ref.p_normal_stage1, ref.p1) and np.allclose(ref.p_normal_stage2, ref.p2)

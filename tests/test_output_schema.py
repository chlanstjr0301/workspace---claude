# -*- coding: utf-8 -*-
"""불변식: 산출물 스키마가 분석 프로토콜 §10 예측 파일 최소 열을 만족한다."""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "outputs")

REQUIRED = ["source", "original_row_start", "original_row_end", "burst_id",
            "time_start", "time_end", "score_stage1", "score_stage2",
            "p_normal_stage1", "p_normal_stage2", "risk_score",
            "alarm_level", "top_reason_1", "top_reason_2", "label"]

REQUIRED_TABLES = [
    "e0_data_quality.csv", "e0_bursts.csv", "e0_burst_summary.csv",
    "e1_window_sensitivity.csv", "e2_model_comparison.csv",
    "e2_selection_gates.csv", "e3_normal_block_cv.csv",
    "e4_feature_ablation.csv", "e5_robustness.csv",
    "e7_error_conditions.csv", "e8_conformal_coverage.csv",
]


def _need(p):
    if not os.path.exists(p):
        pytest.skip("run_all.py 를 먼저 실행해야 한다: %s" % os.path.basename(p))
    return p


def test_predictions_schema():
    df = pd.read_csv(_need(os.path.join(OUT, "predictions.csv")))
    for c in REQUIRED:
        assert c in df.columns, "필수 열 누락: %s" % c
    assert set(df["alarm_level"].unique()) <= {"green", "yellow", "red"}
    assert df["p_normal_stage1"].between(0, 1).all()
    assert df["p_normal_stage2"].between(0, 1).all()
    assert df["risk_score"].between(0, 1).all()
    assert set(df["source"].unique()) == {"normal", "outlier"}
    assert df.isna().sum().sum() == 0


def test_all_tables_present():
    for t in REQUIRED_TABLES:
        p = os.path.join(OUT, "tables", t)
        _need(p)
        assert len(pd.read_csv(p)) > 0, "빈 표: %s" % t


def test_manifest_records_hashes():
    with open(_need(os.path.join(OUT, "run_manifest.json")), encoding="utf-8") as f:
        m = json.load(f)
    for k in ("mode", "seed", "data_sha256", "config", "chosen_stage1_model"):
        assert k in m, "manifest 항목 누락: %s" % k
    assert len(m["data_sha256"]) == 2
    for h in m["data_sha256"].values():
        assert len(h) == 64


def test_no_absolute_paths_in_sources():
    """상대경로만 사용 (§10 재현성 통과 조건)."""
    # 이 파일 자신은 검사 패턴을 문자열로 담고 있으므로 제외한다.
    me = os.path.basename(__file__)
    pats = ["C:" + "\\Users", "/Users" + "/", "D:" + "\\"]
    bad = []
    for d, _, fs in os.walk(ROOT):
        if "outputs" in d or "__pycache__" in d:
            continue
        for f in fs:
            if not f.endswith(".py") or f == me:
                continue
            txt = open(os.path.join(d, f), encoding="utf-8").read()
            if any(p in txt for p in pats):
                bad.append(f)
    assert not bad, "절대경로 발견: %s" % bad

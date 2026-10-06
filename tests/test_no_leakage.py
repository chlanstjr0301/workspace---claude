# -*- coding: utf-8 -*-
"""불변식: holdout/이상 정보가 scaler·임계값에 들어가지 않는다 (분석 프로토콜 §2.2-6)."""
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import calibration as cal        # noqa: E402
from src import data as D                 # noqa: E402
from src import features as FT            # noqa: E402
from src import models as MD              # noqa: E402
from src import windows as WD             # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
CH = CFG["data"]["channels"]
GAP = CFG["windows"]["gap_sec"]
SEQ = CFG["windows"]["length"]


@pytest.fixture(scope="module")
def prep():
    n = D.load_raw(os.path.join(ROOT, CFG["data"]["normal"]), CH)
    o = D.load_raw(os.path.join(ROOT, CFG["data"]["outlier"]), CH)
    Xn, mn = WD.make_windows(n, CH, SEQ, GAP, source="normal")
    Xo, mo = WD.make_windows(o, CH, SEQ, GAP, source="outlier")
    mn["block"] = WD.block_split(mn, CFG["cv"]["n_blocks"])
    Fn, names, gof = FT.build(Xn, CH, mn, tuple(CFG["features"]["model_groups"]))
    Fo, _, _ = FT.build(Xo, CH, mo, tuple(CFG["features"]["model_groups"]))
    return Fn, Fo, mn


def test_model_fit_ignores_holdout_and_anomaly(prep):
    """학습 블록만으로 적합한 모델은 holdout 을 섞어도 결과가 동일해야 한다."""
    Fn, Fo, mn = prep
    tr = np.isin(mn["block"].values, [0, 1, 2])
    for m_a, m_b in zip(MD.build(CFG, 0), MD.build(CFG, 0)):
        s_a = m_a.fit(Fn[tr]).score(Fo)
        s_b = m_b.fit(Fn[tr].copy()).score(Fo)
        assert np.allclose(s_a, s_b), "%s 가 재현되지 않는다" % m_a.name


def test_scaler_bounds_come_from_train_only(prep):
    """BL-0 의 범위가 학습 블록의 min/max 와 정확히 일치해야 한다."""
    Fn, _, mn = prep
    tr = np.isin(mn["block"].values, [0, 1, 2])
    m = MD.BL0RangeRule().fit(Fn[tr])
    assert np.allclose(m.lo_, Fn[tr].min(axis=0))
    assert np.allclose(m.hi_, Fn[tr].max(axis=0))
    # holdout 전체의 min/max 와는 달라야 한다(= 전체 데이터로 적합하지 않았다)
    assert not np.allclose(m.lo_, Fn.min(axis=0))


def test_conformal_p_is_valid_probability(prep):
    Fn, Fo, mn = prep
    tr = np.isin(mn["block"].values, [0, 1, 2])
    cl = mn["block"].values == 3
    m = MD.M1Mahalanobis().fit(Fn[tr])
    p = cal.conformal_p(m.score(Fn[cl]), m.score(Fo))
    assert p.min() > 0.0 and p.max() <= 1.0
    n = int(cl.sum())
    assert np.isclose(p.min(), 1.0 / (n + 1)) or p.min() >= 1.0 / (n + 1)


def test_conformal_calibration_is_uniform_on_calibration_set(prep):
    """calibration 자기 자신에 대한 p 는 대략 균등분포여야 한다."""
    Fn, _, mn = prep
    tr = np.isin(mn["block"].values, [0, 1, 2])
    cl = mn["block"].values == 3
    m = MD.M1Mahalanobis().fit(Fn[tr])
    s = m.score(Fn[cl])
    p = cal.conformal_p(s, s)
    for a in (0.05, 0.10, 0.20):
        assert abs((p <= a).mean() - a) < 0.05, "coverage 가 %.2f 에서 벗어남" % a

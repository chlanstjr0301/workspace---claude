from __future__ import annotations

import numpy as np

from src.calibration import empirical_p_values


def test_empirical_p_values_are_monotone() -> None:
    calibration = np.array([1.0, 2.0, 3.0, 4.0])
    scores = np.array([0.0, 1.0, 2.5, 5.0])
    p = empirical_p_values(calibration, scores)
    assert np.all(np.diff(p) <= 0)
    assert np.all((p > 0) & (p <= 1))

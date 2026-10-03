from __future__ import annotations

import numpy as np

from .features import FeatureSet


def repair_group_contributions(model, train_x: np.ndarray, x: np.ndarray, features: FeatureSet) -> dict[str, np.ndarray]:
    baseline = model.score(x)
    center = np.median(train_x, axis=0)
    out: dict[str, np.ndarray] = {}
    for group, indices in features.groups.items():
        repaired = x.copy()
        repaired[:, indices] = center[indices]
        out[group] = np.maximum(baseline - model.score(repaired), 0.0)
    return out


def top_group(contributions: dict[str, np.ndarray]) -> np.ndarray:
    names = list(contributions)
    matrix = np.column_stack([contributions[name] for name in names])
    return np.asarray(names, dtype=object)[np.argmax(matrix, axis=1)]

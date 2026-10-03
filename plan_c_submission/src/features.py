from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import SIGNAL_COLUMNS, WindowSet


@dataclass(frozen=True)
class FeatureSet:
    values: np.ndarray
    names: list[str]
    groups: dict[str, np.ndarray]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1))
    return np.divide((ac * bc).sum(axis=1), denom, out=np.zeros(len(a)), where=denom > 1e-12)


def _acf(x: np.ndarray, lag: int) -> np.ndarray:
    if x.shape[1] <= lag:
        return np.zeros(x.shape[0])
    return _safe_corr(x[:, :-lag], x[:, lag:])


def extract_features(windows: WindowSet) -> FeatureSet:
    w = windows.values
    if len(w) == 0:
        return FeatureSet(np.empty((0, 0)), [], {})
    columns: list[np.ndarray] = []
    names: list[str] = []
    group_names: list[str] = []

    for j, channel in enumerate(SIGNAL_COLUMNS):
        x = w[:, :, j]
        stats = {
            "std": np.std(x, axis=1, ddof=0),
            "rms": np.sqrt(np.mean(x * x, axis=1)),
            "ptp": np.ptp(x, axis=1),
            "mad": np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1),
        }
        for key, value in stats.items():
            columns.append(value); names.append(f"A_{key}_{channel}"); group_names.append("A")
        shape = {
            "acf1": _acf(x, 1),
            "acf2": _acf(x, 2),
            "zcr": np.mean(np.signbit(x[:, 1:]) != np.signbit(x[:, :-1]), axis=1),
            "diff_mad": np.median(np.abs(np.diff(x, axis=1)), axis=1),
        }
        for key, value in shape.items():
            columns.append(value); names.append(f"S_{key}_{channel}"); group_names.append("S")
        offsets = {
            "mean": np.mean(x, axis=1),
            "median": np.median(x, axis=1),
            "absmean": np.mean(np.abs(x), axis=1),
        }
        for key, value in offsets.items():
            columns.append(value); names.append(f"O_{key}_{channel}"); group_names.append("O")

    x0, x1 = w[:, :, 0], w[:, :, 1]
    relations = {
        "corr01": _safe_corr(x0, x1),
        "abscorr01": _safe_corr(np.abs(x0), np.abs(x1)),
        "std_ratio01": np.std(x0, axis=1) / (np.std(x1, axis=1) + 1e-12),
    }
    for key, value in relations.items():
        columns.append(value); names.append(f"R_{key}"); group_names.append("R")

    q = {
        "burst_position_fraction": windows.burst_position / np.maximum(windows.burst_length - 1, 1),
        "burst_length": windows.burst_length.astype(float),
    }
    for key, value in q.items():
        columns.append(value); names.append(f"Q_{key}"); group_names.append("Q")

    matrix = np.column_stack(columns).astype(np.float64)
    matrix[~np.isfinite(matrix)] = 0.0
    groups = {g: np.flatnonzero(np.asarray(group_names) == g) for g in sorted(set(group_names))}
    return FeatureSet(matrix, names, groups)


def feature_indices(names: list[str], include_groups: tuple[str, ...]) -> np.ndarray:
    prefixes = tuple(f"{g}_" for g in include_groups)
    return np.asarray([i for i, name in enumerate(names) if name.startswith(prefixes)], dtype=int)


def z_normalize_windows(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.where(std > 1e-9, std, 1.0)

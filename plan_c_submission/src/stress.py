from __future__ import annotations

import re

import numpy as np


def _interp_jitter(values: np.ndarray, sigma_ms: float, rng: np.random.Generator) -> np.ndarray:
    n, length, channels = values.shape
    base = np.arange(length, dtype=float)
    sigma_steps = sigma_ms / 100.0
    out = np.empty_like(values)
    for i in range(n):
        jittered = np.clip(base + rng.normal(0.0, sigma_steps, size=length), 0, length - 1)
        jittered.sort()
        for c in range(channels):
            out[i, :, c] = np.interp(base, jittered, values[i, :, c])
    return out


def perturb(
    values: np.ndarray,
    name: str,
    train_channel_std: np.ndarray,
    seed: int,
) -> np.ndarray:
    x = values.copy()
    rng = np.random.default_rng(seed)
    if name.startswith("gain_"):
        factor = float(name.split("_")[1])
        return x * factor
    if name.startswith("offset_"):
        token = name.removeprefix("offset_").removesuffix("std")
        token = token.replace("minus", "-")
        factor = float(token)
        return x + factor * train_channel_std.reshape(1, 1, -1)
    if name.startswith("polarity_ai"):
        channel = int(name[-1])
        x[:, :, channel] *= -1
        return x
    if name.startswith("noise_"):
        pct = float(name.removeprefix("noise_").removesuffix("pct")) / 100.0
        noise = rng.normal(size=x.shape) * train_channel_std.reshape(1, 1, -1) * pct
        return x + noise
    if name.startswith("jitter_"):
        sigma = float(re.sub("ms$", "", name.removeprefix("jitter_")))
        return _interp_jitter(x, sigma, rng)
    if name.startswith("dropout_ai"):
        channel = int(name[-1])
        x[:, :, channel] = 0.0
        return x
    raise KeyError(name)

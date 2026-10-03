from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler

from .features import z_normalize_windows


class Detector(ABC):
    @abstractmethod
    def fit(self, x: np.ndarray) -> "Detector": ...

    @abstractmethod
    def score(self, x: np.ndarray) -> np.ndarray: ...


class RawRangeDetector(Detector):
    def fit(self, x: np.ndarray) -> "RawRangeDetector":
        flat = x.reshape(-1, x.shape[-1])
        self.low_ = np.min(flat, axis=0)
        self.high_ = np.max(flat, axis=0)
        q25, q75 = np.quantile(flat, [0.25, 0.75], axis=0)
        self.scale_ = np.maximum(q75 - q25, 1e-9)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        below = np.maximum((self.low_ - x) / self.scale_, 0.0)
        above = np.maximum((x - self.high_) / self.scale_, 0.0)
        return np.maximum(below, above).max(axis=(1, 2))


class MahalanobisDetector(Detector):
    def __init__(self) -> None:
        self.scaler = RobustScaler(quantile_range=(10, 90))
        self.cov = LedoitWolf()

    def fit(self, x: np.ndarray) -> "MahalanobisDetector":
        z = self.scaler.fit_transform(x)
        self.cov.fit(z)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(x)
        d = z - self.cov.location_
        return np.einsum("ij,jk,ik->i", d, self.cov.precision_, d)


class IsolationForestDetector(Detector):
    def __init__(self, seed: int, n_estimators: int = 300, max_samples: int = 256) -> None:
        self.scaler = RobustScaler(quantile_range=(10, 90))
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination="auto",
            random_state=seed,
            # A single worker avoids Windows named-pipe permission failures in
            # restricted native environments and keeps runs deterministic.
            n_jobs=1,
        )

    def fit(self, x: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(self.scaler.fit_transform(x))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(self.scaler.transform(x))


class PCAMSPCDetector(Detector):
    def __init__(self, variance: float = 0.95) -> None:
        self.variance = variance
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance, svd_solver="full")

    @staticmethod
    def _robust_location_scale(x: np.ndarray) -> tuple[float, float]:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)) * 1.4826)
        return med, max(mad, 1e-9)

    def fit(self, x: np.ndarray) -> "PCAMSPCDetector":
        z = self.scaler.fit_transform(x)
        scores = self.pca.fit_transform(z)
        reconstructed = self.pca.inverse_transform(scores)
        t2 = np.sum((scores * scores) / np.maximum(self.pca.explained_variance_, 1e-12), axis=1)
        spe = np.sum((z - reconstructed) ** 2, axis=1)
        self.t2_loc_, self.t2_scale_ = self._robust_location_scale(t2)
        self.spe_loc_, self.spe_scale_ = self._robust_location_scale(spe)
        return self

    def components(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = self.scaler.transform(x)
        scores = self.pca.transform(z)
        reconstructed = self.pca.inverse_transform(scores)
        t2 = np.sum((scores * scores) / np.maximum(self.pca.explained_variance_, 1e-12), axis=1)
        spe = np.sum((z - reconstructed) ** 2, axis=1)
        return t2, spe

    def score(self, x: np.ndarray) -> np.ndarray:
        t2, spe = self.components(x)
        z_t2 = (t2 - self.t2_loc_) / self.t2_scale_
        z_spe = (spe - self.spe_loc_) / self.spe_scale_
        return np.maximum(z_t2, z_spe)


class DynamicPCADetector(PCAMSPCDetector):
    def _prepare(self, x: np.ndarray) -> np.ndarray:
        return z_normalize_windows(x).reshape(len(x), -1)

    def fit(self, x: np.ndarray) -> "DynamicPCADetector":
        super().fit(self._prepare(x))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return super().score(self._prepare(x))


def make_feature_detector(
    model_id: str,
    seed: int,
    pca_variance: float,
    iforest_estimators: int,
    iforest_max_samples: int,
) -> Detector:
    if model_id == "B1_IF":
        return IsolationForestDetector(seed, iforest_estimators, iforest_max_samples)
    if model_id == "M1_Mahalanobis":
        return MahalanobisDetector()
    if model_id == "M2_PCA_MSPC":
        return PCAMSPCDetector(pca_variance)
    raise KeyError(model_id)

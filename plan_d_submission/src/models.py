# -*- coding: utf-8 -*-
"""모델 포트폴리오 (분석 프로토콜 §4).

공통 규약: fit(F_train) 후 score(F) 가 '클수록 이상'인 점수를 돌려준다.
모든 적합은 해당 fold 의 정상 학습구간에서만 이루어진다 (§2.2-6).
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class BL0RangeRule:
    """학습 정상 범위 이탈 규칙. 설명 가능한 최저 기준선."""
    name = "BL0"
    pretty = "BL-0 학습 정상 범위 이탈"

    def fit(self, F):
        self.lo_ = F.min(axis=0)
        self.hi_ = F.max(axis=0)
        self.rng_ = np.maximum(self.hi_ - self.lo_, 1e-12)
        return self

    def score(self, F):
        below = (self.lo_ - F) / self.rng_
        above = (F - self.hi_) / self.rng_
        return np.maximum(np.maximum(below, above), 0).max(axis=1)


class M1Mahalanobis:
    """Shrinkage(Ledoit-Wolf) Mahalanobis / Hotelling T^2."""
    name = "M1"
    pretty = "M1 Shrinkage Mahalanobis"

    def fit(self, F):
        self.sc_ = StandardScaler().fit(F)
        Z = self.sc_.transform(F)
        self.lw_ = LedoitWolf().fit(Z)
        return self

    def score(self, F):
        return self.lw_.mahalanobis(self.sc_.transform(F))

    def contrib(self, F):
        """특징별 기여도 분해 (§8.3)."""
        Z = self.sc_.transform(F) - self.lw_.location_
        P = self.lw_.precision_
        return Z * (Z @ P)


class M2IsolationForest:
    name = "M2"
    pretty = "M2 Isolation Forest"

    def __init__(self, n_estimators=200, max_samples=256, seed=0):
        self.kw = dict(n_estimators=n_estimators, max_samples=max_samples,
                       random_state=seed, n_jobs=-1)

    def fit(self, F):
        self.sc_ = StandardScaler().fit(F)
        self.m_ = IsolationForest(**self.kw).fit(self.sc_.transform(F))
        return self

    def score(self, F):
        return -self.m_.score_samples(self.sc_.transform(F))


class M3PCAMSPC:
    """PCA-MSPC: Hotelling T^2 + SPE(Q). 제조현장 표준."""
    name = "M3"
    pretty = "M3 PCA-MSPC (T2+SPE)"

    def __init__(self, var_ratio=0.95):
        self.var_ratio = var_ratio

    def fit(self, F):
        self.sc_ = StandardScaler().fit(F)
        Z = self.sc_.transform(F)
        self.p_ = PCA(n_components=self.var_ratio).fit(Z)
        T = self.p_.transform(Z)
        self.lam_ = T.var(axis=0) + 1e-12
        spe = ((Z - self.p_.inverse_transform(T)) ** 2).sum(axis=1)
        t2 = (T ** 2 / self.lam_).sum(axis=1)
        # 두 통계를 학습분포 95분위로 정규화해 합산 (스케일 통일)
        self.n_t2_ = np.quantile(t2, 0.95) + 1e-12
        self.n_spe_ = np.quantile(spe, 0.95) + 1e-12
        return self

    def _parts(self, F):
        Z = self.sc_.transform(F)
        T = self.p_.transform(Z)
        R = Z - self.p_.inverse_transform(T)
        return (T ** 2 / self.lam_).sum(axis=1), (R ** 2).sum(axis=1), R

    def score(self, F):
        t2, spe, _ = self._parts(F)
        return np.maximum(t2 / self.n_t2_, spe / self.n_spe_)

    def contrib(self, F):
        _, _, R = self._parts(F)
        return R ** 2


def build(cfg, seed):
    m = cfg["models"]
    out = []
    if "BL0" in m["enabled"]:
        out.append(BL0RangeRule())
    if "M1" in m["enabled"]:
        out.append(M1Mahalanobis())
    if "M2" in m["enabled"]:
        out.append(M2IsolationForest(
            n_estimators=m["isolation_forest"]["n_estimators"],
            max_samples=m["isolation_forest"]["max_samples"], seed=seed))
    if "M3" in m["enabled"]:
        out.append(M3PCAMSPC(var_ratio=m["pca_mspc"]["var_ratio"]))
    return out

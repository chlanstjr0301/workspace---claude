# -*- coding: utf-8 -*-
"""
단일클래스 이상탐지 후보 — 전부 정상 학습구간만으로 적합한다.

공격4 반영
  §2.1  Mahalanobis / PCA-SPE 가 IsolationForest 를 이긴다 -> 전부 비교 대상에 포함
  §2.4  IF 는 축평행 분할이라 스케일 불변. 나머지는 표준화 필요 -> 모델별 조건 명시
  §5    "진폭 모델의 FPR 0 은 견고함이 아니라 방향의 운" -> 단측(upper-tail) 변형 제공
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

SEED = 42


class Detector:
    """score(X) 가 클수록 이상. fit 은 정상 학습 데이터만 받는다."""
    name = "base"
    needs_scaling = True

    def fit(self, Xtr):
        raise NotImplementedError

    def score(self, X):
        raise NotImplementedError

    @property
    def n_params(self):
        return 0


class RangeRule(Detector):
    """BL-0. 학습 min/max 이탈 폭(정규화) 의 채널 최댓값.

    주의(공격4 §1.3): 이 데이터에서 테스트 정상구간은 학습 범위 내부이므로
    FP=0 이 구조적으로 보장된다. 공정 비교용 베이스라인이 아니라 참고값이다.
    """
    name = "BL0_Range"
    needs_scaling = False

    def fit(self, Xtr):
        self.lo_, self.hi_ = Xtr.min(0), Xtr.max(0)
        self.rng_ = np.maximum(self.hi_ - self.lo_, 1e-12)
        return self

    def score(self, X):
        below = (self.lo_ - X) / self.rng_
        above = (X - self.hi_) / self.rng_
        return np.maximum(below, above).max(axis=1)


class Mahalanobis(Detector):
    name = "Mahalanobis"

    def __init__(self, shrink=1e-6):
        self.shrink = shrink

    def fit(self, Xtr):
        self.mu_ = Xtr.mean(0)
        S = np.cov(Xtr.T) + self.shrink * np.eye(Xtr.shape[1])
        self.Si_ = np.linalg.inv(S)
        return self

    def score(self, X):
        D = X - self.mu_
        return np.einsum('ij,jk,ik->i', D, self.Si_, D)

    @property
    def n_params(self):
        d = len(self.mu_)
        return d + d * (d + 1) // 2


class MahalanobisOneSided(Mahalanobis):
    """단측 변형 — 학습 평균보다 '커진' 성분만 거리에 넣는다.

    저부하 에피소드(공격4 §4.1)는 진폭을 평균 아래로 내리고 고장은 위로 올린다.
    양측 거리는 저부하를 이상으로 오인하지만 단측은 구조적으로 면역이다.
    이 선택은 '정상 운전변화가 진폭을 낮추는 방향' 이라는 가정에 의존하며
    보고서에 한계로 명시해야 한다.
    """
    name = "Mahalanobis1S"

    def score(self, X):
        D = np.maximum(X - self.mu_, 0.0)
        return np.einsum('ij,jk,ik->i', D, self.Si_, D)


class PCAModel(Detector):
    """Hotelling T^2 (주성분 공간 내) / SPE·Q (주성분 공간 밖)."""

    def __init__(self, n_components=None, stat="SPE", var_target=0.95):
        self.nc, self.stat, self.var_target = n_components, stat, var_target
        self.name = "PCA_" + stat

    def fit(self, Xtr):
        U, S, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
        self.mu_ = Xtr.mean(0)
        ev = S ** 2 / max(len(Xtr) - 1, 1)
        if self.nc is None:
            r = np.cumsum(ev) / ev.sum()
            self.nc_ = int(np.searchsorted(r, self.var_target) + 1)
        else:
            self.nc_ = min(self.nc, Xtr.shape[1])
        self.nc_ = max(1, min(self.nc_, Xtr.shape[1] - 1))
        self.P_ = Vt[:self.nc_].T
        self.lam_ = np.maximum(ev[:self.nc_], 1e-12)
        return self

    def score(self, X):
        Y = X - self.mu_
        t = Y @ self.P_
        if self.stat == "T2":
            return (t ** 2 / self.lam_).sum(1)
        return ((Y - t @ self.P_.T) ** 2).sum(1)      # SPE / Q

    @property
    def n_params(self):
        return self.P_.size + self.lam_.size


class IForest(Detector):
    name = "IsolationForest"
    needs_scaling = False          # 축평행 분할이라 단조변환 불변

    def __init__(self, n_estimators=500, seed=SEED):
        self.ne, self.seed = n_estimators, seed

    def fit(self, Xtr):
        self.m_ = IsolationForest(n_estimators=self.ne, random_state=self.seed).fit(Xtr)
        return self

    def score(self, X):
        return -self.m_.score_samples(X)

    @property
    def n_params(self):
        return self.ne


class LOFNovelty(Detector):
    name = "LOF"

    def __init__(self, n_neighbors=35):
        self.k = n_neighbors

    def fit(self, Xtr):
        self.m_ = LocalOutlierFactor(n_neighbors=self.k, novelty=True).fit(Xtr)
        return self

    def score(self, X):
        return -self.m_.score_samples(X)


class OCSVMDet(Detector):
    name = "OCSVM"

    def __init__(self, nu=0.01, gamma="scale"):
        self.nu, self.gamma = nu, gamma

    def fit(self, Xtr):
        self.m_ = OneClassSVM(nu=self.nu, gamma=self.gamma).fit(Xtr)
        return self

    def score(self, X):
        return -self.m_.decision_function(X)

    @property
    def n_params(self):
        return int(self.m_.support_vectors_.shape[0])


def registry():
    """비교 대상 후보. 순서가 보고서 표 순서."""
    return [
        RangeRule(),
        Mahalanobis(),
        MahalanobisOneSided(),
        PCAModel(stat="T2"),
        PCAModel(stat="SPE"),
        IForest(n_estimators=500),
        LOFNovelty(),
        OCSVMDet(),
    ]


class Standardizer:
    def fit(self, X):
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        return self

    def transform(self, X):
        return (X - self.mu_) / self.sd_

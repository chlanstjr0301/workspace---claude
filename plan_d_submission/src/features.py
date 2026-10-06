# -*- coding: utf-8 -*-
"""A/S/R/O/Q 특징군 (분석 프로토콜 §3.1).

A 진폭  : 센서 이득에 민감        std, p2p, rms
S 형태  : 이득/오프셋에 비교적 강건 lag-1/lag-2 자기상관, zero-crossing rate
R 관계  : 상/하부 센서 상호작용    AI0-AI1 상관, |x| 상관, std 비
O 수준  : 오프셋/이득에 민감       전류 평균, 전류 절댓값 평균
Q 품질  : 수집 품질 (진단 전용)    window 내 dt 변동, 버스트 내 위치

주의: Q 는 고장 특징과 분리해 사용한다. 기본 모델 입력은 A/S/R/O 이다.
"""
import numpy as np

SHORT = {"AI0_Vibration": "AI0", "AI1_Vibration": "AI1", "AI2_Current": "CUR"}


def _ac(Z, k):
    """lag-k 자기상관 (window 내 평균 제거 후)."""
    num = (Z[:, :-k, :] * Z[:, k:, :]).mean(axis=1)
    den = (Z ** 2).mean(axis=1) + 1e-12
    return num / den


def _zcr(Z):
    s = np.sign(Z)
    s[s == 0] = 1
    return (s[:, :-1, :] != s[:, 1:, :]).mean(axis=1)


def _corr(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).mean(axis=1)
    den = np.sqrt((a ** 2).mean(axis=1) * (b ** 2).mean(axis=1)) + 1e-12
    return num / den


def build(X, channels, meta=None, groups=("A", "S", "R", "O")):
    """X=(n,seq,ch) -> (F, names, group_of)"""
    Z = X - X.mean(axis=1, keepdims=True)
    cols, names, gof = [], [], []
    sn = [SHORT.get(c, c) for c in channels]

    def add(mat, tmpl, g):
        for j, c in enumerate(sn):
            cols.append(mat[:, j])
            names.append(tmpl % c)
            gof.append(g)

    if "A" in groups:
        add(X.std(axis=1), "A_std_%s", "A")
        add(X.max(axis=1) - X.min(axis=1), "A_p2p_%s", "A")
        add(np.sqrt((X ** 2).mean(axis=1)), "A_rms_%s", "A")
    if "S" in groups:
        add(_ac(Z, 1), "S_ac1_%s", "S")
        add(_ac(Z, 2), "S_ac2_%s", "S")
        add(_zcr(Z), "S_zcr_%s", "S")
    if "R" in groups:
        a, b = X[:, :, 0], X[:, :, 1]
        cols.append(_corr(a, b)); names.append("R_corr_AI0_AI1"); gof.append("R")
        cols.append(_corr(np.abs(a), np.abs(b)))
        names.append("R_abscorr_AI0_AI1"); gof.append("R")
        cols.append(a.std(axis=1) / (b.std(axis=1) + 1e-12))
        names.append("R_stdratio_AI0_AI1"); gof.append("R")
    if "O" in groups:
        cols.append(X[:, :, 2].mean(axis=1)); names.append("O_mean_CUR"); gof.append("O")
        cols.append(np.abs(X[:, :, 2]).mean(axis=1))
        names.append("O_absmean_CUR"); gof.append("O")
    if "Q" in groups and meta is not None:
        cols.append(meta["dt_std"].values.astype(float))
        names.append("Q_dt_std"); gof.append("Q")
        cols.append(meta["pos_in_burst_sec"].values.astype(float))
        names.append("Q_pos_in_burst"); gof.append("Q")

    F = np.column_stack(cols).astype(float)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F, names, np.array(gof)


def select_groups(F, gof, keep):
    idx = np.where(np.isin(gof, list(keep)))[0]
    return F[:, idx], idx


def load_regime(X, q_lo, q_hi):
    """정상 학습 분위수로 만든 저/중/고 부하 상태 (§3.2).
    반환 0=저, 1=중, 2=고.  경계는 학습 fold 에서만 계산한다."""
    rms = np.sqrt((X ** 2).mean(axis=1)).mean(axis=1)
    return np.digitize(rms, [q_lo, q_hi])


def regime_bounds(X_train):
    rms = np.sqrt((X_train ** 2).mean(axis=1)).mean(axis=1)
    return float(np.quantile(rms, 1 / 3.0)), float(np.quantile(rms, 2 / 3.0))

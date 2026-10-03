# -*- coding: utf-8 -*-
"""
데이터 적재 · gap-aware windowing · 특징 추출

설계 근거는 `previous research/공격4 - Plan A 적대적 검증.md`:
  §1.3  BL-0 의 FP=0 은 구조적 -> 범위규칙은 참고 베이스라인으로만
  §2.3  AI2 채널 전체가 세션 간 비교 불가 -> D1 규칙으로 기본 제외
  §3.1  정상 진동은 백색잡음이 아니라 1.8/3.6 Hz 선 스펙트럼 -> 대역 파워 특징
  §3.2  corr01 단독 금지, `corr01 x 진폭` 상호작용으로 사용
  §4.1  16,000-19,000 은 가역 저부하 에피소드 -> 부하 대리변수로 식별
  §4.2  상·하부 진폭비는 극성·게인 불변
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "data", "raw"))

COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
SEQ = 20                 # window 길이(샘플). gap-aware 이므로 정확히 1.9 s
GAP_SEC = 0.5            # 이 이상 벌어지면 연속 구간을 끊는다
FS = 10.0                # Hz
TRAIN_END, VALID_END = 12000, 15000


# --------------------------------------------------------------------------- #
def load():
    n = pd.read_csv(os.path.join(RAW, "press_data_normal.csv"),
                    index_col=0, parse_dates=["TimeStamp"])
    o = pd.read_csv(os.path.join(RAW, "press_data_outlier.csv"),
                    index_col=0, parse_dates=["TimeStamp"])
    return n, o


def runs(df):
    """dt > GAP_SEC 로 끊은 연속 수집 구간(burst) [(start, end)] — end 배타."""
    dt = df.TimeStamp.diff().dt.total_seconds().values
    e = np.concatenate(([0], np.where(dt > GAP_SEC)[0], [len(df)]))
    return [(int(a), int(b)) for a, b in zip(e[:-1], e[1:]) if b > a]


def _acf(x, lag):
    """lag-k 지연 상관 (표준 ACF 가 아니라 lagged Pearson). 부분배열 기준으로 가드."""
    a, b = x[:-lag], x[lag:]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _band(x, lo, hi):
    """window 내 [lo, hi) Hz 대역 파워 비중."""
    xc = x - x.mean()
    if xc.std() < 1e-12:
        return 0.0
    p = np.abs(np.fft.rfft(xc * np.hanning(len(xc)))) ** 2
    f = np.fft.rfftfreq(len(xc), 1.0 / FS)
    tot = p.sum()
    if tot <= 0:
        return 0.0
    return float(p[(f >= lo) & (f < hi)].sum() / tot)


# 특징 이름 — 접두사가 채널, D1_ 은 AI2 유래(기본 제외 대상)
FEATURES = [
    "AI0_std", "AI0_ptp", "AI0_acf1", "AI0_acf2", "AI0_kurt", "AI0_crest",
    "AI0_bp18", "AI0_bp36",
    "AI1_std", "AI1_ptp", "AI1_acf1", "AI1_acf2", "AI1_kurt", "AI1_crest",
    "AI1_bp18", "AI1_bp36",
    "corr01", "amp_ratio", "corr01_x_amp",
    "D1_AI2_std", "D1_AI2_ptp", "D1_AI2_acf1", "D1_AI2_acf2", "D1_AI2_carrier",
]


def _window_feat(W):
    f = []
    for k in (0, 1):
        x = W[:, k]
        sd = float(x.std())
        xc = x - x.mean()
        kurt = float(((xc / (sd + 1e-12)) ** 4).mean() - 3.0)
        crest = float(np.abs(xc).max() / (sd + 1e-12))
        f += [sd, float(np.ptp(x)), _acf(x, 1), _acf(x, 2), kurt, crest,
              _band(x, 1.5, 2.5), _band(x, 3.0, 4.0)]
    a, b = W[:, 0] - W[:, 0].mean(), W[:, 1] - W[:, 1].mean()
    c01 = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0
    amp = float(W[:, 0].std() / (W[:, 1].std() + 1e-12))     # 극성·게인 불변
    # 공격4 §3.2: 역위상 단독은 정상에서도 나온다. 진폭과의 결합이 판별력
    f += [c01, amp, -c01 * float(W[:, 0].std())]
    x2 = W[:, 2]
    f += [float(x2.std()), float(np.ptp(x2)), _acf(x2, 1), _acf(x2, 2),
          _band(x2, 0.4, 0.9)]                                # 전원 캐리어 대역
    return f


def window_features(df):
    """gap-aware window 별 특징 + window 시작 행 인덱스 + 시작 시각."""
    X = df[COLS].values
    ts = df.TimeStamp.values
    F, T, S = [], [], []
    for a, b in runs(df):
        for i in range(a, b - SEQ + 1):
            F.append(_window_feat(X[i:i + SEQ]))
            T.append(i)
            S.append(ts[i])
    return np.array(F), np.array(T), np.array(S)


# --------------------------------------------------------------------------- #
# 특징집합 — D1 규칙(공격4 §1.1, §2.3)에 따라 AI2 유래는 기본 제외
# --------------------------------------------------------------------------- #
def idx(names):
    return [FEATURES.index(x) for x in names]


FEATURE_SETS = {
    "S0_AMP4":   idx(["AI0_std", "AI0_ptp", "AI1_std", "AI1_ptp"]),
    "S1_VIB":    idx(["AI0_std", "AI0_ptp", "AI0_acf1", "AI0_acf2",
                      "AI1_std", "AI1_ptp", "AI1_acf1", "AI1_acf2",
                      "corr01", "amp_ratio", "corr01_x_amp"]),
    "S2_SHAPE":  idx(["AI0_std", "AI0_ptp", "AI0_kurt", "AI0_crest",
                      "AI1_std", "AI1_ptp", "AI1_kurt", "AI1_crest",
                      "amp_ratio", "corr01_x_amp"]),
    "S3_SPEC":   idx(["AI0_std", "AI0_ptp", "AI0_bp18", "AI0_bp36",
                      "AI1_std", "AI1_ptp", "AI1_bp18", "AI1_bp36",
                      "amp_ratio", "corr01_x_amp"]),
    # 민감도 분석 전용 — D1 위반, 최종모델 후보 아님
    "X_WITH_AI2": idx(["AI0_std", "AI0_ptp", "AI1_std", "AI1_ptp",
                       "D1_AI2_std", "D1_AI2_ptp"]),
    "X_AI2_ONLY": idx(["D1_AI2_acf1", "D1_AI2_carrier"]),
}
D1_VIOLATING = {"X_WITH_AI2", "X_AI2_ONLY"}


def split_masks(T):
    return (T < TRAIN_END,
            (T >= TRAIN_END) & (T < VALID_END),
            T >= VALID_END)


def load_proxy(F):
    """부하 대리변수 (공격4 §4.1). AI2 는 세션 오염이 있으므로 진동으로만 구성."""
    return F[:, FEATURES.index("AI1_std")]


def build():
    """전처리 일괄 수행 후 dict 반환."""
    n, o = load()
    Fn, Tn, Sn = window_features(n)
    Fa, Ta, Sa = window_features(o)
    mtr, mva, mte = split_masks(Tn)
    return dict(normal=n, outlier=o,
                Fn=Fn, Tn=Tn, Sn=Sn, Fa=Fa, Ta=Ta, Sa=Sa,
                tr=Fn[mtr], va=Fn[mva], te=Fn[mte],
                Ttr=Tn[mtr], Tva=Tn[mva], Tte=Tn[mte],
                Ste=Sn[mte], t0=o.TimeStamp.values[0],
                runs_normal=runs(n), runs_outlier=runs(o))

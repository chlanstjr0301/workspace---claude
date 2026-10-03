# -*- coding: utf-8 -*-
"""섭동 시험(E5/E6)과 설명 가능성(§8.3)."""
import numpy as np

CH = {"AI0": 0, "AI1": 1, "AI2": 2}


def perturb(X, kind, amount, train_std=None, rng=None):
    """센서 조건 섭동. 원본을 바꾸지 않는다.

    gain     : 전 채널 이득 변화
    offset   : 전 채널 DC 오프셋 추가 (학습 표준편차 배수)
    polarity : 지정 채널 부호 반전
    jitter   : 샘플 시각 흔들림을 선형보간으로 근사
    """
    Y = X.copy()
    if kind == "gain":
        Y = Y * amount
    elif kind == "offset":
        s = train_std if train_std is not None else X.std(axis=(0, 1))
        Y = Y + (np.asarray(s) * amount)[None, None, :]
    elif kind == "polarity":
        if amount == "all":
            Y = -Y
        else:
            Y[:, :, CH[amount]] *= -1
    elif kind == "jitter":
        # dt=0.1s 기준 amount 초 만큼 시각이 흔들렸을 때의 선형보간 재샘플
        rng = rng or np.random.default_rng(0)
        n, seq, c = Y.shape
        frac = rng.normal(0.0, amount / 0.1, size=(n, seq, 1)).clip(-0.49, 0.49)
        Yp = np.concatenate([Y[:, :1, :], Y[:, :-1, :]], axis=1)
        Yn = np.concatenate([Y[:, 1:, :], Y[:, -1:, :]], axis=1)
        Y = np.where(frac >= 0, Y + frac * (Yn - Y), Y + (-frac) * (Yp - Y))
    else:
        raise ValueError("unknown perturbation: %s" % kind)
    return Y


def top_contributions(contrib, names, k=2):
    """window 별 상위 기여 특징 이름 k개."""
    idx = np.argsort(-np.asarray(contrib), axis=1)[:, :k]
    return [[names[j] for j in row] for row in idx]


def contribution_stability(rank_lists):
    """fold 간 상위 특징 순위 일치도 (§6.1 설명성).
    여러 fold 의 상위 특징 집합 간 Jaccard 평균."""
    if len(rank_lists) < 2:
        return float("nan")
    vals = []
    for i in range(len(rank_lists)):
        for j in range(i + 1, len(rank_lists)):
            a, b = set(rank_lists[i]), set(rank_lists[j])
            vals.append(len(a & b) / max(len(a | b), 1))
    return float(np.mean(vals))


def reason_phrase(name):
    """특징 이름을 현장 점검 문구로 (§9)."""
    if name.startswith("O_") or name.endswith("CUR") or "CUR" in name:
        return "전류 수준/형태 이상 - 전류센서 이득·오프셋·결선 확인"
    if name.startswith("R_"):
        return "상·하부 진동 관계 이상 - 체결·정렬·베어링 유격 점검"
    if name.startswith("S_"):
        return "진동 시간구조 이상 - DAQ 시각 동기화와 샘플링 확인"
    if name.startswith("A_"):
        return "진동 진폭 상승 - 부하·소재·금형 변경 여부 확인"
    if name.startswith("Q_"):
        return "수집 품질 이상 - DAQ 시각/버스트 구조 확인"
    return "원인 미분류 - 원신호 확인"

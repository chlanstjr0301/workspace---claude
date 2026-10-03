# -*- coding: utf-8 -*-
"""
평가 — window / burst / 이벤트 3층, conformal p-value, 블록 부트스트랩

공격4 반영
  §1.1  디바운스 k=5 를 적용한 FAR/h 가 선정 1순위
  §1.2  Lead Time 은 측정 불가 -> 탐지지연(Detection Delay) 으로 교체
  §1.2  Event Recall 은 양성 1건이라 전 후보 1.0 -> 참고값으로만
  §1.4  점추정 금지, 블록(burst) 부트스트랩 CI 병기
  §1.5  conformal 은 '오경보율 통제' 로만 주장. 균등성 점검으로 보정 검증
"""
import numpy as np

DEBOUNCE_K = 5          # 공정 상수로 선언 (0.5 s)
FS = 10.0


# --------------------------------------------------------------------------- #
# 기본 지표
# --------------------------------------------------------------------------- #
def window_metrics(pred_n, pred_a):
    tp = int(pred_a.sum()); fn = len(pred_a) - tp
    fp = int(pred_n.sum()); tn = len(pred_n) - fp
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    mcc_d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_d if mcc_d > 0 else 0.0
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, precision=P, recall=R, f1=F1,
                fpr=fp / (fp + tn) if fp + tn else 0.0, mcc=mcc)


def debounce(pred, k=DEBOUNCE_K):
    """연속 k개가 모두 1인 지점부터 알람 ON. 알람 구간 리스트 [(s, e)] 반환."""
    p = np.asarray(pred).astype(bool)
    if len(p) < k:
        return []
    on = np.zeros(len(p), dtype=bool)
    c = 0
    for i, v in enumerate(p):
        c = c + 1 if v else 0
        if c >= k:
            on[i - k + 1:i + 1] = True
    seg, s = [], None
    for i, v in enumerate(on):
        if v and s is None:
            s = i
        elif not v and s is not None:
            seg.append((s, i)); s = None
    if s is not None:
        seg.append((s, len(on)))
    return seg


def far_per_hour(pred_n, n_rows_normal, k=DEBOUNCE_K):
    """오경보 건수 / 데이터시간(h). 분모는 벽시계가 아니라 rows x 0.1 s (보수적)."""
    hours = n_rows_normal / FS / 3600.0
    alarms = len(debounce(pred_n, k))
    return (alarms / hours if hours > 0 else float("nan")), alarms, hours


def poisson_upper(count, hours, conf=0.95):
    """관측 count 건에 대한 Poisson 상한 (count=0 -> 3.0/hours)."""
    from scipy.stats import chi2
    if hours <= 0:
        return float("nan")
    return chi2.ppf(conf, 2 * (count + 1)) / 2.0 / hours


def detection_delay(pred_a, stamps_a, t0, k=DEBOUNCE_K):
    """최초 알람 시각 - t0 (초). 공격4 §1.2 — 리드타임이 아니라 탐지지연."""
    seg = debounce(pred_a, k)
    if not seg:
        return float("nan")
    first = stamps_a[seg[0][0]]
    return float((np.datetime64(first) - np.datetime64(t0)) / np.timedelta64(1, "s"))


def burst_metrics(pred_n, idx_n, pred_a, idx_a):
    """burst(연속 수집 구간) 1개 = 관측 1건. idx_* 는 window 가 속한 burst id."""
    def agg(pred, idx):
        out = {}
        for p, b in zip(pred, idx):
            out[b] = out.get(b, 0) or int(p)
        return np.array(list(out.values()))
    return window_metrics(agg(pred_n, idx_n), agg(pred_a, idx_a))


# --------------------------------------------------------------------------- #
# conformal
# --------------------------------------------------------------------------- #
def conformal_p(cal_scores, scores):
    """p = (1 + #{s_i >= s}) / (m + 1). 정상 calibration 만 사용."""
    c = np.sort(np.asarray(cal_scores))
    ge = len(c) - np.searchsorted(c, scores, side="left")
    return (1.0 + ge) / (len(c) + 1.0)


def uniformity_ks(p):
    """정상 window 의 p 가 균등분포인지 — KS 통계량. 교환가능성 점검."""
    p = np.sort(np.asarray(p))
    n = len(p)
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(np.max(np.maximum(i / n - p, p - (i - 1) / n)))


# --------------------------------------------------------------------------- #
# 블록 부트스트랩 (burst 단위 재표집)
# --------------------------------------------------------------------------- #
def block_bootstrap_f1(pred_n, idx_n, pred_a, idx_a, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    bn, ba = np.unique(idx_n), np.unique(idx_a)
    gn = {b: pred_n[idx_n == b] for b in bn}
    ga = {b: pred_a[idx_a == b] for b in ba}
    out = np.empty(B)
    for t in range(B):
        sn = np.concatenate([gn[b] for b in rng.choice(bn, len(bn), replace=True)])
        sa = np.concatenate([ga[b] for b in rng.choice(ba, len(ba), replace=True)])
        out[t] = window_metrics(sn, sa)["f1"]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_delta(predA_n, predA_a, predB_n, predB_a, idx_n, idx_a, B=2000, seed=0):
    """대응 블록 부트스트랩으로 F1 차이의 CI 와 P(delta <= 0)."""
    rng = np.random.default_rng(seed)
    bn, ba = np.unique(idx_n), np.unique(idx_a)
    d = np.empty(B)
    for t in range(B):
        cn = rng.choice(bn, len(bn), replace=True)
        ca = rng.choice(ba, len(ba), replace=True)
        mn = np.concatenate([np.where(idx_n == b)[0] for b in cn])
        ma = np.concatenate([np.where(idx_a == b)[0] for b in ca])
        d[t] = (window_metrics(predA_n[mn], predA_a[ma])["f1"]
                - window_metrics(predB_n[mn], predB_a[ma])["f1"])
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float((d <= 0).mean()))


def burst_index(T, run_bounds):
    """window 시작 행 인덱스 -> 소속 burst id."""
    starts = np.array([a for a, _ in run_bounds])
    return np.searchsorted(starts, T, side="right") - 1

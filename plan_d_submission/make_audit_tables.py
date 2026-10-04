# -*- coding: utf-8 -*-
"""제출 전 감사 대응 진단표 (모델·특징·임계값·경보 규칙 불변).

동결 운영 구성(학습 블록 0-2, 보정 블록 3, 1단 M3 / 2단 M1-진동, p<=0.01,
빨강 = 같은 버스트 연속 3 window)을 그대로 다시 만들어 다음을 측정한다.
어떤 결과도 모델 선택이나 임계값 변경에 쓰지 않는다.

  r6  동결 2단 시스템 섭동 시험 (정상 holdout 오경보 / 고장 탐지)
  r7  경보 단계별 사건 수 · 시간당 빈도 · Poisson 95% 상한
  r8  1단 / 2단 / 지속성 효과 분해
  r9  빨강 미탐 원인 분해 + 버스트 단위 탐지율
  r10 사후확률 민감도 (FPR·TPR Clopper-Pearson 구간 포함)
  r11 음성 대조: 1단 M3 를 진동 전용 / 전류 전용 특징으로 적합

    python make_audit_tables.py
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import yaml
from scipy.stats import beta, chi2
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_all as R                      # noqa: E402
from src import calibration as cal      # noqa: E402
from src import data as D               # noqa: E402
from src import evaluation as EV        # noqa: E402
from src import explain as EX           # noqa: E402
from src import features as FT          # noqa: E402
from src import models as MD            # noqa: E402

TAB = os.path.join(HERE, "outputs", "tables")
TRAIN_BLOCKS, CAL_BLOCK, HOLD_BLOCK = [0, 1, 2], 3, 4


def save(df, name):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, name), index=False, encoding="utf-8-sig")
    print("  -> %s (%d행)" % (name, len(df)))


def poisson_upper(k, hours, conf=0.95):
    return chi2.ppf(conf, 2 * (k + 1)) / 2.0 / hours


def cp_interval(k, n, conf=0.95):
    a = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def persist(flag, bid, n):
    """같은 버스트에서 연속 n 이상인 window 만 True (run_all.final_predictions 와 동일)."""
    run = np.zeros(len(flag), dtype=int)
    for i in range(len(flag)):
        if flag[i]:
            run[i] = run[i - 1] + 1 if i > 0 and bid[i] == bid[i - 1] else 1
    return flag & (run >= n)


def confusion(y, pred):
    tp = int((y & pred).sum()); fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum()); tn = int((~y & ~pred).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=p, recall=r,
                f1=2 * p * r / max(p + r, 1e-12), fpr=fp / max(fp + tn, 1))


# --------------------------------------------------------------------------- #
class Frozen:
    """동결 운영 구성 재현. run_all.final_predictions 와 같은 블록·모델·임계값."""

    def __init__(self, cfg, P, stage1_cols=None):
        self.cfg, self.P = cfg, P
        self.seed = cfg["seed"]
        self.idx, _ = R.model_cols(P, tuple(cfg["features"]["model_groups"]))
        names = [P["names"][j] for j in self.idx]
        self.vib = np.array([i for i, n in enumerate(names)
                             if "CUR" not in n and not n.startswith("O_")])
        self.cur = np.array([i for i, n in enumerate(names)
                             if "CUR" in n or n.startswith("O_")])
        mn = P["mn"]
        self.tr = np.isin(mn["block"].values, TRAIN_BLOCKS)
        self.cl = mn["block"].values == CAL_BLOCK
        self.ho = mn["block"].values == HOLD_BLOCK
        self.alpha = cfg["calibration"]["yellow_p"]
        self.red_n = cfg["calibration"]["red_consecutive"]
        s1 = np.arange(len(self.idx)) if stage1_cols is None else stage1_cols
        self.c1 = s1
        self.m1, self.cal1 = self._fit(s1, "M3")
        self.m2, self.cal2 = self._fit(self.vib, "M1")

    def _fit(self, cols, name):
        m = [x for x in MD.build(self.cfg, self.seed) if x.name == name][0]
        F = self.P["Fn"][:, self.idx]
        m.fit(F[self.tr][:, cols])
        return m, m.score(F[self.cl][:, cols])

    def judge(self, F_full, bid):
        F = F_full[:, self.idx]
        p1 = cal.conformal_p(self.cal1, self.m1.score(F[:, self.c1]))
        p2 = cal.conformal_p(self.cal2, self.m2.score(F[:, self.vib]))
        s1, s2 = p1 <= self.alpha, p2 <= self.alpha
        red = persist(s1 & s2, bid, self.red_n)
        return dict(p1=p1, p2=p2, s1=s1, s2=s2, red=red,
                    score1=self.m1.score(F[:, self.c1]))


# --------------------------------------------------------------------------- #
def r6_stress(cfg, P, fz):
    print("R6  동결 2단 시스템 섭동 시험")
    ch = cfg["data"]["channels"]
    groups = tuple(cfg["features"]["groups"])
    rb = cfg["robustness"]
    train_std = P["Xn"].std(axis=(0, 1))
    mn, mo = P["mn"], P["mo"]
    Xh = P["Xn"][fz.ho]
    mh = mn[fz.ho].reset_index(drop=True)
    bh, bo = mh["burst_id"].values, mo["burst_id"].values
    rng = np.random.default_rng(cfg["seed"])

    plans = [("none", "-", None)]
    plans += [("gain_all", a, ("gain", a)) for a in rb["gain"]]
    plans += [("offset_all", a, ("offset", a)) for a in rb["offset_sigma"]]
    plans += [("polarity", a, ("polarity", a)) for a in rb["polarity"]]
    plans += [("jitter", a, ("jitter", a)) for a in rb["jitter_sec"]]
    for c, nm in enumerate(["AI0", "AI1", "AI2"]):
        for a in rb["offset_sigma"]:
            plans.append(("offset_%s_only" % nm, a, ("offset_ch", (c, a))))
        for a in rb["gain"]:
            plans.append(("gain_%s_only" % nm, a, ("gain_ch", (c, a))))

    def apply(X, spec):
        if spec is None:
            return X
        kind, amt = spec
        if kind == "offset_ch":
            c, a = amt
            Y = X.copy(); Y[:, :, c] += a * train_std[c]; return Y
        if kind == "gain_ch":
            c, a = amt
            Y = X.copy(); Y[:, :, c] *= a; return Y
        return EX.perturb(X, kind, amt, train_std, rng)

    base = None
    rows = []
    for name, amt, spec in plans:
        Fh, _, _ = FT.build(apply(Xh, spec), ch, mh, groups)
        Fo, _, _ = FT.build(apply(P["Xo"], spec), ch, mo, groups)
        jh, jo = fz.judge(Fh, bh), fz.judge(Fo, bo)
        row = {"perturbation": name, "amount": str(amt),
               "normal_stage1_rate": jh["s1"].mean(),
               "normal_stage2_rate": jh["s2"].mean(),
               "normal_red_rate": jh["red"].mean(),
               "fault_stage1_recall": jo["s1"].mean(),
               "fault_red_recall": jo["red"].mean()}
        if base is None:
            base = row
        row["normal_red_ratio_vs_none"] = (row["normal_red_rate"] /
                                           max(base["normal_red_rate"], 1e-12))
        rows.append(row)
    df = pd.DataFrame(rows)
    save(df, "r6_frozen_system_stress.csv")
    return df


def holdout_exposure_hours(P, fz):
    mh = P["mn"][fz.ho].reset_index(drop=True)
    return EV.false_alarm_profile(np.zeros(len(mh), bool), mh)["exposure_hours"]


def r7_event_rates(P, pred, fz):
    print("R7  경보 단계별 사건 수·시간당 빈도")
    h = pred[pred.split == "holdout"].reset_index(drop=True)
    hours = holdout_exposure_hours(P, fz)
    rows = []
    for name, flag in (("1단 경보(노랑+빨강)", h.alarm_level != "green"),
                       ("노랑", h.alarm_level == "yellow"),
                       ("빨강", h.alarm_level == "red")):
        n_ev, _ = EV.merge_alarms(flag.values, h["burst_id"].values)
        rows.append({"경보": name, "window": int(flag.sum()),
                     "window 비율": float(flag.mean()),
                     "경보 사건": n_ev,
                     "알람 burst 수": int(h[flag].burst_id.nunique()),
                     "전체 burst 수": int(h.burst_id.nunique()),
                     "관측시간(h)": hours,
                     "사건/h": n_ev / hours,
                     "사건/h 95% 상한": poisson_upper(n_ev, hours)})
    df = pd.DataFrame(rows)
    save(df, "r7_alarm_event_rates.csv")
    return df


def r8_decomposition(P, pred, fz):
    print("R8  1단·2단·지속성 효과 분해")
    op = pred[pred.split.isin(["holdout", "fault"])].reset_index(drop=True)
    y = (op.split == "fault").values
    bid = (op.source.astype(str) + ":" + op.burst_id.astype(str)).values
    s1 = (op.p_normal_stage1 <= fz.alpha).values
    s2 = (op.p_normal_stage2 <= fz.alpha).values
    hours = holdout_exposure_hours(P, fz)
    rules = [("1단 단독", s1),
             ("2단 단독(진동 M1)", s2),
             ("1단 + 지속성3", persist(s1, bid, fz.red_n)),
             ("2단 + 지속성3", persist(s2, bid, fz.red_n)),
             ("1단 AND 2단", s1 & s2),
             ("1단 AND 2단 + 지속성3 (=빨강)", persist(s1 & s2, bid, fz.red_n))]
    red_file = (op.alarm_level == "red").values
    rows = []
    for name, pr in rules:
        m = confusion(y, pr)
        n_ev, _ = EV.merge_alarms(pr[~y], bid[~y])
        rows.append({"규칙": name, **m,
                     "정상 경보 사건": n_ev, "사건/h": n_ev / hours})
    df = pd.DataFrame(rows)
    assert np.array_equal(rules[-1][1], red_file), "빨강 재구성이 예측파일과 다름"
    base = df.loc[0, "fpr"]
    df["1단 대비 FPR 배수"] = base / df["fpr"].replace(0, np.nan)
    save(df, "r8_stage_decomposition.csv")
    return df


def r9_fn(pred, fz):
    print("R9  빨강 미탐 원인 분해")
    f = pred[pred.split == "fault"].reset_index(drop=True)
    f["pos"] = f.groupby("burst_id").cumcount()
    f["n_win"] = f.groupby("burst_id").burst_id.transform("size")
    fn = f[f.alarm_level != "red"].copy()

    def cause(r):
        if r.p_normal_stage1 > fz.alpha:
            return "1단 미검출"
        if r.p_normal_stage2 > fz.alpha:
            return "2단(진동) 미확인"
        if r.n_win < fz.red_n:
            return "버스트 window < 3"
        if r.pos < fz.red_n - 1:
            return "버스트 첫 2 window(지속성 구조)"
        return "연속 3 미충족(중간 끊김)"
    fn["원인"] = fn.apply(cause, axis=1)
    by = fn.groupby("원인").size().reset_index(name="미탐 window")
    by["비중"] = by["미탐 window"] / len(fn)
    save(by, "r9_red_fn_causes.csv")

    bur = f.groupby("burst_id").agg(
        windows=("pos", "size"),
        stage1_detected=("alarm_level", lambda s: bool((s != "green").any())),
        red_detected=("alarm_level", lambda s: bool((s == "red").any())),
        red_windows=("alarm_level", lambda s: int((s == "red").sum()))).reset_index()
    save(bur, "r9_burst_detection.csv")
    print("     빨강 burst 탐지 %d/%d" % (bur.red_detected.sum(), len(bur)))
    return by, bur


def r10_posterior(pred):
    print("R10 사후확률 (구간 포함)")
    op = pred[pred.split.isin(["holdout", "fault"])]
    y = (op.split == "fault").values
    red = (op.alarm_level == "red").values
    tp = int((y & red).sum()); npos = int(y.sum())
    fp = int((~y & red).sum()); nneg = int((~y).sum())
    tpr, fpr = tp / npos, fp / nneg
    tpr_lo, _ = cp_interval(tp, npos)
    _, fpr_hi = cp_interval(fp, nneg)
    rows = []
    for pi in (0.01, 0.001, 0.0001):
        post = lambda t, f: t * pi / (t * pi + f * (1 - pi))
        rows.append({"가정 고장유병률(window당)": pi,
                     "빨강 TPR": tpr, "빨강 FPR": fpr,
                     "TPR 95% 하한(CP)": tpr_lo, "FPR 95% 상한(CP)": fpr_hi,
                     "P(고장|빨강) 점추정": post(tpr, fpr),
                     "P(고장|빨강) 보수(하한 TPR·상한 FPR)": post(tpr_lo, fpr_hi)})
    df = pd.DataFrame(rows)
    save(df, "r10_posterior_bounds.csv")
    return df


def r11_negative_control(cfg, P, fz_base):
    print("R11 음성 대조 — 1단 특징 채널 구성")
    mn, mo = P["mn"], P["mo"]
    Fh, Fo = P["Fn"][fz_base.ho], P["Fo"]
    bh, bo = mn[fz_base.ho]["burst_id"].values, mo["burst_id"].values
    hours = holdout_exposure_hours(P, fz_base)
    rows = []
    for name, cols in (("전체 25 (제출본)", None),
                       ("진동 전용 (전류·O 제외)", fz_base.vib),
                       ("전류 전용 (CUR·O)", fz_base.cur)):
        fz = fz_base if cols is None else Frozen(cfg, P, stage1_cols=cols)
        jh, jo = fz.judge(Fh, bh), fz.judge(Fo, bo)
        y = np.r_[np.zeros(len(Fh), bool), np.ones(len(Fo), bool)]
        sc = np.r_[jh["score1"], jo["score1"]]
        s1 = np.r_[jh["s1"], jo["s1"]]
        red = np.r_[jh["red"], jo["red"]]
        m1, mr = confusion(y, s1), confusion(y, red)
        ev1, _ = EV.merge_alarms(jh["s1"], bh)
        evr, _ = EV.merge_alarms(jh["red"], bh)
        rows.append({"1단 특징": name, "특징 수": len(fz.c1),
                     "1단 AUROC": roc_auc_score(y, sc),
                     "1단 AP": average_precision_score(y, sc),
                     "1단 정상 경보율": m1["fpr"], "1단 Recall": m1["recall"],
                     "1단 정상 사건/h": ev1 / hours,
                     "빨강 정상 경보율": mr["fpr"], "빨강 Recall": mr["recall"],
                     "빨강 F1": mr["f1"], "빨강 정상 사건/h": evr / hours})
    df = pd.DataFrame(rows)
    save(df, "r11_negative_control_channels.csv")
    return df


def main():
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
    normal, outlier, _ = D.load_all(cfg, HERE)
    P = R.prepare(cfg, normal, outlier, cfg["windows"]["length"])
    pred = pd.read_csv(os.path.join(HERE, "outputs", "predictions.csv"))
    fz = Frozen(cfg, P)

    # 동결 구성 재현 검증: 예측파일의 p-value 와 일치해야 한다.
    allF = np.vstack([P["Fn"], P["Fo"]])
    bid = np.r_[P["mn"]["burst_id"].values, P["mo"]["burst_id"].values]
    j = fz.judge(allF, bid)
    assert np.allclose(j["p1"], pred.p_normal_stage1.values)
    assert np.allclose(j["p2"], pred.p_normal_stage2.values)
    print("동결 구성 재현 확인: p1·p2 가 predictions.csv 와 일치")

    r6_stress(cfg, P, fz)
    r7_event_rates(P, pred, fz)
    r8_decomposition(P, pred, fz)
    r9_fn(pred, fz)
    r10_posterior(pred)
    r11_negative_control(cfg, P, fz)
    print("완료")


if __name__ == "__main__":
    main()

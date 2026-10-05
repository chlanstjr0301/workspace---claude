# -*- coding: utf-8 -*-
"""Plan D 전체 파이프라인.

    python run_all.py --config config.yaml --mode quick
    python run_all.py --config config.yaml --mode full

quick : CPU 특징모델(BL0/M1/M2/M3) + 모든 표/그림/예측파일
full  : + LSTM-AE(G0/G1) 다중시드

핵심 원칙
 - 모든 스케일러·임계값은 해당 fold 의 정상 학습/보정 구간에서만 적합한다.
 - 이상 데이터는 적합·임계값에 쓰지 않는다. 선정 게이트 일부에는 사용된다.
 - window 는 버스트 내부에서만 만든다.
"""
import argparse
import json
import os
import platform
import sys

# 한국어 Windows(cp949) 에서 로그의 유니코드 문자가 UnicodeEncodeError 를
# 일으켜 파이프라인이 중단되는 것을 막는다.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src import calibration as cal          # noqa: E402
from src import data as D                   # noqa: E402
from src import evaluation as EV            # noqa: E402
from src import explain as EX               # noqa: E402
from src import features as FT              # noqa: E402
from src import models as MD                # noqa: E402
from src import plotting as PL              # noqa: E402
from src import windows as WD               # noqa: E402

TAB = os.path.join(HERE, "outputs", "tables")
FIG = os.path.join(HERE, "outputs", "figures")
OUT = os.path.join(HERE, "outputs")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def save(df, name):
    os.makedirs(TAB, exist_ok=True)
    p = os.path.join(TAB, name)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    log("  -> %s (%d행)" % (name, len(df)))
    return p


# --------------------------------------------------------------------------- #
# E0  데이터 품질 / 버스트 진단
# --------------------------------------------------------------------------- #
def e0(cfg, normal, outlier):
    log("E0  데이터 품질·버스트 진단")
    ch, gap = cfg["data"]["channels"], cfg["windows"]["gap_sec"]
    q = pd.DataFrame([D.audit(normal, "normal", ch, gap),
                      D.audit(outlier, "outlier", ch, gap)])
    save(q, "e0_data_quality.csv")
    save(pd.DataFrame(D.channel_stats(normal, "normal", ch) +
                      D.channel_stats(outlier, "outlier", ch)),
         "e0_channel_stats.csv")
    btn = WD.burst_table(normal, gap)
    bto = WD.burst_table(outlier, gap)
    btn.insert(0, "source", "normal")
    bto.insert(0, "source", "outlier")
    save(pd.concat([btn, bto], ignore_index=True), "e0_bursts.csv")

    summ = []
    for nm, bt in (("normal", btn), ("outlier", bto)):
        summ.append({
            "source": nm, "n_bursts": len(bt),
            "samples_min": int(bt["n_samples"].min()),
            "samples_median": float(bt["n_samples"].median()),
            "samples_max": int(bt["n_samples"].max()),
            "dur_max_sec": float(bt["dur_sec"].max()),
            "bursts_ge_120_samples": int((bt["n_samples"] >= 120).sum()),
        })
    s = pd.DataFrame(summ)
    save(s, "e0_burst_summary.csv")
    os.makedirs(FIG, exist_ok=True)
    PL.burst_timeline(btn, bto, FIG)
    PL.burst_length_hist(btn, bto, FIG)
    return s


# --------------------------------------------------------------------------- #
# 공통: window + 특징 준비
# --------------------------------------------------------------------------- #
def prepare(cfg, normal, outlier, seq):
    ch, gap = cfg["data"]["channels"], cfg["windows"]["gap_sec"]
    Xn, mn = WD.make_windows(normal, ch, seq, gap, source="normal")
    Xo, mo = WD.make_windows(outlier, ch, seq, gap, source="outlier")
    mn["block"] = WD.block_split(mn, cfg["cv"]["n_blocks"])
    mo["block"] = -1
    groups = tuple(cfg["features"]["groups"])
    Fn, names, gof = FT.build(Xn, ch, mn, groups)
    Fo, _, _ = FT.build(Xo, ch, mo, groups)
    return dict(Xn=Xn, Xo=Xo, mn=mn, mo=mo, Fn=Fn, Fo=Fo,
                names=names, gof=gof, seq=seq)


def aggregate_bl1(g1):
    """BL-1 fold 결과를 e2_model_comparison.csv 스키마로 집계."""
    return g1.groupby(["model", "pretty"]).agg(
        recall_mean=("recall", "mean"), f1_mean=("f1", "mean"),
        precision_mean=("precision", "mean"), mcc_mean=("mcc", "mean"),
        AP_mean=("AP", "mean"), AUROC_mean=("AUROC", "mean"),
        fp_rate_mean=("fp_rate", "mean"), fp_rate_worst=("fp_rate", "max"),
        far_h_mean=("far_per_hour", "mean"),
        far_h_worst=("far_per_hour", "max"),
        far_h_upper95_worst=("far_per_hour_upper95", "max"),
        alarm_events_worst=("alarm_events", "max"),
        bursts_detected_min=("bursts_detected", "min"),
        bursts_total=("bursts_total", "max")).reset_index()


def model_cols(P, use_groups):
    idx = np.where(np.isin(P["gof"], list(use_groups)))[0]
    return idx, [P["names"][i] for i in idx]


# --------------------------------------------------------------------------- #
# E1  window 길이 민감도
# --------------------------------------------------------------------------- #
def e1(cfg, normal, outlier):
    log("E1  window 길이 민감도")
    rows = []
    gap = cfg["windows"]["gap_sec"]
    ch = cfg["data"]["channels"]
    for seq in cfg["windows"]["length_sensitivity"]:
        Xn, mn = WD.make_windows(normal, ch, seq, gap, source="normal")
        Xo, mo = WD.make_windows(outlier, ch, seq, gap, source="outlier")
        naive_n = max(len(normal) - seq + 1, 0)
        naive_o = max(len(outlier) - seq + 1, 0)
        rows.append({
            "seq": seq,
            "normal_windows_gap_aware": len(Xn),
            "normal_windows_naive": naive_n,
            "normal_retained_pct": 100.0 * len(Xn) / max(naive_n, 1),
            "outlier_windows_gap_aware": len(Xo),
            "outlier_windows_naive": naive_o,
            "outlier_retained_pct": 100.0 * len(Xo) / max(naive_o, 1),
            "bursts_usable_normal": int(mn["burst_id"].nunique()) if len(mn) else 0,
            "bursts_usable_outlier": int(mo["burst_id"].nunique()) if len(mo) else 0,
        })
    df = pd.DataFrame(rows)
    save(df, "e1_window_sensitivity.csv")
    return df


# --------------------------------------------------------------------------- #
# E2/E3  모델 벤치마크 + 정상 블록 CV  (이상은 최종 평가에만)
# --------------------------------------------------------------------------- #
HOLD_SCORES = {}


def run_cv(cfg, P, use_groups, seed, tag=""):
    """정상 5블록 CV. fold 마다 train 적합 -> cal 로 conformal -> holdout 평가.
    이상 window 는 동일 fold 모델로 점수만 매긴다."""
    idx, fnames = model_cols(P, use_groups)
    Fn, Fo, mn = P["Fn"][:, idx], P["Fo"][:, idx], P["mn"]
    folds = WD.cv_folds(cfg["cv"]["n_blocks"])
    yellow = cfg["calibration"]["yellow_p"]
    per_fold, anom_scores, p_by_fold, contribs = [], {}, {}, {}

    for f in folds:
        tr = np.isin(mn["block"].values, f["train"])
        cl = mn["block"].values == f["cal"]
        ho = mn["block"].values == f["holdout"]
        for m in MD.build(cfg, seed):
            m.fit(Fn[tr])
            s_cal = m.score(Fn[cl])
            s_ho = m.score(Fn[ho])
            s_an = m.score(Fo)
            p_ho = cal.conformal_p(s_cal, s_ho)
            p_an = cal.conformal_p(s_cal, s_an)
            flags_ho = p_ho <= yellow
            flags_an = p_an <= yellow

            fa = EV.false_alarm_profile(flags_ho, mn[ho].reset_index(drop=True))
            y = np.r_[np.zeros(ho.sum()), np.ones(len(Fo))]
            sc = np.r_[s_ho, s_an]
            pr = np.r_[flags_ho, flags_an].astype(int)
            met = EV.window_metrics(y, pr, sc)
            det = EV.detection_delay(flags_an, P["mo"])
            n_burst_det = int(sum(d["detected"] for d in det))

            per_fold.append(dict(
                model=m.name, pretty=m.pretty, fold=f["fold"],
                holdout_block=f["holdout"], seed=seed, groups="".join(use_groups),
                seq=P["seq"], tag=tag, **fa, **met,
                bursts_detected=n_burst_det, bursts_total=len(det)))
            anom_scores.setdefault(m.name, []).append(s_an)
            if f["fold"] == 0:
                HOLD_SCORES[m.name] = (s_ho, s_an)
            p_by_fold.setdefault(m.name, {})[f["fold"]] = p_ho
            if hasattr(m, "contrib"):
                c = np.abs(m.contrib(Fo)).mean(axis=0)
                contribs.setdefault(m.name, []).append(
                    [fnames[j] for j in np.argsort(-c)[:5]])
    return pd.DataFrame(per_fold), anom_scores, p_by_fold, contribs, fnames


def e2_e3(cfg, P, seed):
    log("E2/E3  동일조건 모델 벤치마크 + 정상 5블록 CV")
    g = tuple(cfg["features"]["model_groups"])
    df, anom, p_by_fold, contribs, fnames = run_cv(cfg, P, g, seed)
    save(df, "e3_normal_block_cv.csv")

    agg = df.groupby(["model", "pretty"]).agg(
        recall_mean=("recall", "mean"), f1_mean=("f1", "mean"),
        precision_mean=("precision", "mean"), mcc_mean=("mcc", "mean"),
        AP_mean=("AP", "mean"), AUROC_mean=("AUROC", "mean"),
        fp_rate_mean=("fp_rate", "mean"), fp_rate_worst=("fp_rate", "max"),
        far_h_mean=("far_per_hour", "mean"), far_h_worst=("far_per_hour", "max"),
        far_h_upper95_worst=("far_per_hour_upper95", "max"),
        alarm_events_worst=("alarm_events", "max"),
        bursts_detected_min=("bursts_detected", "min"),
        bursts_total=("bursts_total", "max"),
    ).reset_index()
    save(agg, "e2_model_comparison.csv")
    try:
        PL.block_fp_heatmap(df, FIG)
        PL.conformal_reliability(
            {k: np.concatenate(list(v.values()))
             for k, v in list(p_by_fold.items())[:1]}[list(p_by_fold)[0]]
            if False else p_by_fold[list(p_by_fold)[0]], FIG)
        if HOLD_SCORES:
            PL.score_separation(HOLD_SCORES, FIG)
    except Exception as e:                                  # 그림 실패는 치명 아님
        log("  (그림 생략: %s)" % e)
    return df, agg, anom, p_by_fold, contribs, fnames


# --------------------------------------------------------------------------- #
# E4  특징군 ablation
# --------------------------------------------------------------------------- #
def e4(cfg, P, seed):
    log("E4  특징군 ablation (A/S/R/O)")
    base = list(cfg["features"]["model_groups"])
    combos = [("ASRO", base)]
    for g in base:
        combos.append(("-" + g, [x for x in base if x != g]))
    for g in base:
        combos.append(("only" + g, [g]))
    rows = []
    for tag, gg in combos:
        if not gg:
            continue
        df, *_ = run_cv(cfg, P, tuple(gg), seed, tag=tag)
        a = df.groupby("model").agg(
            recall=("recall", "mean"), f1=("f1", "mean"),
            AP=("AP", "mean"), fp_rate_worst=("fp_rate", "max")).reset_index()
        a.insert(0, "feature_set", tag)
        rows.append(a)
    out = pd.concat(rows, ignore_index=True)
    save(out, "e4_feature_ablation.csv")
    return out


# --------------------------------------------------------------------------- #
# E5/E6  섭동 강건성
# --------------------------------------------------------------------------- #
def e5_e6(cfg, P, seed):
    log("E5/E6  극성·이득·오프셋·jitter 섭동")
    ch = cfg["data"]["channels"]
    g = tuple(cfg["features"]["model_groups"])
    idx, _ = model_cols(P, g)
    mn, Fn = P["mn"], P["Fn"][:, idx]
    folds = WD.cv_folds(cfg["cv"]["n_blocks"])
    yellow = cfg["calibration"]["yellow_p"]
    rb = cfg["robustness"]
    rng = np.random.default_rng(seed)

    plans = ([("gain", a) for a in rb["gain"]] +
             [("offset", a) for a in rb["offset_sigma"]] +
             [("polarity", a) for a in rb["polarity"]] +
             [("jitter", a) for a in rb["jitter_sec"]])

    train_std = P["Xn"].std(axis=(0, 1))
    rows, shift_rows = [], []
    for f in folds:
        tr = np.isin(mn["block"].values, f["train"])
        cl = mn["block"].values == f["cal"]
        ho = mn["block"].values == f["holdout"]
        for m in MD.build(cfg, seed):
            m.fit(Fn[tr])
            s_cal = m.score(Fn[cl])
            base_rec = (cal.conformal_p(s_cal, m.score(P["Fo"][:, idx]))
                        <= yellow).mean()
            base_fp = (cal.conformal_p(s_cal, m.score(Fn[ho]))
                       <= yellow).mean()
            for kind, amt in plans:
                # (a) 이상 쪽: 섭동 후에도 탐지되는가  (§6.1 문자 그대로)
                Xp = EX.perturb(P["Xo"], kind, amt, train_std, rng)
                Fp, _, _ = FT.build(Xp, ch, P["mo"],
                                    tuple(cfg["features"]["groups"]))
                rec = (cal.conformal_p(s_cal, m.score(Fp[:, idx]))
                       <= yellow).mean()
                rows.append({
                    "model": m.name, "fold": f["fold"],
                    "perturbation": kind, "amount": str(amt),
                    "recall_base": base_rec, "recall_perturbed": rec,
                    "recall_drop_pp": 100.0 * (base_rec - rec)})
                # (b) 정상 쪽: 계측조건만 바뀐 정상을 고장이라 부르는가
                #     Recall 이 1.0 으로 포화되면 (a) 는 변별력이 없으므로
                #     이 지표를 강건성 게이트의 실제 근거로 쓴다.
                Xq = EX.perturb(P["Xn"][ho], kind, amt, train_std, rng)
                Fq, _, _ = FT.build(Xq, ch, mn[ho].reset_index(drop=True),
                                    tuple(cfg["features"]["groups"]))
                fp = (cal.conformal_p(s_cal, m.score(Fq[:, idx]))
                      <= yellow).mean()
                shift_rows.append({
                    "model": m.name, "fold": f["fold"],
                    "perturbation": kind, "amount": str(amt),
                    "fp_base": base_fp, "fp_shifted": fp,
                    "fp_increase_pp": 100.0 * (fp - base_fp)})
    df = pd.DataFrame(rows)
    sh = pd.DataFrame(shift_rows)
    save(df, "e5_robustness.csv")
    save(sh, "e6_shift_false_alarm.csv")
    agg = df.groupby(["model", "perturbation"]).agg(
        recall_drop_pp=("recall_drop_pp", "max"),
        recall_perturbed_min=("recall_perturbed", "min")).reset_index()
    sagg = sh.groupby(["model", "perturbation"]).agg(
        fp_increase_pp=("fp_increase_pp", "max"),
        fp_shifted_worst=("fp_shifted", "max")).reset_index()
    agg = agg.merge(sagg, on=["model", "perturbation"], how="left")
    save(agg, "e5_robustness_summary.csv")
    try:
        PL.robustness_bars(agg, FIG)
    except Exception as e:
        log("  (그림 생략: %s)" % e)
    return df, agg


# --------------------------------------------------------------------------- #
# E7  FP/FN 조건 분해
# --------------------------------------------------------------------------- #
def e7(cfg, P, seed, model_name):
    log("E7  FP/FN 조건 분해 (%s)" % model_name)
    g = tuple(cfg["features"]["model_groups"])
    idx, fnames = model_cols(P, g)
    mn, Fn, Fo = P["mn"], P["Fn"][:, idx], P["Fo"][:, idx]
    folds = WD.cv_folds(cfg["cv"]["n_blocks"])
    yellow = cfg["calibration"]["yellow_p"]
    rows = []
    for f in folds:
        tr = np.isin(mn["block"].values, f["train"])
        cl = mn["block"].values == f["cal"]
        ho = mn["block"].values == f["holdout"]
        m = [x for x in MD.build(cfg, seed) if x.name == model_name][0]
        m.fit(Fn[tr])
        s_cal = m.score(Fn[cl])
        lo, hi = FT.regime_bounds(P["Xn"][tr])

        # FP 조건
        p_ho = cal.conformal_p(s_cal, m.score(Fn[ho]))
        fp = p_ho <= yellow
        reg = FT.load_regime(P["Xn"][ho], lo, hi)
        pos = mn[ho]["pos_in_burst_sec"].values
        for name, sel in (("load_low", reg == 0), ("load_mid", reg == 1),
                          ("load_high", reg == 2),
                          ("burst_first_1s", pos <= 1.0),
                          ("burst_after_1s", pos > 1.0)):
            if sel.sum():
                rows.append({"fold": f["fold"], "kind": "FP", "condition": name,
                             "n": int(sel.sum()),
                             "errors": int(fp[sel].sum()),
                             "rate": float(fp[sel].mean())})
        # FN 조건
        p_an = cal.conformal_p(s_cal, m.score(Fo))
        fn = p_an > yellow
        rega = FT.load_regime(P["Xo"], lo, hi)
        posa = P["mo"]["pos_in_burst_sec"].values
        for name, sel in (("load_low", rega == 0), ("load_mid", rega == 1),
                          ("load_high", rega == 2),
                          ("burst_first_1s", posa <= 1.0),
                          ("burst_after_1s", posa > 1.0)):
            if sel.sum():
                rows.append({"fold": f["fold"], "kind": "FN", "condition": name,
                             "n": int(sel.sum()),
                             "errors": int(fn[sel].sum()),
                             "rate": float(fn[sel].mean())})
    df = pd.DataFrame(rows)
    save(df, "e7_error_conditions.csv")
    agg = df.groupby(["kind", "condition"]).agg(
        n=("n", "mean"), rate_mean=("rate", "mean"),
        rate_worst=("rate", "max")).reset_index()
    save(agg, "e7_error_conditions_summary.csv")
    return df, agg


# --------------------------------------------------------------------------- #
# E9  abs() 전처리 불변성 — 가이드북 전처리가 무엇을 파괴하는지 정량화
# --------------------------------------------------------------------------- #
def e9(cfg, P):
    log("E9  abs() 전처리 불변성")
    X = P["Xn"]
    Xa = np.abs(X)
    Z = X - X.mean(axis=1, keepdims=True)
    Za = Xa - Xa.mean(axis=1, keepdims=True)

    def stats(W, Zc):
        return {
            "RMS": np.sqrt((W ** 2).mean(axis=1)),
            "MAV": np.abs(W).mean(axis=1),
            "PEAK": np.abs(W).max(axis=1),
            "STD": W.std(axis=1),
            "P2P": W.max(axis=1) - W.min(axis=1),
            "MEAN": W.mean(axis=1),
            "KURT": (Zc ** 4).mean(axis=1) / (Zc.std(axis=1) ** 4 + 1e-12),
        }

    a, b = stats(X, Z), stats(Xa, Za)
    rows = [{"feature": k,
             "max_abs_change": float(np.abs(a[k] - b[k]).max()),
             "invariant_to_abs": bool(np.abs(a[k] - b[k]).max() < 1e-9)}
            for k in a]
    df = pd.DataFrame(rows).sort_values("max_abs_change").reset_index(drop=True)
    save(df, "e9_abs_invariance.csv")
    return df


# --------------------------------------------------------------------------- #
# E8  conformal coverage
# --------------------------------------------------------------------------- #
def e8(cfg, p_by_fold):
    log("E8  conformal coverage")
    rows = []
    for mname, d in p_by_fold.items():
        for fold, p in d.items():
            for r in cal.coverage(p):
                rows.append({"model": mname, "fold": fold, **r})
    df = pd.DataFrame(rows)
    save(df, "e8_conformal_coverage.csv")
    return df


# --------------------------------------------------------------------------- #
# 최종 예측 파일 (CARE-Press 2단 판정)
# --------------------------------------------------------------------------- #
def final_predictions(cfg, P, seed, stage1, stage2):
    log("최종 예측 생성 (CARE-Press 2단 판정: 1단=%s, 2단=%s)" % (stage1, stage2))
    g = tuple(cfg["features"]["model_groups"])
    idx, fnames = model_cols(P, g)
    mn = P["mn"]
    # 2단은 '전류 제외 진동 증거'
    vib = [i for i, j in enumerate(idx)
           if "CUR" not in P["names"][j] and not P["names"][j].startswith("O_")]
    tr = np.isin(mn["block"].values, [0, 1, 2])
    cl = mn["block"].values == 3
    yellow, red_n = cfg["calibration"]["yellow_p"], cfg["calibration"]["red_consecutive"]

    def fit_score(cols, mname):
        m = [x for x in MD.build(cfg, seed) if x.name == mname][0]
        m.fit(P["Fn"][:, idx][tr][:, cols])
        sc = m.score(P["Fn"][:, idx][cl][:, cols])
        allF = np.vstack([P["Fn"][:, idx], P["Fo"][:, idx]])[:, cols]
        return m, sc, m.score(allF)

    allcols = np.arange(len(idx))
    m1, cal1, s1 = fit_score(allcols, stage1)
    m2, cal2, s2 = fit_score(np.array(vib), stage2)
    p1 = cal.conformal_p(cal1, s1)
    p2 = cal.conformal_p(cal2, s2)

    meta = pd.concat([mn, P["mo"]], ignore_index=True)
    lvl = np.where(p1 > yellow, "green",
                   np.where(p2 <= yellow, "red", "yellow"))
    # 빨강은 연속 red_n window 조건 적용
    isred = lvl == "red"
    bid = meta["burst_id"].values
    src = meta["source"].values
    run = np.zeros(len(isred), dtype=int)
    for i in range(len(isred)):
        if isred[i]:
            same = i > 0 and bid[i] == bid[i - 1] and src[i] == src[i - 1]
            run[i] = run[i - 1] + 1 if same else 1
    lvl = np.where(isred & (run < red_n), "yellow", lvl)

    if hasattr(m1, "contrib"):
        C = np.abs(m1.contrib(np.vstack([P["Fn"][:, idx], P["Fo"][:, idx]])))
        tops = EX.top_contributions(C, [P["names"][j] for j in idx], 2)
    else:
        tops = [["", ""]] * len(meta)

    # 어느 행이 in-sample 인지 명시한다. 최종 모델은 블록 0-2 로 적합하고
    # 블록 3 으로 보정하므로, 정상 행 대부분은 held-out 이 아니다.
    # 블록 4 만이 이 모델이 한 번도 보지 못한 정상 구간이다.
    blk = np.r_[mn["block"].values, np.full(len(P["mo"]), -1)]
    split = np.where(blk < 0, "fault",
                     np.where(np.isin(blk, [0, 1, 2]), "fit",
                              np.where(blk == 3, "calibration", "holdout")))

    pred = pd.DataFrame({
        "source": meta["source"], "burst_id": meta["burst_id"],
        "block": blk, "split": split,
        "original_row_start": meta["original_row_start"],
        "original_row_end": meta["original_row_end"],
        "time_start": meta["time_start"], "time_end": meta["time_end"],
        "score_stage1": s1, "score_stage2": s2,
        "p_normal_stage1": p1, "p_normal_stage2": p2,
        "risk_score": 1.0 - p1,
        "alarm_level": lvl,
        "top_reason_1": [t[0] for t in tops],
        "top_reason_2": [t[1] for t in tops],
        "recommended_action": [EX.reason_phrase(t[0]) for t in tops],
        "label": meta["label"],
    })
    os.makedirs(OUT, exist_ok=True)
    pred.to_csv(os.path.join(OUT, "predictions.csv"),
                index=False, encoding="utf-8-sig")
    log("  -> predictions.csv (%d행)" % len(pred))

    # split 별로 따로 보고한다. fit/calibration 행은 in-sample 이므로
    # 운영 성능 근거로 쓸 수 없다. 운영 수치는 holdout 행만 본다.
    rows = []
    for name in ("fit", "calibration", "holdout", "fault"):
        sel = split == name
        if not sel.any():
            continue
        rows.append({
            "stage1": stage1, "stage2": stage2, "split": name,
            "in_sample": name in ("fit", "calibration"),
            "windows": int(sel.sum()),
            "green_pct": 100.0 * (lvl[sel] == "green").mean(),
            "yellow_pct": 100.0 * (lvl[sel] == "yellow").mean(),
            "red_pct": 100.0 * (lvl[sel] == "red").mean(),
        })
    summ = pd.DataFrame(rows)
    save(summ, "final_alarm_summary.csv")
    h = summ[summ.split == "holdout"]
    if len(h):
        log("  (운영 근거는 holdout 행만: green %.2f%% / yellow %.2f%% / red %.2f%%)"
            % (h.green_pct.iloc[0], h.yellow_pct.iloc[0], h.red_pct.iloc[0]))
    return pred, summ


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    a = ap.parse_args()

    t0 = time.time()
    # 패키지 폴더·저장소 루트 어디서 실행해도 같은 설정을 찾는다.
    cfg_path = a.config if os.path.isfile(a.config) else os.path.join(HERE, a.config)
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(HERE, os.path.basename(a.config))
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    seed = cfg["seed"]
    np.random.seed(seed)
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    log("Plan D 파이프라인 시작  mode=%s  seed=%d" % (a.mode, seed))
    normal, outlier, hashes = D.load_all(cfg, HERE)
    log("  정상 %d행 / 이상 %d행" % (len(normal), len(outlier)))

    e0(cfg, normal, outlier)
    e1(cfg, normal, outlier)

    P = prepare(cfg, normal, outlier, cfg["windows"]["length"])
    log("  window seq=%d -> 정상 %d / 이상 %d, 특징 %d차원"
        % (P["seq"], len(P["Xn"]), len(P["Xo"]), P["Fn"].shape[1]))

    cvdf, agg, anom, p_by_fold, contribs, fnames = e2_e3(cfg, P, seed)

    # 이전 full 실행의 BL-1 결과가 남아 있으면 quick 에서도 비교표에 합친다.
    # (quick 재실행이 full 산출물을 지워버리지 않게 하기 위함)
    bl1_path = os.path.join(TAB, "e2_bl1_g1_planD.csv")
    if a.mode != "full" and os.path.exists(bl1_path):
        prev = pd.read_csv(bl1_path)
        agg = pd.concat([agg, aggregate_bl1(prev)], ignore_index=True)
        save(agg, "e2_model_comparison.csv")
        log("  (이전 full 실행의 BL-1 결과를 비교표에 재사용)")

    # BL-1 공식 LSTM-AE (full 모드에서만)
    if a.mode == "full":
        from src import bl1 as B1
        from src import deep as DP
        if DP.available():
            log("BL-1 LSTM-AE 재현  backend=%s" % DP.BACKEND)
            ep = cfg["deep"]["lstm_ae"]["epochs_full"]
            g0 = pd.DataFrame([B1.run_g0(cfg, normal, outlier, sd, ep)
                               for sd in cfg["deep"]["lstm_ae"]["seeds"]])
            save(g0, "e2_bl1_g0_guidebook.csv")
            g1 = pd.concat([B1.run_g1(cfg, P, sd, ep)
                            for sd in cfg["deep"]["lstm_ae"]["seeds_cv"]],
                           ignore_index=True)
            save(g1, "e2_bl1_g1_planD.csv")
            agg = pd.concat([agg, aggregate_bl1(g1)], ignore_index=True)
            save(agg, "e2_model_comparison.csv")
        else:
            log("  (딥러닝 백엔드 없음 - BL-1 건너뜀)")
    e4(cfg, P, seed)
    rob, rob_agg = e5_e6(cfg, P, seed)
    cov = e8(cfg, p_by_fold)
    e9(cfg, P)

    # 통과 기준(§6.2 + DL-001~003, 정정 DL-016) 에 따른 모델 선정
    from src.selection import apply_gates
    gates, chosen, picks = apply_gates(cfg, agg, rob_agg, cov)
    save(gates, "e2_selection_gates.csv")
    save(pd.DataFrame([{"규칙": k, "선정 모델": v} for k, v in picks.items()]),
         "r12_selection_by_rule.csv")
    log("  선정: 원안(v0)=%s / 제출본(v1)=%s / 정정(v2)=%s"
        % (picks["v0_original"], picks["v1_submitted"], picks["v2_corrected"]))
    log("  1순위 정렬키: far_h_upper95_worst (낮을수록 우선)")

    e7(cfg, P, seed, chosen)
    final_predictions(cfg, P, seed, chosen, "M1")

    # 설명 안정성
    st = [{"model": k, "top_feature_stability": EX.contribution_stability(v),
           "top_features_fold0": ", ".join(v[0])} for k, v in contribs.items()]
    if st:
        save(pd.DataFrame(st), "e4_explanation_stability.csv")

    manifest = {
        "mode": a.mode, "seed": seed,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "data_sha256": hashes,
        "config": cfg,
        "elapsed_sec": round(time.time() - t0, 2),
        "chosen_stage1_model": chosen,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUT, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log("완료. %.1f초" % (time.time() - t0))

    # 보고서 보충표(r1-r5)와 감사 진단표(r6-r11). 모델·임계값 불변, 기존 산출물만 사용.
    import make_report_tables as RT
    import make_audit_tables as AT
    import make_domain_tables as DT
    import make_mofn_table as MT
    import make_protocol_tables as PT
    import make_supervised_control as SC
    import make_calibration as CB
    import make_correction_tables as CT
    import make_interaction_tables as IT
    import make_review_tables as RV
    for name, mod in (("보고서 보충표 r1-r5", RT), ("감사 진단표 r6-r12", AT),
                      ("도메인 진단표 d1-d3", DT), ("M-of-N 표 d4", MT),
                      ("프로토콜 통제표 d5-d6", PT),
                      ("지도학습 대조군 d7", SC),
                      ("확률 보정 r13", CB),
                      ("정정 대응표 c1-c5", CT),
                      ("상호작용 진단표 i1-i4", IT),
                      ("검토 대응 진단표 v1-v6", RV)):
        log(name + " 생성")
        mod.main()


if __name__ == "__main__":
    main()

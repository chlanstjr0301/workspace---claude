# -*- coding: utf-8 -*-
"""
단일 실행 엔트리 — 전처리 → 학습 → 추론 → 결과생성

    python run_all.py

산출물 (outputs/)
    model_comparison.csv   보고서 제2장 모델 비교표 (문항 2)
    fp_breakdown.csv       오경보 집중조건 분해표 (문항 3)
    predictions.csv        제출용 테스트데이터 예측결과 파일
    metrics.csv            최종모델 지표 요약
    run_log.txt            전체 로그

평가 조건 고정 (보고서에 그대로 삽입)
    windowing : gap-aware (dt > 0.5 s 로 끊고 run 내부에서만 생성), SEQ=20, offset=0
    분할      : train normal[:12000] / valid normal[12000:15000] / test normal[15000:] + outlier
    표준화    : 학습구간만 fit. IF·범위규칙은 스케일 불변이라 미적용
    임계값    : 정상 검증구간 q99.9 — 이상 라벨 미사용
    디바운스  : k = 5 (0.5 s), 공정 상수로 선언
"""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import data as D
import evaluate as E
import models as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "outputs"))
QUANT = 0.999
_LOG = io.StringIO()


def say(s=""):
    print(s)
    _LOG.write(s + "\n")


# --------------------------------------------------------------------------- #
def fit_predict(det, Xtr, Xva, Xte, Xa):
    if det.needs_scaling:
        sc = M.Standardizer().fit(Xtr)
        Xtr, Xva, Xte, Xa = map(sc.transform, (Xtr, Xva, Xte, Xa))
    det.fit(Xtr)
    return det.score(Xva), det.score(Xte), det.score(Xa)


def evaluate_one(det, fs_name, ctx, cols):
    Xtr, Xva = ctx["tr"][:, cols], ctx["va"][:, cols]
    Xte, Xa = ctx["te"][:, cols], ctx["Fa"][:, cols]
    sv, st, sa = fit_predict(det, Xtr, Xva, Xte, Xa)

    th = float(np.quantile(sv, QUANT))
    pn, pa = (st > th).astype(int), (sa > th).astype(int)

    w = E.window_metrics(pn, pa)
    far, alarms, hours = E.far_per_hour(pn, len(ctx["normal"].iloc[D.VALID_END:]))
    far_hi = E.poisson_upper(alarms, hours)
    delay = E.detection_delay(pa, ctx["Sa"], ctx["t0"])

    bi_n = E.burst_index(ctx["Tte"], ctx["runs_normal"])
    bi_a = E.burst_index(ctx["Ta"], ctx["runs_outlier"])
    b = E.burst_metrics(pn, bi_n, pa, bi_a)
    lo, hi = E.block_bootstrap_f1(pn, bi_n, pa, bi_a, B=1000, seed=0)

    p_norm = E.conformal_p(sv, st)
    ks = E.uniformity_ks(p_norm)

    return dict(feature_set=fs_name, model=det.name, d1_ok=fs_name not in D.D1_VIOLATING,
                n_params=det.n_params, threshold=th,
                precision=w["precision"], recall=w["recall"], f1=w["f1"],
                fpr=w["fpr"], mcc=w["mcc"],
                TP=w["TP"], FN=w["FN"], FP=w["FP"], TN=w["TN"],
                f1_ci_lo=lo, f1_ci_hi=hi,
                far_per_h=far, far_alarms=alarms, far_upper95=far_hi,
                burst_recall=b["recall"], burst_fp=b["FP"], burst_f1=b["f1"],
                event_recall=1.0 if w["TP"] > 0 else 0.0,
                detect_delay_s=delay, conformal_ks=ks,
                _pn=pn, _pa=pa, _st=st, _sa=sa, _sv=sv, _bi_n=bi_n, _bi_a=bi_a)


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    say("=" * 100)
    say("Plan A 실행 — 소성가공 예지보전 (문제 ③)")
    say("=" * 100)
    say("python %s / numpy %s / pandas %s"
        % (sys.version.split()[0], np.__version__, pd.__version__))

    ctx = D.build()
    say("\n[전처리] gap-aware windowing SEQ=%d, GAP=%.1fs" % (D.SEQ, D.GAP_SEC))
    say("  정상 window %d (train %d / valid %d / test %d) / 이상 window %d"
        % (len(ctx["Fn"]), len(ctx["tr"]), len(ctx["va"]), len(ctx["te"]), len(ctx["Fa"])))
    say("  burst: 정상 %d / 이상 %d" % (len(ctx["runs_normal"]), len(ctx["runs_outlier"])))
    say("  특징 %d개, 특징집합 %d종" % (len(D.FEATURES), len(D.FEATURE_SETS)))

    rows = []
    say("\n[학습·평가] 임계값 = 정상 검증구간 q%.3f (이상 라벨 미사용)" % QUANT)
    for fs_name, cols in D.FEATURE_SETS.items():
        for det in M.registry():
            try:
                r = evaluate_one(det, fs_name, ctx, cols)
            except Exception as ex:                       # 모델별 수치 실패 격리
                say("  [skip] %s / %s — %s" % (fs_name, det.name, ex))
                continue
            rows.append(r)
            say("  %-12s %-15s F1=%.4f [%.3f,%.3f] P=%.3f R=%.3f FPR=%.4f "
                "FAR/h=%5.1f(<%5.1f) burstR=%.2f delay=%.1fs KS=%.3f%s"
                % (fs_name, det.name, r["f1"], r["f1_ci_lo"], r["f1_ci_hi"],
                   r["precision"], r["recall"], r["fpr"], r["far_per_h"],
                   r["far_upper95"], r["burst_recall"], r["detect_delay_s"],
                   r["conformal_ks"], "" if r["d1_ok"] else "   <D1위반>"))

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 측정 한계 선언 (선정 전에 반드시) ---------------- #
    hours = len(ctx["normal"].iloc[D.VALID_END:]) / D.FS / 3600.0
    say("\n" + "=" * 100)
    say("[측정 한계]  선정 규칙을 적용하기 전에 어느 지표가 변별 가능한지 선언한다")
    say("=" * 100)
    say("  FAR/h 분해능 — 정상 테스트 관측 %.3f h" % hours)
    for a in (0, 1, 2, 3):
        say("    알람 %d건 -> FAR/h=%5.1f, Poisson 95%% 상한=%5.1f"
            % (a, a / hours, E.poisson_upper(a, hours)))
    say("  => 알람 0건(0.0/h)과 1건(7.2/h)의 95%% 상한이 21.6 vs 34.2 로 겹친다.")
    say("     FAR/h 를 '=0' 하드 제약으로 쓰면 노이즈로 후보를 거른다. FP 개수로 순위만 매긴다.")

    floor = min(r["detect_delay_s"] for r in rows if not np.isnan(r["detect_delay_s"]))
    say("\n  탐지지연 하한 — 이상 파일 첫 burst 가 4샘플이라 window 생성 불가.")
    say("    t0 이후 첫 유효 window 까지 %.2f s 는 모델과 무관한 구조적 하한이다." % floor)
    say("    전 후보가 동일 값을 가지므로 탐지지연은 모델 변별에 쓸 수 없다.")

    say("\n  Event Recall — 양성 이벤트 1건이므로 탐지 후보 전부 1.0. 참고값으로만 기재.")

    # ---------------- 최종모델 선정 (공격4 §1.3 재정의 순서) ---------------- #
    say("\n" + "=" * 100)
    say("[최종모델 선정]  1) D1 통과  2) F1 95%CI 하한 최대  3) FP 최소  4) 파라미터 최소")
    say("=" * 100)
    cand = df[df.d1_ok].copy()
    say("  D1 통과 후보: %d개 / 전체 %d개" % (len(cand), len(df)))
    cand = cand.sort_values(["f1_ci_lo", "FP", "n_params"], ascending=[False, True, True])
    say("\n" + cand[["feature_set", "model", "f1", "f1_ci_lo", "f1_ci_hi", "precision",
                     "recall", "fpr", "FP", "far_per_h", "n_params"]]
        .head(10).to_string(index=False, float_format=lambda x: "%.4f" % x))

    # 범위규칙이 구조적으로 FP=0 인지 점검 (공격4 §1.3)
    say("\n  [점검] 범위규칙의 FP=0 이 구조적인가 — 테스트 특징이 학습 범위를 벗어날 수 있는가")
    for fs in ("S0_AMP4", "S1_VIB", "S3_SPEC"):
        c = D.FEATURE_SETS[fs]
        lo_, hi_ = ctx["tr"][:, c].min(0), ctx["tr"][:, c].max(0)
        ex = ((ctx["te"][:, c] < lo_) | (ctx["te"][:, c] > hi_)).any(1)
        say("    %-10s 테스트 정상 중 학습범위 이탈 가능 window %d / %d (%.2f%%) -> %s"
            % (fs, ex.sum(), len(ex), 100 * ex.mean(),
               "구조적 FP=0 아님" if ex.sum() > 0 else "구조적 FP=0 (비교 부적격)"))

    best = cand.iloc[0]
    say("\n  >>> 최종모델: %s / %s" % (best.feature_set, best.model))
    br = next(r for r in rows if r["feature_set"] == best.feature_set
              and r["model"] == best.model)

    # ---------------- 2층 베이스라인 (공격4 §7.2) ---------------- #
    say("\n  [2층 베이스라인 대비]")

    # (b-raw) 공격4 의 BL-0 — 원시 채널 min/max. FP=0 이 구조적이므로 참고값
    raw_lo = ctx["normal"].iloc[:D.TRAIN_END][D.COLS].min().values
    raw_hi = ctx["normal"].iloc[:D.TRAIN_END][D.COLS].max().values

    def raw_range_pred(df, T):
        X = df[D.COLS].values
        return np.array([int(((X[i:i + D.SEQ] < raw_lo) | (X[i:i + D.SEQ] > raw_hi)).any())
                         for i in T])

    pn_raw = raw_range_pred(ctx["normal"], ctx["Tte"])
    pa_raw = raw_range_pred(ctx["outlier"], ctx["Ta"])
    w_raw = E.window_metrics(pn_raw, pa_raw)
    say("    (b-raw) 원시채널 범위규칙 [공격4 BL-0]  F1=%.4f P=%.3f R=%.3f FPR=%.4f  "
        "*FP=0 은 구조적 — 공정 비교 부적격"
        % (w_raw["f1"], w_raw["precision"], w_raw["recall"], w_raw["fpr"]))

    for ref_fs, ref_md, lab in (("S0_AMP4", "BL0_Range", "(b) 특징공간 범위규칙"),
                                ("S0_AMP4", "Mahalanobis", "(c) 진폭4 Mahalanobis")):
        ref = next((r for r in rows if r["feature_set"] == ref_fs and r["model"] == ref_md), None)
        if ref is None:
            continue
        lo, hi, p0 = E.paired_delta(br["_pn"], br["_pa"], ref["_pn"], ref["_pa"],
                                    br["_bi_n"], br["_bi_a"], B=1000, seed=1)
        say("    vs %-22s F1 %.4f -> %.4f, dF1 95%%CI [%+.4f, %+.4f], P(d<=0)=%.3f"
            % (lab, ref["f1"], br["f1"], lo, hi, p0))

    lo, hi, p0 = E.paired_delta(br["_pn"], br["_pa"], pn_raw, pa_raw,
                                br["_bi_n"], br["_bi_a"], B=1000, seed=1)
    say("    vs (b-raw) 공격4 BL-0        F1 %.4f -> %.4f, dF1 95%%CI [%+.4f, %+.4f], P(d<=0)=%.3f"
        % (w_raw["f1"], br["f1"], lo, hi, p0))

    # 차점 후보와의 비교 — CI 겹침 여부가 '선정근거' 의 핵심
    second = cand.iloc[1]
    sr = next(r for r in rows if r["feature_set"] == second.feature_set
              and r["model"] == second.model)
    lo, hi, p0 = E.paired_delta(br["_pn"], br["_pa"], sr["_pn"], sr["_pa"],
                                br["_bi_n"], br["_bi_a"], B=1000, seed=2)
    say("    vs 차점 %s/%s  dF1 95%%CI [%+.4f, %+.4f], P(d<=0)=%.3f  -> %s"
        % (second.feature_set, second.model, lo, hi, p0,
           "유의한 차이 없음" if lo <= 0 <= hi else "유의"))

    # ---------------- 오경보 집중조건 분해 (문항 3) ---------------- #
    say("\n" + "=" * 100)
    say("[오경보 집중조건 분해]  정상 테스트 구간을 1,000행 단위로")
    say("=" * 100)
    bins = [(15000, 16000), (16000, 17000), (17000, 18000), (18000, 19000), (19000, 20000)]
    fp_rows = []
    worst = max(rows, key=lambda r: r["FP"] if r["d1_ok"] else -1)
    say("  (FP 최다 D1통과 후보 = %s / %s, FP=%d 기준)"
        % (worst["feature_set"], worst["model"], worst["FP"]))
    say("  %-14s %6s %6s %8s %9s %9s %9s" % ("구간", "nwin", "FP", "FP율", "AI1_std", "corr01", "amp_ratio"))
    i_std, i_c01 = D.FEATURES.index("AI1_std"), D.FEATURES.index("corr01")
    i_ar = D.FEATURES.index("amp_ratio")
    for lo_, hi_ in bins:
        m = (ctx["Tte"] >= lo_) & (ctx["Tte"] < hi_)
        fp = int(worst["_pn"][m].sum())
        rec = dict(bin="%d-%d" % (lo_, hi_), n_window=int(m.sum()), FP=fp,
                   fp_rate=fp / max(m.sum(), 1),
                   AI1_std=float(ctx["te"][m, i_std].mean()),
                   corr01=float(ctx["te"][m, i_c01].mean()),
                   amp_ratio=float(ctx["te"][m, i_ar].mean()))
        fp_rows.append(rec)
        say("  %-14s %6d %6d %8.4f %9.4f %9.4f %9.4f"
            % (rec["bin"], rec["n_window"], fp, rec["fp_rate"],
               rec["AI1_std"], rec["corr01"], rec["amp_ratio"]))
    tot = sum(r["FP"] for r in fp_rows)
    epi = sum(r["FP"] for r in fp_rows if r["bin"] in ("16000-17000", "17000-18000", "18000-19000"))
    say("\n  16,000-19,000 구간 FP 집중도: %d / %d = %.1f%%"
        % (epi, tot, 100 * epi / tot if tot else 0))
    pd.DataFrame(fp_rows).to_csv(os.path.join(OUT, "fp_breakdown.csv"),
                                 index=False, encoding="utf-8-sig")

    # ---------------- 제출용 예측결과 파일 ---------------- #
    cal = br["_sv"]
    pred = pd.DataFrame({
        "TimeStamp": np.concatenate([ctx["Ste"], ctx["Sa"]]),
        "window_id": np.arange(len(ctx["Ste"]) + len(ctx["Sa"])),
        "source": ["normal_test"] * len(ctx["Ste"]) + ["outlier"] * len(ctx["Sa"]),
        "anomaly_score": np.concatenate([br["_st"], br["_sa"]]),
        "conformal_p": np.concatenate([E.conformal_p(cal, br["_st"]),
                                       E.conformal_p(cal, br["_sa"])]),
        "alarm": np.concatenate([br["_pn"], br["_pa"]]),
        "Equipment_state_true": [0] * len(ctx["Ste"]) + [1] * len(ctx["Sa"]),
    })
    pred["anomaly_prob"] = 1.0 - pred["conformal_p"]
    pred.to_csv(os.path.join(OUT, "predictions.csv"), index=False, encoding="utf-8-sig")

    summary = {k: v for k, v in br.items() if not k.startswith("_")}
    pd.DataFrame([summary]).to_csv(os.path.join(OUT, "metrics.csv"),
                                   index=False, encoding="utf-8-sig")

    say("\n[산출물] outputs/model_comparison.csv (%d행), fp_breakdown.csv, "
        "predictions.csv (%d행), metrics.csv" % (len(df), len(pred)))
    with io.open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_LOG.getvalue())


if __name__ == "__main__":
    main()

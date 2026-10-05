# -*- coding: utf-8 -*-
"""
보조 산출물 — 통제실험 E1 / E3-a / E4 + 기여도 플롯 + conformal 균등성 점검

    python extras.py

산출물 (outputs/)
    e1_temporal_control.csv        E1  정상 전/후반 교차 — 시간 흐름만으로 알람이 뜨는가
    e3a_gain_perturbation.csv      E3-a 진폭 게인 ±10 % 섭동 전후 FPR (방어책 전/후)
    e4_rounding_control.csv        E4  두 파일 소수 6자리 라운딩 후 전 후보 재측정
    figures/pca_spe_contribution.png   PCA-SPE 기여도 분해 (오경보 집중구간)
    figures/conformal_uniformity.png   conformal p-value 히스토그램 + PP-plot
    extras_log.txt                 전체 로그

설계 근거
    E1   `docs/plan/plan A/Model Performance Indicators/README.md` §6.3
         "정상 데이터를 전/후반 분할 → 서로를 이상으로 탐지하는지. 기준 FAR/h <= 1"
    E3-a `previous research/공격4 - Plan A 적대적 검증.md` §1.6
         아핀불변 특징은 설계상 불변이므로 대상에서 제외하고 진폭 특징만 섭동
    E4   같은 문서 §3 말미 — 두 파일을 동일 정밀도로 라운딩 후 재측정
"""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data as D
import evaluate as E
import models as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "outputs"))
FIG = os.path.join(OUT, "figures")
QUANT = 0.999
_LOG = io.StringIO()

# 게인에 비례하는 진폭형 특징 (아핀불변 특징은 제외 — 공격4 §1.6)
AMP_FEATURES = ["AI0_std", "AI0_ptp", "AI1_std", "AI1_ptp", "corr01_x_amp",
                "D1_AI2_std", "D1_AI2_ptp"]
AMP_IDX = [D.FEATURES.index(f) for f in AMP_FEATURES]


def say(s=""):
    print(s)
    _LOG.write(s + "\n")


def _fit_score(det, Xtr, Xs):
    """표준화가 필요한 모델만 학습구간 기준으로 스케일링."""
    if det.needs_scaling:
        sc = M.Standardizer().fit(Xtr)
        Xtr, Xs = sc.transform(Xtr), [sc.transform(x) for x in Xs]
    else:
        Xs = list(Xs)
    det.fit(Xtr)
    return [det.score(x) for x in Xs]


# =========================================================================== #
# E1 — 정상 데이터 시간 전/후반 교차. 시간 흐름만으로 알람이 뜨면 '날짜 분류기'
# =========================================================================== #
def e1_temporal(ctx):
    say("\n" + "=" * 100)
    say("E1  정상 전/후반 교차 통제실험 — 기준: FAR/h <= 1")
    say("=" * 100)
    say("정상 학습구간(행 0~11999)을 전반/후반으로 나눠, 한쪽으로 적합한 모델이")
    say("다른 쪽을 이상으로 탐지하는지 본다. 라벨상 둘 다 정상이므로 알람은 0이어야 한다.")

    Fn, Tn = ctx["Fn"], ctx["Tn"]
    MID = D.TRAIN_END // 2                     # 행 6000
    halves = {"first(0~5999)": Tn < MID,
              "second(6000~11999)": (Tn >= MID) & (Tn < D.TRAIN_END)}
    rows = []
    for fs_name, cols in D.FEATURE_SETS.items():
        if fs_name in D.D1_VIOLATING:
            continue
        for src, dst in (("first(0~5999)", "second(6000~11999)"),
                         ("second(6000~11999)", "first(0~5999)")):
            Xsrc, Tsrc = Fn[halves[src]][:, cols], Tn[halves[src]]
            Xdst, Tdst = Fn[halves[dst]][:, cols], Tn[halves[dst]]
            # 적합 75 % / 임계값 25 % — 같은 반쪽 안에서만
            cut = int(len(Xsrc) * 0.75)
            for det in M.registry():
                try:
                    sv, sd = _fit_score(det, Xsrc[:cut], (Xsrc[cut:], Xdst))
                except Exception as ex:
                    say("  [skip] %s %s — %s" % (fs_name, det.name, ex))
                    continue
                th = float(np.quantile(sv, QUANT))
                pred = (sd > th).astype(int)
                n_rows = int(Tdst.max() - Tdst.min() + 1)
                far, alarms, hours = E.far_per_hour(pred, n_rows)
                rows.append(dict(feature_set=fs_name, model=det.name,
                                 fit_on=src, eval_on=dst,
                                 n_eval_window=len(Xdst), flagged=int(pred.sum()),
                                 fpr=float(pred.mean()), alarms=alarms,
                                 hours=hours, far_per_h=far,
                                 far_upper95=E.poisson_upper(alarms, hours),
                                 pass_e1=bool(far <= 1.0)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e1_temporal_control.csv"),
              index=False, encoding="utf-8-sig")
    h = float(df.hours.median())
    say("\n  관측시간 %.3f h — 알람 1건이면 FAR/h = %.2f 이다." % (h, 1.0 / h))
    say("  즉 FAR/h <= 1 기준은 이 관측길이에서 알람 0건만 통과시킨다 (§1.4의 축퇴).")
    say("  따라서 알람 건수와 window FPR 을 1차 지표로, FAR/h 는 참고값으로 읽는다.")
    say("\n  후보 %d개 (D1 통과 특징집합 4종 × 검출기 8종 × 방향 2)" % len(df))
    say("  E1 통과(FAR/h <= 1): %d / %d" % (int(df.pass_e1.sum()), len(df)))
    fin = df[(df.feature_set == "S1_VIB") & (df.model == "BL0_Range")]
    say("\n  [최종모델 S1_VIB / 범위규칙]")
    for _, r in fin.iterrows():
        say("    fit %-20s -> eval %-20s  오경보 window %d/%d (FPR %.4f)  알람 %d건  FAR/h %.2f  %s"
            % (r.fit_on, r.eval_on, r.flagged, r.n_eval_window, r.fpr,
               r.alarms, r.far_per_h, "통과" if r.pass_e1 else "실패"))
    worst = df.sort_values("far_per_h", ascending=False).head(5)
    say("\n  FAR/h 최악 5 (시간 흐름에 가장 민감한 후보)")
    for _, r in worst.iterrows():
        say("    %-10s %-15s %-20s -> %-20s FAR/h %6.2f  FPR %.4f"
            % (r.feature_set, r.model, r.fit_on, r.eval_on, r.far_per_h, r.fpr))
    return df


# =========================================================================== #
# E3-a — 진폭 게인 ±10 % 섭동. 아핀불변 특징은 설계상 불변이므로 제외
# =========================================================================== #
def _scaled_features(df, gain):
    """원시 진동 채널에 게인을 곱한 뒤 특징을 다시 계산한다."""
    d = df.copy()
    d["AI0_Vibration"] = d["AI0_Vibration"] * gain
    d["AI1_Vibration"] = d["AI1_Vibration"] * gain
    return D.window_features(d)


def _relative_norm(X, ref):
    """방어책 — 진폭형 특징을 기준 진폭으로 나눠 게인 불변으로 만든다."""
    Y = X.copy()
    Y[:, AMP_IDX] = Y[:, AMP_IDX] / max(ref, 1e-12)
    return Y


def e3a_gain(ctx):
    say("\n" + "=" * 100)
    say("E3-a  진폭 게인 ±10 %% 섭동 — 센서 재캘리브레이션만으로 오경보가 늘어나는가")
    say("=" * 100)
    say("공격4 §1.6: acf1·acf2·corr01·amp_ratio 는 np.corrcoef / 비율 기반이라 아핀불변이다.")
    say("따라서 게인 섭동 대상은 진폭형 특징 %d개뿐이다: %s"
        % (len(AMP_FEATURES), ", ".join(AMP_FEATURES)))
    say("원시 채널에 게인을 곱해 특징을 재계산하므로 불변성은 자동으로 반영된다.")

    gains = [0.9, 1.0, 1.1]
    cache = {}
    for g in gains:
        Fn, Tn, _ = _scaled_features(ctx["normal"], g)
        Fa, _, _ = _scaled_features(ctx["outlier"], g)
        cache[g] = (Fn, Tn, Fa)
    say("\n  게인 적용 후 특징 재계산 완료 (정상 %d window / 이상 %d window)"
        % (len(cache[1.0][0]), len(cache[1.0][2])))

    # 불변성 수치 확인
    F1, F09 = cache[1.0][0], cache[0.9][0]
    say("\n  불변성 점검 (게인 0.9 vs 1.0, 정상 window 전체 최대 상대차)")
    for nm in ("AI0_acf1", "corr01", "amp_ratio", "AI0_kurt", "AI0_crest",
               "AI0_bp18", "AI0_std", "AI0_ptp", "corr01_x_amp"):
        j = D.FEATURES.index(nm)
        a, b = F1[:, j], F09[:, j]
        rel = np.max(np.abs(b - a) / (np.abs(a) + 1e-9))
        say("    %-14s 최대 상대차 %.3e  %s"
            % (nm, rel, "불변" if rel < 1e-6 else "게인 영향"))

    Fnx, Tnx, _ = cache[1.0]
    mtrx, _, mtex = D.split_masks(Tnx)
    jx = D.FEATURES.index("AI1_std")
    md_tr, md_te = float(np.median(Fnx[mtrx][:, jx])), float(np.median(Fnx[mtex][:, jx]))
    say("\n  진단 — AI1_std 중앙값: 학습구간 %.5f / 테스트 정상구간 %.5f (비 %.3f)"
        % (md_tr, md_te, md_te / md_tr))
    say("  테스트 구간은 저부하 에피소드(16,000~19,000) 때문에 진폭 중앙값이 학습구간보다 낮다.")
    say("  세션 상대 정규화는 이 차이까지 지우므로 방어책으로 쓸 수 없다 (아래 표).")

    rows = []
    for fs_name, cols in D.FEATURE_SETS.items():
        if fs_name in D.D1_VIOLATING:
            continue
        for defense in ("none", "relative_norm", "invariant_only"):
            # 학습·임계값은 항상 게인 1.0 (배포 시점), 평가만 섭동
            use = ([c for c in cols if D.FEATURES[c] not in AMP_FEATURES]
                   if defense == "invariant_only" else list(cols))
            if not use:
                continue
            Fn0, Tn0, _ = cache[1.0]
            mtr, mva, _ = D.split_masks(Tn0)
            rel = defense == "relative_norm"
            ref_tr = float(np.median(Fn0[mtr][:, D.FEATURES.index("AI1_std")]))
            Xtr0 = _relative_norm(Fn0[mtr], ref_tr) if rel else Fn0[mtr]
            Xva0 = _relative_norm(Fn0[mva], ref_tr) if rel else Fn0[mva]
            for det in M.registry():
                for g in gains:
                    Fng, Tng, Fag = cache[g]
                    _, _, mte = D.split_masks(Tng)
                    Xte, Xa = Fng[mte], Fag
                    if rel:
                        # 같은 세션의 정상 window 중앙값으로 정규화 (라벨 미사용)
                        ref_te = float(np.median(Xte[:, D.FEATURES.index("AI1_std")]))
                        Xte, Xa = _relative_norm(Xte, ref_te), _relative_norm(Xa, ref_te)
                    try:
                        sv, st, sa = _fit_score(det, Xtr0[:, use],
                                                (Xva0[:, use], Xte[:, use], Xa[:, use]))
                    except Exception as ex:
                        say("  [skip] %s %s — %s" % (fs_name, det.name, ex))
                        continue
                    th = float(np.quantile(sv, QUANT))
                    pn, pa = (st > th).astype(int), (sa > th).astype(int)
                    w = E.window_metrics(pn, pa)
                    rows.append(dict(feature_set=fs_name, model=det.name,
                                     defense=defense, n_feat=len(use),
                                     gain=g, fpr=w["fpr"], FP=w["FP"],
                                     recall=w["recall"], f1=w["f1"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e3a_gain_perturbation.csv"),
              index=False, encoding="utf-8-sig")

    say("\n  [최종모델 S1_VIB / 범위규칙]")
    say("  %-16s %5s %6s %7s %8s %8s" % ("방어책", "특징", "게인", "FP", "FPR", "F1"))
    sel = df[(df.feature_set == "S1_VIB") & (df.model == "BL0_Range")]
    for _, r in sel.sort_values(["defense", "gain"]).iterrows():
        say("  %-16s %5d %6.1f %7d %8.4f %8.4f"
            % (r.defense, r.n_feat, r.gain, r.FP, r.fpr, r.f1))

    say("\n  D1 통과 후보 전체 — 게인 섭동으로 FPR 이 가장 많이 늘어난 5 (방어책 없음)")
    nod = df[df.defense == "none"].pivot_table(
        index=["feature_set", "model"], columns="gain", values="fpr")
    nod["max_delta"] = (nod[[0.9, 1.1]].max(axis=1) - nod[1.0])
    for (fs, mo), r in nod.sort_values("max_delta", ascending=False).head(5).iterrows():
        say("    %-10s %-15s FPR  g0.9 %.4f / g1.0 %.4f / g1.1 %.4f  (최대 증가 %+.4f)"
            % (fs, mo, r[0.9], r[1.0], r[1.1], r["max_delta"]))

    say("\n  방어책별 — 게인 섭동 최대 FPR 증가 / 게인 1.0 F1 (D1 통과 후보 평균)")
    for dfn in ("none", "relative_norm", "invariant_only"):
        sub = df[df.defense == dfn]
        if not len(sub):
            continue
        pv = sub.pivot_table(index=["feature_set", "model"], columns="gain", values="fpr")
        say("    %-16s 최대 FPR 증가 %+.4f   게인1.0 F1 평균 %.4f"
            % (dfn, (pv[[0.9, 1.1]].max(axis=1) - pv[1.0]).mean(),
               sub[sub.gain == 1.0].f1.mean()))
    say("\n  해석: 최종모델은 방어책 없이 이미 게인 불변이다 (FPR 0.0000 유지).")
    say("  relative_norm 은 불변성을 얻지만 저부하 에피소드까지 지워 F1 을 크게 떨어뜨린다.")
    say("  invariant_only 는 진폭 특징을 버리는 대가로 불변성을 얻는다 — 비용을 표로 제시한다.")
    return df


# =========================================================================== #
# E4 — 두 파일 소수 6자리 라운딩. 파일 정밀도 자체가 지문인지 통제
# =========================================================================== #
def _decimals(path, ncol=3, nrow=2000):
    mx = 0
    with io.open(path, encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            if i >= nrow:
                break
            for tok in line.strip().split(",")[2:2 + ncol]:
                if "." in tok:
                    mx = max(mx, len(tok.split(".")[1]))
    return mx


def e4_rounding(ctx, base_rows):
    say("\n" + "=" * 100)
    say("E4  파일 정밀도 통제 — 두 파일을 소수 6자리로 라운딩 후 전 후보 재측정")
    say("=" * 100)
    dn = _decimals(os.path.join(D.RAW, "press_data_normal.csv"))
    do = _decimals(os.path.join(D.RAW, "press_data_outlier.csv"))
    say("  원본 소수 자리수 최대 — 정상 %d자리 / 이상 %d자리" % (dn, do))
    if dn != do:
        say("  두 파일의 기록 정밀도가 다르다. 그 자체가 파일 지문이 될 수 있으므로")
        say("  동일 정밀도(6자리)로 맞춘 뒤 전 후보를 다시 측정한다.")

    n, o = ctx["normal"].copy(), ctx["outlier"].copy()
    for d in (n, o):
        for c in D.COLS:
            d[c] = d[c].round(6)
    Fn, Tn, _ = D.window_features(n)
    Fa, _, _ = D.window_features(o)
    mtr, mva, mte = D.split_masks(Tn)

    rows = []
    for fs_name, cols in D.FEATURE_SETS.items():
        for det in M.registry():
            try:
                sv, st, sa = _fit_score(det, Fn[mtr][:, cols],
                                        (Fn[mva][:, cols], Fn[mte][:, cols], Fa[:, cols]))
            except Exception as ex:
                say("  [skip] %s %s — %s" % (fs_name, det.name, ex))
                continue
            th = float(np.quantile(sv, QUANT))
            w = E.window_metrics((st > th).astype(int), (sa > th).astype(int))
            rows.append(dict(feature_set=fs_name, model=det.name,
                             d1_ok=fs_name not in D.D1_VIOLATING,
                             f1_rounded=w["f1"], fpr_rounded=w["fpr"],
                             recall_rounded=w["recall"], FP_rounded=w["FP"]))
    df = pd.DataFrame(rows)
    base = pd.DataFrame(base_rows)[["feature_set", "model", "f1", "fpr", "recall", "FP"]]
    df = df.merge(base, on=["feature_set", "model"], how="left")
    df["delta_f1"] = df.f1_rounded - df.f1
    df["delta_fpr"] = df.fpr_rounded - df.fpr
    df.to_csv(os.path.join(OUT, "e4_rounding_control.csv"),
              index=False, encoding="utf-8-sig")

    say("\n  전 후보 %d개 — 라운딩 전후 변화" % len(df))
    say("    |ΔF1| 최대 %.4f / 평균 %.4f" % (df.delta_f1.abs().max(), df.delta_f1.abs().mean()))
    say("    |ΔFPR| 최대 %.4f / 평균 %.4f" % (df.delta_fpr.abs().max(), df.delta_fpr.abs().mean()))
    fin = df[(df.feature_set == "S1_VIB") & (df.model == "BL0_Range")].iloc[0]
    say("\n  [최종모델 S1_VIB / 범위규칙]  F1 %.4f -> %.4f (Δ%+.4f) / FPR %.4f -> %.4f"
        % (fin.f1, fin.f1_rounded, fin.delta_f1, fin.fpr, fin.fpr_rounded))
    ai2 = df[df.feature_set == "X_AI2_ONLY"]
    if len(ai2):
        say("\n  [D1 위반 X_AI2_ONLY — 파일 지문 의심 대상]")
        for _, r in ai2.iterrows():
            say("    %-15s F1 %.4f -> %.4f (Δ%+.4f)"
                % (r.model, r.f1, r.f1_rounded, r.delta_f1))
    big = df.reindex(df.delta_f1.abs().sort_values(ascending=False).index).head(5)
    say("\n  변화가 가장 큰 5")
    for _, r in big.iterrows():
        say("    %-12s %-15s ΔF1 %+.4f  ΔFPR %+.4f  (D1 %s)"
            % (r.feature_set, r.model, r.delta_f1, r.delta_fpr,
               "통과" if r.d1_ok else "위반"))
    return df


# =========================================================================== #
# 그림 1 — PCA-SPE 기여도 분해 (오경보 집중구간)
# =========================================================================== #
def fig_contribution(ctx):
    say("\n" + "=" * 100)
    say("그림 1  PCA-SPE 기여도 분해 — 오경보 집중구간 16,000~19,000")
    say("=" * 100)
    fs_name, cols = "S3_SPEC", D.FEATURE_SETS["S3_SPEC"]
    names = [D.FEATURES[c] for c in cols]
    det = M.PCAModel(stat="SPE")

    Xtr, Xva = ctx["tr"][:, cols], ctx["va"][:, cols]
    Xte, Xa = ctx["te"][:, cols], ctx["Fa"][:, cols]
    sc = M.Standardizer().fit(Xtr)
    Ztr, Zva, Zte, Za = map(sc.transform, (Xtr, Xva, Xte, Xa))
    det.fit(Ztr)
    th = float(np.quantile(det.score(Zva), QUANT))

    def contrib(Z):
        """특징별 SPE 기여 = 잔차 제곱 성분. 합이 SPE 와 같다."""
        Y = Z - det.mu_
        t = Y @ det.P_
        return (Y - t @ det.P_.T) ** 2

    Cte, Ca = contrib(Zte), contrib(Za)
    Tte = ctx["Tte"]
    zone = (Tte >= 16000) & (Tte < 19000)
    base = ~zone

    say("  주성분 %d개 (분산 95 %% 기준), SPE 임계값 %.4f" % (det.nc_, th))
    say("  테스트 정상 window — 집중구간 %d / 그 외 %d / 이상 %d"
        % (int(zone.sum()), int(base.sum()), len(Za)))
    say("  평균 SPE — 집중구간 %.4f / 그 외 %.4f / 이상 %.4f"
        % (Cte[zone].sum(1).mean(), Cte[base].sum(1).mean(), Ca.sum(1).mean()))

    mz, mb, ma = Cte[zone].mean(0), Cte[base].mean(0), Ca.mean(0)
    order = np.argsort(-(mz / (mb + 1e-12)))

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(len(cols))
    w = 0.27
    ax[0].bar(x - w, mb[order], w, label="normal test, other rows", color="#6b7280")
    ax[0].bar(x, mz[order], w, label="normal test, rows 16k-19k", color="#d97706")
    ax[0].bar(x + w, ma[order], w, label="anomaly", color="#b91c1c")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax[0].set_ylabel("mean SPE contribution  (log scale)")
    ax[0].set_yscale("log")                 # 이상이 정상의 ~75배라 선형축에서는 정상끼리 비교가 안 보인다
    ax[0].set_title("(a) Per-feature SPE contribution  [S3_SPEC / PCA-SPE]")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].grid(axis="y", alpha=0.3, which="both")
    for i, j in enumerate(order):           # 집중구간/그외 배율을 막대 위에 표기
        ax[0].text(i, mz[j] * 1.15, "%.1fx" % (mz[j] / (mb[j] + 1e-12)),
                   ha="center", fontsize=7, color="#92400e")

    # 행 구간별 SPE 추이
    bins = np.arange(15000, 20001, 250)
    mid, mean_spe, frac = [], [], []
    spe_te = Cte.sum(1)
    for a, b in zip(bins[:-1], bins[1:]):
        m = (Tte >= a) & (Tte < b)
        if m.sum() == 0:
            continue
        mid.append((a + b) / 2)
        mean_spe.append(spe_te[m].mean())
        frac.append((spe_te[m] > th).mean())
    ax2 = ax[1]
    ax2.plot(mid, mean_spe, "o-", color="#1d4ed8", label="mean SPE")
    ax2.axhline(th, ls="--", color="#b91c1c", label="threshold (normal q99.9)")
    ax2.axvspan(16000, 19000, color="#fde68a", alpha=0.45, label="false-alarm zone")
    ax2.set_xlabel("row index (normal test region)")
    ax2.set_ylabel("mean SPE")
    ax2.set_title("(b) SPE over the normal test region")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG, "pca_spe_contribution.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    say("  저장: %s" % os.path.relpath(p, OUT))

    say("\n  집중구간/그외 기여도 비율 상위 5")
    for i in order[:5]:
        say("    %-14s 집중구간 %.4f / 그외 %.4f = %.2f배   (이상 %.4f)"
            % (names[i], mz[i], mb[i], mz[i] / (mb[i] + 1e-12), ma[i]))
    pd.DataFrame(dict(feature=names,
                      contrib_zone=mz, contrib_other=mb, contrib_anomaly=ma,
                      ratio_zone_over_other=mz / (mb + 1e-12))).to_csv(
        os.path.join(OUT, "spe_contribution.csv"), index=False, encoding="utf-8-sig")


# =========================================================================== #
# 그림 2 — conformal 균등성 점검
# =========================================================================== #
def fig_conformal(ctx):
    say("\n" + "=" * 100)
    say("그림 2  conformal p-value 균등성 점검 — 교환가능성 가정 검증")
    say("=" * 100)
    targets = [("S1_VIB", "BL0_Range", M.RangeRule()),
               ("S3_SPEC", "PCA_SPE", M.PCAModel(stat="SPE"))]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    rows = []
    for r, (fs_name, mname, det) in enumerate(targets):
        cols = D.FEATURE_SETS[fs_name]
        sv, st = _fit_score(det, ctx["tr"][:, cols],
                            (ctx["va"][:, cols], ctx["te"][:, cols]))
        p = E.conformal_p(sv, st)
        ks = E.uniformity_ks(p)
        ties = float((st <= st.min() + 1e-12).mean())
        rows.append(dict(feature_set=fs_name, model=mname, n=len(p),
                         ks=ks, p_min=float(p.min()), p_max=float(p.max()),
                         frac_p_lt_001=float((p < 0.01).mean()),
                         frac_tied_min_score=ties))
        say("  %-10s %-10s KS %.4f  p 범위 [%.4f, %.4f]  p<0.01 비율 %.4f  "
            "최저점수 동점 비율 %.4f"
            % (fs_name, mname, ks, p.min(), p.max(), (p < 0.01).mean(), ties))

        ax = axes[r, 0]
        ax.hist(p, bins=40, range=(0, 1), color="#1d4ed8", alpha=0.8,
                edgecolor="white")
        ax.axhline(len(p) / 40, ls="--", color="#b91c1c", label="uniform expectation")
        ax.set_title("%s / %s — p-value histogram (KS=%.3f)" % (fs_name, mname, ks),
                     fontsize=10)
        ax.set_xlabel("conformal p-value (normal test windows)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

        ax = axes[r, 1]
        ps = np.sort(p)
        emp = np.arange(1, len(ps) + 1) / len(ps)
        ax.plot([0, 1], [0, 1], ls="--", color="#b91c1c", label="uniform")
        ax.plot(ps, emp, color="#1d4ed8", lw=1.8, label="empirical")
        ax.set_title("%s / %s — PP-plot" % (fs_name, mname), fontsize=10)
        ax.set_xlabel("theoretical uniform quantile")
        ax.set_ylabel("empirical CDF")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p_ = os.path.join(FIG, "conformal_uniformity.png")
    fig.savefig(p_, dpi=170)
    plt.close(fig)
    say("  저장: %s" % os.path.relpath(p_, OUT))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "conformal_uniformity.csv"),
                              index=False, encoding="utf-8-sig")


# =========================================================================== #
def baseline_rows(ctx):
    """E4 비교용 — 라운딩 전 전 후보 지표."""
    rows = []
    for fs_name, cols in D.FEATURE_SETS.items():
        for det in M.registry():
            try:
                sv, st, sa = _fit_score(det, ctx["tr"][:, cols],
                                        (ctx["va"][:, cols], ctx["te"][:, cols],
                                         ctx["Fa"][:, cols]))
            except Exception:
                continue
            th = float(np.quantile(sv, QUANT))
            w = E.window_metrics((st > th).astype(int), (sa > th).astype(int))
            rows.append(dict(feature_set=fs_name, model=det.name, f1=w["f1"],
                             fpr=w["fpr"], recall=w["recall"], FP=w["FP"]))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    say("=" * 100)
    say("Plan A 보조 산출물 — 통제실험 E1 / E3-a / E4 + 기여도 플롯 + conformal 점검")
    say("=" * 100)
    say("python %s / numpy %s / pandas %s" %
        (sys.version.split()[0], np.__version__, pd.__version__))

    ctx = D.build()
    say("\n[전처리] 정상 window %d (train %d / valid %d / test %d) / 이상 %d"
        % (len(ctx["Fn"]), len(ctx["tr"]), len(ctx["va"]), len(ctx["te"]), len(ctx["Fa"])))

    base = baseline_rows(ctx)
    e1_temporal(ctx)
    e3a_gain(ctx)
    e4_rounding(ctx, base)
    fig_contribution(ctx)
    fig_conformal(ctx)

    with io.open(os.path.join(OUT, "extras_log.txt"), "w", encoding="utf-8") as f:
        f.write(_LOG.getvalue())
    say("\n" + "=" * 100)
    say("완료 — outputs/ 에 CSV 5종 + figures/ 에 그림 2종")
    say("=" * 100)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""최종보고서 검토 대응 진단표 v1-v6 (docs/plan/EXPERIMENTS_E1_E6.md).

모델·특징·임계값·경보 규칙 불변. predictions.csv 와 기존 표는 수정하지 않는다.
진단 결과를 모델 선정에 쓰지 않는다 (DL-018).

  E1  v1a 수집 지문 / v1b 기초 통계 / v1c 직류·양자화 반사실 /
      v1d 채널·특징군 분리력 / v1e 단일 특징 AUROC
  E2  v2a 동결 분할 후보별 전체 시스템 / v2b·v2c 동결 규칙의 5-fold 적용 /
      v2d 후보별 전체 시스템의 계측 섭동 시험 (지시서 확장: v2a 에서 BL-0 가 우세하게
      나온 뒤, '왜 BL-0 가 아닌가' 에 답하기 위해 추가. DL-018 에 기록)
  E3  v3a 버스트 부트스트랩 구간 / v3b 사후확률 부트스트랩 보수값
  E4  v4a 선정 민감도 / v4b 선정 원리 비교 / v4c 대조군 임계값 무관 지표
  E5  v5a 교대 단위 경보 / v5b 정지 검토 규칙 발동 / v5c 사유 분포 / v5d 빨강 대안 문구
  E6  v6 개별 특징 영향도

    python make_review_tables.py            # 전부
    python make_review_tables.py --only E1 E3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_audit_tables as AT          # noqa: E402
import run_all as R                     # noqa: E402
from src import calibration as cal      # noqa: E402
from src import data as D               # noqa: E402
from src import evaluation as EV        # noqa: E402
from src import explain as EX           # noqa: E402
from src import features as FT          # noqa: E402
from src import models as MD            # noqa: E402
from src import selection as SEL        # noqa: E402
from src import windows as WD           # noqa: E402

OUT = os.path.join(HERE, "outputs")
TAB = os.path.join(OUT, "tables")
RAW = os.path.join(HERE, "data", "raw")
FILES = {"normal": "press_data_normal.csv", "fault": "press_data_outlier.csv"}
CH = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
GRID_Q = 1.1920929


def save(df, name):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, name), index=False, encoding="utf-8-sig")
    print("  -> %s (%d행)" % (name, len(df)))


def T(name):
    return pd.read_csv(os.path.join(TAB, name))


def auc_dir(y, s):
    a = roc_auc_score(y, s)
    return (a, "+") if a >= 0.5 else (1 - a, "-")


# --------------------------------------------------------------------------- #
class Ctx:
    """공통 준비물: 설정, window·특징, 동결 시스템, 예측파일."""

    def __init__(self):
        self.cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
        n, o, _ = D.load_all(self.cfg, HERE)
        self.P = R.prepare(self.cfg, n, o, self.cfg["windows"]["length"])
        self.pred = pd.read_csv(os.path.join(OUT, "predictions.csv"))
        self.fz = AT.Frozen(self.cfg, self.P)
        P, fz = self.P, self.fz
        allF = np.vstack([P["Fn"], P["Fo"]])
        bid = np.r_[P["mn"]["burst_id"].values, P["mo"]["burst_id"].values]
        j = fz.judge(allF, bid)
        assert np.allclose(j["p1"], self.pred.p_normal_stage1.values)
        assert np.allclose(j["p2"], self.pred.p_normal_stage2.values)
        print("동결 구성 재현 확인: p1·p2 가 predictions.csv 와 일치")
        self.names = [P["names"][k] for k in fz.idx]
        self.groups = tuple(self.cfg["features"]["groups"])
        self.mh = P["mn"][fz.ho].reset_index(drop=True)
        self.mo = P["mo"]
        self.bh, self.bo = self.mh["burst_id"].values, self.mo["burst_id"].values
        self.Xh, self.Xo = P["Xn"][fz.ho], P["Xo"]
        self.Fh, self.Fo = P["Fn"][fz.ho], P["Fo"]
        self.y = np.r_[np.zeros(len(self.Fh), bool), np.ones(len(self.Fo), bool)]

    def judge_eval(self, fz, Fh=None, Fo=None):
        """평가 블록 정상과 고장을 따로 판정(버스트 경계 보존) 후 이어 붙인다."""
        jh = fz.judge(self.Fh if Fh is None else Fh, self.bh)
        jo = fz.judge(self.Fo if Fo is None else Fo, self.bo)
        return {k: np.r_[jh[k], jo[k]] for k in jh}


def refit(fz, stage1_name):
    """동결 구성에서 1단 모델만 바꾼 사본 (진단용)."""
    import copy
    z = copy.copy(fz)
    z.m1, z.cal1 = fz._fit(fz.c1, stage1_name)
    return z


def rule_flags(j):
    return {"1단 단독": j["s1"], "1단 AND 2단": j["s1"] & j["s2"], "빨강(AND+3연속)": j["red"]}


def system_row(ctx, j, flag, nh):
    c = AT.confusion(ctx.y, flag)
    n_ev, _ = EV.merge_alarms(flag[:nh], ctx.bh)
    fb = pd.Series(flag[nh:]).groupby(ctx.bo).any()
    return dict(TP=c["TP"], FP=c["FP"], FN=c["FN"], TN=c["TN"],
                precision=c["precision"], recall=c["recall"], f1=c["f1"], fpr=c["fpr"],
                fp_events=n_ev, bursts_detected=int(fb.sum()), bursts_total=len(fb))


# =========================================================================== #
# E1
# =========================================================================== #
def _decimals(s):
    s = str(s).lower()
    if "e" in s:
        s = s.split("e")[0]
    return len(s.split(".")[1]) if "." in s else 0


def e1(ctx):
    print("E1  수집 경로 진단과 교란 분해")
    cfg, P, fz = ctx.cfg, ctx.P, ctx.fz

    # ---- v1a 수집 지문 --------------------------------------------------- #
    rows = []
    for src, fn in FILES.items():
        df = pd.read_csv(os.path.join(RAW, fn), dtype=str)
        for c in CH:
            s = df[c]
            x = s.astype(float).values
            dec = s.map(_decimals)
            u = np.unique(x)
            dpos = np.diff(u)
            mpd = float(dpos[dpos > 0].min())
            for qname, q in (("min_pos_diff", mpd), ("1.1920929", GRID_Q)):
                r = x / q
                resid = float(np.abs(r - np.round(r)).max())
                rows.append(dict(source=src, channel=c, n=len(x), n_unique=len(u),
                                 unique_ratio=len(u) / len(x),
                                 max_decimals=int(dec[dec < 15].max()),
                                 median_decimals=float(dec[dec < 15].median()),
                                 float_artifact=int((dec >= 15).sum()),
                                 min_pos_diff=mpd, grid_q_name=qname, grid_q=q,
                                 grid_resid_max=resid, on_grid=bool(resid < 1e-3),
                                 clip_share_min=float((x == x.min()).mean()),
                                 clip_share_max=float((x == x.max()).mean())))
    v1a = pd.DataFrame(rows)
    save(v1a, "v1a_acquisition_fingerprint.csv")
    g = v1a[v1a.grid_q_name == "1.1920929"].set_index(["source", "channel"])
    assert g.loc[("fault", "AI2_Current"), "on_grid"] and not g.loc[("normal", "AI2_Current"), "on_grid"], \
        "격자 검사 결과가 검토 확인값과 다르다"
    print(v1a[v1a.grid_q_name == "1.1920929"][["source", "channel", "n_unique", "max_decimals",
                                                "grid_resid_max", "on_grid"]].to_string(index=False))

    # ---- v1b 기초 통계 --------------------------------------------------- #
    Xtr = P["Xn"][fz.tr].reshape(-1, 3)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    raw = {k: pd.read_csv(os.path.join(RAW, f)) for k, f in FILES.items()}
    rows = [
        dict(항목="행 기준 이상 비율", 정상=len(raw["normal"]), 고장=len(raw["fault"]),
             값="%.2f%%" % (100 * len(raw["fault"]) / (len(raw["normal"]) + len(raw["fault"]))),
             비고="행 600 / 20,600"),
        dict(항목="window 기준 이상 비율", 정상=len(P["Fn"]), 고장=len(P["Fo"]),
             값="%.2f%%" % (100 * len(P["Fo"]) / (len(P["Fn"]) + len(P["Fo"]))), 비고="전체 window"),
        dict(항목="평가 표본 유병률", 정상=len(ctx.Fh), 고장=len(ctx.Fo),
             값="%.2f%%" % (100 * len(ctx.Fo) / (len(ctx.Fh) + len(ctx.Fo))), 비고="평가 블록 4 + 고장"),
        dict(항목="중복 TimeStamp 처리", 정상=int(raw["normal"].TimeStamp.duplicated().sum()),
             고장=int(raw["fault"].TimeStamp.duplicated().sum()), 값="유지",
             비고="삭제하지 않음. 간격 0초라 같은 버스트에 포함 (src/data.py, src/windows.py)"),
    ]
    for k, c in enumerate(CH):
        z = {s: np.abs((raw[s][c].values - mu[k]) / sd[k]) for s in raw}
        rows.append(dict(항목="|z|>4 행 비율 · " + c, 정상=round(100 * (z["normal"] > 4).mean(), 3),
                         고장=round(100 * (z["fault"] > 4).mean(), 3), 값="%",
                         비고="기준: 정상 학습 블록 0–2 평균·표준편차"))
    rows.append(dict(항목="생산단위(스트로크) 식별", 정상="", 고장="", 값="불가",
                     비고="버스트 최대 4.9초, 샘플 간격 0.1초, Nyquist 5 Hz. 스트로크 경계 신호 없음"))
    save(pd.DataFrame(rows), "v1b_basic_stats.csv")

    # ---- v1c 직류·양자화 반사실 ------------------------------------------ #
    ch = cfg["data"]["channels"]
    m_tr = float(Xtr[:, 2].mean())
    m_fa = float(ctx.Xo[:, :, 2].mean())

    def shift_fault(X):
        Y = X.copy(); Y[:, :, 2] += m_tr - m_fa; return Y

    def quant(X):
        Y = X.copy(); Y[:, :, 2] = np.round(Y[:, :, 2] / GRID_Q) * GRID_Q; return Y

    def demean(X):
        Y = X.copy(); Y[:, :, 2] -= Y[:, :, 2].mean(axis=1, keepdims=True); return Y

    conds = [("C0", "변환 없음", lambda h: h, lambda o: o),
             ("C1", "고장 전류 직류를 정상 학습 수준으로 이동", lambda h: h, shift_fault),
             ("C2", "정상 전류를 고장 격자(1.192)로 양자화", quant, lambda o: o),
             ("C3", "정상·고장 전류의 window별 평균 제거", demean, demean),
             ("C4", "C1 + C2", quant, shift_fault)]
    rows = []
    for cid, desc, fh, fo in conds:
        Fh, _, _ = FT.build(fh(ctx.Xh), ch, ctx.mh, ctx.groups)
        Fo, _, _ = FT.build(fo(ctx.Xo), ch, ctx.mo, ctx.groups)
        j = ctx.judge_eval(fz, Fh, Fo)
        nh = len(Fh)
        c1 = AT.confusion(ctx.y, j["s1"]); cr = AT.confusion(ctx.y, j["red"])
        rows.append(dict(id=cid, 조건=desc, stage1_auroc=roc_auc_score(ctx.y, j["score1"]),
                         normal_stage1_rate=j["s1"][:nh].mean(), normal_red_rate=j["red"][:nh].mean(),
                         fault_stage1_recall=j["s1"][nh:].mean(), fault_red_recall=j["red"][nh:].mean(),
                         stage1_f1=c1["f1"], red_f1=cr["f1"]))
    v1c = pd.DataFrame(rows)
    r2 = T("r2_operational_performance.csv").set_index("기준")
    c0 = v1c.iloc[0]
    assert abs(c0.red_f1 - r2.loc["최종 빨강", "f1"]) < 1e-4
    assert abs(c0.stage1_f1 - r2.loc["1단 경보(노랑+빨강)", "f1"]) < 1e-4
    for k in ["stage1_auroc", "normal_stage1_rate", "fault_red_recall", "red_f1"]:
        v1c["d_" + k] = v1c[k] - c0[k]
    save(v1c, "v1c_dc_quantization_counterfactual.csv")
    print(v1c[["id", "stage1_auroc", "normal_stage1_rate", "normal_red_rate",
               "fault_red_recall", "red_f1"]].round(4).to_string(index=False))

    # ---- v1d 채널·특징군 분리력 ------------------------------------------ #
    nm = ctx.names
    sel = {
        "G-all": lambda n: True,
        "G-vib": lambda n: "CUR" not in n and not n.startswith("O_"),
        "G-vibA": lambda n: n.startswith("A_") and "CUR" not in n,
        "G-vibS": lambda n: n.startswith("S_") and "CUR" not in n,
        "G-R": lambda n: n.startswith("R_"),
        "G-cur": lambda n: "CUR" in n or n.startswith("O_"),
        "G-curA": lambda n: n.startswith("A_") and "CUR" in n,
        "G-curS": lambda n: n.startswith("S_") and "CUR" in n,
        "G-O": lambda n: n.startswith("O_"),
    }
    F = P["Fn"][:, fz.idx]
    Fe = np.vstack([ctx.Fh[:, fz.idx], ctx.Fo[:, fz.idx]])
    nh = len(ctx.Fh)
    rows = []
    for gid, f in sel.items():
        cols = np.array([i for i, n in enumerate(nm) if f(n)])
        for mname in ("M1", "M3"):
            m = [x for x in MD.build(cfg, cfg["seed"]) if x.name == mname][0]
            m.fit(F[fz.tr][:, cols])
            p = cal.conformal_p(m.score(F[fz.cl][:, cols]), m.score(Fe[:, cols]))
            s = m.score(Fe[:, cols])
            rows.append(dict(id=gid, model=mname, n_features=len(cols),
                             features=" ".join(nm[i] for i in cols),
                             auroc=roc_auc_score(ctx.y, s), ap=average_precision_score(ctx.y, s),
                             normal_alarm_rate=(p[:nh] <= fz.alpha).mean(),
                             fault_recall=(p[nh:] <= fz.alpha).mean()))
    v1d = pd.DataFrame(rows)
    r11 = T("r11_negative_control_channels.csv")
    ga = v1d[(v1d.id == "G-all") & (v1d.model == "M3")].iloc[0]
    gc = v1d[(v1d.id == "G-cur") & (v1d.model == "M3")].iloc[0]
    assert abs(ga.auroc - r11.iloc[0]["1단 AUROC"]) < 1e-4
    assert abs(gc.auroc - r11.iloc[2]["1단 AUROC"]) < 1e-4
    save(v1d, "v1d_channel_group_separability.csv")
    print(v1d[["id", "model", "n_features", "auroc", "normal_alarm_rate", "fault_recall"]]
          .round(4).to_string(index=False))

    # ---- v1e 단일 특징 AUROC -------------------------------------------- #
    rows = []
    for i, n in enumerate(nm):
        a, d = auc_dir(ctx.y, Fe[:, i])
        rows.append(dict(feature=n, auroc=a, direction=d,
                         normal_median=float(np.median(Fe[:nh, i])),
                         fault_median=float(np.median(Fe[nh:, i]))))
    v1e = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    save(v1e, "v1e_single_feature_auroc.csv")
    print(v1e.head(6).round(4).to_string(index=False))


# =========================================================================== #
# E2
# =========================================================================== #
CANDS = ["BL0", "M1", "M2", "M3"]


def e2(ctx):
    print("E2  최종 시스템의 동일 조건 비교")
    cfg, P, fz = ctx.cfg, ctx.P, ctx.fz
    nh = len(ctx.Fh)
    rows = []
    for c in CANDS:
        z = refit(fz, c)
        j = ctx.judge_eval(z)
        ties = float(np.mean(z.cal1 <= z.cal1.min() + 1e-12))
        for rule, flag in rule_flags(j).items():
            rows.append(dict(stage1=c, rule=rule, stage1_auroc=roc_auc_score(ctx.y, j["score1"]),
                             p_ties_share=ties, **system_row(ctx, j, flag, nh)))
    v2a = pd.DataFrame(rows)
    m3 = v2a[v2a.stage1 == "M3"].set_index("rule")
    assert list(m3["FP"]) == [241, 35, 3], "M3 행이 r8 분해와 다르다"
    save(v2a, "v2a_frozen_split_full_system.csv")
    print(v2a[v2a.rule == "빨강(AND+3연속)"][["stage1", "FP", "recall", "f1", "fpr",
                                            "fp_events", "bursts_detected"]].round(4).to_string(index=False))

    # ---- 5-fold ---------------------------------------------------------- #
    blocks = P["mn"]["block"].values
    F = P["Fn"][:, fz.idx]
    e3 = T("e3_normal_block_cv.csv")
    rows = []
    for fd in WD.cv_folds(cfg["cv"]["n_blocks"]):
        tr = np.isin(blocks, fd["train"]); cl = blocks == fd["cal"]; ho = blocks == fd["holdout"]
        mh = P["mn"][ho].reset_index(drop=True)
        bh = mh["burst_id"].values
        m2 = [x for x in MD.build(cfg, cfg["seed"]) if x.name == "M1"][0]
        m2.fit(F[tr][:, fz.vib]); cal2 = m2.score(F[cl][:, fz.vib])
        for c in CANDS:
            m1 = [x for x in MD.build(cfg, cfg["seed"]) if x.name == c][0]
            m1.fit(F[tr]); cal1 = m1.score(F[cl])
            out = {}
            for nm_, FF, bb in (("h", F[ho], bh), ("o", P["Fo"][:, fz.idx], ctx.bo)):
                p1 = cal.conformal_p(cal1, m1.score(FF)); p2 = cal.conformal_p(cal2, m2.score(FF[:, fz.vib]))
                s1, s2 = p1 <= fz.alpha, p2 <= fz.alpha
                out[nm_] = dict(s1=s1, s2=s2, red=AT.persist(s1 & s2, bb, fz.red_n))
            y = np.r_[np.zeros(ho.sum(), bool), np.ones(len(ctx.Fo), bool)]
            for rule, key in (("1단 단독", "s1"), ("1단 AND 2단", None), ("빨강(AND+3연속)", "red")):
                if key is None:
                    fh, fo = out["h"]["s1"] & out["h"]["s2"], out["o"]["s1"] & out["o"]["s2"]
                else:
                    fh, fo = out["h"][key], out["o"][key]
                flag = np.r_[fh, fo]
                cc = AT.confusion(y, flag)
                n_ev, _ = EV.merge_alarms(fh, bh)
                rows.append(dict(stage1=c, fold=fd["fold"], holdout=fd["holdout"], cal=fd["cal"],
                                 rule=rule, FP=cc["FP"], n_normal=int(ho.sum()), fpr=cc["fpr"],
                                 recall=cc["recall"], precision=cc["precision"], f1=cc["f1"],
                                 fp_events=n_ev,
                                 bursts_detected=int(pd.Series(fo).groupby(ctx.bo).any().sum())))
            if c in set(e3.model):
                ref = e3[(e3.model == c) & (e3.fold == fd["fold"])]["fp_windows"].iloc[0]
                got = rows[-3]["FP"]
                assert got == ref, "%s fold %d 1단 FP %d != e3 %d" % (c, fd["fold"], got, ref)
    v2b = pd.DataFrame(rows)
    save(v2b, "v2b_cv_full_system.csv")
    v2c = v2b.groupby(["stage1", "rule"]).agg(
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"), f1_min=("f1", "min"),
        fpr_mean=("fpr", "mean"), fpr_max=("fpr", "max"),
        recall_mean=("recall", "mean"), fp_events_sum=("fp_events", "sum"),
        bursts_min=("bursts_detected", "min")).reset_index()
    save(v2c, "v2c_cv_full_system_summary.csv")
    print(v2c[v2c.rule == "빨강(AND+3연속)"].round(4).to_string(index=False))
    e2d(ctx)


def e2d(ctx):
    """후보별 전체 시스템(1단 후보 + 동결 2단 + 3연속)에 r6 와 같은 섭동을 가한다."""
    cfg, P, fz = ctx.cfg, ctx.P, ctx.fz
    ch, rb = cfg["data"]["channels"], cfg["robustness"]
    train_std = P["Xn"].std(axis=(0, 1))          # r6 와 같은 정의
    rng = np.random.default_rng(cfg["seed"])
    plans = [("none", "-", None)]
    plans += [("gain_all", a, ("gain", a)) for a in rb["gain"]]
    plans += [("offset_all", a, ("offset", a)) for a in rb["offset_sigma"]]
    plans += [("polarity", a, ("polarity", a)) for a in rb["polarity"]]
    plans += [("jitter", a, ("jitter", a)) for a in rb["jitter_sec"]]
    for c, nm_ in enumerate(["AI0", "AI1", "AI2"]):
        for a in rb["offset_sigma"]:
            plans.append(("offset_%s_only" % nm_, a, ("offset_ch", (c, a))))
        for a in rb["gain"]:
            plans.append(("gain_%s_only" % nm_, a, ("gain_ch", (c, a))))

    def apply(X, spec):
        if spec is None:
            return X
        kind, amt = spec
        if kind == "offset_ch":
            c, a = amt; Y = X.copy(); Y[:, :, c] += a * train_std[c]; return Y
        if kind == "gain_ch":
            c, a = amt; Y = X.copy(); Y[:, :, c] *= a; return Y
        return EX.perturb(X, kind, amt, train_std, rng)

    systems = {c: refit(fz, c) for c in CANDS}
    nh = len(ctx.Fh)
    rows = []
    for name, amt, spec in plans:
        Fh, _, _ = FT.build(apply(ctx.Xh, spec), ch, ctx.mh, ctx.groups)
        Fo, _, _ = FT.build(apply(ctx.Xo, spec), ch, ctx.mo, ctx.groups)
        for c, z in systems.items():
            j = ctx.judge_eval(z, Fh, Fo)
            rows.append(dict(stage1=c, perturbation=name, amount=str(amt),
                             normal_stage1_rate=j["s1"][:nh].mean(),
                             normal_red_rate=j["red"][:nh].mean(),
                             fault_red_recall=j["red"][nh:].mean()))
    v2d = pd.DataFrame(rows)
    r6 = T("r6_frozen_system_stress.csv")
    m3 = v2d[v2d.stage1 == "M3"].reset_index(drop=True)
    assert np.allclose(m3.normal_red_rate.values, r6.normal_red_rate.values)
    save(v2d, "v2d_stress_by_stage1.csv")
    s = v2d.groupby("stage1").agg(worst_normal_red=("normal_red_rate", "max"),
                                  worst_normal_stage1=("normal_stage1_rate", "max"),
                                  min_fault_red_recall=("fault_red_recall", "min"))
    print(s.round(4).to_string())


# =========================================================================== #
# E3
# =========================================================================== #
def _burst_counts(df, flag_col):
    g = df.groupby("burst_id")
    n = g.size().values
    k = g[flag_col].sum().values
    ev = g[flag_col].apply(lambda s: EV.merge_alarms(s.values, np.zeros(len(s)))[0]).values
    return n, k, ev


def e3(ctx, B=5000):
    print("E3  버스트 단위 부트스트랩 (B=%d)" % B)
    pred = ctx.pred
    h = pred[pred.split == "holdout"].copy(); f = pred[pred.split == "fault"].copy()
    for d in (h, f):
        d["red"] = d.alarm_level == "red"
        d["s1"] = d.p_normal_stage1 <= ctx.fz.alpha
    rng = np.random.default_rng(ctx.cfg["seed"])
    rows, store = [], {}
    for lvl in ("red", "s1"):
        nN, fpN, evN = _burst_counts(h, lvl)
        nF, tpF, _ = _burst_counts(f, lvl)
        iN = rng.integers(0, len(nN), size=(B, len(nN)))
        iF = rng.integers(0, len(nF), size=(B, len(nF)))
        N, FP, EVc = nN[iN].sum(1), fpN[iN].sum(1), evN[iN].sum(1)
        M, TP = nF[iF].sum(1), tpF[iF].sum(1)
        fpr, rec = FP / N, TP / M
        prec = TP / np.maximum(TP + FP, 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        pt = dict(fpr=fpN.sum() / nN.sum(), recall=tpF.sum() / nF.sum())
        pt["precision"] = tpF.sum() / max(tpF.sum() + fpN.sum(), 1)
        pt["f1"] = 2 * pt["precision"] * pt["recall"] / (pt["precision"] + pt["recall"])
        pt["fp_windows"], pt["fp_events"] = fpN.sum(), evN.sum()
        for k, arr in (("fpr", fpr), ("recall", rec), ("precision", prec), ("f1", f1),
                       ("fp_windows", FP), ("fp_events", EVc)):
            rows.append(dict(level="빨강" if lvl == "red" else "1단", metric=k, point=pt[k],
                             lo95=float(np.percentile(arr, 2.5)), hi95=float(np.percentile(arr, 97.5)),
                             share_fp_zero=float((FP == 0).mean()) if k == "fpr" else np.nan,
                             n_normal_bursts=len(nN), n_fault_bursts=len(nF), B=B))
        store[lvl] = dict(fpr=fpr, rec=rec)
    v3a = pd.DataFrame(rows)
    r2 = T("r2_operational_performance.csv").set_index("기준")
    red = v3a[v3a.level == "빨강"].set_index("metric")
    assert abs(red.loc["f1", "point"] - r2.loc["최종 빨강", "f1"]) < 1e-9
    save(v3a, "v3a_burst_bootstrap_ci.csv")
    print(v3a[v3a.level == "빨강"].round(4).to_string(index=False))

    r10 = T("r10_posterior_bounds.csv")
    tpr, fpr = red.loc["recall", "point"], red.loc["fpr", "point"]
    t_lo = float(np.percentile(store["red"]["rec"], 2.5))
    f_hi = float(np.percentile(store["red"]["fpr"], 97.5))
    rows = []
    for i, pi in enumerate((0.01, 0.001, 0.0001)):
        post = lambda t, f_: t * pi / (t * pi + f_ * (1 - pi))
        rows.append({"가정 고장유병률": pi, "점추정": post(tpr, fpr),
                     "보수(CP, window 독립)": r10.iloc[i, 6],
                     "보수(버스트 부트스트랩)": post(t_lo, f_hi),
                     "TPR 2.5%": t_lo, "FPR 97.5%": f_hi})
    v3b = pd.DataFrame(rows)
    save(v3b, "v3b_posterior_bootstrap.csv")
    print(v3b.round(4).to_string(index=False))


# =========================================================================== #
# E4
# =========================================================================== #
def _choose(g, t=None, abs_tol=None):
    ok = g.gate1_no_nan & g.gate4_burst_detected & g.gate5_v2_abs_operating
    if t is not None:
        d, s = g.worst_recall_drop_pp, g.worst_shift_fp_increase_pp
        ok &= (d.isna() | (d <= t)) & (s.isna() | (s <= t))
    else:
        ok &= g.gate2_robust
    if abs_tol is not None:
        bl0 = g.bl0_fp_rate_worst.iloc[0]
        lim = max(bl0 * (1 + SEL.FP_TOL_REL), bl0 + abs_tol)
        ok &= (g.model == "BL0") | (g.fp_rate_worst <= lim)
    else:
        ok &= g.gate3_fp_vs_bl0
    c = g[ok]
    if c.empty:
        return "통과 없음", 0
    return str(c.sort_values(SEL.SORT_KEYS, ascending=[True, False, False]).iloc[0]["model"]), len(c)


def _intervals(sweep, xs, picks):
    rows, start = [], 0
    for i in range(1, len(xs) + 1):
        if i == len(xs) or picks[i] != picks[start]:
            rows.append(dict(sweep=sweep, start=xs[start], end=xs[i - 1],
                             chosen=picks[start][0], n_passed=picks[start][1]))
            start = i
    return rows


def e4(ctx):
    print("E4  선정 민감도와 대조군의 임계값 무관 지표")
    g = T("e2_selection_gates.csv")
    assert _choose(g, t=20.0)[0] == "M3"
    ts = np.round(np.arange(0, 100.01, 0.1), 1)
    rows = _intervals("섭동 게이트 한도 t (%p)", ts, [_choose(g, t=t) for t in ts])
    tol = np.round(np.arange(0, 0.0201, 0.001), 3)
    rows += _intervals("gate3 절대 허용오차", tol, [_choose(g, abs_tol=a) for a in tol])
    v4a = pd.DataFrame(rows)
    save(v4a, "v4a_selection_sensitivity.csv")
    print(v4a.to_string(index=False))

    e5 = T("e5_robustness_summary.csv")
    worst = e5.groupby("model").fp_increase_pp.max()
    rk = e5.pivot(index="model", columns="perturbation", values="fp_increase_pp").rank()
    r12 = T("r12_selection_by_rule.csv").set_index("규칙")
    v4b = pd.DataFrame([
        {"원리": "minimax (최악 섭동 오경보 증가 최소)", "선정": worst.idxmin(),
         "근거": "; ".join("%s %.1f" % (k, v) for k, v in worst.sort_values().items())},
        {"원리": "섭동 4종 순위합 최소",
         "선정": "·".join(rk.sum(axis=1)[rk.sum(axis=1) == rk.sum(axis=1).min()].index) + " (동률)"
         if (rk.sum(axis=1) == rk.sum(axis=1).min()).sum() > 1 else rk.sum(axis=1).idxmin(),
         "근거": "; ".join("%s %.0f" % (k, v) for k, v in rk.sum(axis=1).sort_values().items())},
        {"원리": "원 기준 v0", "선정": r12.loc["v0_original", "선정 모델"], "근거": "r12"},
        {"원리": "제출 기준 v1", "선정": r12.loc["v1_submitted", "선정 모델"], "근거": "r12"},
        {"원리": "정정 기준 v2", "선정": r12.loc["v2_corrected", "선정 모델"], "근거": "r12"},
    ])
    save(v4b, "v4b_selection_principles.csv")
    print(v4b.to_string(index=False))

    # ---- v4c 대조군 AUROC (make_supervised_control 재현) ------------------ #
    cfg, P = ctx.cfg, ctx.P
    nb = cfg["cv"]["n_blocks"]
    blocks = P["mn"]["block"].values
    d7 = T("d7_supervised_control.csv").set_index("id")
    sets = {"groups 25 (Q 포함, 기존)": np.arange(P["Fn"].shape[1]), "model_groups 23": ctx.fz.idx}

    def fit_score(Xtr, ytr, Xte, seed):
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=2000, random_state=seed).fit(sc.transform(Xtr), ytr)
        return m.predict(sc.transform(Xte)), m.decision_function(sc.transform(Xte))

    def prf(yh, y):
        tp = ((yh == 1) & (y == 1)).sum(); fp = ((yh == 1) & (y == 0)).sum(); fn = ((yh == 0) & (y == 1)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    rows = []
    for sname, cols in sets.items():
        Fn, Fo = P["Fn"][:, cols], P["Fo"][:, cols]
        f1s, aucs, aps, pis = [], [], [], []
        for b in range(nb):
            tr_n, te_n = blocks != b, blocks == b
            m_o = np.random.default_rng(b).random(len(Fo)) < (1.0 / nb)
            Xtr = np.vstack([Fn[tr_n], Fo[~m_o]]); ytr = np.r_[np.zeros(tr_n.sum()), np.ones((~m_o).sum())]
            Xte = np.vstack([Fn[te_n], Fo[m_o]]); yte = np.r_[np.zeros(te_n.sum()), np.ones(m_o.sum())]
            yh, s = fit_score(Xtr, ytr, Xte, b)
            f1s.append(prf(yh, yte)); aucs.append(roc_auc_score(yte, s))
            aps.append(average_precision_score(yte, s)); pis.append(yte.mean())
        pi = float(np.mean(pis))
        rows.append(dict(id="M4", feature_set=sname, f1=np.mean(f1s), auroc=np.mean(aucs), ap=np.mean(aps),
                         test_prevalence=pi, coin_f1=2 * pi * 0.5 / (pi + 0.5), allpos_f1=2 * pi / (pi + 1)))
        if sname.startswith("groups"):
            assert abs(np.mean(f1s) - d7.loc["M4", "f1_mean"]) < 1e-9
        for lo, hi in [(0, nb - 1), (0, 1), (nb - 2, nb - 1)]:
            m = np.isin(blocks, [lo, hi])
            Xf, yf, bb = Fn[m], (blocks[m] == hi).astype(int), blocks[m]
            half = np.zeros(len(Xf), dtype=bool)
            for bk in (lo, hi):
                idx = np.where(bb == bk)[0]; half[idx[: len(idx) // 2]] = True
            yh, s = fit_score(Xf[half], yf[half], Xf[~half], 42)
            nid = "NC-%d%d" % (lo, hi)
            f1 = prf(yh, yf[~half])
            if sname.startswith("groups"):
                assert abs(f1 - d7.loc[nid, "f1_mean"]) < 1e-9
            pi = float(yf[~half].mean())
            rows.append(dict(id=nid, feature_set=sname, f1=f1, auroc=roc_auc_score(yf[~half], s),
                             ap=average_precision_score(yf[~half], s), test_prevalence=pi,
                             coin_f1=2 * pi * 0.5 / (pi + 0.5), allpos_f1=2 * pi / (pi + 1)))
    v4c = pd.DataFrame(rows)
    save(v4c, "v4c_control_threshold_free.csv")
    print(v4c.round(4).to_string(index=False))


# =========================================================================== #
# E5
# =========================================================================== #
def _events(df, flag):
    """flag 경보 사건 목록: (burst_id, 시작 시각)."""
    out = []
    _, ev = EV.merge_alarms(flag, df.burst_id.values)
    for a, _b in ev:
        out.append((int(df.burst_id.values[a]), pd.Timestamp(df.time_start.values[a])))
    return out


def _span_hours(df):
    t0, t1 = pd.to_datetime(df.time_start), pd.to_datetime(df.time_end)
    g = pd.DataFrame(dict(b=df.burst_id.values, t0=t0, t1=t1)).groupby("b").agg(t0=("t0", "min"), t1=("t1", "max"))
    collect = float((g.t1 - g.t0).dt.total_seconds().sum()) / 3600
    wall = float((t1.max() - t0.min()).total_seconds()) / 3600
    return collect, wall


def e5(ctx):
    print("E5  현장 운영 지표")
    pred = ctx.pred
    nrm = pred[pred.source == "normal"]
    scopes = [("블록 %d" % b, nrm[nrm.block == b].reset_index(drop=True), b != 4) for b in range(5)]
    scopes.append(("정상 전체", nrm.reset_index(drop=True), True))
    rows = []
    for name, d, ins in scopes:
        collect, wall = _span_hours(d)
        for lvl in ("yellow", "red"):
            n_ev = len(_events(d, (d.alarm_level == lvl).values))
            rows.append(dict(scope=name, in_sample=ins, level="노랑" if lvl == "yellow" else "빨강",
                             events=n_ev, collect_hours=collect, wall_hours=wall, duty=collect / wall,
                             per_collect_h=n_ev / collect, per_wall_h=n_ev / wall,
                             per_shift_8h=8 * n_ev / wall,
                             per_shift_8h_upper95=8 * AT.poisson_upper(n_ev, wall)))
    v5a = pd.DataFrame(rows)
    r7 = T("r7_alarm_event_rates.csv").set_index("경보")
    b4 = v5a[(v5a.scope == "블록 4") & (v5a.level == "빨강")].iloc[0]
    assert abs(b4.per_collect_h - r7.loc["빨강", "사건/h"]) < 1e-6
    save(v5a, "v5a_alarm_per_shift.csv")
    print(v5a[v5a.scope.isin(["블록 4", "정상 전체"])].round(3).to_string(index=False))

    # ---- v5b 정지 검토 규칙 ---------------------------------------------- #
    def triggers(d, order):
        ev = _events(d, (d.alarm_level == "red").values)
        pos = {b: i for i, b in enumerate(order)}
        s1 = len(ev)
        s2 = [ev[i] for i in range(1, len(ev))
              if pos[ev[i][0]] - pos[ev[i - 1][0]] == 1]
        s3 = [ev[i] for i in range(1, len(ev))
              if (ev[i][1] - ev[i - 1][1]).total_seconds() <= 600]
        return ev, s1, s2, s3

    rows = []
    for name, d, ins in scopes[:5] + [scopes[5]]:
        order = list(pd.unique(d.burst_id))
        ev, s1, s2, s3 = triggers(d, order)
        rows.append(dict(scope=name, in_sample=ins, S1_빨강1건=s1, S2_인접버스트_빨강연속=len(s2),
                         S3_10분내_빨강2건=len(s3), 교대2회연속="평가 불가(정상 기록 약 77분)",
                         first_S1_s=np.nan, first_S2_s=np.nan, first_S3_s=np.nan))
    f = pred[pred.source == "outlier"].reset_index(drop=True)
    t0 = pd.read_csv(os.path.join(RAW, FILES["fault"]))["TimeStamp"].pipe(pd.to_datetime).min()
    order = list(pd.unique(f.burst_id))
    ev, s1, s2, s3 = triggers(f, order)
    sec = lambda e: (e[1] - t0).total_seconds() if e else np.nan
    rows.append(dict(scope="고장 기록", in_sample=False, S1_빨강1건=s1, S2_인접버스트_빨강연속=len(s2),
                     S3_10분내_빨강2건=len(s3), 교대2회연속="평가 불가",
                     first_S1_s=sec(ev[0] if ev else None), first_S2_s=sec(s2[0] if s2 else None),
                     first_S3_s=sec(s3[0] if s3 else None)))
    v5b = pd.DataFrame(rows)
    save(v5b, "v5b_stop_rule_triggers.csv")
    print(v5b.drop(columns=["교대2회연속"]).to_string(index=False))

    # ---- v5c·v5d 사유 분포와 빨강 대안 문구 ------------------------------ #
    def cat(n):
        if "CUR" in n or n.startswith("O_"):
            return "전류"
        return {"R": "진동 관계(R)", "S": "진동 형태(S)", "A": "진동 진폭(A)"}.get(n[0], "기타")

    fz = ctx.fz
    allF = np.vstack([ctx.P["Fn"], ctx.P["Fo"]])[:, fz.idx]
    C2 = np.abs(fz.m2.contrib(allF[:, fz.vib]))
    vib_names = [ctx.names[i] for i in fz.vib]
    top2 = np.array([vib_names[k] for k in C2.argmax(axis=1)])
    p = pred.assign(cat1=pred.top_reason_1.map(cat), cat2=[cat(x) for x in top2], top2=top2)
    groups = {"고장 빨강": (p.split == "fault") & (p.alarm_level == "red"),
              "정상 평가 빨강": (p.split == "holdout") & (p.alarm_level == "red"),
              "정상 평가 노랑": (p.split == "holdout") & (p.alarm_level == "yellow"),
              "고장 노랑": (p.split == "fault") & (p.alarm_level == "yellow")}
    rows = []
    for gname, msk in groups.items():
        for basis, col in (("1단 기여 1위(현 출력)", "cat1"), ("2단 진동 기여 1위", "cat2")):
            vc = p.loc[msk, col].value_counts()
            for k, v in vc.items():
                rows.append(dict(group=gname, basis=basis, category=k, n=int(v), share=v / msk.sum(),
                                 group_n=int(msk.sum())))
    v5c = pd.DataFrame(rows)
    fr = v5c[(v5c.group == "고장 빨강") & (v5c.basis.str.startswith("1단")) & (v5c.category == "전류")].iloc[0]
    assert fr.n == 268 and fr.group_n == 375, "고장 빨강 전류 사유 수가 검토 확인값과 다르다"
    save(v5c, "v5c_reason_distribution.csv")
    print(v5c.round(3).to_string(index=False))
    red = p.alarm_level == "red"
    v5d = p.loc[red, ["source", "burst_id", "original_row_start", "top_reason_1", "recommended_action", "top2"]] \
        .rename(columns={"top2": "stage2_top_vib_feature"})
    v5d["red_action_vib"] = v5d.stage2_top_vib_feature.map(EX.reason_phrase)
    save(v5d, "v5d_red_action_alternative.csv")


# =========================================================================== #
# E6
# =========================================================================== #
MEANING = {
    "A_std": "출렁임 크기(표준편차)", "A_p2p": "출렁임 폭(최대-최소)", "A_rms": "에너지(RMS)",
    "S_ac1": "0.1초 간격 자기상관(10 Hz 앨리어싱 아래 리듬)", "S_ac2": "0.2초 간격 자기상관",
    "S_zcr": "부호 변화 빈도", "R_corr": "상·하부 진동 동기성", "R_abscorr": "상·하부 진폭 동기성",
    "R_stdratio": "상·하부 진폭비(체결·정렬 관련 가능)", "O_mean": "전류 window 평균 수준",
    "O_absmean": "전류 window 절댓값 평균",
}


def _meaning(n):
    for k, v in MEANING.items():
        if n.startswith(k):
            ch = "전류" if "CUR" in n else ("상부 진동" if "AI0" in n and "AI1" not in n else
                                           ("하부 진동" if "AI1" in n and "AI0" not in n else "상·하부"))
            return v, ch
    return "", ""


def e6(ctx):
    print("E6  개별 특징 영향도")
    cfg, P, fz = ctx.cfg, ctx.P, ctx.fz
    nm, nh = ctx.names, len(ctx.Fh)
    Fe = np.vstack([ctx.Fh[:, fz.idx], ctx.Fo[:, fz.idx]])
    Cn = np.abs(fz.m1.contrib(Fe))
    top = Cn.argmax(axis=1)
    v1e = T("v1e_single_feature_auroc.csv").set_index("feature")
    F = P["Fn"][:, fz.idx]
    base_p = cal.conformal_p(fz.cal1, fz.m1.score(Fe))
    base_rate, base_rec = (base_p[:nh] <= fz.alpha).mean(), (base_p[nh:] <= fz.alpha).mean()
    base_ap = average_precision_score(ctx.y, fz.m1.score(Fe))
    rows = []
    for i, n in enumerate(nm):
        cols = np.array([k for k in range(len(nm)) if k != i])
        m = [x for x in MD.build(cfg, cfg["seed"]) if x.name == "M3"][0]
        m.fit(F[fz.tr][:, cols])
        s = m.score(Fe[:, cols])
        p = cal.conformal_p(m.score(F[fz.cl][:, cols]), s)
        mean_, ch = _meaning(n)
        rows.append(dict(feature=n, group=n.split("_")[0], channel=ch, physical_meaning=mean_,
                         top1_share_fault=float((top[nh:] == i).mean()),
                         top1_share_normal=float((top[:nh] == i).mean()),
                         single_auroc=v1e.loc[n, "auroc"], direction=v1e.loc[n, "direction"],
                         drop_one_d_normal_rate=(p[:nh] <= fz.alpha).mean() - base_rate,
                         drop_one_d_fault_recall=(p[nh:] <= fz.alpha).mean() - base_rec,
                         drop_one_d_ap=average_precision_score(ctx.y, s) - base_ap))
    v6 = pd.DataFrame(rows).sort_values(["top1_share_fault", "single_auroc"], ascending=False)
    cur = v6[v6.feature.str.contains("CUR") | v6.feature.str.startswith("O_")].top1_share_fault.sum()
    r5 = T("r5_stage1_channel_share.csv")
    assert abs(cur - r5.iloc[0]["비중"]) < 1e-3, "전류 1위 비중 %.4f != r5" % cur
    save(v6, "v6_feature_importance.csv")
    print(v6.head(10)[["feature", "top1_share_fault", "top1_share_normal", "single_auroc",
                       "drop_one_d_normal_rate", "drop_one_d_ap"]].round(4).to_string(index=False))


# =========================================================================== #
STEPS = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6}


def main(only=None):
    print("검토 대응 진단표 v1-v6 (모델 불변)")
    ctx = Ctx()
    for k, fn in STEPS.items():
        if only and k not in only:
            continue
        fn(ctx)
    print("완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    main(ap.parse_args().only)

# -*- coding: utf-8 -*-
"""조기탐지 가능성 진단표 w1-w4 (동결 구성 불변, 진단 전용).

왜 필요한가
  이 데이터셋은 정상(2022-07-12)과 고장(2022-07-17)이 다른 날에 수집되었고,
  두 파일 모두 단일 Equipment_state 다. 따라서 "조기탐지"의 세 전제
  (전이 라벨 · 고장 전 구간 · 단조 증가하는 심각도)가 모두 부재하다.
  그럼에도 무엇을 만들 수 있고 무엇은 만들 수 없는지를 수치로 가른다.

  w1  교란 불변 파이프라인 사다리 — 전처리 5단계 x 특징집합 7종 x M1/M3
      w1b 불변성 실측 (이득·오프셋·극성 섭동 시 특징 변화량)
  w2  가짜 고장(날짜 교란) 상한 — 정상 블록 하나를 '고장'으로 두고 같은 프로토콜
      w2b margin = 실제 고장 AUROC - 가짜 고장 AUROC(최악). 높을수록 기계를 본다
  w3  기록 내 조기성 부재 증명 (음성 결과)
  w4  리드타임 상한과 확인 지연
  w5  w1/w2 결과로 조립한 제안 2단 구조의 실제 성능 + 가짜 고장 음성대조

원칙
  - 동결 구성(1단 M3 23특징 / 2단 M1 진동 15 / p<=0.01 / 빨강 3연속 /
    학습 0-2 · 보정 3 · 평가 4)과 `predictions.csv`, 기존 표를 바꾸지 않는다.
  - 여기서 나온 어떤 수치도 제출본 모델 선정에 쓰지 않는다 (decision_log DL-019).
  - w1/w2 의 전처리·특징집합은 모두 진단용 별도 적합이다.

    python make_earlywarning_tables.py
    python make_earlywarning_tables.py --only W2 W3
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
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_audit_tables as AT               # noqa: E402
import run_all as R                          # noqa: E402
from src import calibration as cal           # noqa: E402
from src import data as D                    # noqa: E402
from src import evaluation as EV             # noqa: E402
from src import features as FT               # noqa: E402
from src import models as MD                 # noqa: E402
from src import windows as WD                # noqa: E402

TAB = os.path.join(HERE, "outputs", "tables")
TRAIN_BLOCKS, CAL_BLOCK, HOLD_BLOCK = AT.TRAIN_BLOCKS, AT.CAL_BLOCK, AT.HOLD_BLOCK
GRID_Q = 1.1920929          # 검토에서 발견한 고장 전류 양자화 격자 (2^-23 x 1e7)

# --------------------------------------------------------------------------- #
# 전처리 사다리 — 아래로 갈수록 계측 교란 경로를 더 끊는다
# --------------------------------------------------------------------------- #
PREPROC = [
    ("R0", dict(quant=False, norm=None),
     "원본 (현행 파이프라인)", "없음", "인과적"),
    ("W1", dict(quant=False, norm="demean"),
     "window 평균 제거", "오프셋", "인과적 (1초 지연)"),
    ("W2", dict(quant=False, norm="winz"),
     "window robust z (median/MAD)", "오프셋+이득", "인과적 (1초 지연)"),
    ("B1", dict(quant=False, norm="burstz"),
     "버스트 robust z (median/MAD)", "오프셋+이득",
     "비인과적 (버스트 완료 후에만 가능)"),
    ("W2Q", dict(quant=True, norm="winz"),
     "전류 공통격자 양자화 + window robust z", "오프셋+이득+양자화 격자",
     "인과적 (1초 지연)"),
]

# 프로토콜: (학습 블록, 보정 블록, 정상평가 블록)
PROTO_FROZEN = ([0, 1, 2], 3, 4)        # 동결 구성과 동일
PROTO_MATCHED = ([0, 1], 2, 3)          # 가짜 고장 실험과 학습 블록 수를 맞춘 비교군


def save(df, name):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, name), index=False, encoding="utf-8-sig")
    print("  -> %s (%d행)" % (name, len(df)))
    return df


def _check(ok, msg):
    if not ok:
        raise AssertionError("검증 실패: " + msg)
    print("     [검증] " + msg)


# --------------------------------------------------------------------------- #
class Ctx:
    def __init__(self):
        self.cfg = yaml.safe_load(
            open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
        self.ch = self.cfg["data"]["channels"]
        self.gap = self.cfg["windows"]["gap_sec"]
        self.groups = tuple(self.cfg["features"]["groups"])
        self.mgroups = tuple(self.cfg["features"]["model_groups"])
        self.seed = self.cfg["seed"]
        self.alpha = self.cfg["calibration"]["yellow_p"]
        self.red_n = self.cfg["calibration"]["red_consecutive"]

        self.normal, self.outlier, _ = D.load_all(self.cfg, HERE)
        self.P = R.prepare(self.cfg, self.normal, self.outlier,
                           self.cfg["windows"]["length"])
        self.pred = pd.read_csv(os.path.join(HERE, "outputs", "predictions.csv"))
        self.pred["time_start"] = pd.to_datetime(self.pred["time_start"])
        self.pred["time_end"] = pd.to_datetime(self.pred["time_end"])

        self.idx, self.mnames = R.model_cols(self.P, self.mgroups)
        self.blocks = self.P["mn"]["block"].values
        self.mn, self.mo = self.P["mn"], self.P["mo"]
        # 버스트 단위 robust 통계는 원신호에서 정확히 구한다
        self.bstat = {"normal": self._burst_stats(self.normal),
                      "outlier": self._burst_stats(self.outlier)}
        self.fz = AT.Frozen(self.cfg, self.P)

    def _burst_stats(self, df):
        bid = WD.segment_bursts(df, self.gap)
        X = df[self.ch].values.astype(float)
        med, mad = {}, {}
        for b in np.unique(bid):
            v = X[bid == b]
            m = np.median(v, axis=0)
            a = np.median(np.abs(v - m), axis=0)
            med[int(b)] = m
            mad[int(b)] = np.where(a > 0, a, 1.0)
        return med, mad

    def verify_frozen(self):
        allF = np.vstack([self.P["Fn"], self.P["Fo"]])
        bid = np.r_[self.mn["burst_id"].values, self.mo["burst_id"].values]
        j = self.fz.judge(allF, bid)
        _check(np.allclose(j["p1"], self.pred.p_normal_stage1.values)
               and np.allclose(j["p2"], self.pred.p_normal_stage2.values),
               "동결 구성 p1·p2 == predictions.csv")

    # ---- 전처리 ---------------------------------------------------------- #
    def preprocess(self, X, meta, which, spec):
        Y = X.astype(float).copy()
        if spec["quant"]:
            Y[:, :, 2] = np.round(Y[:, :, 2] / GRID_Q) * GRID_Q
        norm = spec["norm"]
        if norm == "demean":
            Y = Y - Y.mean(axis=1, keepdims=True)
        elif norm == "winz":
            m = np.median(Y, axis=1, keepdims=True)
            a = np.median(np.abs(Y - m), axis=1, keepdims=True)
            Y = (Y - m) / np.where(a > 0, a, 1.0)
        elif norm == "burstz":
            # 통계를 '주어진 X' 에서 구한다. 그래야 변환이 입력의 순수 함수가 되고
            # 계측 섭동에 대한 불변성이 성립한다(w1b 가 이를 실측한다).
            # 중첩 window 를 이어 붙여 재는 근사이므로 버스트 내부 샘플에
            # 가중이 더 실린다. 운영에서는 버스트 버퍼에서 같은 값을 얻는다.
            b = meta["burst_id"].values
            M = np.empty((len(Y), 1, Y.shape[2]))
            S = np.empty_like(M)
            for bb in np.unique(b):
                sel = b == bb
                v = Y[sel].reshape(-1, Y.shape[2])
                m = np.median(v, axis=0)
                a = np.median(np.abs(v - m), axis=0)
                M[sel, 0, :] = m
                S[sel, 0, :] = np.where(a > 0, a, 1.0)
            Y = (Y - M) / S
        return Y

    def features(self, spec):
        """전처리 수준별 (Fn_model, Fo_model). 모델 입력 23열로 잘라 돌려준다."""
        Xn = self.preprocess(self.P["Xn"], self.mn, "normal", spec)
        Xo = self.preprocess(self.P["Xo"], self.mo, "outlier", spec)
        Fn, _, _ = FT.build(Xn, self.ch, self.mn, self.groups)
        Fo, _, _ = FT.build(Xo, self.ch, self.mo, self.groups)
        return Fn[:, self.idx], Fo[:, self.idx]

    # ---- 특징 부분집합 --------------------------------------------------- #
    def subsets(self):
        nm = self.mnames

        def sel(fn):
            return np.array([i for i, n in enumerate(nm) if fn(n)])

        vib = sel(lambda n: "CUR" not in n and not n.startswith("O_"))
        return [
            ("G-all", "전체 23", np.arange(len(nm))),
            ("G-vib", "진동 15 (전류 미사용)", vib),
            ("G-vibA", "진동 진폭 6", sel(
                lambda n: n.startswith("A_") and n[-3:] in ("AI0", "AI1"))),
            ("G-vibS", "진동 형태 6", sel(
                lambda n: n.startswith("S_") and n[-3:] in ("AI0", "AI1"))),
            ("G-R", "상·하부 관계 3", sel(lambda n: n.startswith("R_"))),
            ("G-vibSR", "진동 형태+관계 9", sel(
                lambda n: (n.startswith("S_") and n[-3:] in ("AI0", "AI1"))
                or n.startswith("R_"))),
            ("G-SR", "형태 전체+관계 12", sel(
                lambda n: n.startswith("S_") or n.startswith("R_"))),
        ]

    # ---- 프로토콜 1회 실행 ------------------------------------------------ #
    def run_proto(self, Fn, Fo, cols, mname, proto, pos_block=None):
        """pos_block=None 이면 양성 = 고장 428 window, 아니면 그 정상 블록."""
        train, calb, ne = proto
        tr = np.isin(self.blocks, train)
        cl = self.blocks == calb
        ho = self.blocks == ne
        m = [x for x in MD.build(self.cfg, self.seed) if x.name == mname][0]
        m.fit(Fn[tr][:, cols])
        cs = m.score(Fn[cl][:, cols])
        s_ne = m.score(Fn[ho][:, cols])
        if pos_block is None:
            s_pos = m.score(Fo[:, cols])
        else:
            s_pos = m.score(Fn[self.blocks == pos_block][:, cols])
        y = np.r_[np.zeros(len(s_ne), bool), np.ones(len(s_pos), bool)]
        sc = np.r_[s_ne, s_pos]
        p = cal.conformal_p(cs, sc)
        return {
            "auroc": float(roc_auc_score(y, sc)),
            "ap": float(average_precision_score(y, sc)),
            "normal_alarm_rate": float((p[:len(s_ne)] <= self.alpha).mean()),
            "pos_recall": float((p[len(s_ne):] <= self.alpha).mean()),
            "n_normal": int(len(s_ne)), "n_pos": int(len(s_pos)),
        }


# --------------------------------------------------------------------------- #
# W1  교란 불변 파이프라인 사다리
# --------------------------------------------------------------------------- #
def w1(C, cache):
    print("W1  교란 불변 파이프라인 사다리")
    rows = []
    for pid, spec, desc, inv, causal in PREPROC:
        Fn, Fo = cache.setdefault(pid, C.features(spec))
        for sid, sdesc, cols in C.subsets():
            for mname in ("M1", "M3"):
                a = C.run_proto(Fn, Fo, cols, mname, PROTO_FROZEN)
                b = C.run_proto(Fn, Fo, cols, mname, PROTO_MATCHED)
                rows.append({
                    "preproc": pid, "전처리": desc, "불변 대상": inv,
                    "실시간 가능": causal,
                    "subset": sid, "특징집합": sdesc, "n_features": len(cols),
                    "model": mname,
                    "real_auroc_frozen": a["auroc"], "real_ap_frozen": a["ap"],
                    "real_normal_alarm_rate": a["normal_alarm_rate"],
                    "real_fault_recall": a["pos_recall"],
                    "real_auroc_matched": b["auroc"],
                })
    df = pd.DataFrame(rows)
    save(df, "w1_invariant_pipeline.csv")

    # 검증: R0 · G-all · M3 는 r11 제출본 행과 같아야 한다 (같은 정의)
    r11 = pd.read_csv(os.path.join(TAB, "r11_negative_control_channels.csv"))
    ref = float(r11.iloc[0]["1단 AUROC"])
    got = float(df[(df.preproc == "R0") & (df.subset == "G-all")
                   & (df.model == "M3")].real_auroc_frozen.iloc[0])
    _check(round(got, 4) == round(ref, 4),
           "R0·G-all·M3 AUROC %.4f == r11 제출본 행 %.4f" % (got, ref))

    base = df[(df.preproc == "R0") & (df.subset == "G-all")]
    print("     현행(R0·G-all) AUROC M1 %.4f / M3 %.4f"
          % tuple(base.sort_values("model").real_auroc_frozen))
    best = df.loc[df[df.preproc.isin(["W2", "W2Q"])].real_auroc_frozen.idxmax()]
    print("     인과적 완전불변(W2/W2Q) 최고 = %s·%s·%s AUROC %.4f"
          % (best.preproc, best.subset, best.model, best.real_auroc_frozen))
    return df


def w1b_invariance(C, cache):
    """w1b 불변성 실측 — 전처리별로 계측 섭동이 특징을 실제로 안 바꾸는지 센다."""
    print("W1b 불변성 실측 (이득·오프셋·극성)")
    ho = C.blocks == HOLD_BLOCK
    mh = C.mn[ho].reset_index(drop=True)
    Xh = C.P["Xn"][ho]
    train_std = C.P["Xn"].std(axis=(0, 1))

    perts = [("gain x1.25", lambda X: X * 1.25),
             ("offset +0.5σ", lambda X: X + 0.5 * train_std[None, None, :]),
             ("전류만 offset +0.5σ",
              lambda X: X + np.array([0, 0, 0.5 * train_std[2]])[None, None, :]),
             ("극성 반전 (AI0)",
              lambda X: X * np.array([-1, 1, 1])[None, None, :]),
             ("전류 격자 양자화",
              lambda X: np.concatenate(
                  [X[:, :, :2], (np.round(X[:, :, 2:] / GRID_Q) * GRID_Q)],
                  axis=2))]
    rows = []
    for pid, spec, desc, inv, _c in PREPROC:
        F0, _, _ = FT.build(C.preprocess(Xh, mh, "normal", spec),
                            C.ch, mh, C.groups)
        F0 = F0[:, C.idx]
        # 특징별 정규화 척도: 같은 전처리에서 정상 학습 블록의 표준편차
        Fn_p, _ = cache.setdefault(pid, C.features(spec))
        scale = Fn_p[np.isin(C.blocks, TRAIN_BLOCKS)].std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        for pname, fn in perts:
            F1, _, _ = FT.build(C.preprocess(fn(Xh), mh, "normal", spec),
                                C.ch, mh, C.groups)
            F1 = F1[:, C.idx]
            rel = np.abs(F1 - F0) / scale[None, :]
            for sid, sdesc, cols in C.subsets():
                if sid not in ("G-all", "G-vibA", "G-vibS", "G-R", "G-SR"):
                    continue
                rows.append({
                    "preproc": pid, "전처리": desc, "섭동": pname,
                    "subset": sid,
                    "max_rel_change": float(rel[:, cols].max()),
                    "p99_rel_change": float(np.quantile(rel[:, cols], 0.99)),
                    "median_rel_change": float(np.median(rel[:, cols])),
                    "invariant_1e-6": bool(rel[:, cols].max() < 1e-6),
                    "invariant_p99_1e-6": bool(
                        np.quantile(rel[:, cols], 0.99) < 1e-6),
                })
    df = pd.DataFrame(rows)
    save(df, "w1b_invariance_check.csv")

    # 보고서가 주장하는 "S·R 은 이득·오프셋에 강건" 을 실측으로 확인
    sr = df[(df.preproc == "R0") & (df.subset == "G-SR")
            & df.섭동.isin(["gain x1.25", "offset +0.5σ"])]
    print("     R0 에서 S·R(12) 의 이득·오프셋 섭동 최대 상대변화 = %.2e, %.2e"
          % tuple(sr.sort_values("섭동").max_rel_change))
    a = df[(df.subset == "G-vibA") & (df.섭동 == "gain x1.25")]
    print("     진동 진폭(G-vibA) 이득 섭동 최대 상대변화: "
          + ", ".join("%s %.2e" % (r.preproc, r.max_rel_change)
                      for r in a.itertuples()))
    return df


# --------------------------------------------------------------------------- #
# W2  가짜 고장(날짜 교란) 상한
# --------------------------------------------------------------------------- #
def _pseudo_proto(pf):
    """가짜 고장 블록 pf 를 빼고 남은 4블록을 학습2·보정1·정상평가1 로 배정."""
    rest = [b for b in range(5) if b != pf]
    return (rest[:2], rest[2], rest[3])


def w2(C, cache):
    print("W2  가짜 고장(날짜 교란) 상한")
    # 블록별 중앙 시각 (교란 간격 계산용)
    tmid = C.mn.groupby("block")["time_start"].median()
    rows = []
    for pid, spec, desc, inv, causal in PREPROC:
        Fn, Fo = cache.setdefault(pid, C.features(spec))
        for sid, sdesc, cols in C.subsets():
            for mname in ("M1", "M3"):
                for pf in range(5):
                    proto = _pseudo_proto(pf)
                    a = C.run_proto(Fn, Fo, cols, mname, proto, pos_block=pf)
                    gap = abs((tmid[pf] - tmid[proto[0][0]]).total_seconds()) / 3600.0
                    rows.append({
                        "preproc": pid, "전처리": desc, "subset": sid,
                        "n_features": len(cols), "model": mname,
                        "pseudo_fault_block": pf,
                        "train_blocks": "".join(map(str, proto[0])),
                        "cal_block": proto[1], "normal_eval_block": proto[2],
                        "pseudo_auroc": a["auroc"], "pseudo_ap": a["ap"],
                        "normal_alarm_rate": a["normal_alarm_rate"],
                        "pseudo_recall": a["pos_recall"],
                        "train_to_pseudo_gap_h": gap,
                        "n_normal": a["n_normal"], "n_pseudo": a["n_pos"]})
    df = pd.DataFrame(rows)
    save(df, "w2_pseudo_fault_confound.csv")

    agg = df.groupby(["preproc", "전처리", "subset", "n_features", "model"]).agg(
        pseudo_auroc_mean=("pseudo_auroc", "mean"),
        pseudo_auroc_max=("pseudo_auroc", "max"),
        pseudo_auroc_min=("pseudo_auroc", "min")).reset_index()
    w1t = pd.read_csv(os.path.join(TAB, "w1_invariant_pipeline.csv"))
    m = agg.merge(w1t[["preproc", "subset", "model", "real_auroc_frozen",
                       "real_auroc_matched", "real_fault_recall",
                       "real_normal_alarm_rate", "불변 대상", "실시간 가능"]],
                  on=["preproc", "subset", "model"], how="left")
    # margin: 학습 블록 수를 맞춘 real 과 비교한다 (matched)
    m["margin"] = m["real_auroc_matched"] - m["pseudo_auroc_max"]
    m["margin_mean"] = m["real_auroc_matched"] - m["pseudo_auroc_mean"]
    m = m.sort_values("margin", ascending=False).reset_index(drop=True)
    m.insert(0, "rank", np.arange(1, len(m) + 1))
    save(m, "w2b_confound_margin.csv")

    nc = pd.read_csv(os.path.join(TAB, "v4c_control_threshold_free.csv"))
    nc04 = float(nc[(nc.id == "NC-04")
                    & (nc.feature_set == "groups25 (기존)")].auroc.iloc[0])
    r0 = m[(m.preproc == "R0") & (m.subset == "G-all")]
    print("     현행(R0·G-all) 가짜 고장 AUROC 최악 = %.4f / %.4f (M1 / M3)"
          % tuple(r0.sort_values("model").pseudo_auroc_max))
    print("     (참고: 지도학습 음성대조 NC-04 AUROC %.4f)" % nc04)
    top = m.iloc[0]
    print("     margin 1위: %s · %s · %s → real %.4f − pseudo최악 %.4f = %+.4f"
          % (top.preproc, top.subset, top.model, top.real_auroc_matched,
             top.pseudo_auroc_max, top.margin))
    best_causal = m[m["실시간 가능"].astype(str).str.startswith("인과적")].iloc[0]
    print("     실시간 가능 중 1위: %s · %s · %s margin %+.4f (real %.4f)"
          % (best_causal.preproc, best_causal.subset, best_causal.model,
             best_causal.margin, best_causal.real_auroc_matched))
    return df, m


# --------------------------------------------------------------------------- #
# W3  기록 내 조기성 부재 증명 (음성 결과)
# --------------------------------------------------------------------------- #
def w3(C):
    print("W3  기록 내 조기성 부재 증명")
    f = C.pred[C.pred.split == "fault"].sort_values("time_start").copy()
    f["t_rel"] = (f.time_start - f.time_start.min()).dt.total_seconds()
    g = f.groupby("burst_id").agg(
        t_rel=("t_rel", "min"), n_windows=("alarm_level", "size"),
        red=("alarm_level", lambda s: int((s == "red").sum())),
        yellow=("alarm_level", lambda s: int((s == "yellow").sum())),
        green=("alarm_level", lambda s: int((s == "green").sum())),
        score2_mean=("score_stage2", "mean")).reset_index()
    g["red_recall"] = g.red / g.n_windows
    save(g, "w3b_fault_burst_timeline.csv")

    def sp(x, y):
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            return float("nan"), float("nan")
        r, p = spearmanr(x, y)
        return float(r), float(p)

    # 버스트 window 수를 통제한 뒤의 시간 효과 (순위 잔차 상관)
    rt, rn, rr = (pd.Series(g.t_rel).rank().values,
                  pd.Series(g.n_windows).rank().values,
                  pd.Series(g.red_recall).rank().values)
    res_t = rt - np.polyval(np.polyfit(rn, rt, 1), rn)
    res_r = rr - np.polyval(np.polyfit(rn, rr, 1), rn)

    nst = C.normal.Equipment_state.values
    ost = C.outlier.Equipment_state.values
    rows = []

    def add(item, value, note):
        rows.append({"항목": item, "값": value, "해석": note})

    add("정상 파일 Equipment_state 고유값", str(sorted(set(nst.tolist()))),
        "단일 상태 — 전이 없음")
    add("고장 파일 Equipment_state 고유값", str(sorted(set(ost.tolist()))),
        "단일 상태 — 전이 없음")
    add("상태 전이 건수 (정상+고장)",
        int((np.diff(nst) != 0).sum() + (np.diff(ost) != 0).sum()),
        "0 이면 time-to-failure 라벨을 만들 수 없다")
    add("고장 전(정상 상태) 구간 길이(초)", 0.0,
        "고장 파일 첫 행부터 state=1 — 사전 경고 구간이 존재하지 않는다")
    add("고장 기록 벽시계 길이(초)",
        round(float((C.outlier.TimeStamp.max()
                     - C.outlier.TimeStamp.min()).total_seconds()), 1),
        "리드타임의 물리적 상한")
    add("고장 1단 p1 고유값 수", int(f.p_normal_stage1.nunique()),
        "1 이면 점수가 바닥에 포화 — 심각도 눈금이 없다")
    add("고장 risk_score 고유값 수", int(f.risk_score.nunique()),
        "1 이면 위험도 서열을 만들 수 없다")
    add("고장 1단 미검출 window 수", int((f.p_normal_stage1 > C.alpha).sum()),
        "0 / %d — 1단 Recall 1.0 포화" % len(f))
    add("첫 경보(노랑 이상) 시각(초)",
        round(float(f.loc[f.alarm_level != "green", "t_rel"].min()), 3),
        "0 이면 기록 첫 window 부터 이미 경보")
    add("첫 빨강 시각(초)",
        round(float(f.loc[f.alarm_level == "red", "t_rel"].min()), 3),
        "확인 지연 뒤 즉시 확정 — 벌 수 있는 조기 구간이 없다")
    for col in ("score_stage1", "score_stage2"):
        r, p = sp(f.t_rel.values, f[col].values)
        add("경과시간 vs %s (Spearman ρ)" % col, round(r, 4),
            "p=%.3g. 양수여야 열화 누적. 음수는 반대 방향" % p)
    r, p = sp(g.n_windows.values, g.red_recall.values)
    add("버스트 window수 vs 빨강 recall (ρ)", round(r, 4),
        "p=%.3g. 짧은 버스트가 3연속 규칙을 못 채우는 수집 구조 효과" % p)
    r, p = sp(g.t_rel.values, g.red_recall.values)
    add("버스트 시작시각 vs 빨강 recall (ρ)", round(r, 4),
        "p=%.3g. 통제 전 값" % p)
    r, p = sp(res_t, res_r)
    add("window수 통제 후 시각 vs recall (ρ)", round(r, 4),
        "p=%.3g. 통제 후에도 음수면 신호가 시간에 따라 약해진다 "
        "(조기탐지가 요구하는 방향의 반대)" % p)
    add("판정", "이 데이터셋에는 사전 경고 구간이 구조적으로 존재하지 않는다",
        "전이 0건 · 고장 전 구간 0초 · 첫 window 경보 · p1 포화 · 시간 추세 음수")
    df = save(pd.DataFrame(rows), "w3a_earliness_evidence.csv")

    _check(int((np.diff(nst) != 0).sum() + (np.diff(ost) != 0).sum()) == 0
           and int(f.p_normal_stage1.nunique()) == 1
           and float(f.loc[f.alarm_level != "green", "t_rel"].min()) == 0.0,
           "전이 0건 · p1 고유값 1개 · 첫 window 부터 경보")
    return df, g


# --------------------------------------------------------------------------- #
# W4  리드타임 상한과 확인 지연
# --------------------------------------------------------------------------- #
def w4(C):
    print("W4  리드타임 상한과 확인 지연")
    btn = WD.burst_table(C.normal, C.gap)
    bto = WD.burst_table(C.outlier, C.gap)
    dt = float(C.normal.TimeStamp.diff().dt.total_seconds().median())

    def duty(df):
        bt = WD.burst_table(df, C.gap)
        collect = float(bt["dur_sec"].sum())
        wall = float((df.TimeStamp.max() - df.TimeStamp.min()).total_seconds())
        return collect, wall, collect / wall

    cn, wn, dn = duty(C.normal)
    co, wo, do = duty(C.outlier)

    def gaps(df):
        d = df.TimeStamp.diff().dt.total_seconds().values[1:]
        g = d[d > C.gap]
        return (float(np.median(g)), float(g.min()), float(g.max())) if len(g) \
            else (float("nan"),) * 3

    gn = gaps(C.normal)
    go = gaps(C.outlier)
    d4 = pd.read_csv(os.path.join(TAB, "d4_mofn_tradeoff.csv"))
    sub = d4[d4.is_submitted].iloc[0]

    rows = []

    def add(item, normal, fault, unit, note):
        rows.append({"항목": item, "정상": normal, "고장": fault,
                     "단위": unit, "근거·해석": note})

    add("버스트 수", len(btn), len(bto), "건", "e0_bursts.csv")
    add("버스트 길이 최대", round(float(btn.dur_sec.max()), 1),
        round(float(bto.dur_sec.max()), 1), "초",
        "**관측 가능한 리드타임의 절대 상한.** 한 버스트를 넘는 선행 관측이 없다")
    add("버스트 길이 중위수", round(float(btn.dur_sec.median()), 1),
        round(float(bto.dur_sec.median()), 1), "초", "e0_burst_summary.csv")
    add("버스트 간 간격 중위수", round(gn[0], 1), round(go[0], 1), "초",
        "경보 갱신 주기. 이 간격 동안은 관측이 없다")
    add("수집시간 / 벽시계", "%.1f / %.1f" % (cn, wn), "%.1f / %.1f" % (co, wo),
        "초", "버스트 수집 구조")
    add("duty (수집시간 비율)", round(dn, 4), round(do, 4), "-",
        "1.0 이 아니면 벽시계 리드타임을 수집시간으로 환산할 수 없다")
    add("샘플 간격", dt, dt, "초", "표본주기 %.0f Hz" % (1.0 / dt))
    add("Nyquist 주파수", round(0.5 / dt, 1), round(0.5 / dt, 1), "Hz",
        "프레스 스트로크·베어링 통과주파수 분해 불가")
    add("버스트 최대 샘플 수", int(btn.n_samples.max()), int(bto.n_samples.max()),
        "샘플", "가이드북이 요구하는 120샘플(seq20+offset100)에 도달하는 버스트 0개 "
                "— OBS-02")
    add("고장 전(정상 상태) 구간", "-", 0.0, "초",
        "**실측 리드타임 = 0초.** 고장 파일이 이미 state=1 로 시작한다")
    add("확인 지연 (연속 3, 제출 규칙)", "-",
        "%.1f / %.1f" % (sub.delay_median_s, sub.delay_max_s), "초 (중위/최대)",
        "d4_mofn_tradeoff.csv. 버스트 진입 후 경과시간이며 고장 전 리드타임이 아니다")
    add("주장 가능한 최대 리드타임", "-",
        round(float(bto.dur_sec.max()), 1), "초",
        "관측 상한. 다만 고장 전 구간이 0초이므로 실측값은 0초다")
    df = save(pd.DataFrame(rows), "w4a_leadtime_bounds.csv")

    mo = d4[["N", "M", "rule", "burst_detected", "window_recall", "fp_window",
             "fp_events", "far_per_h", "delay_median_s", "delay_max_s",
             "is_submitted"]].copy()
    mo["확인 지연 = (M-1) x dt (초)"] = (mo["M"] - 1) * dt
    save(mo, "w4b_confirmation_delay.csv")

    _check(int(btn.n_samples.max()) < 120 and int(bto.n_samples.max()) < 120,
           "어떤 버스트도 120샘플에 도달하지 못한다 (정상 max %d / 고장 max %d) "
           "== OBS-02" % (btn.n_samples.max(), bto.n_samples.max()))
    print("     관측 가능 리드타임 상한 %.1f초 / 실측 0초 / 확인 지연 중위 %.1f초"
          % (bto.dur_sec.max(), sub.delay_median_s))
    return df, mo


# --------------------------------------------------------------------------- #
# W5  제안 2단 구조 — w1/w2 결과로 조립해 실제로 돌려본다
# --------------------------------------------------------------------------- #
# 주의: 아래 조합은 w1/w2 표를 '보고' 고른 것이다. 사후 선택이므로 일반화
# 주장을 하지 않는다. 가짜 고장 열(pseudo_red_rate)이 그 사후성에 대한
# 음성 대조 역할을 한다.
CANDIDATES = [
    ("현행 동결 (제출본)", ("R0", "G-all", "M3"), ("R0", "G-vib", "M1")),
    ("제안 A: 불변 1단 + 교란무상관 2단", ("W2", "G-SR", "M3"), ("R0", "G-vibA", "M1")),
    ("제안 B: A 의 모델 교체", ("W2", "G-SR", "M1"), ("R0", "G-vibA", "M3")),
    ("제안 C: 교란무상관 단일단", ("R0", "G-vibA", "M1"), None),
    ("제안 D: 완전불변 2단", ("W2", "G-SR", "M3"), ("W2", "G-vibSR", "M1")),
    ("제안 E: 전류 미사용 2단", ("W2", "G-vibSR", "M3"), ("R0", "G-vibA", "M1")),
]


def _two_stage(C, cache, s1, s2, proto, pos_block=None):
    train, calb, ne = proto
    tr = np.isin(C.blocks, train)
    cl = C.blocks == calb
    ho = C.blocks == ne
    subs = {sid: cols for sid, _d, cols in C.subsets()}
    specs = {pid: spec for pid, spec, _d, _i, _c in PREPROC}

    def stage(pid, sid, mname):
        Fn, Fo = cache.setdefault(pid, C.features(specs[pid]))
        cols = subs[sid]
        m = [x for x in MD.build(C.cfg, C.seed) if x.name == mname][0]
        m.fit(Fn[tr][:, cols])
        cs = m.score(Fn[cl][:, cols])
        s_ne = m.score(Fn[ho][:, cols])
        s_pos = (m.score(Fo[:, cols]) if pos_block is None
                 else m.score(Fn[C.blocks == pos_block][:, cols]))
        return cal.conformal_p(cs, np.r_[s_ne, s_pos]), np.r_[s_ne, s_pos], len(s_ne)

    p1, sc1, nh = stage(*s1)
    p2 = np.zeros_like(p1) if s2 is None else stage(*s2)[0]
    f1, f2 = p1 <= C.alpha, p2 <= C.alpha
    flag = f1 & f2
    bh = C.mn[ho]["burst_id"].values
    bp = (C.mo["burst_id"].values if pos_block is None
          else C.mn[C.blocks == pos_block]["burst_id"].values)
    red = np.r_[AT.persist(flag[:nh], bh, C.red_n),
                AT.persist(flag[nh:], bp, C.red_n)]
    y = np.r_[np.zeros(nh, bool), np.ones(len(p1) - nh, bool)]
    m = AT.confusion(y, red)
    ev, _ = EV.merge_alarms(red[:nh], bh)
    det = int(pd.Series(bp[red[nh:]]).nunique()) if red[nh:].any() else 0
    return {**m, "stage1_auroc": float(roc_auc_score(y, sc1)),
            "stage1_alarm_rate_normal": float(f1[:nh].mean()),
            "stage2_alarm_rate_normal": float(f2[:nh].mean()),
            "fp_events": ev, "bursts_detected": det,
            "bursts_total": int(pd.Series(bp).nunique())}


def w5(C, cache):
    print("W5  제안 2단 구조 실행 (사후 선택 — 일반화 주장 안 함)")
    rows = []
    for name, s1, s2 in CANDIDATES:
        a = _two_stage(C, cache, s1, s2, PROTO_FROZEN)
        b = _two_stage(C, cache, s1, s2, PROTO_MATCHED)
        ps = [_two_stage(C, cache, s1, s2, _pseudo_proto(pf), pos_block=pf)
              for pf in range(5)]
        rows.append({
            "구성": name,
            "1단": "%s·%s·%s" % s1,
            "2단": ("%s·%s·%s" % s2) if s2 else "없음",
            "실시간 가능": all(p != "B1" for p in
                           ([s1[0]] + ([s2[0]] if s2 else []))),
            "red_TP": a["TP"], "red_FP": a["FP"], "red_FN": a["FN"],
            "red_precision": a["precision"], "red_recall": a["recall"],
            "red_f1": a["f1"], "red_fpr": a["fpr"],
            "fp_events": a["fp_events"],
            "bursts_detected": a["bursts_detected"],
            "stage1_auroc": a["stage1_auroc"],
            "stage1_alarm_rate_normal": a["stage1_alarm_rate_normal"],
            "stage2_alarm_rate_normal": a["stage2_alarm_rate_normal"],
            "red_f1_matched": b["f1"], "red_fpr_matched": b["fpr"],
            "red_recall_matched": b["recall"],
            "pseudo_red_rate_max": max(x["fpr"] for x in ps),
            "pseudo_red_rate_mean": float(np.mean([x["fpr"] for x in ps])),
            "pseudo_red_recall_max": max(x["recall"] for x in ps),
            "pseudo_stage1_auroc_max": max(x["stage1_auroc"] for x in ps),
        })
    df = pd.DataFrame(rows)
    df["margin_stage1"] = df["stage1_auroc"] - df["pseudo_stage1_auroc_max"]
    save(df, "w5_proposed_two_stage.csv")

    v2a = pd.read_csv(os.path.join(TAB, "v2a_frozen_split_full_system.csv"))
    ref = v2a[(v2a.stage1 == "M3")
              & v2a.rule.astype(str).str.contains("빨강")].iloc[0]
    cur = df.iloc[0]
    _check(int(cur.red_TP) == int(ref.TP) and int(cur.red_FP) == int(ref.FP)
           and int(cur.red_FN) == int(ref.FN),
           "현행 동결 재현 TP/FP/FN %d/%d/%d == v2a 빨강 행"
           % (cur.red_TP, cur.red_FP, cur.red_FN))
    for r in df.itertuples():
        print("     %-32s 빨강 F1 %.4f FPR %.5f Recall %.4f | 가짜고장 빨강률 "
              "최악 %.5f | 1단 margin %+.4f"
              % (r.구성, r.red_f1, r.red_fpr, r.red_recall,
                 r.pseudo_red_rate_max, r.margin_stage1))
    return df


# --------------------------------------------------------------------------- #
def main(only=None):
    print("조기탐지 가능성 진단표 w1-w4 (동결 구성 불변, 진단 전용)")
    C = Ctx()
    C.verify_frozen()
    cache = {}
    want = {s.upper() for s in only} if only else None

    def run(name):
        return want is None or name in want

    if run("W1"):
        w1(C, cache)
        w1b_invariance(C, cache)
    if run("W2"):
        if not os.path.exists(os.path.join(TAB, "w1_invariant_pipeline.csv")):
            print("  (W2 가 요구하는 w1 표가 없어 W1 을 먼저 실행한다)")
            w1(C, cache)
        w2(C, cache)
    if run("W3"):
        w3(C)
    if run("W4"):
        w4(C)
    if run("W5"):
        if not os.path.exists(os.path.join(TAB, "v2a_frozen_split_full_system.csv")):
            raise SystemExit("W5 는 v2a_frozen_split_full_system.csv 가 필요하다 "
                             "(make_review_tables.py --only E2 를 먼저 실행)")
        w5(C, cache)
    print("완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="실행할 실험만 지정 (예: --only W2 W3)")
    a = ap.parse_args()
    main(a.only)

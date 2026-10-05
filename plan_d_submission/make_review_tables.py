# -*- coding: utf-8 -*-
"""최종보고서 적대적 검토 대응 진단표 v1-v6 (모델·특징·임계값·경보 규칙 불변).

검토에서 문구 수정만으로는 해결되지 않는 지적에 근거 표를 만든다.
모든 실험은 진단용이며, 결과를 모델 선정에 쓰지 않는다 (decision_log DL-018).

  E1 / v1a-v1e  수집 경로 진단과 교란 분해      (보고서 1장 · 3.1)
  E2 / v2a-v2d  최종 시스템 구조의 동일 조건 비교 (2장). v2d 는 후보별 섭동 시험(DL-020)
  E3 / v3a-v3b  버스트 단위 층화 클러스터 부트스트랩
  E4 / v4a-v4c  선정 민감도 + 대조군 임계값 무관 지표
  E5 / v5a-v5d  현장 운영 지표                   (4장)
  E6 / v6       개별 영향변수 순위               (3.1)

동결 구성(바꾸지 않는 것): 1단 M3 PCA-MSPC(특징 23), 2단 M1 Mahalanobis(진동 15),
conformal p <= 0.01, 빨강 = 같은 버스트 3 window 연속, 학습 블록 0-2, 보정 블록 3,
평가 블록 4.

    python make_review_tables.py
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

import make_audit_tables as AT               # noqa: E402  Frozen/persist/confusion 등
import run_all as R                          # noqa: E402  prepare/model_cols
from src import calibration as cal           # noqa: E402
from src import data as D                    # noqa: E402
from src import evaluation as EV             # noqa: E402
from src import explain as EX                # noqa: E402
from src import features as FT               # noqa: E402
from src import models as MD                 # noqa: E402
from src import windows as WD                # noqa: E402
from src.selection import FP_TOL_ABS, FP_TOL_REL, SORT_KEYS, _pick   # noqa: E402

TAB = os.path.join(HERE, "outputs", "tables")
TRAIN_BLOCKS, CAL_BLOCK, HOLD_BLOCK = AT.TRAIN_BLOCKS, AT.CAL_BLOCK, AT.HOLD_BLOCK

# 검토에서 발견한 고장 전류 양자화 격자 (2^-23 x 1e7)
GRID_Q = 1.1920929
# E3 부트스트랩 반복
N_BOOT = 5000

# --------------------------------------------------------------------------- #
# E6 물리 해석 사전 (코드 상수. 결과를 보고 바꾸지 않는다)
# --------------------------------------------------------------------------- #
CHANNEL_KO = {"AI0": "상부진동", "AI1": "하부진동", "CUR": "전류",
              "AI0_AI1": "상·하부진동"}
PHYSICS = {
    "A_std": "출렁임 크기(표준편차)",
    "A_p2p": "출렁임 크기(최대-최소)",
    "A_rms": "출렁임 크기(실효값)",
    "S_ac1": "0.1초 간격 자기상관. 10 Hz 앨리어싱 아래의 리듬. "
             "회전수·전원 주파수·수집 클록과 구분 불가",
    "S_ac2": "0.2초 간격 자기상관. 10 Hz 앨리어싱 아래의 리듬. "
             "회전수·전원 주파수·수집 클록과 구분 불가",
    "S_zcr": "부호 변화 빈도",
    "R_corr": "상·하부 진동의 동기성",
    "R_abscorr": "상·하부 진동의 동기성(절댓값)",
    "R_stdratio": "상·하부 진폭비. 체결·정렬과 관련 가능",
    "O_mean": "전류의 window 평균 수준",
    "O_absmean": "전류의 window 평균 수준(절댓값)",
}


def save(df, name):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, name), index=False, encoding="utf-8-sig")
    print("  -> %s (%d행)" % (name, len(df)))
    return df


def _check(ok, msg):
    if not ok:
        raise AssertionError("검증 실패: " + msg)
    print("     [검증] " + msg)


def feature_channel(name):
    if name.endswith("_AI0_AI1"):
        return CHANNEL_KO["AI0_AI1"]
    for k in ("AI0", "AI1", "CUR"):
        if name.endswith("_" + k):
            return CHANNEL_KO[k]
    return "기타"


def feature_physics(name):
    for pref in sorted(PHYSICS, key=len, reverse=True):
        if name.startswith(pref):
            return PHYSICS[pref]
    return "미분류"


def n_runs(flag):
    """연속 True 구간(=경보 사건) 개수. 단일 버스트 안에서 쓴다."""
    f = np.asarray(flag, dtype=bool)
    if not f.any():
        return 0
    return int(np.sum(f & ~np.r_[False, f[:-1]]))


def f1_of(prec, rec):
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def coin_f1(pi):
    """유병률 pi 에서 동전던지기(양성예측률 0.5)의 기대 F1."""
    return 2 * pi * 0.5 / (pi + 0.5) if pi + 0.5 else 0.0


def allpos_f1(pi):
    """유병률 pi 에서 '전부 양성' 규칙의 F1."""
    return 2 * pi / (pi + 1.0) if pi + 1.0 else 0.0


# --------------------------------------------------------------------------- #
# 공통 컨텍스트
# --------------------------------------------------------------------------- #
class Ctx:
    """데이터·window·특징·동결 구성을 한 번만 준비한다."""

    def __init__(self):
        self.cfg = yaml.safe_load(
            open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
        self.ch = self.cfg["data"]["channels"]
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
        self.vib = np.array([i for i, n in enumerate(self.mnames)
                             if "CUR" not in n and not n.startswith("O_")])
        self.cur = np.array([i for i, n in enumerate(self.mnames)
                             if "CUR" in n or n.startswith("O_")])
        mnb = self.P["mn"]["block"].values
        self.tr = np.isin(mnb, TRAIN_BLOCKS)
        self.cl = mnb == CAL_BLOCK
        self.ho = mnb == HOLD_BLOCK
        self.fz = AT.Frozen(self.cfg, self.P)

        # 평가 표본 (정상 블록 4 + 고장 428)
        self.mh = self.P["mn"][self.ho].reset_index(drop=True)
        self.Xh, self.Xo = self.P["Xn"][self.ho], self.P["Xo"]
        self.bh = self.mh["burst_id"].values
        self.bo = self.P["mo"]["burst_id"].values
        self.nh, self.no = int(self.ho.sum()), len(self.P["Xo"])
        self.y_eval = np.r_[np.zeros(self.nh, bool), np.ones(self.no, bool)]

    # ---- 0.4 동결 구성 재현 검증 ---------------------------------------- #
    def verify_frozen(self):
        allF = np.vstack([self.P["Fn"], self.P["Fo"]])
        bid = np.r_[self.P["mn"]["burst_id"].values,
                    self.P["mo"]["burst_id"].values]
        j = self.fz.judge(allF, bid)
        _check(np.allclose(j["p1"], self.pred.p_normal_stage1.values),
               "동결 1단 p1 == predictions.csv")
        _check(np.allclose(j["p2"], self.pred.p_normal_stage2.values),
               "동결 2단 p2 == predictions.csv")

    # ---- 보조: 모델 특징행렬 ------------------------------------------- #
    def Fn_m(self):
        return self.P["Fn"][:, self.idx]

    def Fo_m(self):
        return self.P["Fo"][:, self.idx]

    def fit_stage(self, model_name, cols, train_mask, cal_mask):
        """지정 특징 부분집합으로 모델을 적합하고 보정점수를 돌려준다(진단용)."""
        F = self.Fn_m()
        m = [x for x in MD.build(self.cfg, self.seed) if x.name == model_name][0]
        m.fit(F[train_mask][:, cols])
        return m, m.score(F[cal_mask][:, cols])

    def build_features(self, X, meta):
        F, _, _ = FT.build(X, self.ch, meta, self.groups)
        return F


# --------------------------------------------------------------------------- #
# E1  수집 경로 진단과 교란 분해
# --------------------------------------------------------------------------- #
def e1a_fingerprint(C):
    """v1a 수집 지문 — 원본 CSV 를 문자열로 읽어 소수 자릿수·격자를 센다."""
    print("E1a 수집 지문 (원본 데이터)")
    rows = []
    for src, key in (("normal", "normal"), ("outlier", "outlier")):
        path = os.path.join(HERE, C.cfg["data"][key])
        raw = pd.read_csv(path, dtype=str)
        for c in C.ch:
            s = raw[c].astype(str).str.strip()
            x = s.astype(float).values
            # 원문 문자열의 소수점 아래 자릿수 (지수표기는 가수부만 센다)
            mant = s.str.split("e", n=1).str[0].str.split("E", n=1).str[0]
            dec = mant.str.split(".").str[1].fillna("").str.len().values
            u = np.unique(x)
            d = np.diff(u)
            mpd = float(d[d > 0].min()) if (d > 0).any() else float("nan")
            base = {
                "source": src, "channel": c, "n": len(x),
                "n_unique": int(len(u)), "unique_ratio": len(u) / len(x),
                "max_decimals": int(dec.max()), "median_decimals": float(np.median(dec)),
                "float_artifact": int((dec >= 15).sum()),
                "min_pos_diff": mpd,
                "clip_share_min": float((x == x.min()).mean()),
                "clip_share_max": float((x == x.max()).mean()),
            }
            for gsrc, q in (("min_pos_diff", mpd), ("review_q", GRID_Q)):
                if not np.isfinite(q) or q <= 0:
                    resid = float("nan")
                else:
                    resid = float(np.abs(x / q - np.round(x / q)).max())
                rows.append({**base, "grid_source": gsrc, "grid_q": q,
                             "grid_resid_max": resid,
                             "on_grid": bool(np.isfinite(resid) and resid < 1e-3)})
    df = pd.DataFrame(rows)[[
        "source", "channel", "n", "n_unique", "unique_ratio", "max_decimals",
        "median_decimals", "float_artifact", "min_pos_diff", "grid_source",
        "grid_q", "grid_resid_max", "on_grid", "clip_share_min",
        "clip_share_max"]]
    save(df, "v1a_acquisition_fingerprint.csv")

    g = df[df.grid_source == "review_q"].set_index(["source", "channel"])
    fo = g.loc[("outlier", "AI2_Current")]
    no = g.loc[("normal", "AI2_Current")]
    _check(bool(fo["on_grid"]) and not bool(no["on_grid"]),
           "고장 전류 on_grid=True (resid %.3g) / 정상 전류 on_grid=False "
           "(resid %.4f) @ q=%.7f" % (fo["grid_resid_max"],
                                      no["grid_resid_max"], GRID_Q))
    return df


def e1b_basic_stats(C):
    """v1b 1장 기초 통계 보강."""
    print("E1b 1장 기초 통계 보강")
    n_norm, n_out = len(C.normal), len(C.outlier)
    n_tot = n_norm + n_out
    wn, wo = len(C.P["Xn"]), len(C.P["Xo"])
    rows = [
        {"항목": "원본 행 수", "정상": n_norm, "고장": n_out,
         "비고": "고장 비율 %.3f%% (%d / %s)" % (100.0 * n_out / n_tot, n_out,
                                            format(n_tot, ","))},
        {"항목": "window 수 (seq=10, gap-aware)", "정상": wn, "고장": wo,
         "비고": "고장 비율 %.3f%% (%d / %s)" % (100.0 * wo / (wn + wo), wo,
                                            format(wn + wo, ","))},
        {"항목": "평가 표본 유병률", "정상": C.nh, "고장": C.no,
         "비고": "%.4f (%d / %s) — 정상은 블록 4 만" % (
             C.no / (C.nh + C.no), C.no, format(C.nh + C.no, ","))},
        {"항목": "중복 TimeStamp 건수",
         "정상": int(C.normal.TimeStamp.duplicated().sum()),
         "고장": int(C.outlier.TimeStamp.duplicated().sum()),
         "비고": "처리 방식: 유지(제거하지 않음). src/data.py load_raw 는 중복 "
                "TimeStamp 를 삭제하지 않고, audit() 이 n_dup_timestamp 로만 센다. "
                "버스트 분할은 diff>gap 기준이므로 중복(diff=0)은 같은 버스트에 남는다"},
    ]
    # 이상치: 정상 학습 블록 0-2 원신호의 평균·표준편차 기준 |z| > 4
    tr_rows = np.zeros(n_norm, dtype=bool)
    sub = C.P["mn"][np.isin(C.P["mn"]["block"].values, TRAIN_BLOCKS)]
    for a, b in zip(sub["original_row_start"].values,
                    sub["original_row_end"].values):
        tr_rows[a:b + 1] = True
    for c in C.ch:
        xt = C.normal[c].values.astype(float)[tr_rows]
        mu, sd = xt.mean(), xt.std()
        rn = float((np.abs((C.normal[c].values.astype(float) - mu) / sd) > 4).mean())
        ro = float((np.abs((C.outlier[c].values.astype(float) - mu) / sd) > 4).mean())
        rows.append({"항목": "이상치 |z|>4 비율 (%s)" % c,
                     "정상": "%.4f%%" % (100 * rn), "고장": "%.4f%%" % (100 * ro),
                     "비고": "기준: 정상 학습 블록 0-2 원신호 (n=%s) mean=%.4f sd=%.4f"
                             % (format(int(tr_rows.sum()), ","), mu, sd)})
    # 생산단위 식별 가능성
    gap = C.cfg["windows"]["gap_sec"]
    btn = WD.burst_table(C.normal, gap)
    bto = WD.burst_table(C.outlier, gap)
    dt_n = C.normal["TimeStamp"].diff().dt.total_seconds().median()
    rows.append({"항목": "버스트 최대 길이(초)",
                 "정상": round(float(btn["dur_sec"].max()), 1),
                 "고장": round(float(bto["dur_sec"].max()), 1),
                 "비고": "샘플 간격 %.1f초 → 표본주기 10 Hz, Nyquist 5 Hz" % dt_n})
    rows.append({"항목": "생산단위(프레스 스트로크) 식별 가능성",
                 "정상": "불가", "고장": "불가",
                 "비고": "10 Hz 버스트 수집에서 프레스 스트로크 단위 식별은 불가. "
                        "버스트 최대 %.1f초·간격 %.1f초·Nyquist 5 Hz 로는 스트로크 "
                        "경계를 분해할 수 없다"
                        % (max(float(btn["dur_sec"].max()),
                               float(bto["dur_sec"].max())), dt_n)})
    return save(pd.DataFrame(rows), "v1b_basic_stats.csv")


def _judge_pair(C, Xh, Xo):
    """동결 시스템으로 (정상 블록4, 고장) 한 쌍을 판정하고 지표를 돌려준다."""
    Fh = C.build_features(Xh, C.mh)
    Fo = C.build_features(Xo, C.P["mo"])
    jh, jo = C.fz.judge(Fh, C.bh), C.fz.judge(Fo, C.bo)
    sc = np.r_[jh["score1"], jo["score1"]]
    red = np.r_[jh["red"], jo["red"]]
    m = AT.confusion(C.y_eval, red)
    return {
        "stage1_auroc": float(roc_auc_score(C.y_eval, sc)),
        "normal_stage1_rate": float(jh["s1"].mean()),
        "normal_red_rate": float(jh["red"].mean()),
        "fault_stage1_recall": float(jo["s1"].mean()),
        "fault_red_recall": float(jo["red"].mean()),
        "red_f1": float(m["f1"]),
    }


def e1c_counterfactual(C):
    """v1c 직류·양자화 반사실 시험 (동결 시스템. 모델·보정점수 불변)."""
    print("E1c 직류·양자화 반사실 시험")
    Fn_tr_cur = C.P["Xn"][C.tr][:, :, 2]
    shift = float(Fn_tr_cur.mean() - C.Xo[:, :, 2].mean())

    def quant(X):
        Y = X.copy()
        Y[:, :, 2] = np.round(Y[:, :, 2] / GRID_Q) * GRID_Q
        return Y

    def demean(X):
        Y = X.copy()
        Y[:, :, 2] -= Y[:, :, 2].mean(axis=1, keepdims=True)
        return Y

    def c1(X):
        Y = X.copy()
        Y[:, :, 2] += shift
        return Y

    conds = [
        ("C0", "없음 (재현 확인)", lambda X: X, lambda X: X,
         "표 2-8과 같아야 함"),
        ("C1", "고장 전류 상수 이동 %+.4f" % shift, lambda X: X, c1,
         "파일 간 직류 차이를 없애도 판정이 유지되는가"),
        ("C2", "정상(평가 블록) 전류를 고장 격자로 양자화 q=%.7f" % GRID_Q,
         quant, lambda X: X, "양자화 자체가 경보를 만드는가"),
        ("C3", "정상·고장 모두 window별 전류 평균 제거", demean, demean,
         "window 단위 직류 정보의 역할"),
        ("C4", "C1 + C2 동시", quant, c1,
         "수집 경로 차이 두 가지를 함께 없앤 경우"),
    ]
    rows = []
    for cid, desc, fh, fo, ask in conds:
        met = _judge_pair(C, fh(C.Xh), fo(C.Xo))
        rows.append({"ID": cid, "변환": desc, "묻는 것": ask, **met})
    df = pd.DataFrame(rows)
    base = df.iloc[0]
    df["d_stage1_auroc"] = df["stage1_auroc"] - base["stage1_auroc"]
    df["d_fault_red_recall"] = df["fault_red_recall"] - base["fault_red_recall"]
    df["d_normal_stage1_rate_pp"] = 100.0 * (df["normal_stage1_rate"]
                                             - base["normal_stage1_rate"])
    df["d_normal_red_rate_pp"] = 100.0 * (df["normal_red_rate"]
                                          - base["normal_red_rate"])

    # 0.5 사전 판정 규칙 (결과를 보기 전에 정한 해석)
    def verdict(r):
        if r["ID"] == "C0":
            return "재현 확인 행"
        if r["ID"] == "C1":
            return ("판정은 전류 직류 성분 차이에 의존하지 않는다"
                    if abs(r["d_stage1_auroc"]) < 0.005
                    and abs(r["d_fault_red_recall"]) < 0.02
                    else "판정의 일부가 전류 직류 차이에 의존한다 "
                         "(ΔAUROC %+.4f, Δ빨강Recall %+.4f)"
                         % (r["d_stage1_auroc"], r["d_fault_red_recall"]))
        if r["ID"] == "C2":
            return ("양자화 자체가 오경보를 만들지 않는다"
                    if abs(r["d_normal_stage1_rate_pp"]) < 0.5
                    else "양자화가 정상 1단 경보율을 %+.2f%%p 바꾼다"
                         % r["d_normal_stage1_rate_pp"])
        return "사전 규칙 없음 (보조 조건)"
    df["사전판정"] = df.apply(verdict, axis=1)
    save(df, "v1c_dc_quantization_counterfactual.csv")

    r2 = pd.read_csv(os.path.join(TAB, "r2_operational_performance.csv"))
    s1 = r2[r2.iloc[:, 0].astype(str).str.startswith("1단 경보")].iloc[0]
    rd = r2[r2.iloc[:, 0].astype(str).str.startswith("최종 빨강")].iloc[0]
    c0 = df.iloc[0]
    _check(round(c0["normal_stage1_rate"], 4) == round(float(s1["fpr"]), 4)
           and round(c0["normal_red_rate"], 4) == round(float(rd["fpr"]), 4)
           and round(c0["fault_red_recall"], 4) == round(float(rd["recall"]), 4)
           and round(c0["red_f1"], 4) == round(float(rd["f1"]), 4),
           "C0 == r2_operational_performance (1단 FPR %.4f / 빨강 FPR %.4f / "
           "빨강 Recall %.4f / 빨강 F1 %.4f)"
           % (c0["normal_stage1_rate"], c0["normal_red_rate"],
              c0["fault_red_recall"], c0["red_f1"]))
    return df


def _subsets(C):
    nm = C.mnames
    def sel(fn):
        return np.array([i for i, n in enumerate(nm) if fn(n)])
    return [
        ("G-all", "전체 23", np.arange(len(nm))),
        ("G-vib", "진동 전체 15 (=2단 입력)", C.vib),
        ("G-vibA", "진동 진폭 6",
         sel(lambda n: n.startswith("A_") and (n.endswith("_AI0") or n.endswith("_AI1")))),
        ("G-vibS", "진동 형태 6",
         sel(lambda n: n.startswith("S_") and (n.endswith("_AI0") or n.endswith("_AI1")))),
        ("G-R", "상·하부 관계 3", sel(lambda n: n.startswith("R_"))),
        ("G-cur", "전류 전체 8", C.cur),
        ("G-curA", "전류 진폭 3",
         sel(lambda n: n.startswith("A_") and n.endswith("_CUR"))),
        ("G-curS", "전류 형태 3",
         sel(lambda n: n.startswith("S_") and n.endswith("_CUR"))),
        ("G-O", "전류 수준 2", sel(lambda n: n.startswith("O_"))),
    ]


def e1d_separability(C):
    """v1d 채널·특징군별 분리력 (진단용 별도 적합. 동결 모델과 별개)."""
    print("E1d 채널·특징군별 분리력")
    Fn, Fo = C.Fn_m(), C.Fo_m()
    rows = []
    for sid, desc, cols in _subsets(C):
        for mname in ("M1", "M3"):
            m, cs = C.fit_stage(mname, cols, C.tr, C.cl)
            sh = m.score(Fn[C.ho][:, cols])
            so = m.score(Fo[:, cols])
            sc = np.r_[sh, so]
            p = cal.conformal_p(cs, sc)
            rows.append({
                "subset": sid, "설명": desc, "model": mname,
                "n_features": len(cols),
                "auroc": float(roc_auc_score(C.y_eval, sc)),
                "ap": float(average_precision_score(C.y_eval, sc)),
                "normal_alarm_rate": float((p[:C.nh] <= C.alpha).mean()),
                "fault_recall": float((p[C.nh:] <= C.alpha).mean()),
            })
    df = pd.DataFrame(rows)
    save(df, "v1d_channel_group_separability.csv")

    r11 = pd.read_csv(os.path.join(TAB, "r11_negative_control_channels.csv"))
    ref_all = float(r11.iloc[0]["1단 AUROC"])
    ref_cur = float(r11[r11.iloc[:, 0].astype(str).str.startswith("전류 전용")]
                    .iloc[0]["1단 AUROC"])
    got_all = float(df[(df.subset == "G-all") & (df.model == "M3")].auroc.iloc[0])
    got_cur = float(df[(df.subset == "G-cur") & (df.model == "M3")].auroc.iloc[0])
    _check(round(got_all, 4) == round(ref_all, 4)
           and round(got_cur, 4) == round(ref_cur, 4),
           "G-all·M3 AUROC %.4f == r11 전체 행 / G-cur·M3 %.4f == r11 전류 전용 행"
           % (got_all, got_cur))

    # 사전 판정 규칙
    va = df[df.subset == "G-vibA"]
    hit = va[(va.auroc >= 0.98) & (va.normal_alarm_rate <= 0.02)]
    if len(hit):
        print("     [사전판정] 전류를 전혀 보지 않는 진동 진폭만으로도 고장 기록이 "
              "갈린다 (%s)" % ", ".join("%s AUROC %.4f, 정상경보율 %.2f%%"
                                      % (r.model, r.auroc, 100 * r.normal_alarm_rate)
                                      for r in hit.itertuples()))
    else:
        print("     [사전판정] 진동 진폭 단독으로는 기준(AUROC>=0.98 & 정상경보율<=2%) "
              "미달 — 해당 문구를 쓰지 않는다")
    return df


def e1e_single_feature(C):
    """v1e 단일 특징 AUROC 순위 (E6 에서도 사용)."""
    print("E1e 단일 특징 AUROC")
    Fn, Fo = C.Fn_m(), C.Fo_m()
    rows = []
    for j, nm in enumerate(C.mnames):
        v = np.r_[Fn[C.ho][:, j], Fo[:, j]]
        a = float(roc_auc_score(C.y_eval, v))
        rows.append({
            "feature": nm, "group": nm.split("_")[0],
            "channel": feature_channel(nm),
            "single_auroc": max(a, 1 - a),
            "auroc_raw": a,
            "direction": "고장이 높음" if a >= 0.5 else "고장이 낮음",
            "normal_median": float(np.median(Fn[C.ho][:, j])),
            "fault_median": float(np.median(Fo[:, j])),
        })
    df = pd.DataFrame(rows).sort_values(
        "single_auroc", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    save(df, "v1e_single_feature_auroc.csv")
    top = df.iloc[0]
    if top["feature"].startswith("S_") and top["feature"].endswith("_CUR"):
        print("     [사전판정] 단일 특징 1위가 전류 형태(%s) — 1.5·3.1 교란 서술을 "
              "'전류 파형 형태'로 바꾼다" % top["feature"])
    else:
        print("     [사전판정] 단일 특징 1위 = %s (AUROC %.4f, %s) — 전류 형태가 "
              "1위가 아니므로 해당 교체 규칙은 발동하지 않는다"
              % (top["feature"], top["single_auroc"], top["channel"]))
    return df


def e1(C):
    e1a_fingerprint(C)
    e1b_basic_stats(C)
    e1c_counterfactual(C)
    e1d_separability(C)
    return e1e_single_feature(C)


# --------------------------------------------------------------------------- #
# E2  최종 시스템의 동일 조건 비교
# --------------------------------------------------------------------------- #
class FrozenAny(AT.Frozen):
    """Frozen 의 1단 모델만 교체한다. 2단·임계값·연속 규칙·블록은 동결과 같다.

    AT.Frozen.__init__ 이 1단을 먼저, 2단을 나중에 적합하는 순서에 의존한다.
    """

    def __init__(self, cfg, P, stage1_name="M3", stage1_cols=None):
        self.stage1_name = stage1_name
        self._stage1_fitted = False
        super().__init__(cfg, P, stage1_cols=stage1_cols)

    def _fit(self, cols, name):
        if not self._stage1_fitted:
            self._stage1_fitted = True
            name = self.stage1_name
        return super()._fit(cols, name)


def _rule_metrics(C, s1h, s2h, s1o, s2o):
    """규칙 3종에 대한 전체 시스템 지표. 연속 규칙은 arm 별 버스트로 적용."""
    out = []
    rules = [
        ("① 1단 단독", s1h, s1o, False),
        ("② 1단 AND 2단", s1h & s2h, s1o & s2o, False),
        ("③ ② + 3연속 (=빨강)", s1h & s2h, s1o & s2o, True),
    ]
    for name, ph, po, use_persist in rules:
        if use_persist:
            ph = AT.persist(ph, C.bh, C.red_n)
            po = AT.persist(po, C.bo, C.red_n)
        pred = np.r_[ph, po]
        m = AT.confusion(C.y_eval, pred)
        ev, _ = EV.merge_alarms(ph, C.bh)
        det = int(pd.Series(C.bo[po]).nunique()) if po.any() else 0
        out.append({"rule": name, **m, "fp_events": ev,
                    "bursts_detected": det,
                    "bursts_total": int(pd.Series(C.bo).nunique())})
    return out


def e2a_frozen_split(C):
    """v2a 동결 분할에서 후보별 전체 시스템 (BL-1 제외: 재학습 3.6시간)."""
    print("E2a 동결 분할 · 후보별 전체 시스템")
    rows = []
    for name in ("BL0", "M1", "M2", "M3"):
        fz = FrozenAny(C.cfg, C.P, stage1_name=name)
        jh = fz.judge(C.P["Fn"][C.ho], C.bh)
        jo = fz.judge(C.P["Fo"], C.bo)
        auroc = float(roc_auc_score(C.y_eval, np.r_[jh["score1"], jo["score1"]]))
        ties = float((fz.cal1 == fz.cal1.min()).mean())
        for m in _rule_metrics(C, jh["s1"], jh["s2"], jo["s1"], jo["s2"]):
            rows.append({"stage1": name, "stage2": "M1(진동15)", **m,
                         "stage1_auroc": auroc, "p_ties_share": ties})
    df = pd.DataFrame(rows)
    save(df, "v2a_frozen_split_full_system.csv")

    r8 = pd.read_csv(os.path.join(TAB, "r8_stage_decomposition.csv"))
    ref = {"① 1단 단독": int(r8.iloc[0]["FP"]),
           "② 1단 AND 2단": int(r8[r8.iloc[:, 0] == "1단 AND 2단"].iloc[0]["FP"]),
           "③ ② + 3연속 (=빨강)": int(
               r8[r8.iloc[:, 0].astype(str).str.contains("=빨강")].iloc[0]["FP"])}
    m3 = df[df.stage1 == "M3"].set_index("rule")["FP"].to_dict()
    _check(all(int(m3[k]) == v for k, v in ref.items()),
           "M3 행 FP %s == r8_stage_decomposition %s"
           % ([int(m3[k]) for k in ref], list(ref.values())))
    print("     (BL-0 conformal p 동률 비중 p_ties_share = %.4f)"
          % df[df.stage1 == "BL0"].p_ties_share.iloc[0])
    return df


def e2b_cv_full_system(C):
    """v2b/v2c 동결 규칙의 5-fold 적용 (1단 후보 4종 x 5 fold)."""
    print("E2b 동결 규칙의 5-fold 적용")
    Fn, Fo = C.Fn_m(), C.Fo_m()
    mnb = C.P["mn"]["block"].values
    folds = WD.cv_folds(C.cfg["cv"]["n_blocks"])
    rows = []
    for f in folds:
        tr = np.isin(mnb, f["train"])
        clm = mnb == f["cal"]
        hom = mnb == f["holdout"]
        mh = C.P["mn"][hom].reset_index(drop=True)
        bh = mh["burst_id"].values
        nh = int(hom.sum())
        y = np.r_[np.zeros(nh, bool), np.ones(C.no, bool)]
        # 2단(진동 M1)은 fold 안에서 후보와 무관하게 한 번만 적합한다
        m2, c2 = C.fit_stage("M1", C.vib, tr, clm)
        p2h = cal.conformal_p(c2, m2.score(Fn[hom][:, C.vib]))
        p2o = cal.conformal_p(c2, m2.score(Fo[:, C.vib]))
        s2h, s2o = p2h <= C.alpha, p2o <= C.alpha
        allc = np.arange(len(C.idx))
        for name in ("BL0", "M1", "M2", "M3"):
            m1, c1 = C.fit_stage(name, allc, tr, clm)
            sch, sco = m1.score(Fn[hom]), m1.score(Fo)
            p1h = cal.conformal_p(c1, sch)
            p1o = cal.conformal_p(c1, sco)
            s1h, s1o = p1h <= C.alpha, p1o <= C.alpha
            auroc = float(roc_auc_score(y, np.r_[sch, sco]))
            ties = float((c1 == c1.min()).mean())
            for rname, ph0, po0, use_p in (
                    ("① 1단 단독", s1h, s1o, False),
                    ("② 1단 AND 2단", s1h & s2h, s1o & s2o, False),
                    ("③ ② + 3연속 (=빨강)", s1h & s2h, s1o & s2o, True)):
                ph = AT.persist(ph0, bh, C.red_n) if use_p else ph0
                po = AT.persist(po0, C.bo, C.red_n) if use_p else po0
                m = AT.confusion(y, np.r_[ph, po])
                ev, _ = EV.merge_alarms(ph, bh)
                det = int(pd.Series(C.bo[po]).nunique()) if po.any() else 0
                rows.append({"stage1": name, "fold": f["fold"],
                             "holdout_block": f["holdout"], "rule": rname,
                             **m, "fp_events": ev, "bursts_detected": det,
                             "bursts_total": int(pd.Series(C.bo).nunique()),
                             "stage1_auroc": auroc, "p_ties_share": ties,
                             "n_normal": nh})
    df = pd.DataFrame(rows)
    save(df, "v2b_cv_full_system.csv")

    summ = df.groupby(["stage1", "rule"]).agg(
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"), f1_worst=("f1", "min"),
        fpr_mean=("fpr", "mean"), fpr_sd=("fpr", "std"), fpr_worst=("fpr", "max"),
        recall_mean=("recall", "mean"), recall_worst=("recall", "min"),
        precision_mean=("precision", "mean"),
        fp_mean=("FP", "mean"), fp_worst=("FP", "max"),
        fp_events_worst=("fp_events", "max"),
        bursts_detected_min=("bursts_detected", "min"),
        auroc_mean=("stage1_auroc", "mean")).reset_index()
    save(summ, "v2c_cv_full_system_summary.csv")

    e3 = pd.read_csv(os.path.join(TAB, "e3_normal_block_cv.csv"))
    ref = (e3[e3.model == "M3"].sort_values("fold")["fp_windows"]
           .astype(int).tolist())
    got = (df[(df.stage1 == "M3") & (df.rule == "① 1단 단독")]
           .sort_values("fold")["FP"].astype(int).tolist())
    _check(got == ref, "규칙① M3 fold별 FP %s == e3_normal_block_cv %s"
           % (got, ref))

    red = summ[summ.rule == "③ ② + 3연속 (=빨강)"]
    bf1 = red.loc[red.f1_mean.idxmax(), "stage1"]
    bfpr = red.loc[red.fpr_mean.idxmin(), "stage1"]
    m3 = red[red.stage1 == "M3"].iloc[0]
    print("     [사전판정] 주 지표(빨강 F1 fold평균 / 빨강 FPR fold평균) 최상 = "
          "%s / %s. M3 = F1 %.4f, FPR %.5f" % (bf1, bfpr, m3.f1_mean, m3.fpr_mean))
    if bf1 != "M3" or bfpr != "M3":
        print("     → 사후 비교에서 %s·%s 가 우수했으나 선정은 사전 기준"
              "(섭동 강건성)에 따랐다. 모델은 바꾸지 않는다." % (bf1, bfpr))
    print("     fold 간 빨강 FPR 최댓값(운영 상한 참고값) = %.5f (M3 %.5f)"
          % (red.fpr_worst.max(), m3.fpr_worst))
    return df, summ


def e2d_stress_by_stage1(C):
    """v2d 후보별 전체 시스템(1단 후보 + 동결 2단 + 3연속)에 r6 와 같은 섭동 (DL-020).

    지시서 외 확장: v2a 에서 1단 BL-0 시스템이 빨강 지표에서 우세하게 나온 뒤,
    '왜 BL-0 가 아닌가' 에 답하기 위해 추가했다. 섭동 정의·난수 소비 순서는
    make_audit_tables.r6_stress 와 같아서 M3 행이 r6 와 일치해야 한다.
    """
    print("E2d 후보별 전체 시스템 섭동 시험")
    cfg, P = C.cfg, C.P
    rb = cfg["robustness"]
    train_std = P["Xn"].std(axis=(0, 1))          # r6 와 같은 정의
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

    systems = {n: FrozenAny(cfg, P, stage1_name=n) for n in ("BL0", "M1", "M2", "M3")}
    rows = []
    for name, amt, spec in plans:
        Fh = C.build_features(apply(C.Xh, spec), C.mh)
        Fo = C.build_features(apply(C.Xo, spec), C.P["mo"])
        for n, fz in systems.items():
            jh, jo = fz.judge(Fh, C.bh), fz.judge(Fo, C.bo)
            rows.append({"stage1": n, "perturbation": name, "amount": str(amt),
                         "normal_stage1_rate": float(jh["s1"].mean()),
                         "normal_red_rate": float(jh["red"].mean()),
                         "fault_red_recall": float(jo["red"].mean())})
    df = pd.DataFrame(rows)
    r6 = pd.read_csv(os.path.join(TAB, "r6_frozen_system_stress.csv"))
    m3 = df[df.stage1 == "M3"].reset_index(drop=True)
    _check(np.allclose(m3.normal_red_rate.values, r6.normal_red_rate.values) and
           np.allclose(m3.fault_red_recall.values, r6.fault_red_recall.values),
           "v2d M3 행 == r6_frozen_system_stress")
    save(df, "v2d_stress_by_stage1.csv")
    s = df.groupby("stage1").agg(worst_normal_red=("normal_red_rate", "max"),
                                 worst_normal_stage1=("normal_stage1_rate", "max"),
                                 min_fault_red_recall=("fault_red_recall", "min"))
    print(s.round(4).to_string())
    return df


def e2(C):
    e2a_frozen_split(C)
    out = e2b_cv_full_system(C)
    e2d_stress_by_stage1(C)
    return out


# --------------------------------------------------------------------------- #
# E3  버스트 단위 층화 클러스터 부트스트랩
# --------------------------------------------------------------------------- #
def _burst_agg(sub, flag_col):
    """버스트별 (window 수, 경보 window 수, 경보 사건 수)."""
    n, k, ev = [], [], []
    for _, g in sub.groupby("burst_id", sort=True):
        f = g[flag_col].values
        n.append(len(f)); k.append(int(f.sum())); ev.append(n_runs(f))
    return np.array(n), np.array(k), np.array(ev)


def e3(C):
    """v3a/v3b 버스트를 재표집 단위로 둔 층화 클러스터 부트스트랩 (모델 재적합 없음)."""
    print("E3  버스트 단위 층화 클러스터 부트스트랩 (B=%d)" % N_BOOT)
    op = C.pred[C.pred.split.isin(["holdout", "fault"])].copy()
    op["red"] = (op.alarm_level == "red")
    op["st1"] = (op.p_normal_stage1 <= C.alpha)
    h = op[op.split == "holdout"]
    f = op[op.split == "fault"]
    rng = np.random.default_rng(C.seed)
    nb_h, nb_f = h.burst_id.nunique(), f.burst_id.nunique()
    ih = rng.integers(0, nb_h, size=(N_BOOT, nb_h))
    jf = rng.integers(0, nb_f, size=(N_BOOT, nb_f))

    rows, boot = [], {}
    for level, col in (("빨강", "red"), ("1단 경보", "st1")):
        hn, hk, hev = _burst_agg(h, col)
        fn, fk, _ = _burst_agg(f, col)
        N = hn[ih].sum(axis=1).astype(float)        # 정상 window 수
        FP = hk[ih].sum(axis=1).astype(float)
        EVt = hev[ih].sum(axis=1).astype(float)
        M = fn[jf].sum(axis=1).astype(float)        # 고장 window 수
        TP = fk[jf].sum(axis=1).astype(float)
        fpr = FP / N
        rec = TP / M
        prec = np.where(TP + FP > 0, TP / np.maximum(TP + FP, 1e-12), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        pt = {"fpr": hk.sum() / hn.sum(), "recall": fk.sum() / fn.sum(),
              "precision": fk.sum() / max(fk.sum() + hk.sum(), 1),
              "fp_windows": float(hk.sum()), "fp_events": float(hev.sum())}
        pt["f1"] = f1_of(pt["precision"], pt["recall"])
        dist = {"fpr": fpr, "recall": rec, "precision": prec, "f1": f1,
                "fp_windows": FP, "fp_events": EVt}
        share0 = float((FP == 0).mean())
        for met in ("fpr", "recall", "precision", "f1", "fp_windows", "fp_events"):
            d = dist[met]
            rows.append({"level": level, "metric": met, "point": pt[met],
                         "ci_lo_2.5pct": float(np.quantile(d, 0.025)),
                         "ci_hi_97.5pct": float(np.quantile(d, 0.975)),
                         "boot_mean": float(d.mean()), "boot_sd": float(d.std()),
                         "share_fp_zero": share0, "B": N_BOOT,
                         "n_bursts_normal": int(nb_h), "n_bursts_fault": int(nb_f)})
        boot[level] = {"tpr": rec, "fpr": fpr, "point": pt}
    df = pd.DataFrame(rows)
    df["해석상 주의"] = ("고장 버스트 %d개는 한 고장 기록의 조각이다. 이 구간은 "
                     "기록 안의 변동만 반영하며, 고장 사건 간 변동은 반영하지 않는다"
                     % nb_f)
    save(df, "v3a_burst_bootstrap_ci.csv")

    r2 = pd.read_csv(os.path.join(TAB, "r2_operational_performance.csv"))
    rd = r2[r2.iloc[:, 0].astype(str).str.startswith("최종 빨강")].iloc[0]
    pr = boot["빨강"]["point"]
    _check(round(pr["f1"], 4) == round(float(rd["f1"]), 4)
           and round(pr["fpr"], 4) == round(float(rd["fpr"]), 4)
           and round(pr["recall"], 4) == round(float(rd["recall"]), 4),
           "점추정 == 표 2-8 (빨강 F1 %.4f / FPR %.4f / Recall %.4f)"
           % (pr["f1"], pr["fpr"], pr["recall"]))

    # v3b 사후확률 — CP 보수값 vs 부트스트랩 보수값
    r10 = pd.read_csv(os.path.join(TAB, "r10_posterior_bounds.csv"))
    tpr, fpr_ = pr["recall"], pr["fpr"]
    tpr_b = float(np.quantile(boot["빨강"]["tpr"], 0.025))
    fpr_b = float(np.quantile(boot["빨강"]["fpr"], 0.975))
    tpr_cp = float(r10["TPR 95% 하한(CP)"].iloc[0])
    fpr_cp = float(r10["FPR 95% 상한(CP)"].iloc[0])

    def post(t, fp, pi):
        den = t * pi + fp * (1 - pi)
        return t * pi / den if den > 0 else float("nan")

    rows = []
    for pi in (0.01, 0.001, 0.0001):
        rows.append({
            "가정 고장유병률(window당)": pi,
            "빨강 TPR": tpr, "빨강 FPR": fpr_,
            "TPR 하한 (CP 95%)": tpr_cp, "FPR 상한 (CP 95%)": fpr_cp,
            "TPR 하한 (부트스트랩 2.5%)": tpr_b,
            "FPR 상한 (부트스트랩 97.5%)": fpr_b,
            "P(고장|빨강) 점추정": post(tpr, fpr_, pi),
            "P(고장|빨강) CP 보수": post(tpr_cp, fpr_cp, pi),
            "P(고장|빨강) 부트스트랩 보수": post(tpr_b, fpr_b, pi)})
    v3b = save(pd.DataFrame(rows), "v3b_posterior_bootstrap.csv")
    print("     부트스트랩 보수값: TPR 하한 %.4f / FPR 상한 %.5f "
          "(CP: %.4f / %.5f)" % (tpr_b, fpr_b, tpr_cp, fpr_cp))
    return df, v3b


# --------------------------------------------------------------------------- #
# E4  선정 민감도와 대조군의 임계값 무관 지표
# --------------------------------------------------------------------------- #
def _ranges(vals, chosen, npass, sweep, ndigits=4):
    """연속한 스윕 지점에서 (chosen, n_passed) 가 같은 구간으로 묶는다."""
    rows, i = [], 0
    while i < len(vals):
        j = i
        while (j + 1 < len(vals) and chosen[j + 1] == chosen[i]
               and npass[j + 1] == npass[i]):
            j += 1
        rows.append({"sweep": sweep, "from": round(vals[i], ndigits),
                     "to": round(vals[j], ndigits), "chosen": chosen[i],
                     "n_passed": int(npass[i])})
        i = j + 1
    return rows


def e4a_selection_sensitivity(C):
    """v4a/v4b 선정 기준 민감도 (모델 재계산 없음. e2_selection_gates.csv 만 사용)."""
    print("E4a 선정 기준 민감도")
    g = pd.read_csv(os.path.join(TAB, "e2_selection_gates.csv"))
    fixed = (g.gate1_no_nan.astype(bool) & g.gate4_burst_detected.astype(bool)
             & g.gate5_v2_abs_operating.astype(bool))
    bl0_fp = float(g.loc[g.model == "BL0", "fp_rate_worst"].iloc[0])

    def pick(flag):
        t = g.copy()
        t["ok"] = flag
        return _pick(t, "ok", exclude_bl0=False), int(flag.sum())

    rows = []
    # (1) 섭동 게이트 한도 t : 0~100 %p, 0.1 간격
    ts = np.round(np.arange(0, 1000.5, 1) / 10.0, 1)
    ch, npz = [], []
    d = g.worst_recall_drop_pp.values.astype(float)
    s = g.worst_shift_fp_increase_pp.values.astype(float)
    for t in ts:
        g2 = ((~np.isfinite(d)) | (d <= t)) & ((~np.isfinite(s)) | (s <= t))
        flag = fixed.values & g2 & g.gate3_fp_vs_bl0.astype(bool).values
        c, n = pick(pd.Series(flag, index=g.index))
        ch.append(c); npz.append(n)
    rows += _ranges(ts, ch, npz, "gate2_limit_pp", 1)

    # (2) gate3 허용오차(절대값) : 0~0.02, 0.001 간격
    tol = np.round(np.arange(0, 21) / 1000.0, 3)
    fw = g.fp_rate_worst.values.astype(float)
    ch, npz = [], []
    for v in tol:
        lim = max(bl0_fp * (1 + FP_TOL_REL), bl0_fp + v)
        g3 = (g.model.values == "BL0") | (fw <= lim)
        flag = fixed.values & g.gate2_robust.astype(bool).values & g3
        c, n = pick(pd.Series(flag, index=g.index))
        ch.append(c); npz.append(n)
    rows += _ranges(tol, ch, npz, "gate3_tol_abs", 3)
    df = save(pd.DataFrame(rows), "v4a_selection_sensitivity.csv")

    # 검증: t=20 에서 M3, 손계산 구간과 일치
    t20 = [r for r in rows if r["sweep"] == "gate2_limit_pp"
           and r["from"] <= 20.0 <= r["to"]]
    m3r = [r for r in rows if r["sweep"] == "gate2_limit_pp" and r["chosen"] == "M3"]
    m1r = [r for r in rows if r["sweep"] == "gate2_limit_pp" and r["chosen"] == "M1"]
    width = sum(r["to"] - r["from"] + 0.1 for r in m3r)
    _check(bool(t20) and t20[0]["chosen"] == "M3"
           and round(min(r["from"] for r in m3r), 1) == 17.0
           and bool(m1r) and round(min(r["from"] for r in m1r), 1) >= 89.7,
           "t=20 → M3 / M3 선정 하한 t=%.1f / M1 선정 시작 t=%.1f (손계산: "
           "17.0 미만 통과 없음, 17.0<=t<89.7 M3, 89.7 이상 M1)"
           % (min(r["from"] for r in m3r), min(r["from"] for r in m1r)))
    none_rows = [r for r in rows if r["sweep"] == "gate2_limit_pp"
                 and r["n_passed"] == 0]
    if none_rows:
        print("     (t < %.1f 구간은 통과 모델 0개 — _pick 의 기본값 BL0 가 "
              "표시된다)" % (none_rows[-1]["to"] + 0.1))

    # 대안 원리 3가지
    rob = pd.read_csv(os.path.join(TAB, "e5_robustness_summary.csv"))
    pr = []
    mm = g[np.isfinite(g.worst_shift_fp_increase_pp)]
    pr.append({"원리": "(a) minimax — 최악 섭동 오경보 증가 최소",
               "선정 모델": str(mm.loc[mm.worst_shift_fp_increase_pp.idxmin(), "model"]),
               "근거": "worst_shift_fp_increase_pp 최소 (%.3f pp)"
                       % mm.worst_shift_fp_increase_pp.min(),
               "제외": "BL1 (섭동 시험 미실시, NaN)"})
    piv = rob.pivot(index="model", columns="perturbation", values="fp_increase_pp")
    rk = piv.rank(axis=0, method="min").sum(axis=1)
    # 동률은 임의 순서가 되지 않도록 1순위 정렬키(far_h_upper95_worst, 낮을수록
    # 우선)로 깬다. 사전에 고정한 규칙이며 결과를 보고 정한 것이 아니다.
    tb = g.set_index("model")["far_h_upper95_worst"]
    ordered = sorted(rk.index, key=lambda m: (rk[m], tb.get(m, np.inf)))
    best = ordered[0]
    tied = [m for m in rk.index if rk[m] == rk[best]]
    pr.append({"원리": "(b) 섭동 4종 순위합 최소",
               "선정 모델": str(best),
               "근거": "순위합 " + ", ".join("%s=%.0f" % (m, rk[m]) for m in ordered)
                       + ("; 동률 %s → 1순위 정렬키 far_h_upper95_worst 로 결정 "
                          "(%s)" % ("=".join(tied),
                                    ", ".join("%s %.1f" % (m, tb.get(m, float("nan")))
                                              for m in tied))
                          if len(tied) > 1 else "; 동률 없음"),
               "제외": "BL1 (e5_robustness_summary 에 행 없음)"})
    pr.append({"원리": "(c) 원 기준 v0", "선정 모델": _pick(g, "passed_v0", False),
               "근거": "passed_v0 통과: " + ", ".join(g[g.passed_v0].model),
               "제외": "-"})
    pr.append({"원리": "(참고) 정정 기준 v2 (제출본 선정)",
               "선정 모델": _pick(g, "passed_v2", False),
               "근거": "passed_v2 통과: " + ", ".join(g[g.passed_v2].model),
               "제외": "-"})
    v4b = save(pd.DataFrame(pr), "v4b_selection_principles.csv")

    mnmx = pr[0]["선정 모델"]
    if width >= 50.0 and mnmx == "M3":
        print("     [사전판정] 섭동 한도를 %.1f~%.1f%%p 어디에 두어도, 최악 증가 "
              "최소화 원리로도 M3 가 선정된다. 다만 원 기준(v0)으로는 %s 다. "
              "(M3 구간 폭 %.1f%%p)"
              % (min(r["from"] for r in m3r), max(r["to"] for r in m3r),
                 pr[2]["선정 모델"], width))
    else:
        print("     [사전판정] 조건 미충족 (M3 구간 폭 %.1f%%p, minimax=%s) — "
              "해당 문장을 2.6 에 넣지 않는다" % (width, mnmx))
    return df, v4b


def e4b_control_threshold_free(C):
    """v4c 지도학습 대조군·음성대조의 AUROC/AP + 같은 유병률의 우연 수준."""
    print("E4b 대조군 임계값 무관 지표")
    nb = C.cfg["cv"]["n_blocks"]
    blocks = C.P["mn"]["block"].values
    sets = {"groups25 (기존)": tuple(C.groups),
            "model_groups23": tuple(C.mgroups)}
    rows = []
    for sname, gg in sets.items():
        gi = np.where(np.isin(C.P["gof"], list(gg)))[0]
        Fn, Fo = C.P["Fn"][:, gi], C.P["Fo"][:, gi]

        def fit_eval(Xtr, ytr, Xte, yte, seed):
            sc = StandardScaler().fit(Xtr)
            m = LogisticRegression(max_iter=2000, random_state=seed).fit(
                sc.transform(Xtr), ytr)
            Z = sc.transform(Xte)
            yh = m.predict(Z)
            s = m.decision_function(Z)
            tp = int(((yh == 1) & (yte == 1)).sum())
            fp = int(((yh == 1) & (yte == 0)).sum())
            fn = int(((yh == 0) & (yte == 1)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            return (p, r, f1_of(p, r), float(roc_auc_score(yte, s)),
                    float(average_precision_score(yte, s)), float(yte.mean()))

        # M4 지도 로지스틱 회귀: 블록 CV (make_supervised_control 과 동일 분할)
        acc = []
        for b in range(nb):
            tr_n, te_n = blocks != b, blocks == b
            rs = np.random.default_rng(b)
            m_o = rs.random(len(Fo)) < (1.0 / nb)
            Xtr = np.vstack([Fn[tr_n], Fo[~m_o]])
            ytr = np.r_[np.zeros(tr_n.sum()), np.ones((~m_o).sum())]
            Xte = np.vstack([Fn[te_n], Fo[m_o]])
            yte = np.r_[np.zeros(te_n.sum()), np.ones(m_o.sum())]
            acc.append(fit_eval(Xtr, ytr, Xte, yte, b))
        a = np.array(acc)
        pi = float(a[:, 5].mean())
        rows.append({"id": "M4", "task": "고장 vs 정상 (지도)",
                     "feature_set": sname, "uses_fault_label": True,
                     "precision": a[:, 0].mean(), "recall": a[:, 1].mean(),
                     "f1": a[:, 2].mean(), "f1_sd": a[:, 2].std(),
                     "auroc": a[:, 3].mean(), "ap": a[:, 4].mean(),
                     "test_prevalence": pi, "coin_f1": coin_f1(pi),
                     "allpos_f1": allpos_f1(pi), "n_folds": nb})

        # NC 음성대조: 고장 미사용, 정상 내부 가짜 라벨
        for lo, hi in [(0, nb - 1), (0, 1), (nb - 2, nb - 1)]:
            m = np.isin(blocks, [lo, hi])
            Xf, yf = Fn[m], (blocks[m] == hi).astype(int)
            bb = blocks[m]
            half = np.zeros(len(Xf), dtype=bool)
            for b in (lo, hi):
                ii = np.where(bb == b)[0]
                half[ii[: len(ii) // 2]] = True
            p, r, f1, au, ap, pi = fit_eval(Xf[half], yf[half],
                                            Xf[~half], yf[~half], 42)
            rows.append({"id": "NC-%d%d" % (lo, hi),
                         "task": "정상 블록 %d vs %d (가짜 라벨)" % (lo, hi),
                         "feature_set": sname, "uses_fault_label": False,
                         "precision": p, "recall": r, "f1": f1, "f1_sd": 0.0,
                         "auroc": au, "ap": ap, "test_prevalence": pi,
                         "coin_f1": coin_f1(pi), "allpos_f1": allpos_f1(pi),
                         "n_folds": 1})
    df = pd.DataFrame(rows)[[
        "id", "task", "feature_set", "uses_fault_label", "f1", "f1_sd",
        "auroc", "ap", "precision", "recall", "test_prevalence", "coin_f1",
        "allpos_f1", "n_folds"]]
    save(df, "v4c_control_threshold_free.csv")

    d7 = pd.read_csv(os.path.join(TAB, "d7_supervised_control.csv"))
    old = df[df.feature_set == "groups25 (기존)"].set_index("id")
    ok = all(round(float(old.loc[r["id"], "f1"]), 4) == round(float(r["f1_mean"]), 4)
             for _, r in d7.iterrows())
    _check(ok, "기존 특징집합 F1 == d7_supervised_control (%s)"
           % ", ".join("%s %.4f" % (i, old.loc[i, "f1"]) for i in d7.id))

    nc = df[~df.uses_fault_label]
    hi = nc[nc.auroc >= 0.6]
    if len(hi):
        print("     [사전판정] 임계값 기준 F1 은 우연 부근이나, 순위 기준으로는 "
              "정상 시간블록이 일부 구분된다 (AUROC %.3f~%.3f)"
              % (hi.auroc.min(), hi.auroc.max()))
    else:
        print("     [사전판정] 순위 기준으로도 구분되지 않는다 (NC AUROC 최대 %.3f)"
              % nc.auroc.max())
    m4 = df[df.id == "M4"]
    for _, r in m4.iterrows():
        print("     M4 %s: 실제 평가 유병률 %.4f → 동전던지기 F1 %.4f / "
              "전부양성 F1 %.4f (표 2-4 의 0.2032 는 운영 유병률 0.1275 기준값)"
              % (r.feature_set, r.test_prevalence, r.coin_f1, r.allpos_f1))
    return df


def e4(C):
    e4a_selection_sensitivity(C)
    return e4b_control_threshold_free(C)


# --------------------------------------------------------------------------- #
# E5  현장 운영 지표
# --------------------------------------------------------------------------- #
def _scope_hours(sub):
    """(수집시간 h, 벽시계 h). 수집시간 = 버스트 길이 합."""
    bt = sub.groupby("burst_id").agg(t0=("time_start", "min"),
                                     t1=("time_end", "max"))
    collect = float((bt["t1"] - bt["t0"]).dt.total_seconds().sum()) / 3600.0
    wall = float((sub["time_end"].max() - sub["time_start"].min())
                 .total_seconds()) / 3600.0
    return collect, wall


def e5a_per_shift(C):
    """v5a 교대 단위 경보 빈도 (duty 기반 벽시계 환산)."""
    print("E5a 교대 단위 경보 빈도")
    nm = C.pred[C.pred.source == "normal"].sort_values(
        ["burst_id", "time_start"]).reset_index(drop=True)
    scopes = [("정상 전체(블록 0-4)", nm, True)]
    for b in sorted(nm.block.unique()):
        scopes.append(("블록 %d" % b, nm[nm.block == b].reset_index(drop=True),
                       b != HOLD_BLOCK))
    rows = []
    for name, sub, in_sample in scopes:
        collect, wall = _scope_hours(sub)
        duty = collect / wall if wall > 0 else float("nan")
        for level in ("yellow", "red"):
            flag = (sub.alarm_level == level).values
            ev, _ = EV.merge_alarms(flag, sub["burst_id"].values)
            per_c = ev / collect if collect > 0 else float("nan")
            per_w = per_c * duty
            rows.append({
                "scope": name, "in_sample": in_sample, "level": level,
                "windows": int(flag.sum()), "events": ev,
                "collect_hours": collect, "wall_hours": wall, "duty": duty,
                "per_collect_h": per_c, "per_wall_h": per_w,
                "per_shift_8h": per_w * 8.0,
                "per_shift_8h_upper95": AT.poisson_upper(ev, wall) * 8.0})
    df = pd.DataFrame(rows)
    df["해석상 주의"] = ("현재 수집 방식(버스트 수집)이 현장에서도 같다고 가정한 "
                     "환산이다. 연속 수집이면 duty = 1 로 다시 계산해야 한다")
    save(df, "v5a_alarm_per_shift.csv")

    r7 = pd.read_csv(os.path.join(TAB, "r7_alarm_event_rates.csv"))
    ref = float(r7[r7.iloc[:, 0] == "빨강"].iloc[0]["사건/h"])
    got = float(df[(df.scope == "블록 %d" % HOLD_BLOCK)
                   & (df.level == "red")].per_collect_h.iloc[0])
    _check(round(got, 2) == round(ref, 2),
           "평가 블록 빨강 수집시간당 %.4f == r7_alarm_event_rates %.4f"
           % (got, ref))
    h = df[(df.scope == "블록 %d" % HOLD_BLOCK) & (df.level == "red")].iloc[0]
    print("     평가 블록: duty %.4f → 빨강 %.3f 건/벽시계h = %.2f 건/8h교대 "
          "(95%% 상한 %.2f)" % (h.duty, h.per_wall_h, h.per_shift_8h,
                              h.per_shift_8h_upper95))
    return df


def _red_events(sub):
    """빨강 사건 목록. (burst_id, 시작시각, 끝시각) 리스트."""
    out = []
    for b, g in sub.groupby("burst_id", sort=True):
        g = g.sort_values("time_start")
        f = (g.alarm_level == "red").values
        ts = g["time_start"].values
        te = g["time_end"].values
        i = 0
        while i < len(f):
            if f[i]:
                j = i
                while j + 1 < len(f) and f[j + 1]:
                    j += 1
                out.append((int(b), pd.Timestamp(ts[i]), pd.Timestamp(te[j])))
                i = j + 1
            else:
                i += 1
    return sorted(out, key=lambda r: r[1])


def e5b_stop_rules(C):
    """v5b 정지 검토 규칙 S1-S3 발동 횟수 (4.3 문구를 실행 가능한 형태로 고정)."""
    print("E5b 정지 검토 규칙 발동")
    nm = C.pred[C.pred.source == "normal"]
    fl = C.pred[C.pred.source == "outlier"]
    rows = []

    def s_counts(ev):
        s1 = len(ev)
        bl = sorted({b for b, _, _ in ev})
        s2 = sum(1 for b in bl if (b + 1) in bl)
        ts = [t for _, t, _ in ev]
        s3 = sum(1 for i in range(len(ts))
                 if any((ts[i] - ts[j]).total_seconds() <= 600 for j in range(i)))
        return s1, s2, s3

    for b in sorted(nm.block.unique()):
        sub = nm[nm.block == b]
        ev = _red_events(sub)
        s1, s2, s3 = s_counts(ev)
        for rid, desc, v in (("S1", "빨강 1건", s1),
                             ("S2", "빨강 2회 연속(인접 버스트)", s2),
                             ("S3", "10분 내 빨강 2건", s3)):
            rows.append({"scope": "정상 블록 %d" % b, "in_sample": b != HOLD_BLOCK,
                         "rule": rid, "정의": desc, "발동 횟수": v,
                         "빨강 사건": s1, "첫 발동(첫 평가 window 시작 후 초)": "",
                         "첫 발동까지 버스트 수": ""})
        rows.append({"scope": "정상 블록 %d" % b, "in_sample": b != HOLD_BLOCK,
                     "rule": "S4", "정의": "2개 교대 연속", "발동 횟수": "평가 불가",
                     "빨강 사건": s1, "첫 발동(첫 평가 window 시작 후 초)": "",
                     "첫 발동까지 버스트 수": "",
                     })
    # 고장 기록: 첫 발동 시각 / 그때까지의 버스트 수
    ev = _red_events(fl)
    t0 = fl["time_start"].min()
    bl_order = sorted(fl.burst_id.unique())
    s1, s2, s3 = s_counts(ev)

    def first_hit(rid):
        if rid == "S1":
            return ev[0][1] if ev else None
        if rid == "S2":
            seen = set()
            for b, t, _ in ev:
                if (b - 1) in seen or (b + 1) in seen:
                    return t
                seen.add(b)
            return None
        ts = [t for _, t, _ in ev]
        for i in range(len(ts)):
            if any((ts[i] - ts[j]).total_seconds() <= 600 for j in range(i)):
                return ts[i]
        return None

    for rid, desc, v in (("S1", "빨강 1건", s1),
                         ("S2", "빨강 2회 연속(인접 버스트)", s2),
                         ("S3", "10분 내 빨강 2건", s3)):
        t = first_hit(rid)
        rows.append({
            "scope": "고장 기록", "in_sample": False, "rule": rid, "정의": desc,
            "발동 횟수": v, "빨강 사건": s1,
            "첫 발동(첫 평가 window 시작 후 초)": round((t - t0).total_seconds(), 3)
            if t is not None else "미발동",
            "첫 발동까지 버스트 수": (1 + bl_order.index(
                next(b for b, tt, _ in ev if tt == t))) if t is not None else ""})
    rows.append({"scope": "고장 기록", "in_sample": False, "rule": "S4",
                 "정의": "2개 교대 연속", "발동 횟수": "평가 불가", "빨강 사건": s1,
                 "첫 발동(첫 평가 window 시작 후 초)": "", "첫 발동까지 버스트 수": ""})
    df = save(pd.DataFrame(rows), "v5b_stop_rule_triggers.csv")

    hs2 = int(df[(df.scope == "정상 블록 %d" % HOLD_BLOCK)
                 & (df.rule == "S2")]["발동 횟수"].iloc[0])
    fs2 = int(df[(df.scope == "고장 기록") & (df.rule == "S2")]["발동 횟수"].iloc[0])
    if hs2 == 0 and fs2 > 0:
        print("     [사전판정] 정상 평가 블록 S2 발동 0 · 고장 S2 발동 %d → 4.3 의 "
              "'유병률 0.1%% 가정 시 S2 를 정지 검토 조건으로' 권고를 유지한다" % fs2)
    else:
        hs3 = int(df[(df.scope == "정상 블록 %d" % HOLD_BLOCK)
                     & (df.rule == "S3")]["발동 횟수"].iloc[0])
        print("     [사전판정] 정상 평가 블록 S2 발동 %d회 (고장 %d회) → 권고를 S3"
              "(정상 %d회) 또는 시범운영 확정으로 낮춘다" % (hs2, fs2, hs3))
    return df


def _reason_cat(name):
    n = str(name)
    if "CUR" in n or n.startswith("O_"):
        return "전류"
    if n.startswith("R_"):
        return "진동관계 R"
    if n.startswith("S_"):
        return "진동형태 S"
    if n.startswith("A_"):
        return "진동진폭 A"
    return "기타"


def e5c_reasons(C):
    """v5c/v5d 경보 사유 분포와 빨강 권고 문구 대안 (predictions.csv 는 수정 안 함)."""
    print("E5c 경보 사유 분포와 빨강 권고 문구")
    op = C.pred[C.pred.split.isin(["holdout", "fault"])].copy()
    op["cat1"] = op.top_reason_1.map(_reason_cat)
    op["arm"] = np.where(op.split == "fault", "고장", "정상")

    # 2단 M1 진동 기여 1위 (빨강 window)
    vnames = [C.mnames[i] for i in C.vib]
    F_all = np.vstack([C.P["Fn"], C.P["Fo"]])[:, C.idx][:, C.vib]
    cv = np.abs(C.fz.m2.contrib(F_all))
    top_vib = np.array(vnames, dtype=object)[np.argmax(cv, axis=1)]
    C.pred["_vib_top1"] = top_vib
    op = op.join(C.pred["_vib_top1"], how="left")
    op["cat_vib"] = op["_vib_top1"].map(_reason_cat)

    rows = []
    for arm in ("고장", "정상"):
        for lvl in ("red", "yellow"):
            sub = op[(op.arm == arm) & (op.alarm_level == lvl)]
            if not len(sub):
                continue
            for src, col in (("1단 M3 기여 1위", "cat1"),
                             ("2단 M1 진동 기여 1위", "cat_vib")):
                if src.startswith("2단") and lvl != "red":
                    continue        # 2단 기여는 빨강에 대해서만 센다
                vc = sub[col].value_counts()
                for k, v in vc.items():
                    rows.append({"기여 출처": src, "표본": "%s %s" % (arm, lvl),
                                 "window 수": int(len(sub)), "범주": k,
                                 "건수": int(v), "비율": v / len(sub)})
    df = save(pd.DataFrame(rows), "v5c_reason_distribution.csv")

    fr = op[(op.arm == "고장") & (op.alarm_level == "red")]
    k = int((fr.cat1 == "전류").sum())
    _check(k == 268 and len(fr) == 375,
           "고장 빨강 중 전류 범주 %d/%d = %.1f%% (검토 확인값 268/375)"
           % (k, len(fr), 100.0 * k / len(fr)))

    red = op[op.alarm_level == "red"].copy()
    alt = pd.DataFrame({
        "source": red.source, "burst_id": red.burst_id,
        "original_row_start": red.original_row_start,
        "split": red.split,
        "top_reason_1": red.top_reason_1,
        "recommended_action_current": red.recommended_action,
        "stage2_vib_top1": red["_vib_top1"],
        "red_action_vib": [EX.reason_phrase(x) for x in red["_vib_top1"]],
    })
    _check(all(a.endswith(b) for a, b in zip(alt.recommended_action_current, alt.red_action_vib)),
           "빨강 recommended_action == 2단 진동 기여 1위 문구 (DL-022 적용 확인)")
    save(alt, "v5d_red_action_alternative.csv")
    print("     (DL-022: 빨강 window 의 recommended_action 은 2단 진동 기여 1위 기준. "
          "이 표는 그 대조 기록이다)")
    return df, alt


def e5(C):
    e5a_per_shift(C)
    e5b_stop_rules(C)
    return e5c_reasons(C)


# --------------------------------------------------------------------------- #
# E6  개별 영향변수 순위
# --------------------------------------------------------------------------- #
def e6(C):
    """v6 특징 23개의 기여 비중·단일 분리력·제거 영향 (진단용 23회 재적합)."""
    print("E6  개별 영향변수 순위")
    Fn, Fo = C.Fn_m(), C.Fo_m()
    Fh = Fn[C.ho]
    nmx = np.array(C.mnames, dtype=object)
    # 1단 M3 기여 1위 비중
    t_f = nmx[np.argmax(np.abs(C.fz.m1.contrib(Fo)), axis=1)]
    t_n = nmx[np.argmax(np.abs(C.fz.m1.contrib(Fh)), axis=1)]
    sh_f = pd.Series(t_f).value_counts() / len(t_f)
    sh_n = pd.Series(t_n).value_counts() / len(t_n)

    single = pd.read_csv(os.path.join(TAB, "v1e_single_feature_auroc.csv")
                         ).set_index("feature")

    allc = np.arange(len(C.mnames))

    def evaluate(cols):
        m, cs = C.fit_stage("M3", cols, C.tr, C.cl)
        sh = m.score(Fh[:, cols]); so = m.score(Fo[:, cols])
        p = cal.conformal_p(cs, np.r_[sh, so])
        return (float((p[:C.nh] <= C.alpha).mean()),
                float((p[C.nh:] <= C.alpha).mean()),
                float(average_precision_score(C.y_eval, np.r_[sh, so])))

    b_rate, b_rec, b_ap = evaluate(allc)
    rows = []
    for j, nm in enumerate(C.mnames):
        cols = np.array([k for k in allc if k != j])
        rate, rec, ap = evaluate(cols)
        rows.append({
            "feature": nm, "group": nm.split("_")[0],
            "channel": feature_channel(nm),
            "physical_meaning": feature_physics(nm),
            "top1_share_fault": float(sh_f.get(nm, 0.0)),
            "top1_share_normal": float(sh_n.get(nm, 0.0)),
            "single_auroc": float(single.loc[nm, "single_auroc"]),
            "direction": str(single.loc[nm, "direction"]),
            "drop_one_normal_rate": rate,
            "drop_one_fault_recall": rec,
            "drop_one_ap": ap,
            "drop_one_d_normal_rate": rate - b_rate,
            "drop_one_d_fault_recall": rec - b_rec,
            "drop_one_d_ap": ap - b_ap,
        })
    df = pd.DataFrame(rows).sort_values(
        ["top1_share_fault", "single_auroc"], ascending=[False, False]
    ).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["baseline_normal_rate"] = b_rate
    df["baseline_fault_recall"] = b_rec
    df["baseline_ap"] = b_ap
    save(df, "v6_feature_importance.csv")

    r5 = pd.read_csv(os.path.join(TAB, "r5_stage1_channel_share.csv"))
    ref_f = float(r5[r5.iloc[:, 0] == "고장 기록"].iloc[0]["비중"])
    ref_n = float(r5[r5.iloc[:, 0] == "정상 holdout"].iloc[0]["비중"])
    got_f = float(df[df.channel == "전류"].top1_share_fault.sum())
    got_n = float(df[df.channel == "전류"].top1_share_normal.sum())
    _check(round(got_f, 4) == round(ref_f, 4) and round(got_n, 4) == round(ref_n, 4),
           "전류 채널 top1_share 합 고장 %.4f / 정상 %.4f == r5 %.4f / %.4f"
           % (got_f, got_n, ref_f, ref_n))
    print("     [해석상 주의] 탐지가 포화되어 drop-one 의 AP 변화는 작다 "
          "(최대 |ΔAP| %.2e). 순위는 기여도 비중으로 정하고 drop-one 은 보조 열이다"
          % df.drop_one_d_ap.abs().max())
    print("     1위 특징: %s (%s, 고장 기여 1위 비중 %.1f%%, 단일 AUROC %.4f)"
          % (df.iloc[0].feature, df.iloc[0].physical_meaning,
             100 * df.iloc[0].top1_share_fault, df.iloc[0].single_auroc))
    return df


# --------------------------------------------------------------------------- #
def main(only=None):
    print("검토 대응 진단표 v1-v6 생성 (모델·특징·임계값·경보 규칙 불변)")
    C = Ctx()
    print("  정상 window %d / 고장 window %d / 특징 %d (모델 입력 %d)"
          % (len(C.P["Xn"]), len(C.P["Xo"]), C.P["Fn"].shape[1], len(C.idx)))
    C.verify_frozen()

    steps = [("E1", e1), ("E2", e2), ("E3", e3),
             ("E4", e4), ("E5", e5), ("E6", e6)]
    want = {s.upper() for s in only} if only else None
    # E6 는 v1e(E1) 산출물을 읽는다. E6 단독 실행 시 없으면 먼저 만든다.
    if want and "E6" in want and "E1" not in want and not os.path.exists(
            os.path.join(TAB, "v1e_single_feature_auroc.csv")):
        print("  (E6 가 요구하는 v1e 가 없어 E1e 를 먼저 실행한다)")
        e1e_single_feature(C)
    for name, fn in steps:
        if want and name not in want:
            continue
        fn(C)
    print("완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="실행할 실험만 지정 (예: --only E1 E3)")
    a = ap.parse_args()
    main(a.only)

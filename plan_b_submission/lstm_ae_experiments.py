# -*- coding: utf-8 -*-
"""
LSTM-Autoencoder 실험 — 가이드북 재현 + 선행연구 공격 항목 실증 + Plan B 설정

가이드북(「소성가공 예지보전 AI 데이터셋」 분석실습 가이드북) 설정 G0 에서 출발해
공격 문서(previous research/공격1·공격2)가 지적한 항목을 한 번에 하나씩만 바꾼다(OFAT).
각 실험이 공격 주장 하나를 실증하거나 반박하도록 설계했다.

  ID     바꾼 것                                          검증하는 주장
  G0     없음 (가이드북 그대로)                             재현: 가이드북 F1 0.7476
  A1     절댓값(abs) 제거                                  공격1-1, 공격2-[1]
  A2     gap-aware windowing                               Plan A §1.1
  A3     이상 점수 = window 전체 MSE (가이드북은 마지막 1시점)  공격2-[3]
  A4     임계값 = 정상 검증 q99.9 (가이드북은 이상 섞인 P=R 교점) 공격2-[2]
  A5     손실함수 Huber                                     공격1-3, 공격2-[3]
  A6     RobustScaler                                      공격2-[6]
  P20    Plan B: abs 제거 + gap-aware + offset 0 + 전체 MSE
                + 정상 전용 임계값 + Plan B 분할, window 20
  P10    P20 과 같고 window 10 (마할라노비스 모델과 같은 길이)

usage (Colab / 로컬 공통)
  python lstm_ae_experiments.py                      # 전 설정 × 시드 3개 (기본 예산)
  python lstm_ae_experiments.py --configs G0 A1      # 일부만
  python lstm_ae_experiments.py --replicate          # G0 를 가이드북 원래 예산(800 epoch)으로 1회 추가
  python lstm_ae_experiments.py --blockcv            # P20·P10 블록 교차검증 오경보 (시드 42, 각 5회 학습)
  python lstm_ae_experiments.py --smoke              # 2 epoch 로 전 설정이 도는지만 확인
  python lstm_ae_experiments.py --summary            # 학습 없이 결과 요약표만 다시 만듦

재시작: 끝난 실험은 out/runs/<ID>_s<seed>.json 이 있으면 건너뛴다.
       Colab 연결이 끊겨도 같은 명령을 다시 실행하면 이어서 돈다.
"""
import argparse
import glob
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler, RobustScaler

COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
GAP_SEC = 0.5
SEEDS = [42, 1337, 2026]

# --------------------------------------------------------------------------- #
# 실험 설정
# --------------------------------------------------------------------------- #
G0 = dict(abs=True, gap=False, offset=100, seq=20, score="last", thr="pr",
          loss="mse", scaler="minmax", split="guide")


def _cfg(**kw):
    c = dict(G0)
    c.update(kw)
    return c


PLANB = dict(abs=False, gap=True, offset=0, score="full", thr="q999", split="planb")

CONFIGS = {
    "G0": _cfg(),
    "A1": _cfg(abs=False),
    "A2": _cfg(gap=True),
    "A3": _cfg(score="full"),
    "A4": _cfg(thr="q999"),
    "A5": _cfg(loss="huber"),
    "A6": _cfg(scaler="robust"),
    "P20": _cfg(seq=20, **PLANB),
    "P10": _cfg(seq=10, **PLANB),
}

BUDGET_STD = dict(epochs=300, es_patience=30, lr_patience=15)      # 모든 비교 실험 공통
BUDGET_GUIDE = dict(epochs=800, es_patience=120, lr_patience=50)   # 가이드북 원래 예산


# --------------------------------------------------------------------------- #
# 데이터
# --------------------------------------------------------------------------- #
def load(data_dir):
    n = pd.read_csv(os.path.join(data_dir, "press_data_normal.csv"), index_col=0,
                    parse_dates=["TimeStamp"]).reset_index(drop=True)
    o = pd.read_csv(os.path.join(data_dir, "press_data_outlier.csv"), index_col=0,
                    parse_dates=["TimeStamp"]).reset_index(drop=True)
    return n, o


def run_bounds(t):
    """연속 구간 [(a, b)]. 간격 > GAP_SEC 이거나 시간이 거꾸로 가면 끊는다
    (블록 교차검증에서 떨어진 블록을 이어 붙일 때 경계를 끊기 위함)."""
    dt = np.diff(t.astype("datetime64[ns]").astype(np.int64)) / 1e9
    brk = np.where((dt > GAP_SEC) | (dt < 0))[0] + 1
    edges = np.concatenate(([0], brk, [len(t)]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def make_windows(X, t, seq, offset, gap):
    """반환: W (n, seq, 3), 시작 행, window 끝 시각, run id
    naive : 가이드북과 같음. range(len - seq - offset), 공백 무시
    gap   : run 안에서만 생성, 전체 끝에서 offset 만큼 잘라냄"""
    if not gap:
        starts = np.arange(max(len(X) - seq - offset, 0))
        rid = np.zeros(len(starts), dtype=int)
    else:
        limit = len(X) - offset
        s, r = [], []
        for k, (a, b) in enumerate(run_bounds(t)):
            b = min(b, limit)
            if b - a >= seq:
                s += list(range(a, b - seq + 1))
                r += [k] * (b - seq + 1 - a)
        starts, rid = np.array(s, dtype=int), np.array(r, dtype=int)
    W = (np.stack([X[i:i + seq] for i in starts]) if len(starts)
         else np.empty((0, seq, X.shape[1])))
    tend = t[starts + seq - 1] if len(starts) else np.array([], dtype=t.dtype)
    return W.astype(np.float32), starts, tend, rid


class Prep:
    """abs → scaler(학습 구간 행에만 fit)"""

    def __init__(self, cfg, train_rows):
        self.abs = cfg["abs"]
        self.sc = MinMaxScaler() if cfg["scaler"] == "minmax" else RobustScaler()
        self.sc.fit(self._a(train_rows))

    def _a(self, X):
        return np.abs(X) if self.abs else X

    def __call__(self, X):
        return self.sc.transform(self._a(X)).astype(np.float32)


def build_sets(cfg, n, o):
    """설정에 맞춰 학습·조기종료·임계값·테스트 window 를 만든다."""
    seq, off, gap = cfg["seq"], cfg["offset"], cfg["gap"]
    Xn, tn = n[COLS].values, n.TimeStamp.values
    Xo, to = o[COLS].values, o.TimeStamp.values
    S = {}
    if cfg["split"] == "guide":
        prep = Prep(cfg, Xn[:15000])
        Wtr, *_ = make_windows(prep(Xn[:15000]), tn[:15000], seq, off, gap)
        Wte_n, st_n, te_n, rid_n = make_windows(prep(Xn[15000:]), tn[15000:], seq, off, gap)
        Wa, st_a, te_a, rid_a = make_windows(prep(Xo), to, seq, off, gap)
        nv = 880
        # 가이드북은 이상 window 480개 중 300개를 검증에 씀(62.5 %). gap-aware 에서는
        # 이상 window 가 300개보다 적으므로 같은 비율을 적용한다.
        na = 300 if len(Wa) > 300 + 50 else int(round(0.625 * len(Wa)))
        S["train"] = Wtr
        S["es_val"] = Wte_n[:nv]                        # 가이드북: X_valid_0
        S["thr_n"], S["thr_a"] = Wte_n[:nv], Wa[:na]
        S["test_n"], S["test_a"] = Wte_n[nv:], Wa[na:]
        S["meta_n"] = (st_n[nv:] + 15000, te_n[nv:], rid_n[nv:])
        S["meta_a"] = (st_a[na:], te_a[na:], rid_a[na:])
    else:  # planb
        prep = Prep(cfg, Xn[:12000])
        Wtr_all, *_ = make_windows(prep(Xn[:12000]), tn[:12000], seq, off, gap)
        cut = int(len(Wtr_all) * 0.9)                    # 학습 구간 마지막 10 % = 조기종료용
        S["train"], S["es_val"] = Wtr_all[:cut], Wtr_all[cut:]
        S["thr_n"], *_ = make_windows(prep(Xn[12000:15000]), tn[12000:15000], seq, off, gap)
        S["thr_a"] = None
        Wte_n, st_n, te_n, rid_n = make_windows(prep(Xn[15000:]), tn[15000:], seq, off, gap)
        Wa, st_a, te_a, rid_a = make_windows(prep(Xo), to, seq, off, gap)
        S["test_n"], S["test_a"] = Wte_n, Wa
        S["meta_n"] = (st_n + 15000, te_n, rid_n)
        S["meta_a"] = (st_a, te_a, rid_a)
        S["prep"] = prep
    return S


# --------------------------------------------------------------------------- #
# 모델
# --------------------------------------------------------------------------- #
def lstm_ae(seq, n_feat, loss):
    """가이드북 코드 25 와 같은 구조"""
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    m = models.Sequential([
        layers.Input(shape=(seq, n_feat)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32, return_sequences=False),
        layers.RepeatVector(seq),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(64, return_sequences=True),
        layers.TimeDistributed(layers.Dense(n_feat)),
    ])
    lossf = "mse" if loss == "mse" else tf.keras.losses.Huber(delta=1.0)
    m.compile(loss=lossf, optimizer=optimizers.Adam(0.001))
    return m


def train(cfg, S, seed, budget, verbose=0):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.keras.utils.set_random_seed(seed)
    m = lstm_ae(cfg["seq"], len(COLS), cfg["loss"])
    cb = [ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=budget["lr_patience"]),
          EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=budget["es_patience"],
                        mode="min", restore_best_weights=True)]
    t0 = time.time()
    h = m.fit(S["train"], S["train"], epochs=budget["epochs"], batch_size=128,
              validation_data=(S["es_val"], S["es_val"]), callbacks=cb,
              verbose=verbose, shuffle=True)
    return m, dict(epochs_run=len(h.history["loss"]), train_sec=round(time.time() - t0, 1),
                   best_val_loss=float(np.min(h.history["val_loss"])))


def errors(m, W):
    """window·시점·채널별 제곱오차 (n, seq, 3)"""
    if len(W) == 0:
        return np.empty((0,) + W.shape[1:])
    R = m.predict(W, batch_size=1024, verbose=0)
    return (W - R) ** 2


def score(cfg, E):
    """이상 점수와 채널별 기여.
    last: 가이드북 코드 28 — 마지막 1시점만 사용
    full: window 전체 평균"""
    C = E[:, -1, :] if cfg["score"] == "last" else E.mean(1)
    return C.mean(1), C


# --------------------------------------------------------------------------- #
# 임계값·지표
# --------------------------------------------------------------------------- #
def threshold(cfg, s_n, s_a):
    if cfg["thr"] == "pr":
        # 가이드북 코드 29: 이상이 섞인 검증셋에서 precision == recall 인 첫 지점.
        # 부동소수점 등호가 성립하지 않을 수 있어 |P-R| 최소 지점으로 대체.
        y = np.r_[np.zeros(len(s_n)), np.ones(len(s_a))]
        p, r, th = precision_recall_curve(y, np.r_[s_n, s_a])
        i = int(np.argmin(np.abs(p[:-1] - r[:-1])))
        return float(th[i])
    return float(np.quantile(s_n, 0.999))


def metrics(s_n, s_a, th):
    yt = np.r_[np.zeros(len(s_n)), np.ones(len(s_a))]
    s = np.r_[s_n, s_a]
    yp = (s > th).astype(int)
    tp = int(yp[len(s_n):].sum()); fn = len(s_a) - tp
    fp = int(yp[:len(s_n)].sum()); tn = len(s_n) - fp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(n_test_normal=len(s_n), n_test_anom=len(s_a), tp=tp, fn=fn, fp=fp, tn=tn,
                precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0,
                accuracy=(tp + tn) / len(s), fpr=fp / len(s_n), mcc=float(matthews_corrcoef(yt, yp)),
                auroc=float(roc_auc_score(yt, s)), ap=float(average_precision_score(yt, s)),
                threshold=th)


def count_events(alarm, rid, k=3):
    """같은 run 안에서 연속 k개 window 알람 = 이벤트 1건"""
    ev, streak, fired, prev = 0, 0, False, None
    for a, r in zip(alarm, rid):
        if r != prev:
            streak, fired, prev = 0, False, r
        streak = streak + 1 if a else 0
        if not a:
            fired = False
        if streak >= k and not fired:
            ev += 1
            fired = True
    return ev


def entry_delay(alarm, rid, tend, run_start_times, k=3):
    """이상 기록의 각 run 을 '고장 시작'으로 가정했을 때 첫 알람까지의 벽시계 초"""
    out = []
    for k_run, t0 in enumerate(run_start_times):
        idx = np.where(rid >= k_run)[0]
        streak, prev, got = 0, None, None
        for i in idx:
            if rid[i] != prev:
                streak, prev = 0, rid[i]
            streak = streak + 1 if alarm[i] else 0
            if streak >= k:
                got = float((tend[i] - t0) / np.timedelta64(1, "s"))
                break
        out.append(got)
    return out


def poisson_ub(x, alpha=0.05):
    from scipy.stats import chi2
    return float(chi2.ppf(1 - alpha / 2, 2 * (x + 1)) / 2)


# --------------------------------------------------------------------------- #
# 실험 1회
# --------------------------------------------------------------------------- #
def run_one(cid, cfg, seed, n, o, budget, out, tag=""):
    key = "%s%s_s%d" % (cid, tag, seed)
    jpath = os.path.join(out, "runs", key + ".json")
    if os.path.exists(jpath):
        print("  [skip] %s (이미 있음)" % key)
        return
    print("  [run ] %s  %s" % (key, json.dumps(cfg, ensure_ascii=False)))
    S = build_sets(cfg, n, o)
    m, info = train(cfg, S, seed, budget)

    s_thr_n, _ = score(cfg, errors(m, S["thr_n"]))
    s_thr_a = score(cfg, errors(m, S["thr_a"]))[0] if S["thr_a"] is not None else None
    th = threshold(cfg, s_thr_n, s_thr_a)
    s_n, c_n = score(cfg, errors(m, S["test_n"]))
    s_a, c_a = score(cfg, errors(m, S["test_a"]))
    res = dict(config=cid + tag, seed=seed, channels=list(COLS),
               **{k: cfg[k] for k in cfg}, **info,
               **metrics(s_n, s_a, th),
               n_train=len(S["train"]))

    # 오경보 이벤트 (정상 테스트, 연속 3 window)
    al_n = s_n > th
    res["fa_events_test"] = count_events(al_n, S["meta_n"][2], 3)

    # Plan B 분할 전용: 진입시점 탐지지연, 센서 섭동(E3)
    if cfg["split"] == "planb":
        rid_a, tend_a = S["meta_a"][2], S["meta_a"][1]
        runs = run_bounds(o.TimeStamp.values)
        d = entry_delay(s_a > th, rid_a, tend_a, [o.TimeStamp.values[a] for a, _ in runs])
        ok = [x for x in d if x is not None]
        res.update(delay_n_start=len(d), delay_missed=sum(x is None for x in d),
                   delay_median=float(np.median(ok)) if ok else None,
                   delay_max=float(np.max(ok)) if ok else None)
        prep = S["prep"]
        Xo = o[COLS].values.copy()
        # E3-a: 하부 진동(AI1) 부호 반전
        Xf = Xo.copy(); Xf[:, 1] = -Xf[:, 1]
        # E3-b: 이상 데이터 채널별 평균·표준편차를 정상 학습구간과 같게 맞춤
        Xtr = n[COLS].values[:12000]
        Xg = (Xo - Xo.mean(0)) / Xo.std(0) * Xtr.std(0) + Xtr.mean(0)
        for name, Xp in (("e3a_flip_recall", Xf), ("e3b_gain_recall", Xg)):
            Wp, *_ = make_windows(prep(Xp), o.TimeStamp.values, cfg["seq"], cfg["offset"], cfg["gap"])
            res[name] = float((score(cfg, errors(m, Wp))[0] > th).mean())

    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out, "scores"), exist_ok=True)
    np.savez_compressed(os.path.join(out, "scores", key + ".npz"),
                        s_n=s_n, s_a=s_a, chan_n=c_n, chan_a=c_a, th=th,
                        row_n=S["meta_n"][0], row_a=S["meta_a"][0],
                        rid_n=S["meta_n"][2], rid_a=S["meta_a"][2])
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print("         → epochs %d, %.0fs | R %.3f FPR %.4f F1 %.3f AUROC %.3f"
          % (info["epochs_run"], info["train_sec"], res["recall"], res["fpr"],
             res["f1"], res["auroc"]))


# --------------------------------------------------------------------------- #
# 블록 교차검증 오경보 (P20·P10)
# --------------------------------------------------------------------------- #
def block_cv(cid, cfg, n, budget, out, seed=42):
    jpath = os.path.join(out, "runs", "%s_blockcv_s%d.json" % (cid, seed))
    if os.path.exists(jpath):
        print("  [skip] %s blockcv" % cid)
        return
    blocks = np.array_split(np.arange(len(n)), 5)
    total_ev, hours, per = 0, 0.0, []
    for b in range(5):
        rest = np.concatenate([blocks[j] for j in range(5) if j != b])
        cut = int(len(rest) * 0.75)
        tr_idx, th_idx, ho_idx = rest[:cut], rest[cut:], blocks[b]
        X, t = n[COLS].values, n.TimeStamp.values
        prep = Prep(cfg, X[tr_idx])
        Wtr, *_ = make_windows(prep(X[tr_idx]), t[tr_idx], cfg["seq"], 0, True)
        c = int(len(Wtr) * 0.9)
        S = dict(train=Wtr[:c], es_val=Wtr[c:])
        m, _ = train(cfg, S, seed, budget)
        Wth, *_ = make_windows(prep(X[th_idx]), t[th_idx], cfg["seq"], 0, True)
        th = float(np.quantile(score(cfg, errors(m, Wth))[0], 0.999))
        Who, _, _, rid = make_windows(prep(X[ho_idx]), t[ho_idx], cfg["seq"], 0, True)
        ev = count_events(score(cfg, errors(m, Who))[0] > th, rid, 3)
        h = 0.1 * len(ho_idx) / 3600
        per.append(dict(block=b, events=ev, hours=h))
        total_ev += ev; hours += h
        print("    block %d: 오경보 %d건" % (b, ev))
    res = dict(config=cid, seed=seed, channels=list(COLS), events=total_ev, hours=hours,
               far_per_h=total_ev / hours, far_ub95=poisson_ub(total_ev) / hours, per_block=per)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("  %s blockcv: 오경보 %d건 / %.3f h → %.2f 건/h (95%% 상한 %.2f)"
          % (cid, total_ev, hours, res["far_per_h"], res["far_ub95"]))


# --------------------------------------------------------------------------- #
# 요약
# --------------------------------------------------------------------------- #
def summarize(out):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "runs", "*.json"))):
        if "_blockcv_" in p:
            continue
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        print("결과 없음")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "all_runs.csv"), index=False, encoding="utf-8-sig")
    keys = ["recall", "fpr", "precision", "f1", "auroc", "ap", "mcc", "fa_events_test",
            "epochs_run", "train_sec"]
    extra = [k for k in ("delay_median", "delay_missed", "e3a_flip_recall", "e3b_gain_recall")
             if k in df]
    agg = df.groupby("config")[keys + extra].agg(["mean", "std", "count"])
    agg.to_csv(os.path.join(out, "summary.csv"), encoding="utf-8-sig")
    order = [c for c in list(CONFIGS) + ["G0orig"] if c in df.config.unique()]
    lines = ["| 설정 | n | Recall | FPR | F1 | AUROC | AP | 오경보(테스트) | E3 반전 | E3 이득 | epoch |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in order:
        g = df[df.config == c]

        def ms(k, f="%.3f"):
            if k not in g or g[k].isna().all():
                return "—"
            v = g[k].astype(float)
            return (f % v.mean()) + ("" if len(v) < 2 else (" ± " + f % v.std()))
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (c, len(g), ms("recall"), ms("fpr", "%.4f"), ms("f1"), ms("auroc"),
                        ms("ap"), ms("fa_events_test", "%.1f"), ms("e3a_flip_recall"),
                        ms("e3b_gain_recall"), ms("epochs_run", "%.0f")))
    for p in sorted(glob.glob(os.path.join(out, "runs", "*_blockcv_*.json"))):
        with open(p, encoding="utf-8") as f:
            b = json.load(f)
        lines.append("")
        lines.append("블록 교차검증 %s: 오경보 %d건 / %.3f h → %.2f 건/h (95%% 상한 %.2f 건/h)"
                     % (b["config"], b["events"], b["hours"], b["far_per_h"], b["far_ub95"]))
    md = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


# --------------------------------------------------------------------------- #
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="csv 2개가 있는 폴더")
    ap.add_argument("--out", default=os.path.join(here, "results"))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--replicate", action="store_true")
    ap.add_argument("--blockcv", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--channels", nargs="*", default=None,
                    help="입력 채널 목록. 기본 AI0+AI1+AI2. 예: --channels AI0_Vibration AI1_Vibration")
    a = ap.parse_args()

    if a.channels:
        COLS[:] = a.channels
        print("입력 채널: %s" % COLS)

    if a.summary:
        summarize(a.out)
        return
    data = a.data
    if data is None:
        for cand in (os.path.join(here, "data"),
                     os.path.join(here, "..", "..", "..", "..", "data", "raw")):
            if os.path.exists(os.path.join(cand, "press_data_normal.csv")):
                data = cand
                break
    if data is None:
        raise SystemExit("데이터 폴더를 찾지 못함. --data 로 지정")

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    print("TensorFlow %s / GPU: %s" % (tf.__version__, tf.config.list_physical_devices("GPU")))
    n, o = load(data)
    budget = BUDGET_STD
    if a.smoke:
        budget = dict(epochs=2, es_patience=1, lr_patience=1)
        a.seeds = [42]
        a.out = a.out + "_smoke"
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    for cid in a.configs:
        for seed in a.seeds:
            run_one(cid, CONFIGS[cid], seed, n, o, budget, a.out)
    if a.replicate and not a.smoke:
        run_one("G0", CONFIGS["G0"], 42, n, o, BUDGET_GUIDE, a.out, tag="orig")
    if a.blockcv:
        for cid in ("P20", "P10"):
            block_cv(cid, CONFIGS[cid], n, budget, a.out)
    print("\n총 %.1f 분" % ((time.time() - t0) / 60))
    summarize(a.out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""변수 간 상호작용 진단표 (서면평가 문항 3 '변수 간 상호작용').

모델·특징·임계값·경보 규칙 불변. 기존 predictions.csv 와 원본 데이터만 사용한다.
구간 경계(3분위)는 동결 구성의 정상 학습 블록 0-2 에서만 계산한다.

  i1 : 상부 진동 RMS 3분위 × 전류 RMS 3분위 — 정상 holdout 경보율, 고장 빨강 Recall
  i2 : 부하 3분위 × 버스트 내 위치(첫 1초 / 이후) — 같은 지표
  i3 : 두 변수의 주효과만으로 예상한 경보율과 실제 경보율의 차이(상호작용 크기)
  i4 : 저부하 정상 버스트 vs 고장 버스트를 가르는 단일 지표의 AUC (버스트 단위)
  i5 : 버스트 시작부 미탐의 구조 분해(첫 2 window 제외 Recall)와 빨강 오경보 목록 (DL-021)
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs")
TAB = os.path.join(OUT, "tables")
LV = ["하", "중", "상"]


def save(df, name):
    df.to_csv(os.path.join(TAB, name), index=False, encoding="utf-8-sig")
    print("  -> %s (%d행)" % (name, len(df)))


def _rms(X, c):
    return np.sqrt((X[:, :, c] ** 2).mean(axis=1))


def _frame():
    import yaml
    from src import data as D, features as FT, windows as WD
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
    ch, gap, seq = cfg["data"]["channels"], cfg["windows"]["gap_sec"], cfg["windows"]["length"]
    n, o, _ = D.load_all(cfg, HERE)
    Xn, mn = WD.make_windows(n, ch, seq, gap, source="normal")
    Xo, mo = WD.make_windows(o, ch, seq, gap, source="outlier")
    mn["block"] = WD.block_split(mn, cfg["cv"]["n_blocks"])
    tr = np.isin(mn["block"].values, [0, 1, 2])
    ho = mn["block"].values == 4

    pred = pd.read_csv(os.path.join(OUT, "predictions.csv"))
    ph = pred[pred.split == "holdout"].reset_index(drop=True)
    pf = pred[pred.split == "fault"].reset_index(drop=True)
    # 예측파일과 window 의 정렬이 같은지 원본 행 번호로 확인한다.
    assert (ph.original_row_start.values == mn.loc[ho, "original_row_start"].values).all()
    assert (pf.original_row_start.values == mo["original_row_start"].values).all()

    vib = _rms(Xn[tr], 0)
    cur = _rms(Xn[tr], 2)
    b_vib = np.quantile(vib, [1 / 3, 2 / 3])
    b_cur = np.quantile(cur, [1 / 3, 2 / 3])
    lo, hi = FT.regime_bounds(Xn[tr])

    def attach(p, X, meta):
        p = p.copy()
        p["vib_t"] = np.digitize(_rms(X, 0), b_vib)
        p["cur_t"] = np.digitize(_rms(X, 2), b_cur)
        p["load_t"] = FT.load_regime(X, lo, hi)
        p["pos"] = np.where(meta["pos_in_burst_sec"].values <= 1.0, "첫 1초", "1초 이후")
        p["win_idx"] = p.groupby("burst_id").cumcount()
        p["s1"] = (p.p_normal_stage1 <= 0.01).astype(int)
        p["red"] = (p.alarm_level == "red").astype(int)
        return p

    H = attach(ph, Xn[ho], mn[ho].reset_index(drop=True))
    F = attach(pf, Xo, mo)
    bounds = dict(vib=b_vib.tolist(), cur=b_cur.tolist(), load=[lo, hi])
    return H, F, bounds


def _grid(H, F, a, b, la, lb, names_a, names_b):
    rows = []
    for i, va in enumerate(names_a):
        for j, vb in enumerate(names_b):
            h = H[(H[a] == (i if a != "pos" else va)) & (H[b] == (j if b != "pos" else vb))]
            f = F[(F[a] == (i if a != "pos" else va)) & (F[b] == (j if b != "pos" else vb))]
            rows.append({
                la: va if a == "pos" else LV[i], lb: vb if b == "pos" else LV[j],
                "정상 window": len(h),
                "1단 경보율": round(h.s1.mean(), 4) if len(h) else np.nan,
                "빨강 경보율": round(h.red.mean(), 4) if len(h) else np.nan,
                "빨강 경보 수": int(h.red.sum()),
                "고장 window": len(f),
                "고장 빨강 Recall": round(f.red.mean(), 4) if len(f) else np.nan,
                "정상 버스트 수": int(h.burst_id.nunique()),
                "빨강 오경보 버스트 수": int(h[h.red == 1].burst_id.nunique()),
                "고장 버스트 수": int(f.burst_id.nunique()),
            })
    return pd.DataFrame(rows)


def _interaction(H, a, b, la, lb, na, nb):
    """로그오즈 가법모형(주효과만)으로 예상한 1단 경보율과 실제의 차이."""
    from sklearn.linear_model import LogisticRegression
    Xa = pd.get_dummies(H[a].astype(str), prefix=a, drop_first=True)
    Xb = pd.get_dummies(H[b].astype(str), prefix=b, drop_first=True)
    X = pd.concat([Xa, Xb], axis=1).astype(float).values
    m = LogisticRegression(C=1e6, max_iter=2000).fit(X, H.s1.values)
    H = H.assign(exp=m.predict_proba(X)[:, 1])
    rows = []
    for (va, vb), g in H.groupby([a, b]):
        rows.append({"요인 쌍": "%s × %s" % (la, lb),
                     "요인 A 수준": LV[va] if a != "pos" else va,
                     "요인 B 수준": LV[vb] if b != "pos" else vb,
                     "n": len(g), "실제 1단 경보율": round(g.s1.mean(), 4),
                     "주효과만의 예상": round(g.exp.mean(), 4),
                     "차이(실제-예상, %p)": round(100 * (g.s1.mean() - g.exp.mean()), 2)})
    return pd.DataFrame(rows)


def i4_burst_auc():
    """저부하 정상(d1 의 정의)과 고장을 버스트 단위 단일 지표로 가를 수 있는가."""
    from sklearn.metrics import roc_auc_score
    import make_domain_tables as DM
    n, o = DM.load("press_data_normal.csv"), DM.load("press_data_outlier.csv")
    Fn, Fo = DM.burst_table(n, "normal"), DM.burst_table(o, "fault")
    low = Fn[(Fn.t_start_s >= DM.LOWLOAD_T0) & (Fn.t_start_s <= DM.LOWLOAD_T1)]
    y = np.r_[np.zeros(len(low)), np.ones(len(Fo))]
    spec = [("상부 진동 RMS", "rms_AI0", 1), ("전류 RMS", "rms_AI2", 1),
            ("전류 DC 평균", "mean_AI2", 1), ("상·하부 상관(부호 반전)", "corr01", -1),
            ("전류 변동폭(평균 제거 표준편차)", "std_AI2", 1)]
    rows = []
    for name, col, sgn in spec:
        a, b = low[col].values, Fo[col].values
        rows.append({"지표": name, "저부하 정상 버스트": len(a), "고장 버스트": len(b),
                     "저부하 범위": "%.3g ~ %.3g" % (a.min(), a.max()),
                     "고장 범위": "%.3g ~ %.3g" % (b.min(), b.max()),
                     "AUC": round(roc_auc_score(y, sgn * np.r_[a, b]), 3),
                     "범위 겹침": bool(max(a.min(), b.min()) <= min(a.max(), b.max()))})
    return pd.DataFrame(rows).sort_values("AUC", ascending=False)


def i5_start_and_fp(H, F):
    """시작부 미탐이 3연속 규칙의 구조 때문인지, 빨강 오경보가 몇 사건인지."""
    rows = []
    first = F[F.pos == "첫 1초"]
    later = F[F.pos == "1초 이후"]
    for name, d in (("첫 1초 전체", first), ("첫 1초 · 각 버스트 첫 2 window 제외", first[first.win_idx >= 2]),
                    ("첫 1초 · 각 버스트 첫 2 window", first[first.win_idx < 2]), ("1초 이후", later)):
        rows.append({"구분": "고장 빨강 Recall", "대상": name, "window": len(d),
                     "버스트": int(d.burst_id.nunique()), "값": round(d.red.mean(), 4) if len(d) else np.nan,
                     "미탐 window": int((d.red == 0).sum())})
    fp = H[H.red == 1].sort_values("original_row_start")
    for b, g in fp.groupby("burst_id"):
        rows.append({"구분": "빨강 오경보", "대상": "버스트 %d (원본 행 %s)" % (b, ", ".join(map(str, g.original_row_start))),
                     "window": len(g), "버스트": 1, "값": np.nan, "미탐 window": np.nan})
    return pd.DataFrame(rows)


def main():
    print("변수 간 상호작용 진단표 (모델 불변)")
    H, F, bounds = _frame()
    print("  3분위 경계(학습 블록 0-2): 상부진동 RMS %s / 전류 RMS %s"
          % (np.round(bounds["vib"], 4), np.round(bounds["cur"], 1)))
    i1 = _grid(H, F, "vib_t", "cur_t", "상부 진동 RMS", "전류 RMS", LV, LV)
    save(i1, "i1_vib_x_current.csv")
    print(i1.to_string(index=False))
    i2 = _grid(H, F, "load_t", "pos", "부하(3채널 RMS)", "버스트 내 위치", LV, ["첫 1초", "1초 이후"])
    save(i2, "i2_load_x_position.csv")
    print(i2.to_string(index=False))
    i3 = pd.concat([_interaction(H, "vib_t", "cur_t", "상부 진동 RMS", "전류 RMS", LV, LV),
                    _interaction(H, "load_t", "pos", "부하(3채널 RMS)", "버스트 내 위치", LV, None)],
                   ignore_index=True)
    save(i3, "i3_interaction_size.csv")
    print(i3.to_string(index=False))
    i5 = i5_start_and_fp(H, F)
    save(i5, "i5_start_misses_and_red_fp.csv")
    print(i5.to_string(index=False))
    i4 = i4_burst_auc()
    save(i4, "i4_burst_indicator_auc.csv")
    print(i4.to_string(index=False))


if __name__ == "__main__":
    main()

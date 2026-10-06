# -*- coding: utf-8 -*-
"""내부 검증 2차 보완 표.

모델·임계값 불변. 기존 predictions.csv 와 원본 데이터만 사용한다.
  c1 : NC 음성대조를 '같은 유병률의 우연 수준'과 비교
  c2 : FP 집중조건을 동결 운영 구성에서 재계산
  c3 : 블록별 1단/빨강 경보율
  c4 : d2 의 재색인 burst 와 원시 burst_id 매핑
  c5 : 1단·2단 점수 상관
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


def save(df, name):
    df.to_csv(os.path.join(TAB, name), index=False, encoding="utf-8-sig")
    print("  -> %s (%d행)" % (name, len(df)))


def c1_nc_chance():
    """NC 는 균형 과제다. 같은 유병률의 우연 수준과 비교해야 한다."""
    d7 = pd.read_csv(os.path.join(TAB, "d7_supervised_control.csv"))
    rows = []
    for _, r in d7.iterrows():
        if str(r["id"]).startswith("NC"):
            pi = 0.5          # 블록을 시간순 반으로 가른 균형 과제
        else:
            pi = 0.1275       # 고장 vs 정상 운영 표본
        # 동전던지기: P=pi, R=0.5 -> F1 = 2*pi*0.5/(pi+0.5)
        coin = 2 * pi * 0.5 / (pi + 0.5)
        # 전부 양성: P=pi, R=1 -> F1 = 2*pi/(pi+1)
        allpos = 2 * pi / (pi + 1)
        rows.append({
            "id": r["id"], "과제": r["task"], "유병률(근사)": round(pi, 4),
            "보고 F1": round(float(r["f1_mean"]), 4),
            "동전던지기 F1": round(coin, 4),
            "전부양성 F1": round(allpos, 4),
            "우연 초과": bool(float(r["f1_mean"]) > coin + 0.02),
        })
    df = pd.DataFrame(rows)
    save(df, "c1_nc_vs_chance.csv")
    print(df.to_string(index=False))
    return df


def _regimes(pred, Xn, tr_mask):
    from src import features as FT
    lo, hi = FT.regime_bounds(Xn[tr_mask])
    return lo, hi


def c2_fp_conditions_frozen():
    """FP 집중조건을 동결 구성(학습 0-2, 보정 3, holdout 4)에서 재계산."""
    import yaml
    from src import data as D, features as FT, windows as WD
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
    ch, gap, seq = cfg["data"]["channels"], cfg["windows"]["gap_sec"], cfg["windows"]["length"]
    n, o, _ = D.load_all(cfg, HERE)
    Xn, mn = WD.make_windows(n, ch, seq, gap, source="normal")
    mn["block"] = WD.block_split(mn, cfg["cv"]["n_blocks"])
    tr = np.isin(mn["block"].values, [0, 1, 2])
    lo, hi = FT.regime_bounds(Xn[tr])

    pred = pd.read_csv(os.path.join(OUT, "predictions.csv"))
    ho = pred[pred.split == "holdout"].reset_index(drop=True)
    Xho = Xn[mn["block"].values == 4]
    reg = FT.load_regime(Xho, lo, hi)
    pos = ho["time_start"].values  # 위치는 meta 의 pos_in_burst 대체로 burst 내 순번 사용
    posn = ho.groupby("burst_id").cumcount().values * 0.1

    s1 = (ho.p_normal_stage1 <= 0.01).values
    red = (ho.alarm_level == "red").values
    rows = []
    for name, sel in (("저부하(RMS 하위 3분위)", reg == 0),
                      ("중부하", reg == 1),
                      ("고부하", reg == 2),
                      ("버스트 첫 1초", posn <= 1.0),
                      ("버스트 1초 이후", posn > 1.0)):
        if sel.sum() == 0:
            continue
        rows.append({"조건": name, "n": int(sel.sum()),
                     "1단 경보율": round(float(s1[sel].mean()), 4),
                     "빨강 경보율": round(float(red[sel].mean()), 4)})
    df = pd.DataFrame(rows)
    save(df, "c2_fp_conditions_frozen.csv")
    print(df.to_string(index=False))
    return df


def c3_block_alarm_rates():
    """간판 FPR 은 holdout 블록 하나의 값이다. 블록별로 공개한다."""
    pred = pd.read_csv(os.path.join(OUT, "predictions.csv"))
    nm = pred[pred.source == "normal"]
    rows = []
    for b, g in nm.groupby("block"):
        rows.append({"블록": int(b), "split": g.split.iloc[0], "window": len(g),
                     "1단 경보율": round(float((g.p_normal_stage1 <= 0.01).mean()), 4),
                     "빨강 경보율": round(float((g.alarm_level == "red").mean()), 4)})
    rows.append({"블록": -1, "split": "정상 전체", "window": len(nm),
                 "1단 경보율": round(float((nm.p_normal_stage1 <= 0.01).mean()), 4),
                 "빨강 경보율": round(float((nm.alarm_level == "red").mean()), 4)})
    df = pd.DataFrame(rows)
    save(df, "c3_block_alarm_rates.csv")
    print(df.to_string(index=False))
    return df


def c4_burst_id_map():
    """d2 의 burst 는 '10샘플 이상만 추려 0부터 재색인'한 값이다."""
    import yaml
    from src import data as D, windows as WD
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
    gap = cfg["windows"]["gap_sec"]
    _, o, _ = D.load_all(cfg, HERE)
    bid = WD.segment_bursts(o, gap)
    sizes = pd.Series(bid).value_counts().sort_index()
    keep = sizes[sizes >= 10]
    mapping = {i: int(raw) for i, raw in enumerate(keep.index)}

    pred = pd.read_csv(os.path.join(OUT, "predictions.csv"))
    fa = pred[pred.split == "fault"]
    d2 = pd.read_csv(os.path.join(TAB, "d2_quiet_bursts.csv"))
    rows = []
    for _, r in d2.iterrows():
        raw = mapping.get(int(r["burst"]), None)
        sub = fa[fa.burst_id == raw] if raw is not None else fa.iloc[0:0]
        rows.append({"d2 표기(재색인)": int(r["burst"]),
                     "원시 burst_id": raw,
                     "샘플": int(r["n_sample"]),
                     "window": len(sub),
                     "빨강 window": int((sub.alarm_level == "red").sum()),
                     "빨강 비율": round(float((sub.alarm_level == "red").mean()), 3)
                     if len(sub) else np.nan})
    df = pd.DataFrame(rows)
    save(df, "c4_burst_id_map.csv")
    print(df.to_string(index=False))
    return df


def c5_stage_correlation():
    """1단·2단이 독립 증거인지."""
    pred = pd.read_csv(os.path.join(OUT, "predictions.csv"))
    rows = []
    for name, sel in (("정상 holdout", pred.split == "holdout"),
                      ("고장 기록", pred.split == "fault")):
        g = pred[sel]
        sp = g.score_stage1.corr(g.score_stage2, method="spearman")
        a1 = g.p_normal_stage1 <= 0.01
        a2 = g.p_normal_stage2 <= 0.01
        rows.append({
            "대상": name, "n": len(g),
            "Spearman(1단,2단)": round(float(sp), 4),
            "P(2단경보|1단경보)": round(float(a2[a1].mean()), 4) if a1.sum() else np.nan,
            "P(1단경보|2단경보)": round(float(a1[a2].mean()), 4) if a2.sum() else np.nan,
        })
    df = pd.DataFrame(rows)
    save(df, "c5_stage_correlation.csv")
    print(df.to_string(index=False))
    return df


def main():
    print("정정 대응표 생성 (c1-c5)")
    c1_nc_chance()
    print()
    c2_fp_conditions_frozen()
    print()
    c3_block_alarm_rates()
    print()
    c4_burst_id_map()
    print()
    c5_stage_correlation()


if __name__ == "__main__":
    main()

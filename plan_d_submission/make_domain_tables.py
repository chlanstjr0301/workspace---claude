# -*- coding: utf-8 -*-
"""
도메인 지식 적용용 보조 표 생성 (P0)

  outputs/tables/d1_operating_modes.csv   운전모드 3분할 (주운전/저부하/이상)
  outputs/tables/d2_quiet_bursts.csv      이상 이벤트 내 조용한 버스트
  outputs/tables/d3_failure_signature.csv 고장모드 대조표의 '본 데이터' 행

규약은 config.yaml 과 동일하다: gap 0.5 s 로 버스트 분할, window length L=10
→ 길이 10샘플 미만 버스트는 window 를 만들 수 없으므로 제외(이상 17개).

usage: python plan_d_submission/make_domain_tables.py
"""
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "outputs", "tables")

COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
GAP_SEC, WIN_LEN = 0.5, 10
LOWLOAD_T0, LOWLOAD_T1 = 3600.0, 4280.0       # 저부하 운전모드 구간 (초, 파일 시작 기준)


def load(fn):
    return pd.read_csv(os.path.join(RAW, fn), index_col=0, parse_dates=["TimeStamp"])


def bursts(df):
    dt = df.TimeStamp.diff().dt.total_seconds().values
    e = np.concatenate(([0], np.where(dt > GAP_SEC)[0], [len(df)]))
    return [(int(a), int(b)) for a, b in zip(e[:-1], e[1:]) if b - a >= WIN_LEN]


def burst_table(df, label):
    X, ts = df[COLS].values, df.TimeStamp.values
    t0 = ts[0]
    rows = []
    for i, (a, b) in enumerate(bursts(df)):
        W = X[a:b]
        s0 = W[:, 0].std()
        rows.append(dict(
            source=label, burst=i, n_sample=b - a,
            t_start_s=float((ts[a] - t0) / np.timedelta64(1, "s")),
            # 주의: 아래 std_* 는 '버스트 평균을 제거한' 표준편차다.
            # 전류처럼 DC 오프셋이 큰 채널에서는 RMS 와 크게 달라지므로
            # 진짜 RMS(=sqrt(mean(x^2))) 를 별도 열로 함께 기록한다. (DL-007)
            std_AI0=float(s0), std_AI1=float(W[:, 1].std()),
            std_AI2=float(W[:, 2].std()),
            rms_AI0=float(np.sqrt((W[:, 0] ** 2).mean())),
            rms_AI1=float(np.sqrt((W[:, 1] ** 2).mean())),
            rms_AI2=float(np.sqrt((W[:, 2] ** 2).mean())),
            mean_AI2=float(W[:, 2].mean()),
            kurt_AI0=float(((W[:, 0] - W[:, 0].mean()) ** 4).mean() / (s0 ** 4 + 1e-24) - 3.0),
            corr01=float(np.corrcoef(W[:, 0], W[:, 1])[0, 1]) if s0 > 1e-9 else 0.0,
            acf1_AI0=float(np.corrcoef(W[:-1, 0], W[1:, 0])[0, 1]) if s0 > 1e-9 else 0.0,
        ))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    n, o = load("press_data_normal.csv"), load("press_data_outlier.csv")
    Fn, Fo = burst_table(n, "normal"), burst_table(o, "fault")

    low = (Fn.t_start_s >= LOWLOAD_T0) & (Fn.t_start_s <= LOWLOAD_T1)
    Fn["mode"] = np.where(low, "normal_lowload", "normal_main")
    Fo["mode"] = "fault"
    allb = pd.concat([Fn, Fo], ignore_index=True)

    # --- d1: 운전모드 3분할 -------------------------------------------------- #
    g = allb.groupby("mode")
    d1 = pd.DataFrame({
        "n_burst": g.size(),
        "rms_AI0": g.rms_AI0.mean(), "rms_AI1": g.rms_AI1.mean(), "rms_AI2": g.rms_AI2.mean(),
        "std_AI0": g.std_AI0.mean(), "std_AI2": g.std_AI2.mean(),
        "mean_AI2": g.mean_AI2.mean(),
        "corr01": g.corr01.mean(),
        "rms_AI2_min": g.rms_AI2.min(), "rms_AI2_max": g.rms_AI2.max(),
        "corr01_min": g.corr01.min(), "corr01_max": g.corr01.max(),
    }).reindex(["normal_main", "normal_lowload", "fault"])
    d1.to_csv(os.path.join(OUT, "d1_operating_modes.csv"), encoding="utf-8-sig")
    print("[d1] 운전모드 3분할")
    print(d1[["n_burst", "rms_AI0", "rms_AI1", "rms_AI2", "corr01"]]
          .to_string(float_format=lambda x: "%.3f" % x))
    print("\n  겹침 점검 — 저부하 정상 vs 고장")
    for c in ("rms_AI2", "corr01"):
        a = d1.loc["normal_lowload", [c + "_min", c + "_max"]].values
        b = d1.loc["fault", [c + "_min", c + "_max"]].values
        print("    %-8s 저부하 [%.2f, %.2f] / 고장 [%.2f, %.2f] -> %s"
              % (c, a[0], a[1], b[0], b[1],
                 "겹침" if a[0] <= b[1] and b[0] <= a[1] else "분리"))
    sep = d1.loc["fault", "rms_AI0"] / d1.loc["normal_lowload", "rms_AI0"]
    print("    rms_AI0  저부하 %.3f vs 고장 %.3f -> %.1f배 분리"
          % (d1.loc["normal_lowload", "rms_AI0"], d1.loc["fault", "rms_AI0"], sep))

    # --- d2: 이상 이벤트 내 조용한 버스트 ------------------------------------ #
    thr = Fn.rms_AI0.max()
    quiet = Fo[Fo.rms_AI0 <= thr].sort_values("t_start_s")
    dur = float((o.TimeStamp.values[-1] - o.TimeStamp.values[0]) / np.timedelta64(1, "s"))
    quiet = quiet.assign(position=np.where(quiet.t_start_s < dur / 3, "이벤트 초반",
                                           np.where(quiet.t_start_s > dur * 2 / 3,
                                                    "종료 직전", "중반")))
    cols = ["burst", "n_sample", "t_start_s", "rms_AI0", "rms_AI2", "corr01", "position"]
    quiet[cols].to_csv(os.path.join(OUT, "d2_quiet_bursts.csv"), index=False, encoding="utf-8-sig")
    print("\n[d2] 조용한 이상 버스트 (상부 RMS <= 정상 최댓값 %.4f)" % thr)
    print(quiet[cols].to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("  이벤트 길이 %.1f s / 이상 버스트 %d개 중 %d개" % (dur, len(Fo), len(quiet)))

    # --- d3: 고장모드 서명 --------------------------------------------------- #
    m, f = d1.loc["normal_main"], d1.loc["fault"]
    d3 = pd.DataFrame([
        dict(indicator="진동 RMS (상부)", normal_main=m.rms_AI0, fault=f.rms_AI0,
             ratio=f.rms_AI0 / m.rms_AI0),
        dict(indicator="첨도 (상부)", normal_main=Fn[~low].kurt_AI0.mean(),
             fault=Fo.kurt_AI0.mean(), ratio=np.nan),
        dict(indicator="자기상관 lag1 (상부)", normal_main=Fn[~low].acf1_AI0.mean(),
             fault=Fo.acf1_AI0.mean(), ratio=np.nan),
        dict(indicator="상·하부 상관", normal_main=m.corr01, fault=f.corr01, ratio=np.nan),
        dict(indicator="전류 RMS (sqrt(mean(x^2)))", normal_main=m.rms_AI2,
             fault=f.rms_AI2, ratio=f.rms_AI2 / m.rms_AI2),
        dict(indicator="전류 표준편차 (평균 제거)", normal_main=m.std_AI2,
             fault=f.std_AI2, ratio=f.std_AI2 / m.std_AI2),
        dict(indicator="전류 평균 (DC 오프셋)", normal_main=m.mean_AI2,
             fault=f.mean_AI2, ratio=np.nan),
    ])
    d3.to_csv(os.path.join(OUT, "d3_failure_signature.csv"), index=False, encoding="utf-8-sig")
    print("\n[d3] 고장모드 대조표의 '본 데이터' 행")
    print(d3.to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\n저장: outputs/tables/d1_operating_modes.csv, d2_quiet_bursts.csv, d3_failure_signature.csv")


if __name__ == "__main__":
    main()

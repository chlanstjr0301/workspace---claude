# -*- coding: utf-8 -*-
"""실시간 재생 검증 (s1): 원본 기록을 시각 순서대로 한 샘플씩 흘려 넣어 판정한다.

학습·보정은 run_all.final_predictions 와 같은 방식으로 한 번만 한다
(학습 블록 0-2, 보정 블록 3, 1단 = 선정 모델, 2단 = 전류 제외 진동 M1).
재생 단계에서는 새 샘플이 도착할 때마다 다음만 수행한다.

  1) 직전 샘플과 시각 간격 > 0.5초면 새 버스트 시작 (버퍼·연속 카운터 초기화)
  2) 버퍼에 샘플 추가. 10샘플이 차면 최근 10샘플 window 1개를 판정
  3) 특징 23개 → 1단·2단 점수 → 고정된 보정 점수로 conformal p → 초록·노랑·빨강 후보
  4) 같은 버스트에서 빨강 후보가 연속 3개면 빨강

판정에는 현재 샘플까지의 값만 쓴다(미래 샘플·파일 전체 통계 미사용).
재생 결과를 outputs/predictions.csv 와 window 단위로 대조해 표 s1 을 만든다(블록별).
보정 블록 3 의 window 는 자기 점수가 보정 점수에 포함되어 동점 비교가 부동소수점 끝자리에
따라 1순위 갈릴 수 있으므로 따로 보고한다.
처리시간은 실행 환경마다 다르므로 표가 아닌 outputs/stream_replay_timing.json 에 둔다.

    python stream_replay.py
"""
import json
import os
import platform
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs")
TAB = os.path.join(OUT, "tables")

from src import data as D, features as FT, calibration as CAL, models as MD  # noqa: E402


def fit_frozen(cfg):
    """run_all.final_predictions 와 같은 데이터·블록·모델로 1단·2단을 적합한다."""
    import run_all as RA
    normal, outlier, _ = D.load_all(cfg, HERE)
    P = RA.prepare(cfg, normal, outlier, cfg["windows"]["length"])
    mg = tuple(cfg["features"]["model_groups"])
    idx, names = RA.model_cols(P, mg)
    # 재생 단계는 모델 특징군만 계산하므로, 특징 순서가 전체 특징의 앞부분과 같아야 한다
    assert list(idx) == list(range(len(idx))), "모델 특징이 전체 특징의 앞 순서가 아님"
    vib = np.array([i for i, n in enumerate(names) if "CUR" not in n and not n.startswith("O_")])
    blk = P["mn"]["block"].values
    tr, cl = np.isin(blk, [0, 1, 2]), blk == 3
    with open(os.path.join(OUT, "run_manifest.json"), encoding="utf-8") as f:
        stage1 = json.load(f).get("chosen_stage1_model", "M3")

    def fit(cols, name):
        m = [x for x in MD.build(cfg, cfg["seed"]) if x.name == name][0]
        F = P["Fn"][:, idx][:, cols]
        m.fit(F[tr])
        return m, np.sort(m.score(F[cl]))

    m1, cal1 = fit(np.arange(len(idx)), stage1)
    m2, cal2 = fit(vib, "M1")
    return normal, outlier, mg, vib, m1, cal1, m2, cal2, stage1


class StreamJudge:
    """샘플 1개가 도착할 때마다 호출하는 실시간 판정기."""

    def __init__(self, cfg, mg, vib, m1, cal1, m2, cal2):
        self.ch = cfg["data"]["channels"]
        self.seq = cfg["windows"]["length"]
        self.gap = cfg["windows"]["gap_sec"]
        self.alpha = cfg["calibration"]["yellow_p"]
        self.red_n = cfg["calibration"]["red_consecutive"]
        self.mg, self.vib = mg, vib
        self.m1, self.cal1, self.m2, self.cal2 = m1, cal1, m2, cal2
        self.buf = deque(maxlen=self.seq)
        self.last_t, self.burst, self.run = None, -1, 0

    def push(self, t, x, row):
        """t: 시각, x: 채널값 3개, row: 원본 행 번호. window 가 완성되면 판정 dict 를 돌려준다."""
        if self.last_t is None or (t - self.last_t).total_seconds() > self.gap:
            self.burst += 1
            self.buf.clear()
            self.run = 0
        self.last_t = t
        self.buf.append((x, row))
        if len(self.buf) < self.seq:
            return None
        W = np.array([b[0] for b in self.buf], dtype=float)[None]
        F, _, _ = FT.build(W, self.ch, None, self.mg)
        p1 = float(CAL.conformal_p(self.cal1, self.m1.score(F))[0])
        p2 = float(CAL.conformal_p(self.cal2, self.m2.score(F[:, self.vib]))[0])
        if p1 > self.alpha:
            lvl, self.run = "green", 0
        elif p2 <= self.alpha:
            self.run += 1
            lvl = "red" if self.run >= self.red_n else "yellow"
        else:
            lvl, self.run = "yellow", 0
        return {"burst_id": self.burst, "original_row_start": int(self.buf[0][1]),
                "p1": p1, "p2": p2, "alarm_level": lvl}


def replay(df, judge):
    """기록을 시각 순서대로 한 샘플씩 흘려 넣는다."""
    X = df[judge.ch].values.astype(float)
    ts = df["TimeStamp"].tolist()
    rows = df["original_row"].values
    out, lat = [], []
    for i in range(len(df)):
        t0 = time.perf_counter()
        r = judge.push(ts[i], X[i], rows[i])
        if r is not None:
            lat.append(time.perf_counter() - t0)
            out.append(r)
    return pd.DataFrame(out), np.array(lat)


def main():
    print("실시간 재생 검증 s1")
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
    normal, outlier, mg, vib, m1, cal1, m2, cal2, stage1 = fit_frozen(cfg)
    pred = pd.read_csv(os.path.join(OUT, "predictions.csv"))
    parts = []
    lats = []
    for df, is_normal in ((normal, True), (outlier, False)):
        judge = StreamJudge(cfg, mg, vib, m1, cal1, m2, cal2)   # 기록마다 새 판정기
        rp, lat = replay(df, judge)
        lats.append(lat)
        ref = pred[(pred.source == "normal") == is_normal]
        parts.append(ref.merge(rp, on="original_row_start", how="outer", indicator=True,
                               suffixes=("", "_stream")))
    m = pd.concat(parts, ignore_index=True)
    names = {"fit": "정상 학습 블록 0-2", "calibration": "정상 보정 블록 3",
             "holdout": "정상 평가 블록 4", "fault": "고장 기록"}
    rows = []
    for sp in ("fit", "calibration", "holdout", "fault"):
        d = m[m.split == sp]
        both = d[d["_merge"] == "both"]
        if sp == "calibration":
            # 임계값을 정한 자료 자신이라 비교에서 뺀다 (동점 비교가 부동소수점 끝자리에 좌우됨)
            rows.append({"대상": names[sp], "저장 예측 window": len(d),
                         "재생 판정 window": int((d["_merge"] != "left_only").sum()),
                         "짝지은 window": len(both), "경보 상태 일치": "비교 제외",
                         "p값 차이 window": "비교 제외", "p값 최대 차이": "비교 제외",
                         "빨강 (재생)": "비교 제외", "빨강 (저장)": int((both.alarm_level == "red").sum())})
            continue
        pd1 = np.abs(both.p_normal_stage1 - both.p1)
        pd2 = np.abs(both.p_normal_stage2 - both.p2)
        rows.append({
            "대상": names[sp], "저장 예측 window": len(d), "재생 판정 window": int((d["_merge"] != "left_only").sum()),
            "짝지은 window": len(both),
            "경보 상태 일치": int((both.alarm_level == both.alarm_level_stream).sum()),
            "p값 차이 window": int(((pd1 > 1e-9) | (pd2 > 1e-9)).sum()),
            "p값 최대 차이": round(float(np.maximum(pd1, pd2).max()), 6),
            "빨강 (재생)": int((both.alarm_level_stream == "red").sum()),
            "빨강 (저장)": int((both.alarm_level == "red").sum()),
        })
    df = pd.DataFrame(rows)
    extra = int((m["_merge"] == "right_only").sum())
    assert extra == 0, "저장 예측에 없는 재생 window %d개" % extra
    df.to_csv(os.path.join(TAB, "s1_stream_replay.csv"), index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    lat = np.concatenate(lats) * 1000.0
    timing = {"stage1": stage1, "windows": int(len(lat)),
              "ms_per_window_median": round(float(np.median(lat)), 3),
              "ms_per_window_p99": round(float(np.percentile(lat, 99)), 3),
              "ms_per_window_max": round(float(lat.max()), 3),
              "sample_interval_ms": 100.0,
              "python": sys.version.split()[0], "platform": platform.platform(),
              "processor": platform.processor() or platform.machine()}
    with open(os.path.join(OUT, "stream_replay_timing.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, ensure_ascii=False, indent=2)
    print("  처리시간 window당 중앙값 %.3f ms, 99%% %.3f ms (샘플 간격 100 ms)"
          % (timing["ms_per_window_median"], timing["ms_per_window_p99"]))
    # 보정 블록 3 의 window 는 자기 점수가 보정 점수 집합에 들어 있어 동점 비교가 부동소수점
    # 마지막 자리에 따라 1순위(1/(n+1)) 갈린다. 운영 중 새 데이터에는 해당하지 않으므로
    # 일치 판정은 학습·평가 블록과 고장 기록으로 한다.
    ok = all(r["짝지은 window"] == r["저장 예측 window"] == r["재생 판정 window"]
             and r["경보 상태 일치"] == r["짝지은 window"] and r["p값 차이 window"] == 0
             for r in rows if r["대상"] != "정상 보정 블록 3")
    print("  재생 판정 = 저장 예측 (보정 블록 제외):", "일치" if ok else "불일치")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

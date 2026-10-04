# -*- coding: utf-8 -*-
"""결과보고서.md 의 핵심 수치가 outputs/ 의 CSV 와 일치하는지 검사한다.

각 항목은 CSV 에서 값을 읽어 보고서 표기 형식으로 바꾼 뒤, 그 문자열이
보고서 본문에 있는지 확인한다. 정정 전 수치나 표현이 남아 있는지도 확인한다.

    python report/check_numbers.py      # 실패가 있으면 종료코드 1
"""
import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAB = os.path.join(ROOT, "outputs", "tables")
REPORT = os.path.join(HERE, "결과보고서.md")


def T(name):
    return pd.read_csv(os.path.join(TAB, name + ".csv"))


def pct(x, d=2):
    return ("%." + str(d) + "f%%") % (100 * x)


def f(x, d=3):
    return ("%." + str(d) + "f") % x


def comma(n):
    return "{:,}".format(int(n))


def checks():
    out = []
    add = lambda label, s: out.append((label, s))

    q = T("e0_data_quality").set_index("source")
    add("정상 관측 기간", "%s초" % "{:,.1f}".format(q.loc["normal", "span_sec"]))
    add("정상 공백 수", "**%d**" % q.loc["normal", "n_gaps_gt_threshold"])
    b = T("e0_burst_summary").set_index("source")
    add("정상 버스트 수", "**%d개 수집 버스트**" % b.loc["normal", "n_bursts"])
    w = T("e1_window_sensitivity").set_index("seq")
    add("window 10 잔존율", "**%.1f%%**" % w.loc[10, "normal_retained_pct"])
    add("window 10 개수", comma(w.loc[10, "normal_windows_gap_aware"]))
    add("가짜 window 비율", "%.1f%%" % (100 - w.loc[10, "normal_retained_pct"]))
    st = T("e0_channel_stats").set_index(["source", "channel"])
    add("고장 전류 평균", "+%.2f" % st.loc[("outlier", "AI2_Current"), "mean"])

    d1 = T("d1_operating_modes").set_index("mode")
    add("저부하 버스트 수", "| 정상 — 저부하 | %d |" % d1.loc["normal_lowload", "n_burst"])
    add("고장 상부 RMS", "**%.3f**" % d1.loc["fault", "rms_AI0"])
    add("고장 전류 RMS", "%.1f" % d1.loc["fault", "rms_AI2"])

    m = T("e2_model_comparison").set_index("model")
    for k in ["BL0", "M1", "M2", "M3", "BL1"]:
        add("F1 " + k, f(m.loc[k, "f1_mean"]))
        add("최악블록 FP " + k, f(m.loc[k, "fp_rate_worst"], 4))

    r1 = T("r1_gate_before_after").set_index("모델")
    for k in ["BL0", "M1", "M2", "M3"]:
        add("섭동 오경보증가 " + k, "%.2f%%p" % r1.loc[k, "섭동 오경보증가(pp)"])

    r2 = T("r2_operational_performance").set_index("기준")
    red, s1 = r2.loc["최종 빨강"], r2.loc["1단 경보(노랑+빨강)"]
    add("빨강 F1", "**%s**" % f(red.f1))
    add("빨강 Recall", f(red.recall))
    add("빨강 Precision", f(red.precision))
    add("빨강 TP/FP/FN", "**%d** | **%d** | **%d**" % (red.TP, red.FP, red.FN))
    add("1단 F1", f(s1.f1))
    add("1단 FPR", f(s1.fpr, 4))

    r8 = T("r8_stage_decomposition")
    last = r8.iloc[-1]
    add("빨강 감소배수", "%.1f배" % last["1단 대비 FPR 배수"])
    both = r8[r8["규칙"] == "1단 AND 2단"].iloc[0]
    add("2단 확인 배수", "**%.1f배**" % both["1단 대비 FPR 배수"])
    add("3연속 추가 배수", "**%.1f배**" % (both["fpr"] / last["fpr"]))

    c3 = T("c3_block_alarm_rates")
    allrow = c3[c3["블록"] == -1].iloc[0]
    add("정상 전체 빨강율", "**%s**" % pct(allrow["빨강 경보율"]))
    b0 = c3[c3["블록"] == 0].iloc[0]
    add("블록0 빨강율", "**%s**" % pct(b0["빨강 경보율"]))

    r9 = T("r9_burst_detection")
    add("빨강 탐지 버스트", "%d개 중 15" % len(r9) if r9.red_detected.sum() == 15 else "불일치")
    fn = T("r9_red_fn_causes")
    for _, r in fn.iterrows():
        add("미탐 " + r["원인"], "| %d | %.1f%% |" % (r["미탐 window"], 100 * r["비중"]))

    r10 = T("r10_posterior_bounds")
    add("사후확률 1%", "**%s**" % f(r10.iloc[0, 5]))
    add("사후확률 0.1%", f(r10.iloc[1, 5]))
    add("보수 1%", f(r10.iloc[0, 6]))

    r13 = T("r13_probability_calibration")
    add("보정 Brier", "**%.4f**" % r13.iloc[1]["Brier"])
    add("순위 Brier", "%.4f" % r13.iloc[0]["Brier"])

    r11 = T("r11_negative_control_channels")
    add("전류만 AUROC", "%.4f" % r11.iloc[2].filter(like="AUROC").iloc[0])
    r5 = T("r5_stage1_channel_share")
    add("전류 기여 비중", "**%.1f%%**" % (100 * r5.iloc[0]["비중"]))
    c5 = T("c5_stage_correlation")
    add("단계 순위상관", "%.3f" % c5.iloc[0, 2])

    i4 = T("i4_burst_indicator_auc").set_index("지표")
    add("상부 RMS AUC", "%.3f" % i4.loc["상부 진동 RMS", "AUC"])
    add("전류 RMS AUC", "%.3f" % i4.loc["전류 RMS", "AUC"])
    i2 = T("i2_load_x_position")
    mid1 = i2[(i2.iloc[:, 0] == "중") & (i2.iloc[:, 1] == "첫 1초")].iloc[0]
    add("중부하 첫1초 빨강율", "**%s**" % pct(mid1["빨강 경보율"]))

    r7 = T("r7_alarm_event_rates").set_index("경보")
    add("관측시간", "%.3f시간" % r7.loc["빨강", "관측시간(h)"])
    add("빨강 시간당", "| %d | %d |" % (round(r7.loc["빨강", "사건/h"]), round(r7.loc["빨강", "사건/h 95% 상한"])))

    c2 = T("c2_fp_conditions_frozen").set_index("조건")
    add("저부하 1단 경보율", "**%s**" % pct(c2.iloc[0]["1단 경보율"]))

    d4 = T("d4_mofn_tradeoff")
    sub = d4[d4.is_submitted].iloc[0]
    add("연속3 오경보", "**%d** | **%d**" % (sub.fp_window, sub.fp_events))
    return out


FORBIDDEN = [
    "82배", "25개 특징", "표 44개", "0.081시간", "0.139", "5개 모델을 동일 조건으로 비교",
    "17개 중 3개는 구조적으로", "AC 커플링", "평균 부하 수준 정보는 데이터에 존재하지 않는다",
    "작성 요령", "DL-006 정정",
]


def main():
    text = open(REPORT, encoding="utf-8").read()
    fails = []
    items = checks()
    for label, s in items:
        if s not in text:
            fails.append("누락/불일치: %-16s 기대 '%s'" % (label, s))
    for s in FORBIDDEN:
        if s in text:
            fails.append("정정 전 표현 잔존: '%s'" % s)
    print("수치 검사 %d건, 금지 표현 %d건" % (len(items), len(FORBIDDEN)))
    for x in fails:
        print("  FAIL " + x)
    print("결과: %s" % ("통과" if not fails else "실패 %d건" % len(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

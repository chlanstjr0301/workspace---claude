# -*- coding: utf-8 -*-
"""결과보고서.md 의 인용 수치가 outputs/ 의 CSV 와 일치하는지 검사한다.

각 항목은 CSV 에서 값을 읽어 보고서 표기(반올림 자릿수)로 바꾼 뒤, 굵게·공백 표기를
지운 본문에서 그 문자열을 찾는다. 정정 전 수치·표현이 남아 있는지도 검사한다.

    python report/check_numbers.py      # 실패가 있으면 종료코드 1
"""
import os
import re
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(os.path.dirname(HERE), "outputs", "tables")
REPORT = os.path.join(HERE, "결과보고서.md")


def T(name):
    return pd.read_csv(os.path.join(TAB, name + ".csv"))


def p(x, d=2):
    return ("%." + str(d) + "f%%") % (100 * x)


def f(x, d=3):
    return ("%." + str(d) + "f") % x


def c(n):
    return "{:,}".format(int(round(n)))


def checks():
    out = []
    add = lambda label, s: out.append((label, s))

    # ---- 1장 ----------------------------------------------------------- #
    q = T("e0_data_quality").set_index("source")
    add("정상 관측 기간", "{:,.1f}초".format(q.loc["normal", "span_sec"]))
    add("정상 공백 수", "공백 %d개" % q.loc["normal", "n_gaps_gt_threshold"])
    b = T("e0_burst_summary").set_index("source")
    add("정상 버스트", "버스트 %d개" % b.loc["normal", "n_bursts"])
    add("버스트 최장 샘플", "%d샘플(%.1f초)" % (b.loc["normal", "samples_max"], b.loc["normal", "dur_max_sec"]))
    w = T("e1_window_sensitivity").set_index("seq")
    add("window10 잔존율", "%.1f%%" % w.loc[10, "normal_retained_pct"])
    add("window10 수", c(w.loc[10, "normal_windows_gap_aware"]))
    add("공백 넘는 window 비율 10", "%.1f%%" % (100 - w.loc[10, "normal_retained_pct"]))
    add("공백 넘는 window 비율 20", "%.1f%%" % (100 - w.loc[20, "normal_retained_pct"]))
    st = T("e0_channel_stats").set_index(["source", "channel"])
    add("고장 전류 평균", "+%.2f" % st.loc[("outlier", "AI2_Current"), "mean"])
    add("정상 전류 평균", "+%.2f" % st.loc[("normal", "AI2_Current"), "mean"])
    fp_ = T("v1a_acquisition_fingerprint")
    cur = fp_[(fp_.channel == "AI2_Current")].drop_duplicates("source").set_index("source")
    add("정상 전류 고유값", "%s / %s" % (c(cur.loc["normal", "n_unique"]), c(cur.loc["normal", "n"])))
    add("고장 전류 고유값", "%s / %s" % (c(cur.loc["outlier", "n_unique"]), c(cur.loc["outlier", "n"])))
    d1 = T("d1_operating_modes").set_index("mode")
    add("저부하 버스트", "버스트 %d개" % d1.loc["normal_lowload", "n_burst"])
    add("고장 상부 RMS", f(d1.loc["fault", "rms_AI0"]))
    add("고장 전류 RMS", "%.1f" % d1.loc["fault", "rms_AI2"])
    bs = T("v1b_basic_stats")
    zrow = bs[bs.iloc[:, 0].astype(str).str.contains("AI0")].iloc[0]
    add("고장 상부 |z|>4", "%.1f%%" % float(str(zrow["고장"]).rstrip("%")))

    # ---- 2장 ----------------------------------------------------------- #
    m = T("e2_model_comparison").set_index("model")
    for k in ["BL0", "M1", "M2", "M3", "BL1"]:
        add("F1 " + k, f(m.loc[k, "f1_mean"]))
        add("최악블록 FP " + k, f(m.loc[k, "fp_rate_worst"], 4))
    add("Recall 1.000 모델 수", "5종 중 %d종이 Recall 1.000" % int((m.loc[["BL0", "M1", "M2", "M3", "BL1"], "recall_mean"] >= 0.9995).sum()))
    r2 = T("r2_operational_performance").set_index("기준")
    add("CV 유병률", "유병률 %.1f%%" % (100 * r2.loc["M3 (CV 5-fold 평균)", "유병률"]))
    red, s1 = r2.loc["최종 빨강"], r2.loc["1단 경보(노랑+빨강)"]
    add("빨강 TP/FP/FN", "| %d | %d | %d |" % (red.TP, red.FP, red.FN))
    add("빨강 F1", f(red.f1)); add("빨강 Recall", f(red.recall)); add("1단 F1", f(s1.f1))
    add("1단 FPR", p(s1.fpr))
    r1 = T("r1_gate_before_after").set_index("모델")
    for k in ["BL0", "M1", "M2", "M3"]:
        add("섭동 오경보증가 " + k, "%.2f%%p" % r1.loc[k, "섭동 오경보증가(pp)"])
    gates = T("e2_selection_gates").set_index("model")
    for k, lab in (("BL0", "BL-0"), ("M1", "M1"), ("M2", "M2"), ("M3", "**M3**"), ("BL1", "BL-1")):
        add("시간당 경보 상한 " + k, "| %s | %.4f | %d |" % (lab, gates.loc[k, "fp_rate_worst"], round(gates.loc[k, "far_h_upper95_worst"])))
    v4a = T("v4a_selection_sensitivity")
    m3 = v4a[(v4a.chosen == "M3") & (v4a.iloc[:, 0].astype(str).str.contains("gate2"))]
    add("선정 민감도 구간", "%.1f~%.1f%%p" % (m3["from"].min(), m3["to"].max()))
    v4c = T("v4c_control_threshold_free")
    v4c = v4c[v4c.feature_set.astype(str).str.contains("25")].set_index("id")
    add("M4 무작위 F1", f(v4c.loc["M4", "coin_f1"]))
    add("NC AUROC 범위", "%.2f~%.2f" % (v4c.loc[["NC-01", "NC-34", "NC-04"], "auroc"].min(),
                                       v4c.loc[["NC-01", "NC-34", "NC-04"], "auroc"].max()))
    v3 = T("v3a_burst_bootstrap_ci")
    rb = v3[v3.level == "빨강"].set_index("metric")
    add("빨강 F1 구간", "0.931 [%.3f, %.3f]" % (rb.loc["f1", "ci_lo_2.5pct"], rb.loc["f1", "ci_hi_97.5pct"]))
    add("빨강 FPR 상한", "%.2f]" % (100 * rb.loc["fpr", "ci_hi_97.5pct"]))
    add("빨강 Recall 구간", "%.3f–%.3f" % (rb.loc["recall", "ci_lo_2.5pct"], rb.loc["recall", "ci_hi_97.5pct"]))
    r8 = T("r8_stage_decomposition").set_index("규칙")
    add("2단 확인 배수", "%.1f배" % r8.loc["1단 AND 2단", "1단 대비 FPR 배수"])
    add("3연속 배수", "%.1f배" % (r8.loc["1단 AND 2단", "fpr"] / r8.iloc[-1]["fpr"]))
    c3 = T("c3_block_alarm_rates")
    add("정상 전체 빨강율", p(c3[c3["블록"] == -1].iloc[0]["빨강 경보율"]))
    add("블록0 빨강율", p(c3[c3["블록"] == 0].iloc[0]["빨강 경보율"]))
    add("정상 전체 1단율", p(c3[c3["블록"] == -1].iloc[0]["1단 경보율"]))
    v2a = T("v2a_frozen_split_full_system")
    v2a = v2a[v2a.rule.astype(str).str.contains("빨강")].set_index("stage1")
    v2c = T("v2c_cv_full_system_summary")
    v2c = v2c[v2c.rule.astype(str).str.contains("빨강")].set_index("stage1")
    v2d_all = T("v2d_stress_by_stage1")
    v2d0 = v2d_all[v2d_all.perturbation == "none"].set_index("stage1")
    v2d = v2d_all[v2d_all.perturbation != "none"].groupby("stage1").agg(
        s1=("normal_stage1_rate", "max"), red=("normal_red_rate", "max"), rec=("fault_red_recall", "min"))
    for k in ["BL0", "M1", "M2", "M3"]:
        add("후보 행 " + k, "| %s | %d | %s | %s | %s |" % (
            p(v2d0.loc[k, "normal_stage1_rate"]), v2a.loc[k, "FP"], f(v2a.loc[k, "f1"]),
            f(v2c.loc[k, "f1_mean"]), p(v2c.loc[k, "fpr_mean"])))
        add("후보 섭동 " + k, "| %.1f%% | %s | %s |" % (100 * v2d.loc[k, "s1"], p(v2d.loc[k, "red"]), f(v2d.loc[k, "rec"])))
    for k, lab in (("BL0", "BL-0"), ("M1", "M1"), ("M2", "M2"), ("M3", "M3 (제출)")):
        add("구조 기여 " + k, "| %s | %s | %s | %d |" % (lab, p(v2d0.loc[k, "normal_stage1_rate"]), p(v2a.loc[k, "fpr"]), v2a.loc[k, "FP"]))
    add("BL-0 동점 비율", "%.1f%%" % (100 * v2a.loc["BL0", "p_ties_share"]))
    v3b = T("v3b_posterior_bootstrap")
    add("사후 1% 점", f(v3b.iloc[0, 7])); add("사후 1% 보수", f(v3b.iloc[0, 9]))
    add("사후 0.1% 점", f(v3b.iloc[1, 7])); add("사후 0.1% 보수", f(v3b.iloc[1, 9]))
    r13 = T("r13_probability_calibration")
    add("보정 Brier", "| %.4f |" % r13.iloc[1]["Brier"])
    add("순위 Brier", "Brier %.3f" % r13.iloc[0]["Brier"])

    # ---- 3장 ----------------------------------------------------------- #
    v6 = T("v6_feature_importance").set_index("feature")
    for k in ["S_ac1_CUR", "R_stdratio_AI0_AI1", "S_ac2_CUR"]:
        add("기여 " + k, "%.1f%%" % (100 * v6.loc[k, "top1_share_fault"]))
    add("S_ac1 AUROC", f(v6.loc["S_ac1_CUR", "single_auroc"]))
    r5 = T("r5_stage1_channel_share")
    add("전류 기여 비중", "%.1f%%" % (100 * r5.iloc[0]["비중"]))
    v1c = T("v1c_dc_quantization_counterfactual").set_index("ID")
    add("C2 1단 경보율", p(v1c.loc["C2", "normal_stage1_rate"]))
    add("C1 F1 불변", f(v1c.loc["C1", "red_f1"], 4))
    v1d = T("v1d_channel_group_separability").set_index(["subset", "model"])
    add("진동 진폭 AUROC", "AUROC %.3f" % v1d.loc[("G-vibA", "M1"), "auroc"])
    add("전류 8개 AUROC", "AUROC %.3f" % v1d.loc[("G-cur", "M3"), "auroc"])
    w2b = T("w2b_confound_margin")
    w2b = w2b[(w2b.preproc == "R0") & (w2b.model == "M1")].set_index("subset")
    add("가짜 고장 최대", "최대 %.3f" % w2b.loc["G-vibA", "pseudo_auroc_max"])
    i1 = T("i1_vib_x_current")
    add("상호작용 칸", "%s (%s)" % (p(i1.iloc[0]["1단 경보율"]), c(i1.iloc[0]["정상 window"])))
    c2 = T("c2_fp_conditions_frozen")
    add("저부하 1단", p(c2.iloc[0]["1단 경보율"]))
    lo_fp = int(round(c2.iloc[0]["n"] * c2.iloc[0]["1단 경보율"]))
    add("저부하 3분위 오경보 수", "| %d (%d%%) |" % (lo_fp, round(100 * lo_fp / 241)))
    i6 = T("i6_lowload_overlap").set_index("구분")
    for k, lab in (("저부하 구간", "1.6 저부하 구간"), ("하위 3분위 ∩ 저부하 구간", "두 집합의 교집합"),
                   ("저부하 구간 밖", "저부하 구간 밖")):
        r_ = i6.loc[k]
        add("i6 " + k, "| %s | %s (%d%%) | %d | %d (%d%%) | %s | %s |" % (
            lab, c(r_["window"]), round(100 * r_["window 비중"]), r_["버스트"], r_["1단 오경보"],
            round(100 * r_["1단 오경보 비중"]), p(r_["1단 경보율"]), p(r_["빨강 경보율"])))
    ll = i6.loc["저부하 구간"]
    add("저부하 헤드", "1단 오경보 241개 중 %d개(%d%%)가 window %d%%인 저부하 구간에 있고 경보율은 구간 밖의 %.1f배다" % (
        ll["1단 오경보"], round(100 * ll["1단 오경보 비중"]), round(100 * ll["window 비중"]),
        ll["1단 경보율"] / i6.loc["저부하 구간 밖", "1단 경보율"]))
    add("하위3분위∩구간", "1,847개 중 %s개(%d%%)" % (c(i6.loc["하위 3분위 ∩ 저부하 구간", "window"]),
        round(100 * i6.loc["하위 3분위 ∩ 저부하 구간", "window"] / i6.loc["부하 하위 3분위", "window"])))
    i5 = T("i5_start_misses_and_red_fp").set_index("대상")
    add("시작부 3번째부터 Recall", f(i5.loc["첫 1초 · 각 버스트 첫 2 window 제외", "값"]))
    add("첫 2 window 미탐", "%d개(62%%)" % i5.loc["첫 1초 · 각 버스트 첫 2 window", "미탐 window"])
    r6 = T("r6_frozen_system_stress")
    a1 = r6[(r6.perturbation == "offset_AI1_only") & (r6.amount.astype(str) == "0.5")].iloc[0]
    add("하부 진동 오프셋 배수", "%.1f배" % a1.normal_red_ratio_vs_none)
    add("하부 진동 2단 경보율", "%.2f%%" % (100 * a1.normal_stage2_rate))
    oa = r6[(r6.perturbation == "offset_all") & (r6.amount.astype(str) == "0.5")].iloc[0]
    add("전 채널 오프셋 최악", "%s(%.1f배)" % (p(oa.normal_red_rate), oa.normal_red_ratio_vs_none))
    _cur = r6[r6.perturbation.isin(["offset_AI2_only", "gain_AI2_only"]) |
              ((r6.perturbation == "polarity") & (r6.amount.astype(str) == "AI2"))]
    add("전류 단독 조건 수", "전류 단독 계측 변화 %d조건" % len(_cur))
    add("전류 단독 최대 배수", "2.3배 이내" if round(_cur.normal_red_ratio_vs_none.max(), 1) == 2.3 else "불일치")
    add("섭동 조건 수", "섭동 35조건" if len(r6) - 1 == 35 else "불일치")
    _r = r6[r6.perturbation != "none"].copy(); _r["amount"] = _r.amount.astype(str)
    def _grp(x):
        q, m = x.perturbation, x.amount
        for ch, lab in (("AI2", "전류 단독"), ("AI0", "상부 진동 단독"), ("AI1", "하부 진동 단독")):
            if q in ("offset_%s_only" % ch, "gain_%s_only" % ch) or (q == "polarity" and m == ch):
                return lab
        if q in ("offset_all", "gain_all") or (q == "polarity" and m == "all"):
            return "전 채널 동시"
        return "샘플링 흔들림"
    _r["g"] = _r.apply(_grp, axis=1)
    for g, d in _r.groupby("g"):
        w = d.loc[d.normal_red_rate.idxmax()]
        add("섭동군 " + g, "| %d | %.1f~%.1f배 |" % (len(d), d.normal_red_ratio_vs_none.min(), d.normal_red_ratio_vs_none.max()))
        add("섭동군 최악 " + g, "| %s | %.1f%% | %s |" % (p(w.normal_red_rate), 100 * d.normal_stage1_rate.max(), f(d.fault_red_recall.min())))

    # ---- 4장 ----------------------------------------------------------- #
    v5a = T("v5a_alarm_per_shift")
    b4 = v5a[(v5a.scope == "블록 4")].set_index("level")
    add("교대당 빨강", "| %s | %s |" % (c(b4.loc["red", "per_shift_8h"]), c(b4.loc["red", "per_shift_8h_upper95"])))
    add("교대당 노랑", c(b4.loc["yellow", "per_shift_8h"]))
    add("수집 비율", "%.2f" % b4.loc["red", "duty"])
    al = v5a[v5a.scope.str.startswith("정상 전체")].iloc[0]
    s1 = T("s1_stream_replay").set_index("대상")
    for k, lab in (("정상 학습 블록 0-2", "정상 학습 블록 0–2"), ("정상 평가 블록 4", "정상 평가 블록 4"), ("고장 기록", "고장 기록")):
        r_ = s1.loc[k]
        assert int(r_["경보 상태 일치"]) == int(r_["짝지은 window"]) == int(r_["저장 예측 window"])
        add("재생 " + k, "| %s | %s | %s | 0 | %d / %d |" % (lab, c(int(r_["짝지은 window"])), c(int(r_["경보 상태 일치"])),
                                                       int(r_["빨강 (재생)"]), int(r_["빨강 (저장)"])))
    n_rep = sum(int(s1.loc[k, "경보 상태 일치"]) for k in ("정상 학습 블록 0-2", "정상 평가 블록 4", "고장 기록"))
    add("재생 일치 합계", "일괄 예측 %s window와 모두 같다" % c(n_rep))
    pr = pd.read_csv(os.path.join(os.path.dirname(TAB), "predictions.csv"))
    rd = pr[pr.alarm_level == "red"]
    add("빨강 행 구성", "빨강 %d행(고장 %d, 정상 %d)" % (len(rd), (rd.source != "normal").sum(), (rd.source == "normal").sum()))
    up = 3.0 / al.collect_hours
    add("S2 0회 상한", "수집 1시간당 %.1f회, 8시간 교대당 %.1f회" % (up, up * 8 * al.duty))
    d4 = T("d4_mofn_tradeoff").set_index("rule")
    add("판정 시각 지연", "0.9초 → %.1f초" % (d4.loc["연속 3", "delay_median_s"] + 0.9))
    v5b_ = T("v5b_stop_rule_triggers")
    _c = [x for x in v5b_.columns if "첫 발동" in x][0]
    _t = float(v5b_[(v5b_.scope == "고장 기록") & (v5b_.rule == "S2")][_c].iloc[0])
    add("S2 판정 시각", "0.9초 뒤인 %.1f초" % (_t + 0.9))
    add("정상 수집·벽시계 분", "벽시계 %d분·수집 %d분" % (round(60 * al.wall_hours), round(60 * al.collect_hours)))
    v5b = T("v5b_stop_rule_triggers")
    col = [x for x in v5b.columns if "첫 발동" in x][0]
    s2f = v5b[(v5b.scope == "고장 기록") & (v5b.rule == "S2")].iloc[0]
    add("S2 고장 첫 발동", "%.1f초" % float(s2f[col]))
    v5c = T("v5c_reason_distribution")
    g = v5c[(v5c["표본"] == "고장 red")]
    add("1단 전류 비율", "%.1f%%" % (100 * g[(g["범주"] == "전류")].iloc[0]["비율"]))
    add("2단 진폭 비율", "%.1f%%" % (100 * g[g["기여 출처"].str.contains("2단") & (g["범주"] == "진동진폭 A")].iloc[0]["비율"]))

    # ---- 6장 ----------------------------------------------------------- #
    n_tab = len([x for x in os.listdir(TAB) if x.endswith(".csv")])
    add("표 개수", "표 %d개" % n_tab)
    return out


FORBIDDEN = [
    "고장 신호를 지지한다", "수집 시점 차이로 생기는 크기", "재고 교대당", "4상태", "결정 22건", "36조건", "0.9999988", "3.6시간 ×", "±2.6%p", "무작위 수준(0.49", "수십 건", "10건 이상", "5개 중 3개가 Recall", "5종 중 3종이 Recall", "82배", "25개 특징", "표 44개", "0.081시간", "0.139",
    "17개 중 3개는 구조적으로", "AC 커플링", "평균 부하 수준 정보는 데이터에 존재하지 않는다",
    "작성 요령", "상호작용에서 나온다", "한 번도 보지 않은", "우연 수준이다", "직류 성분이 다르며",
    "60배 이상", "약 12.0%", "0.203", "80분의 1",
]


def norm(s):
    return re.sub(r"\*\*", "", s)


def main():
    text = norm(open(REPORT, encoding="utf-8").read())
    fails, items = [], checks()
    for label, s in items:
        if norm(s) not in text:
            fails.append("누락/불일치: %-18s 기대 '%s'" % (label, s))
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

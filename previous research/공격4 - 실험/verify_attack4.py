# -*- coding: utf-8 -*-
"""
공격4 — Plan A 적대적 검증의 재현 스크립트 (단일 엔트리)

`공격4 - Plan A 적대적 검증.md` 에서 ✅(직접 재현)로 표시된 모든 수치를 한 번에 재생성한다.

usage
    python verify_attack4.py              # 전부 실행 + results/ 에 저장
    python verify_attack4.py --only V3    # 특정 검증만
    python verify_attack4.py --list       # 검증 목록

출력
    results/verify_output.txt   사람이 읽는 전체 로그
    results/findings.csv        기계가 읽는 결과 (문서 표와 1:1)

평가 조건 (Plan A / Model Performance Indicators §6.1, §7.1-D 와 동일)
    gap-aware windowing, SEQ=20, offset=0
    train normal[:12000] / valid normal[12000:15000] / test normal[15000:] + outlier 전체
    임계값 = 정상 검증구간 q99.9 (이상 라벨 미사용)

환경
    Python 3.13.9 / numpy 2.5.1 / pandas 2.2.3 / scikit-learn 1.7.2
    Windows 콘솔에서 한글이 깨지면: set PYTHONIOENCODING=utf-8
"""
import argparse
import csv
import io
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):          # 한국어 Windows 콘솔(cp949) 대응
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "data", "raw"))
OUT = os.path.join(HERE, "results")

COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
SEQ, GAP_SEC = 20, 0.5
TRAIN_END, VALID_END = 12000, 15000
FEAT = [f"{c[:3]}_{m}" for c in COLS for m in ("acf1", "acf2", "std", "ptp")] + ["corr01"]

FINDINGS = []          # (검증ID, 항목, 값, 문서기재값, 판정)
_LOG = io.StringIO()


def say(s=""):
    print(s)
    _LOG.write(s + "\n")


def record(vid, item, value, doc=None):
    ok = "" if doc is None else ("일치" if str(value) == str(doc) else "불일치")
    FINDINGS.append((vid, item, value, "" if doc is None else doc, ok))


# --------------------------------------------------------------------------- #
# 공통
# --------------------------------------------------------------------------- #
def load():
    n = pd.read_csv(os.path.join(RAW, "press_data_normal.csv"),
                    index_col=0, parse_dates=["TimeStamp"])
    o = pd.read_csv(os.path.join(RAW, "press_data_outlier.csv"),
                    index_col=0, parse_dates=["TimeStamp"])
    return n, o


def runs(df):
    """dt > GAP_SEC 로 끊은 연속 구간 [(start, end)] (end 배타)."""
    dt = df.TimeStamp.diff().dt.total_seconds().values
    e = np.concatenate(([0], np.where(dt > GAP_SEC)[0], [len(df)]))
    return [(int(a), int(b)) for a, b in zip(e[:-1], e[1:]) if b > a]


def acf(x, lag):
    xc = x - x.mean()
    if xc.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(xc[:-lag], xc[lag:])[0, 1])


def window_features(df):
    """gap-aware window 별 13특징 + window 시작 행 인덱스."""
    X = df[COLS].values
    F, T = [], []
    for a, b in runs(df):
        for i in range(a, b - SEQ + 1):
            W = X[i:i + SEQ]
            f = []
            for k in range(3):
                f += [acf(W[:, k], 1), acf(W[:, k], 2),
                      float(W[:, k].std()), float(np.ptp(W[:, k]))]
            u, v = W[:, 0] - W[:, 0].mean(), W[:, 1] - W[:, 1].mean()
            f.append(float(np.corrcoef(u, v)[0, 1])
                     if u.std() > 1e-9 and v.std() > 1e-9 else 0.0)
            F.append(f)
            T.append(i)
    return np.array(F), np.array(T)


def split(Fn, Tn):
    return (Fn[Tn < TRAIN_END],
            Fn[(Tn >= TRAIN_END) & (Tn < VALID_END)],
            Fn[Tn >= VALID_END])


def confusion(s_norm, s_anom, th):
    pn, pa = (s_norm > th).astype(int), (s_anom > th).astype(int)
    tp, fn = int(pa.sum()), len(pa) - int(pa.sum())
    fp, tn = int(pn.sum()), len(pn) - int(pn.sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, P=P, R=R, F1=F1,
                FPR=fp / (fp + tn) if fp + tn else 0.0, pn=pn)


def line(label, c):
    say("  %-38s TP=%3d FN=%3d FP=%4d TN=%4d | P=%.3f R=%.3f F1=%.4f FPR=%.4f"
        % (label, c["tp"], c["fn"], c["fp"], c["tn"], c["P"], c["R"], c["F1"], c["FPR"]))


# --------------------------------------------------------------------------- #
# V1  디바운스 k=5 → 금지 모델이 선정 규칙을 통과한다      (문서 §1.1)
# --------------------------------------------------------------------------- #
def V1(n, o, Fn, Tn, Fa):
    say("\n" + "=" * 94)
    say("V1  디바운스 k=5 적용 FAR/h — 금지한 AI2_acf1이 선정 규칙을 통과하는가  (문서 §1.1)")
    say("=" * 94)
    tr, va, te = split(Fn, Tn)
    i = FEAT.index
    dur_h = 0.1 * len(n.iloc[VALID_END:]) / 3600.0

    def max_streak(p):
        m = c = 0
        for v in p:
            c = c + 1 if v else 0
            m = max(m, c)
        return m

    say("  관측 데이터시간 = %.3f h" % dur_h)
    say("  %-24s %4s %6s %8s %9s" % ("후보", "FP", "최장연속", "k=5알람", "FAR/h"))
    cands = [("-AI2_acf1 (금지)", i("AI2_acf1"), -1),
             ("C2 AI1_acf2", i("AI1_acf2"), +1),
             ("C3 -corr01", i("corr01"), -1),
             ("C4 AI0_std", i("AI0_std"), +1)]
    for label, j, sgn in cands:
        c = confusion(sgn * te[:, j], sgn * Fa[:, j],
                      np.quantile(sgn * va[:, j], 0.999))
        pn = c["pn"]
        k5 = sum(1 for x in range(len(pn) - 4) if pn[x:x + 5].all())
        ms = max_streak(pn)
        say("  %-24s %4d %6d %8d %9.1f" % (label, c["fp"], ms, k5, k5 / dur_h))
        record("V1", label + " FP", c["fp"])
        record("V1", label + " 최장연속FP", ms)
        record("V1", label + " FAR/h(k=5)", round(k5 / dur_h, 1))

    from sklearn.ensemble import IsolationForest
    AMP6 = [i("AI0_std"), i("AI0_ptp"), i("AI1_std"), i("AI1_ptp"),
            i("AI2_std"), i("AI2_ptp")]
    m = IsolationForest(n_estimators=300, random_state=42).fit(tr[:, AMP6])
    c = confusion(-m.score_samples(te[:, AMP6]), -m.score_samples(Fa[:, AMP6]),
                  np.quantile(-m.score_samples(va[:, AMP6]), 0.999))
    say("  %-24s %4d %6d %8d %9.1f" % ("IF 진폭6 (주력)", c["fp"], max_streak(c["pn"]), 0, 0.0))
    record("V1", "IF 진폭6 FP", c["fp"], "0")
    say("\n  판정: 금지 모델의 FP가 전부 고립 단발이면 k=5가 전멸시켜 FAR/h=0 → 선정 규칙 통과")


# --------------------------------------------------------------------------- #
# V2  BL-0 의 FP=0 은 구조적 항등식                        (문서 §1.3)
# --------------------------------------------------------------------------- #
def V2(n, o):
    say("\n" + "=" * 94)
    say("V2  BL-0 의 FPR 0.0000 은 측정값인가 구조적 항등식인가  (문서 §1.3)")
    say("=" * 94)
    te = n.iloc[VALID_END:]
    for end, lab in ((TRAIN_END, "train[:12000] (프로토콜 §6.1)"),
                     (15000, "train[:15000] (가이드북)")):
        lo, hi = n.iloc[:end][COLS].min(), n.iloc[:end][COLS].max()
        say("\n  %s" % lab)
        for c in COLS:
            inside = (te[c].min() >= lo[c]) and (te[c].max() <= hi[c])
            say("    %-14s 학습[%9.4f,%9.4f] 테스트[%9.4f,%9.4f] %s 여유 %.1f%%/%.1f%%"
                % (c, lo[c], hi[c], te[c].min(), te[c].max(),
                   "포함→FP불가" if inside else "이탈가능",
                   100 * (te[c].min() - lo[c]) / abs(lo[c]),
                   100 * (hi[c] - te[c].max()) / abs(hi[c])))
            record("V2", "%s %s 테스트가 학습범위 내부" % (lab, c), inside)
    say("\n  극값 위치 (전부 15000 이전이면 BL-0 의 FP=0 은 구조적)")
    for c in COLS:
        a, b = int(n[c].values.argmin()), int(n[c].values.argmax())
        say("    %-14s argmin=%5d argmax=%5d" % (c, a, b))
        record("V2", "%s argmin/argmax" % c, "%d/%d" % (a, b))


# --------------------------------------------------------------------------- #
# V3  알고리즘 교차 — 주력 후보가 더 나은 후보에 패배       (문서 §2.1, §2.2)
# --------------------------------------------------------------------------- #
def V3(Fn, Tn, Fa):
    say("\n" + "=" * 94)
    say("V3  알고리즘 × 특징집합 교차 — 동일 분할·동일 임계값(검증 q99.9)  (문서 §2.1, §2.2)")
    say("=" * 94)
    from sklearn.ensemble import IsolationForest
    tr, va, te = split(Fn, Tn)
    i = FEAT.index
    AMP6 = [i("AI0_std"), i("AI0_ptp"), i("AI1_std"), i("AI1_ptp"), i("AI2_std"), i("AI2_ptp")]
    VIB4 = [i("AI0_std"), i("AI0_ptp"), i("AI1_std"), i("AI1_ptp")]
    ALL13 = list(range(13))
    NOACF1 = [k for k in ALL13 if k != i("AI2_acf1")]

    def maha(ix):
        mu, sd = tr[:, ix].mean(0), tr[:, ix].std(0) + 1e-12
        Z = (tr[:, ix] - mu) / sd
        Si = np.linalg.inv(np.cov(Z.T) + 1e-6 * np.eye(len(ix)))
        f = lambda F: np.einsum('ij,jk,ik->i', (F[:, ix] - mu) / sd, Si, (F[:, ix] - mu) / sd)
        return f(va), f(te), f(Fa)

    def pca_spe(ix, nc):
        mu, sd = tr[:, ix].mean(0), tr[:, ix].std(0) + 1e-12
        Z = (tr[:, ix] - mu) / sd
        P = np.linalg.svd(Z, full_matrices=False)[2][:nc].T
        def f(F):
            Y = (F[:, ix] - mu) / sd
            return ((Y - Y @ P @ P.T) ** 2).sum(1)
        return f(va), f(te), f(Fa)

    say("\n  [Mahalanobis]")
    for lab, ix in (("진폭6 (M-1 특징)", AMP6), ("진동진폭4 (AI2 제거)", VIB4), ("전체13", ALL13)):
        v, t, a = maha(ix)
        c = confusion(t, a, np.quantile(v, 0.999))
        line("Mahalanobis " + lab, c)
        record("V3", "Mahalanobis " + lab + " F1", round(c["F1"], 4))

    say("\n  [PCA-SPE]")
    for lab, ix, nc in (("12특징 (AI2_acf1 제외)", NOACF1, 8), ("전체13", ALL13, 8), ("진폭6", AMP6, 4)):
        v, t, a = pca_spe(ix, nc)
        c = confusion(t, a, np.quantile(v, 0.999))
        line("PCA-SPE " + lab, c)
        record("V3", "PCA-SPE " + lab + " F1", round(c["F1"], 4))

    say("\n  [IsolationForest — AI2 의존성, seed 10개]")
    for lab, ix in (("진폭6 (AI2 포함)", AMP6), ("진동진폭4 (AI2 제거)", VIB4)):
        f1 = []
        for s in range(10):
            m = IsolationForest(n_estimators=300, random_state=s).fit(tr[:, ix])
            c = confusion(-m.score_samples(te[:, ix]), -m.score_samples(Fa[:, ix]),
                          np.quantile(-m.score_samples(va[:, ix]), 0.999))
            f1.append(c["F1"])
        below = sum(1 for x in f1 if x < 0.818)
        say("    IF %-24s F1 mean=%.4f sd=%.4f   BL-0(0.818) 미달 seed %d/10"
            % (lab, np.mean(f1), np.std(f1), below))
        record("V3", "IF " + lab + " F1 mean", round(float(np.mean(f1)), 4))
        record("V3", "IF " + lab + " BL-0 미달 seed", "%d/10" % below)


# --------------------------------------------------------------------------- #
# V4  정상 파일 내부 운전변화 — 체제 변화인가 가역 에피소드인가  (문서 §4.1)
# --------------------------------------------------------------------------- #
def V4(Fn, Tn, Fa):
    say("\n" + "=" * 94)
    say("V4  정상 파일 내부 운전변화 — 지속적 체제 변화인가 가역 에피소드인가  (문서 §4.1)")
    say("=" * 94)
    i = FEAT.index
    show = ["AI0_std", "AI1_std", "corr01", "AI0_acf1", "AI1_acf1", "AI2_acf1"]
    ix = [i(s) for s in show]
    say("\n  구간(행)        nwin " + "".join("%10s" % s for s in show))
    for lo, hi in [(0, 15000), (15000, 16000), (16000, 17000), (17000, 18000),
                   (18000, 19000), (19000, 20000)]:
        m = (Tn >= lo) & (Tn < hi)
        say("  %5d-%-6d %5d " % (lo, hi, m.sum())
            + "".join("%10.3f" % v for v in Fn[m][:, ix].mean(0)))
        record("V4", "%d-%d AI0_std" % (lo, hi), round(float(Fn[m][:, i("AI0_std")].mean()), 3))
        record("V4", "%d-%d corr01" % (lo, hi), round(float(Fn[m][:, i("corr01")].mean()), 3))
    say("  이상 전체      %5d " % len(Fa) + "".join("%10.3f" % v for v in Fa[:, ix].mean(0)))
    say("\n  판정: 19,000 이후가 학습구간(0-15,000)으로 복귀하면 체제 변화가 아니라 가역 에피소드")


# --------------------------------------------------------------------------- #
# V5  정상 진동은 백색잡음인가 — 버스트 평균 주기도           (문서 §3.1)
# --------------------------------------------------------------------------- #
def V5(n, o):
    say("\n" + "=" * 94)
    say("V5  정상 진동이 백색잡음인가 — 버스트 평균 주기도 (50샘플, 분해능 0.2 Hz)  (문서 §3.1)")
    say("=" * 94)

    def spec(d, ch):
        X = d[COLS].values
        P = []
        for a, b in runs(d):
            if b - a < 50:
                continue
            x = X[a:a + 50, ch] - X[a:a + 50, ch].mean()
            p = np.abs(np.fft.rfft(x * np.hanning(50))) ** 2
            if p.sum() > 0:
                P.append(p / p.sum())
        return np.mean(P, 0), np.fft.rfftfreq(50, 0.1), len(P)

    f0 = np.fft.rfftfreq(50, 0.1)
    say("  균등기대 파워 = %.1f%% (bin %d개)" % (100 / len(f0), len(f0)))
    for ch, nm in ((0, "AI0"), (1, "AI1"), (2, "AI2")):
        pn_, f, cn = spec(n, ch)
        po, _, co = spec(o, ch)
        k1, k2 = int(pn_.argmax()), int(po.argmax())
        say("  %s  정상: 최대선 %.1f Hz 파워 %4.1f%% (n=%d)   이상: 최대선 %.1f Hz 파워 %4.1f%% (n=%d)"
            % (nm, f[k1], 100 * pn_[k1], cn, f[k2], 100 * po[k2], co))
        record("V5", "%s 정상 최대선(Hz)" % nm, f[k1])
        record("V5", "%s 정상 최대선 파워(%%)" % nm, round(100 * pn_[k1], 1))
    say("\n  판정: 한 bin 파워가 균등기대를 크게 초과하면 백색잡음이 아니라 선 스펙트럼")


# --------------------------------------------------------------------------- #
# V6  corr01 이 gap 을 가로질러 계산됐는가                   (문서 §2.5)
# --------------------------------------------------------------------------- #
def V6(n, o):
    say("\n" + "=" * 94)
    say("V6  corr(AI0,AI1) — 전체 파일 vs run 단위  (문서 §2.5)")
    say("=" * 94)
    for nm, d in (("normal", n), ("outlier", o)):
        X = d[COLS].values
        R = [(a, b) for a, b in runs(d) if b - a >= SEQ]
        whole = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
        rw = float(np.mean([np.corrcoef(X[a:b, 0], X[a:b, 1])[0, 1] for a, b in R]))
        say("  %-8s 전체=%+.4f   run단위평균=%+.4f   차이 %.0f%%"
            % (nm, whole, rw, 100 * (whole - rw) / abs(rw)))
        record("V6", "%s corr01 전체파일" % nm, round(whole, 4))
        record("V6", "%s corr01 run단위" % nm, round(rw, 4))


# --------------------------------------------------------------------------- #
# V7  파일 지문 — 소수 자릿수                                (문서 §4.2)
# --------------------------------------------------------------------------- #
def V7():
    say("\n" + "=" * 94)
    say("V7  파일 지문 — 소수 자릿수 분포  (문서 §4.2)")
    say("=" * 94)
    for nm, fn in (("normal", "press_data_normal.csv"), ("outlier", "press_data_outlier.csv")):
        dec = []
        with open(os.path.join(RAW, fn), encoding="utf-8") as fh:
            next(fh)
            for ln in fh:
                p = ln.rstrip("\n").split(",")
                if len(p) < 5:
                    continue
                for v in p[2:5]:
                    dec.append(len(v.split(".")[1]) if "." in v else 0)
        dec = np.array(dec)
        say("  %-8s 자릿수 중앙값=%d  최대=%d  7자리 초과=%.1f%%"
            % (nm, np.median(dec), dec.max(), 100 * (dec > 7).mean()))
        record("V7", "%s 소수자릿수 중앙값" % nm, int(np.median(dec)))
        record("V7", "%s 7자리초과 비율(%%)" % nm, round(100 * float((dec > 7).mean()), 1))
    say("\n  판정: 두 파일의 직렬화 정밀도가 다르면 날짜 교란이 파일 수준에 존재")


# --------------------------------------------------------------------------- #
# V8  E3 섭동에 대한 불변성                                  (문서 §1.6)
# --------------------------------------------------------------------------- #
def V8(n):
    say("\n" + "=" * 94)
    say("V8  E3 게인·오프셋 섭동 불변성 — 통제실험이 거꾸로 판정하는가  (문서 §1.6)")
    say("=" * 94)
    x = n[COLS].values[:50, 0]
    a1, a2 = acf(x, 1), acf(2.5 * x + 100, 1)
    say("  ACF : 원본=%.6f   (2.5x+100)=%.6f   차이=%.2e  → 아핀불변" % (a1, a2, abs(a1 - a2)))
    say("  std : 원본=%.6f   (2.5x+100)=%.6f               → 게인 비례 (섭동에 취약)"
        % (x.std(), (2.5 * x + 100).std()))
    record("V8", "ACF 아핀불변 차이", "%.2e" % abs(a1 - a2))
    record("V8", "std 게인 비례", "예")
    say("\n  아핀불변 특징 : acf1×3, acf2×3, corr01 = 7개  (AI2_acf1 포함 → E3 무조건 통과)")
    say("  게인 비례 특징: std×3, ptp×3        = 6개  (= M-1 진폭6  → E3 무조건 실패)")


# --------------------------------------------------------------------------- #
VERIFS = {
    "V1": ("디바운스 k=5 FAR/h (§1.1)", lambda ctx: V1(ctx["n"], ctx["o"], ctx["Fn"], ctx["Tn"], ctx["Fa"])),
    "V2": ("BL-0 구조적 FP=0 (§1.3)", lambda ctx: V2(ctx["n"], ctx["o"])),
    "V3": ("알고리즘 교차 (§2.1, §2.2)", lambda ctx: V3(ctx["Fn"], ctx["Tn"], ctx["Fa"])),
    "V4": ("운전변화 가역성 (§4.1)", lambda ctx: V4(ctx["Fn"], ctx["Tn"], ctx["Fa"])),
    "V5": ("선 스펙트럼 (§3.1)", lambda ctx: V5(ctx["n"], ctx["o"])),
    "V6": ("corr01 gap 가로지름 (§2.5)", lambda ctx: V6(ctx["n"], ctx["o"])),
    "V7": ("파일 지문 (§4.2)", lambda ctx: V7()),
    "V8": ("E3 아핀불변 (§1.6)", lambda ctx: V8(ctx["n"])),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="실행할 검증 ID (예: V1 V3)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for k, (d, _) in VERIFS.items():
            print("  %s  %s" % (k, d))
        return

    say("공격4 — Plan A 적대적 검증 재현")
    say("데이터: %s" % RAW)
    say("환경  : python %s / numpy %s / pandas %s"
        % (sys.version.split()[0], np.__version__, pd.__version__))

    n, o = load()
    Fn, Tn = window_features(n)
    Fa, _ = window_features(o)
    tr, va, te = split(Fn, Tn)
    say("window: 정상 전체 %d (train %d / valid %d / test %d) / 이상 %d"
        % (len(Fn), len(tr), len(va), len(te), len(Fa)))
    record("V0", "정상 window 전체", len(Fn), "9978")
    record("V0", "train/valid/test", "%d/%d/%d" % (len(tr), len(va), len(te)))
    record("V0", "이상 window", len(Fa), "276")

    ctx = dict(n=n, o=o, Fn=Fn, Tn=Tn, Fa=Fa)
    keys = a.only if a.only else list(VERIFS)
    for k in keys:
        if k not in VERIFS:
            say("\n[무시] 알 수 없는 검증 ID: %s" % k)
            continue
        VERIFS[k][1](ctx)

    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "verify_output.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_LOG.getvalue())
    with io.open(os.path.join(OUT, "findings.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["verification_id", "item", "value", "doc_value", "match"])
        w.writerows(FINDINGS)
    say("\n저장: results/verify_output.txt , results/findings.csv (%d행)" % len(FINDINGS))


if __name__ == "__main__":
    main()

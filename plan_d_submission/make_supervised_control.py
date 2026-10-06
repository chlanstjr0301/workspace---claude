# -*- coding: utf-8 -*-
"""
M4 지도학습 대조군 + 가짜 라벨 음성대조

왜 필요한가
  본 보고서의 비교(표 2-4)는 정상만으로 학습하는 모델 5종이다. 다른 파이프라인이
  채택한 지도학습(로지스틱 회귀) 계열이 빠져 있어 "왜 지도학습을 쓰지 않았는가"가
  주장으로만 남아 있다. 여기서 두 가지를 측정한다.

  M4  지도 로지스틱 회귀 — 고장 라벨을 써서 학습. 블록 CV 로 평가한다.
      양성 이벤트가 1건이므로 어떤 분할을 해도 같은 이벤트의 조각이 학습과
      평가 양쪽에 들어간다. 높은 점수가 나와도 일반화 근거가 되지 못한다.

  NC  음성대조 — 고장 데이터를 전혀 쓰지 않고, 정상 데이터 안에서
      "앞 블록 vs 뒤 블록"이라는 가짜 라벨을 만들어 같은 분류기를 학습한다.
      이 가짜 과제의 점수가 높다면, 이 특징공간은 임의의 시간 구간을 가르는
      능력이 있다는 뜻이고 고장-정상 F1 의 정보량은 그만큼 줄어든다.

usage: python make_supervised_control.py
출력:  outputs/tables/d7_supervised_control.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import data as DT          # noqa: E402
from src import features as FT      # noqa: E402
from src import windows as WD       # noqa: E402

from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402

OUT = os.path.join(HERE, "outputs", "tables")


def prf(yhat, y):
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def fit_eval(Xtr, ytr, Xte, yte, seed=42):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, random_state=seed).fit(sc.transform(Xtr), ytr)
    yh = m.predict(sc.transform(Xte))
    return prf(yh, yte)


def main():
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
    normal, outlier, _ = DT.load_all(cfg, HERE)
    ch, gap = cfg["data"]["channels"], cfg["windows"]["gap_sec"]
    seq, nb = cfg["windows"]["length"], cfg["cv"]["n_blocks"]

    Xn, mn = WD.make_windows(normal, ch, seq, gap, source="normal")
    Xo, mo = WD.make_windows(outlier, ch, seq, gap, source="outlier")
    mn["block"] = WD.block_split(mn, nb)
    mo["block"] = -1
    groups = tuple(cfg["features"]["groups"])
    Fn, names, _ = FT.build(Xn, ch, mn, groups)
    Fo, _, _ = FT.build(Xo, ch, mo, groups)
    blocks = mn["block"].values
    print("정상 window %d (블록 %d) · 고장 window %d · 특징 %d"
          % (len(Fn), nb, len(Fo), Fn.shape[1]))

    rows = []

    # --- M4 지도 로지스틱 회귀: 블록 CV ---------------------------------- #
    f1s, rcs, pcs = [], [], []
    for b in range(nb):
        tr_n, te_n = blocks != b, blocks == b
        # 고장은 블록이 없으므로 같은 비율로 임의 분할 (어떤 분할이든 같은 이벤트)
        rs = np.random.default_rng(b)
        m_o = rs.random(len(Fo)) < (1.0 / nb)
        Xtr = np.vstack([Fn[tr_n], Fo[~m_o]])
        ytr = np.r_[np.zeros(tr_n.sum()), np.ones((~m_o).sum())]
        Xte = np.vstack([Fn[te_n], Fo[m_o]])
        yte = np.r_[np.zeros(te_n.sum()), np.ones(m_o.sum())]
        p, r, f = fit_eval(Xtr, ytr, Xte, yte, seed=b)
        pcs.append(p); rcs.append(r); f1s.append(f)
    rows.append(dict(id="M4", task="고장 vs 정상 (지도)", uses_fault_label=True,
                     precision=np.mean(pcs), recall=np.mean(rcs),
                     f1_mean=np.mean(f1s), f1_sd=np.std(f1s),
                     note="양성 이벤트 1건 — 학습·평가가 같은 이벤트의 조각"))

    # --- NC 음성대조: 고장 미사용, 정상 내부 가짜 라벨 -------------------- #
    for lo, hi in [(0, nb - 1), (0, 1), (nb - 2, nb - 1)]:
        m = np.isin(blocks, [lo, hi])
        Xf, yf = Fn[m], (blocks[m] == hi).astype(int)
        # 중첩 window 누수를 막기 위해 무작위가 아니라 각 블록 내부를 시간순 절반으로 나눈다.
        bb = blocks[m]
        half = np.zeros(len(Xf), dtype=bool)
        for b in (lo, hi):
            idx = np.where(bb == b)[0]
            half[idx[: len(idx) // 2]] = True      # 앞 절반 학습, 뒤 절반 평가
        p, r, f = fit_eval(Xf[half], yf[half], Xf[~half], yf[~half])
        rows.append(dict(id="NC-%d%d" % (lo, hi),
                         task="정상 블록 %d vs %d (가짜 라벨)" % (lo, hi),
                         uses_fault_label=False,
                         precision=p, recall=r, f1_mean=f, f1_sd=0.0,
                         note="고장 데이터 미사용"))

    t = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(os.path.join(OUT, "d7_supervised_control.csv"),
             index=False, encoding="utf-8-sig")

    print("\n%-8s %-34s %6s %8s %8s %8s" % ("ID", "과제", "고장라벨", "P", "R", "F1"))
    for r_ in rows:
        print("%-8s %-34s %6s %8.3f %8.3f %8.3f"
              % (r_["id"], r_["task"], "사용" if r_["uses_fault_label"] else "미사용",
                 r_["precision"], r_["recall"], r_["f1_mean"]))
    nc = [r_ for r_ in rows if not r_["uses_fault_label"]]
    print("\n음성대조 F1 최대 %.3f — 고장을 전혀 쓰지 않고 정상 안에서 시간 블록만 가른 결과다."
          % max(r_["f1_mean"] for r_ in nc))
    print("저장: outputs/tables/d7_supervised_control.csv")


if __name__ == "__main__":
    main()

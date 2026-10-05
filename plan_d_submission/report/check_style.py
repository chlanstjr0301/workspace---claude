# -*- coding: utf-8 -*-
"""결과보고서 작성 규약 자동 검사 (docs/plan/REPORT_WRITING_CONVENTION.md §2).

    python report/check_style.py           # 실패가 있으면 종료코드 1
    python report/check_style.py --heads   # 헤드 메시지만 순서대로 출력

S1 헤드 메시지 존재 / S2 길이·숫자 / S3 정도 표현·지시어 / S4 내부 작업 용어 /
S5 본문 속 파일명 / S6 CSS 글자 크기 / S7 표 열 수 / S8 그림 색상군 / S9 긴 문장(경고)
"""
import argparse
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "결과보고서.md")
BUILD = os.path.join(HERE, "build_report_pdf.py")

HEAD_MAX = 80
SENT_WARN = 120
MAX_COLS = 7
MAX_HUES = 3

NB = r"(?<![가-힣A-Za-z0-9])"          # 한글 단어 경계 (앞)
DEGREE = ["매우", "크게", "상당히", "상당한", "뚜렷", "심각", "엄청", "탁월", "현저", "극히",
          "훨씬", "아주", "대폭", "막대", "강력", "충분히", "충분한", "다소", "약간", "거의",
          "굉장", "우수"]
DEMONSTRATIVE = ["이것", "그것", "저것", "이는", "이를", "이에", "이로", "이러한", "그러한",
                 "이런", "그런", "해당", "상기", "위의", "아래의"]
RE_DEGREE = re.compile(NB + "(" + "|".join(DEGREE) + ")")
RE_DEMO = re.compile(NB + "(" + "|".join(DEMONSTRATIVE) + ")")
RE_DET = re.compile("(?:(?<=\\s)|^)(이|그|저)\\s+(?=[가-힣])")   # 관형사 '이 ○○', '그 ○○'
RE_INTERNAL = re.compile(NB + "(검토자|감사|클라우드|노트북|사전판정)|Plan [A-H]\\b|DL-\\d+")
RE_FILE = re.compile(r"`[^`]*\.(csv|json|py)`")

# 헤드 메시지 규약을 적용하지 않는 장·절
NO_HEAD = ("표지", "□ 부록", "□ 경진대회 만족도")
# 공식 양식의 고정 문구 (수정 불가)
FORM_FIXED = ("상기 본인(팀)은",)
# 파일명·내부 용어를 허용하는 장 (재현성, 부록)
FILE_OK = ("□ 제6장", "□ 부록")


def parse(md):
    """줄 단위로 (장, 절, 줄, 코드블록 여부)를 붙인다."""
    out, ch, sec, in_code = [], "", "", False
    for ln in md.splitlines():
        if ln.startswith("```"):
            in_code = not in_code
            out.append((ch, sec, ln, True)); continue
        if not in_code and ln.startswith("## "):
            ch, sec = ln[3:].strip(), ""
        elif not in_code and ln.startswith("### "):
            sec = ln[4:].strip()
        out.append((ch, sec, ln, in_code))
    return out


def heads(md):
    """(제목, 헤드 메시지 또는 None) 목록."""
    lines = md.splitlines()
    res, in_code = [], False
    for i, ln in enumerate(lines):
        if ln.startswith("```"):
            in_code = not in_code
        if in_code or not (ln.startswith("## ") or ln.startswith("### ")):
            continue
        title = ln.lstrip("#").strip()
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        nxt = lines[j].strip() if j < len(lines) else ""
        m = re.fullmatch(r"\*\*(.+)\*\*", nxt)
        res.append((ln.startswith("## "), title, m.group(1) if m else None))
    return res


def chapter_of(title, cur):
    return title if title.startswith("□") or title in ("표지",) else cur


def check(md, css_src):
    fails, warns = [], []
    # ---- S1·S2 헤드 메시지 ------------------------------------------------ #
    cur_ch = ""
    for is_ch, title, h in heads(md):
        if is_ch:
            cur_ch = title
        scope = cur_ch if not is_ch else title
        if any(scope.startswith(x) or title.startswith(x) for x in NO_HEAD):
            continue
        if h is None:
            fails.append("S1 헤드 메시지 없음: %s" % title); continue
        if len(h) > HEAD_MAX:
            fails.append("S2 헤드 메시지 %d자 > %d: %s" % (len(h), HEAD_MAX, title))
        if not re.search(r"\d", h):
            fails.append("S2 헤드 메시지에 숫자 없음: %s" % title)
        if h.count("다.") + h.count("다 ") > 1 and h.count(". ") > 0:
            fails.append("S2 헤드 메시지가 한 문장이 아님: %s" % title)

    # ---- S3·S4·S5·S7·S9 본문 ---------------------------------------------- #
    for ch, sec, ln, code in parse(md):
        if code or ln.startswith("#") or ln.startswith("<!--") or ln.startswith("![") \
                or ln.startswith(FORM_FIXED):
            continue
        where = "%s / %s" % (ch[:14], sec[:20])
        text = re.sub(r"`[^`]*`", "", ln)            # 코드·파일 표기는 금지어 검사 제외
        for rx, code_ in ((RE_DEGREE, "S3 정도 표현"), (RE_DEMO, "S3 지시어"), (RE_DET, "S3 지시 관형사")):
            for m in rx.finditer(text):
                fails.append("%s '%s' [%s] %s" % (code_, m.group(0).strip(), where, ln.strip()[:60]))
        if not ch.startswith("□ 부록"):
            for m in RE_INTERNAL.finditer(text):
                fails.append("S4 내부 용어 '%s' [%s]" % (m.group(0), where))
        is_source = ln.lstrip().startswith("*근거") or ln.lstrip().startswith("*출처")
        if not is_source and not ch.startswith(FILE_OK) and RE_FILE.search(ln):
            fails.append("S5 본문 파일명 [%s] %s" % (where, ln.strip()[:60]))
        if ln.startswith("|") and not re.fullmatch(r"\|[\s:|-]+\|", ln.strip()):
            n = ln.strip().strip("|").count("|") + 1
            if n > MAX_COLS:
                fails.append("S7 표 열 %d개 > %d [%s] %s" % (n, MAX_COLS, where, ln.strip()[:50]))
        if not ln.startswith("|") and not is_source:
            for s_ in re.split(r"(?<=다\.)\s+", text.strip()):
                if len(s_) > SENT_WARN:
                    warns.append("S9 긴 문장 %d자 [%s] %s…" % (len(s_), where, s_[:40]))

    # ---- S6 CSS 글자 크기 -------------------------------------------------- #
    small_ok = ("p.source", ".pagefoot")
    for sel, size in re.findall(r"([^{}\n]+)\{[^}]*?font-size:\s*([\d.]+)pt", css_src):
        sel, size = sel.strip(), float(size)
        if sel == "body" and size < 14:
            fails.append("S6 본문 %spt < 14pt" % size)
        elif size < 12 and not any(sel.startswith(x) for x in small_ok):
            fails.append("S6 '%s' %spt < 12pt" % (sel, size))
    for sel in re.findall(r"([^{}\n]+)\{[^}]*?font-size:\s*0\.\d+em", css_src):
        fails.append("S6 '%s' 상대 크기(em<1) 사용 — pt 로 지정" % sel.strip())

    # ---- S8 그림 색상군 ---------------------------------------------------- #
    for img in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md):
        n = hue_groups(os.path.join(HERE, img))
        if n is None:
            fails.append("S8 그림 없음: %s" % img)
        elif n > MAX_HUES:
            fails.append("S8 그림 유채색 %d군 > %d: %s" % (n, MAX_HUES, img))
    return fails, warns


def hue_groups(path):
    """회색을 뺀 유채색 픽셀의 색상(30° 구간) 군집 수."""
    if not os.path.isfile(path):
        return None
    import numpy as np
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3) / 255.0
    mx, mn = im.max(1), im.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0)
    col = im[(sat > 0.25) & (mx > 0.25)]
    if len(col) < 200:
        return 0
    r, g, b = col.T
    mxc, mnc = col.max(1), col.min(1)
    d = np.maximum(mxc - mnc, 1e-9)
    h = np.where(mxc == r, ((g - b) / d) % 6, np.where(mxc == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    hist = np.bincount((h // 30).astype(int) % 12, minlength=12) / len(col)
    on = hist > 0.02
    groups, i = 0, 0
    # 인접 구간은 한 군으로 묶는다 (원형)
    marks = on.astype(int)
    for k in range(12):
        if marks[k] and not marks[(k - 1) % 12]:
            groups += 1
    if marks.all():
        groups = 1
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", action="store_true")
    a = ap.parse_args()
    md = open(REPORT, encoding="utf-8").read()
    if a.heads:
        for is_ch, title, h in heads(md):
            print(("\n■ " if is_ch else "  · ") + title + ("\n      " + h if h else ""))
        return
    css = open(BUILD, encoding="utf-8").read()
    fails, warns = check(md, css)
    for w in warns:
        print("  WARN " + w)
    for f in fails:
        print("  FAIL " + f)
    print("작성 규약 검사: 실패 %d건, 경고 %d건 → %s" % (len(fails), len(warns), "통과" if not fails else "실패"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

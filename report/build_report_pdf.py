"""결과보고서.md -> 결과보고서.pdf (A4, 명조체, 그림 포함).

보고서 서식 전용 스크립트이며 분석 결과에는 관여하지 않는다.
필요 패키지: markdown, playwright (Chromium 브라우저 포함), 나눔명조 글꼴.

    python report/build_report_pdf.py            # PDF 생성
    python report/build_report_pdf.py --html-only # HTML만 생성(검토용)
"""
import argparse
import os
import re

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "결과보고서.md")
OUT_PDF = os.path.join(HERE, "결과보고서.pdf")

CSS = r"""
@page { size: A4; margin: 25mm 20mm 25mm 20mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  /* 공식 양식: 본문 휴먼명조 14pt·줄간격 160%, 주석 10pt. 휴먼명조가 없으면 나눔명조 */
  font-family: "휴먼명조", "HumanMyeongjo", "NanumMyeongjo", "나눔명조", "HCR Batang", "Batang", serif;
  font-size: 14pt; line-height: 1.6; color: #000; background: #fff;
  word-break: keep-all; overflow-wrap: break-word;
}
h1 { font-size: 20pt; text-align: center; margin: 0 0 4mm 0; }
h2 { font-family: "HY헤드라인M", "HYHeadLine M", "NanumSquare", "NanumGothic", sans-serif;
     font-size: 15pt; font-weight: bold; line-height: 1.6; margin: 0 0 3mm 0; page-break-after: avoid; }
h2.chapter { page-break-before: always; }
h3 { font-size: 14pt; margin: 5mm 0 2mm 0; page-break-after: avoid; }
p { margin: 1.5mm 0; text-align: justify; }
blockquote { margin: 3mm 0; padding: 2mm 4mm; border-left: 1.2pt solid #555; background: #f3f3f3; font-size: 12pt; }
body > blockquote:first-of-type { text-align: center; background: none; border: none; font-size: 14pt; }
hr { border: none; border-top: 0.6pt solid #888; margin: 5mm 0; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 1mm 0;
        font-size: 12pt; line-height: 1.45; page-break-inside: auto; }
tr { page-break-inside: avoid; }
thead { display: table-header-group; }
th, td { border: 0.5pt solid #444; padding: 1mm 1.5mm; vertical-align: top; }
th { background: #e8e8e8; font-weight: bold; text-align: center; }
p.caption { font-weight: bold; margin: 4mm 0 0 0; page-break-after: avoid; }
p.source { font-size: 10pt; color: #333; margin: 0.5mm 0 3mm 0; }
p.source::before { content: "* "; font-family: "맑은 고딕", "Malgun Gothic", "NanumGothic", sans-serif; font-size: 12pt; }
code { font-family: "NanumGothicCoding", monospace; font-size: 12pt; }
pre { font-family: "NanumGothicCoding", monospace; font-size: 12pt; line-height: 1.35;
      background: #f4f4f4; border: 0.5pt solid #bbb; padding: 2mm 3mm;
      white-space: pre-wrap; page-break-inside: avoid; }
pre code { font-size: inherit; }
figure { margin: 3mm 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 95mm; }
figcaption { font-size: 12pt; font-weight: bold; margin-top: 1mm; }
ul, ol { margin: 1mm 0 1mm 6mm; padding-left: 4mm; }
/* 공식 양식의 목록 기호: 1단계 ◦, 2단계 -, 주석 * (휴먼명조 10pt) */
ul { list-style-type: "◦  "; }
ul ul { list-style-type: "-  "; }
li::marker { font-size: 15pt; }
/* 표지: 공식 양식(1쪽 테두리 표) */
div.division { text-align: right; font-family: "HY헤드라인M", "NanumSquare", sans-serif; font-size: 12pt; margin: 0 0 2mm 0; }
table.cover { width: 100%; border-collapse: collapse; border: 0.8pt solid #000; page-break-after: always; }
table.cover td, table.cover th { border: 0.6pt solid #000; vertical-align: middle; }
table.cover td.ttl { background: #d9d9d9; text-align: center; height: 13mm;
  font-family: "HY헤드라인M", "NanumSquare", sans-serif; font-size: 20pt; font-weight: bold; line-height: 1.4; }
table.cover th { width: 29mm; background: #d9d9d9; text-align: center;
  font-family: "맑은 고딕", "Malgun Gothic", "NanumGothic", sans-serif; font-size: 11pt; font-weight: bold; }
table.cover td.fld { font-size: 12pt; line-height: 1.4; padding: 2mm 3mm; height: 11mm; }
table.cover td.sum { vertical-align: top; padding: 2mm 3mm; font-size: 12pt; line-height: 1.4; }
table.cover td.sum p.sumhead { font-weight: bold; margin: 0 0 1.5mm 0; text-align: justify; }
table.cover td.sum ul { margin: 0; padding-left: 5mm; }
table.cover td.sum li { margin: 0; text-align: justify; }
table.cover td.decl { border-top: 0.6pt solid #000; padding: 3mm 4mm 3mm 4mm; vertical-align: top; }
table.cover p.decl { font-size: 13pt; line-height: 1.4; text-indent: 4mm; text-align: justify; margin: 0 0 3mm 0; }
table.cover p.date { font-size: 13pt; text-align: right; margin: 0 0 3mm 0; letter-spacing: 0.5pt; }
table.cover p.sign { font-size: 12pt; line-height: 1.4; text-align: right; margin: 0; }
table.cover p.to { font-size: 17pt; font-weight: bold; text-align: center; margin: 3mm 0 0 0; }
li { margin: 0.6mm 0; }
nav.toc { page-break-before: always; }
nav.toc h2 { border-bottom: 1.2pt solid #000; }
nav.toc ul { list-style: none; margin: 0; padding: 0; column-count: 2; column-gap: 8mm; }
nav.toc li.l2 { break-after: avoid; }
nav.toc li { line-height: 1.45; margin: 0; }
nav.toc li.l2 { font-weight: bold; margin-top: 2mm; }
nav.toc li.l3 { margin-left: 7mm; font-size: 12pt; }
p.head { font-weight: bold; font-size: 14pt; margin: 1mm 0 3mm 0; padding: 2mm 3mm;
         border-left: 2.5pt solid #2a78d6; background: #eef4fc; page-break-after: avoid; }
p.formula { text-align: center; margin: 2mm 0; font-size: 12pt; }
/* 만족도 조사 캡쳐: 가로 화면 2장(왼쪽 위·아래) + 휴대폰 화면 1장(오른쪽) */
div.survey { display: flex; gap: 4mm; align-items: flex-start; margin-top: 4mm; page-break-inside: avoid; }
div.survey div.col { flex: 0 0 112mm; }
div.survey div.col img { width: 112mm; border: 0.5pt solid #888; margin-bottom: 4mm; display: block; }
div.survey div.phone img { width: 54mm; border: 0.5pt solid #888; display: block; }
"""

FOOTER = ('<div style="width:100%;font-size:8pt;text-align:center;'
          'font-family:NanumMyeongjo,serif;color:#444;">'
          '- <span class="pageNumber"></span> -</div>')


def _inline(t):
    """표지 칸 안의 굵게 표기만 처리한다."""
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)


def cover_html(md):
    """원고의 표지 부분을 공식 양식(1쪽 테두리 표)으로 조판한다."""
    row = lambda k: (re.search(r"^\| %s \| (.*?) ?\|$" % k, md, re.M) or [None, ""])[1].strip()
    head = re.search(r"### 내용요약\s+\*\*(.+?)\*\*", md, re.S).group(1)
    summ = md.split("### 내용요약", 1)[1].split("상기 본인", 1)[0]
    items = [l[2:] for l in summ.splitlines() if l.startswith("- ")]
    decl = re.search(r"^(상기 본인.*)$", md, re.M).group(1)
    date = re.search(r"^(2026년.*)$", md, re.M).group(1).replace(" ", "&nbsp;")
    signs = re.findall(r"^- (팀[장원] : .*)$", md, re.M)
    to = re.search(r"\*\*(\(사\).*?)\*\*", md).group(1)
    li = "".join("<li>%s</li>" % _inline(x) for x in items)
    sg = "".join('<p class="sign">%s</p>' % x.replace("  ", "&nbsp;&nbsp;") for x in signs)
    return ('<div class="division">일반국민/대학(원)생 부문</div><table class="cover">'
            '<tr><td class="ttl" colspan="2">제6회 K-인공지능 제조데이터 분석 경진대회 보고서</td></tr>'
            '<tr><th>프로젝트명</th><td class="fld">%s</td></tr>'
            '<tr><th>팀명</th><td class="fld">%s</td></tr>'
            '<tr><th>내용요약</th><td class="sum"><p class="sumhead">%s</p><ul>%s</ul></td></tr>'
            '<tr><td class="decl" colspan="2"><p class="decl">%s</p><p class="date">%s</p>%s'
            '<p class="to">%s</p></td></tr></table>'
            % (_inline(row("프로젝트명")), _inline(row("팀명")), _inline(head), li, decl, date, sg, to))


def build_html():
    with open(SRC, encoding="utf-8") as f:
        md_text = f.read()
    cut = md_text.index("## □ 제1장")
    cover = cover_html(md_text[:cut])
    md_text = md_text[cut:]
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"],
                             extension_configs={"toc": {"slugify": _slug}})

    # 그림: <p><img alt=..></p> -> <figure> + 캡션
    body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>',
                  r'<figure><img src="\2" alt="\1"><figcaption>\1</figcaption></figure>', body)
    # 헤드 메시지: 장·절 제목 바로 다음의 굵은 한 문장 (작성 규약 6)
    body = re.sub(r'(</h[23]>)\s*<p><strong>([^<]*)</strong></p>', r'\1<p class="head">\2</p>', body)
    # 표 제목(표 x-y.)과 근거줄(*근거: ...*) 서식
    body = re.sub(r'<p>(표 \d+-\d+[a-z]?\..*?)</p>', r'<p class="caption">\1</p>', body)
    body = re.sub(r'<p><em>(근거:.*?)</em></p>', r'<p class="source">\1</p>', body, flags=re.S)
    # 각 장(□ 제n장, 만족도 조사)은 새 페이지에서 시작
    body = re.sub(r'<h2 id="([^"]*)">□', r'<h2 class="chapter" id="\1">□', body)
    # 부록 B·C 는 앞 부록에 이어 쓴다 (짧은 부록마다 빈 쪽이 생기지 않게)
    body = re.sub(r'<h2 class="chapter" (id="[^"]*">□ 부록 [B-Z])', r'<h2 \1', body)
    # 새 페이지 직전의 구분선은 빈 페이지를 만들 수 있으므로 제거
    body = re.sub(r'<hr />\s*(<h2 class="chapter")', r'\1', body)
    # 수식 줄(  p_normal ...)은 가운데 정렬
    body = re.sub(r'<p>\s*(p_normal\(x\).*?)</p>', r'<p class="formula">\1</p>', body)

    # 목차: 장(h2 □)과 절(h3)만, 표지 뒤에 삽입
    items = []
    for lvl, hid, text in re.findall(r'<h([23])[^>]*id="([^"]*)"[^>]*>(.*?)</h\1>', body):
        text = re.sub(r"<[^>]+>", "", text)
        if lvl == "2" and not text.startswith("□"):
            continue
        if lvl == "3" and not re.match(r"\d+\.\d+", text):
            continue
        items.append(f'<li class="l{lvl}"><a href="#{hid}" style="color:#000;'
                     f'text-decoration:none">{text}</a></li>')
    toc = '<nav class="toc"><h2>목차</h2><ul>' + "".join(items) + "</ul></nav>"
    body = cover + toc + body

    html = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            f'<title>결과보고서</title><style>{CSS}</style></head><body>{body}</body></html>')
    out = os.path.join(HERE, "_build_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


_N = [0]


def _slug(value, sep):
    # 짧은 ASCII 앵커(PDF named destination 길이 제한 회피)
    _N[0] += 1
    return f"h{_N[0]}"


def build_pdf(html_path):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("file://" + html_path)
        page.wait_for_load_state("networkidle")
        page.pdf(path=OUT_PDF, format="A4", print_background=True,
                 display_header_footer=True, header_template="<div></div>",
                 footer_template=FOOTER, prefer_css_page_size=True,
                 margin={"top": "25mm", "bottom": "25mm", "left": "20mm", "right": "20mm"})
        browser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-only", action="store_true")
    a = ap.parse_args()
    html_path = build_html()
    if a.html_only:
        print(html_path)
        return
    build_pdf(html_path)
    os.remove(html_path)
    print(OUT_PDF)


if __name__ == "__main__":
    main()

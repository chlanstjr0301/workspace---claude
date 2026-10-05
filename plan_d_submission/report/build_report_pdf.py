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
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  /* 공식 양식: 본문 휴먼명조 14pt·줄간격 160%, 주석 10pt. 휴먼명조가 없으면 나눔명조 */
  font-family: "휴먼명조", "HumanMyeongjo", "NanumMyeongjo", "나눔명조", "HCR Batang", "Batang", serif;
  font-size: 14pt; line-height: 1.6; color: #000; background: #fff;
  word-break: keep-all; overflow-wrap: break-word;
}
h1 { font-size: 20pt; text-align: center; margin: 0 0 4mm 0; }
h2 { font-size: 16pt; margin: 8mm 0 3mm 0; padding-bottom: 1.5mm;
     border-bottom: 1.2pt solid #000; page-break-after: avoid; }
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
code { font-family: "NanumGothicCoding", monospace; font-size: 12pt; }
pre { font-family: "NanumGothicCoding", monospace; font-size: 12pt; line-height: 1.35;
      background: #f4f4f4; border: 0.5pt solid #bbb; padding: 2mm 3mm;
      white-space: pre-wrap; page-break-inside: avoid; }
pre code { font-size: inherit; }
figure { margin: 3mm 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 95mm; }
figcaption { font-size: 12pt; font-weight: bold; margin-top: 1mm; }
ul, ol { margin: 1mm 0 1mm 6mm; padding-left: 4mm; }
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
"""

FOOTER = ('<div style="width:100%;font-size:8pt;text-align:center;'
          'font-family:NanumMyeongjo,serif;color:#444;">'
          '- <span class="pageNumber"></span> -</div>')


def build_html():
    with open(SRC, encoding="utf-8") as f:
        md_text = f.read()
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
    first = body.find('<h2 class="chapter"')
    body = body[:first] + toc + body[first:]

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
                 margin={"top": "20mm", "bottom": "20mm", "left": "18mm", "right": "18mm"})
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

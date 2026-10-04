from pathlib import Path
import pymupdf,json,re
r=Path(__file__).resolve().parents[1];o={}
for name in ['paper_ko','paper_en_icml']:
 d=pymupdf.open(r/'paper'/f'{name}.pdf');o[name]=[]
 for i,p in enumerate(d):
  t=p.get_text();o[name].append(dict(page=i+1,text=t))
(r/'verification/paper_page_text.json').write_text(json.dumps(o,ensure_ascii=False,indent=2))

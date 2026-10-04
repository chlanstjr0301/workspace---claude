from pathlib import Path
import pymupdf as fitz,json
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];O=R/'verification/pdf_render';O.mkdir(exist_ok=True);records=[]
for p in [R/'paper/paper_ko.pdf',R/'paper/paper_en_icml.pdf',R/'reviews/adversarial_review_ko.pdf']:
 if not p.exists():continue
 doc=fitz.open(p);dest=O/p.stem;dest.mkdir(exist_ok=True);texts=[]
 for i,pg in enumerate(doc):
  pix=pg.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False);pix.save(dest/f'page_{i+1:02}.png');texts.append(f'\n===== PAGE {i+1} =====\n'+pg.get_text())
  words=pg.get_text('words');outside=[w[:5] for w in words if w[0]<0 or w[1]<0 or w[2]>pg.rect.width+1 or w[3]>pg.rect.height+1]
  records.append(dict(file=str(p.relative_to(R)),page=i+1,words=len(words),outside_page=outside,rendered=True))
 (dest/'extracted_text.txt').write_text(''.join(texts))
 for start in range(0,len(doc),2):
  imgs=[Image.open(dest/f'page_{j+1:02}.png').convert('RGB') for j in range(start,min(start+2,len(doc)))];out=Image.new('RGB',(sum(i.width for i in imgs),max(i.height for i in imgs)+35),'#e5e5e5');draw=ImageDraw.Draw(out);x=0
  for j,im in enumerate(imgs):out.paste(im,(x,35));draw.text((x+15,10),f'{p.stem} page {start+j+1}',fill='black');x+=im.width
  out.save(dest/f'sheet_{start//2+1:02}.jpg',quality=90)
(R/'verification/pdf_geometry.json').write_text(json.dumps(records,indent=2));print([(x.name,len(fitz.open(x))) for x in [R/'paper/paper_ko.pdf',R/'paper/paper_en_icml.pdf'] if x.exists()])

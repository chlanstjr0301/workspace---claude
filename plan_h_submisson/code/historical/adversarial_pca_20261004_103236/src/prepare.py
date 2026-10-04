from pathlib import Path
import os,sys,json,hashlib,zipfile,platform,shutil,subprocess
import numpy as np,pandas as pd,sklearn,pymupdf
R=Path(__file__).resolve().parents[1];P=R.parents[1];PK=P/'deliverables/pca_paper_20261003_185434';Z=P/'deliverables/hydraulic_pca_paper_code_data_ko_20261003.zip'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,default=str))
for f in ['logs','manifests','models','predictions','figures','tables','references']: (R/f).mkdir(exist_ok=True)
with zipfile.ZipFile(Z) as z:
 names=z.namelist(); js(R/'manifests/zip_members.json',names);assert z.testzip() is None
 manifest=json.loads((PK/'MANIFEST_SHA256.json').read_text());verified=[]
 for rel,h in manifest.items():
  p=PK/rel;assert sha(p)==h,rel
  zn=[n for n in names if n.endswith('/'+rel) or n==rel];assert len(zn)==1,(rel,zn)
  assert hashlib.sha256(z.read(zn[0])).hexdigest()==h,rel
  verified.append(dict(path=str(p),sha256=h))
js(R/'inventory.json',dict(zip=str(Z),zip_sha256=sha(Z),package=str(PK),verified_payload_files=len(verified),files=verified,python=sys.version,platform=platform.platform(),versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),threads=2))
from importlib.metadata import distributions
(R/'requirements.lock.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()))+'\n')
for f in ['models.py','evaluation.py','infer.py']:shutil.copy2(P/'src'/f,R/'src'/f)
rows=pd.read_csv(P/'configs/first_experiment/frozen_rows.csv',keep_default_na=False);rows.TimeStamp=pd.to_datetime(rows.TimeStamp)
rows.to_csv(R/'manifests/original_rows.csv',index=False)
audit=[];dup=[]
for name,g in rows.groupby('source_file',sort=False):
 p=P/'data'/name;d=pd.read_csv(p);d['source_row']=np.arange(1,len(d)+1);d.TimeStamp=pd.to_datetime(d.TimeStamp);key=['TimeStamp','AI0_Vibration','AI1_Vibration','AI2_Current','Equipment_state'];mask=d.duplicated(key);c=d[~mask];gap=c.TimeStamp.diff().dt.total_seconds()
 assert c.source_row.tolist()==g.source_row.tolist();assert np.array_equal(c[key[:-1]].to_numpy(),g[key[:-1]].to_numpy());assert (gap.dropna()>0).all();assert not d.isna().any().any()
 audit.append(dict(file=name,sha256=sha(p),raw_rows=len(d),clean_rows=len(c),duplicates=int(mask.sum()),gap_gt05=int((gap>.5).sum()),max_gap=float(gap.max()),dates=list(c.TimeStamp.dt.date.astype(str).unique()),negative_current=int((c.AI2_Current<0).sum()),numeric_failures=0,time_failures=0,missing=0))
 for i in d[mask].index:dup.append(dict(file=name,removed_row=int(d.loc[i,'source_row']),kept_row=int(d.loc[(d[key]==d.loc[i,key]).all(axis=1),'source_row'].iloc[0])))
js(R/'manifests/data_audit.json',audit);pd.DataFrame(dup).to_csv(R/'manifests/duplicates.csv',index=False)
assert rows.row_id.is_unique;assert rows.groupby('burst_id').split.nunique().max()==1
rows.groupby(['split','label']).agg(rows=('row_id','size'),bursts=('burst_id','nunique')).to_csv(R/'manifests/original_split_counts.csv')
# Predefined temporal burst halves, nearest row balance, independently within class.
def half(ids):
 g=rows.loc[ids]; sizes=g.groupby('burst_id',sort=False).size();j=int(np.argmin(abs(sizes.cumsum().to_numpy()[:-1]-len(g)/2)))+1;j=len(sizes)//2 if g.label.iloc[0]==1 else j;b=set(sizes.index[:j]);return g.index[g.burst_id.isin(b)].tolist(),g.index[~g.burst_id.isin(b)].tolist()
cn=rows.index[(rows.split=='calibration')&(rows.label==0)].tolist();ca=rows.index[(rows.split=='calibration')&(rows.label==1)].tolist();sn=rows.index[(rows.split=='selection')&(rows.label==0)].tolist();sa=rows.index[(rows.split=='selection')&(rows.label==1)].tolist()
cn1,cn2=half(cn);ca1,ca2=half(ca);sn1,sn2=half(sn);sa1,sa2=half(sa)
blocks=dict(inner_early=dict(cal=cn1,eval=sorted(cn2+ca2)),inner_late=dict(cal=cn,eval=sorted(sn1+sa1)),outer_development=dict(cal=cn,eval=sorted(sn2+sa2)),historical=dict(cal=cn,eval=rows.index[rows.split=='test'].tolist()))
js(R/'manifests/blocks.json',blocks)
counts=[]
for name,b in blocks.items():
 for role,idx in b.items():
  for y,g in rows.loc[idx].groupby('label'): counts.append(dict(block=name,role=role,label=int(y),rows=len(g),bursts=g.burst_id.nunique()))
pd.DataFrame(counts).to_csv(R/'manifests/block_counts.csv',index=False)
# New role boundaries prevent windows borrowing across early/late blocks.
rows['review_partition']=rows.split+':'+rows.train_role
for name,ids in [('cal_early',cn1),('cal_late',cn2),('cal_anom_unused',ca1),('cal_anom_eval',ca2),('selection_inner',sn1+sa1),('selection_outer',sn2+sa2)]:rows.loc[ids,'review_partition']=name
rows.to_csv(R/'manifests/review_rows.csv',index=False)
doc=pymupdf.open(PK/'paper/manuscript_ko.pdf');texts=[]
for i,p in enumerate(doc): texts.append(f'PAGE {i+1}\n'+p.get_text())
(R/'references/paper_extracted.txt').write_text('\n\n'.join(texts));shutil.copy2(PK/'paper/manuscript_ko.tex',R/'references/manuscript_ko.tex')
js(R/'checkpoint.json',dict(stage='inventory_complete',verified_files=len(verified),new_test_use='not yet; original reproduction authorized',zip_preserved=True))
print('Prepared',R,'verified',len(verified));print(pd.DataFrame(counts).to_string(index=False))

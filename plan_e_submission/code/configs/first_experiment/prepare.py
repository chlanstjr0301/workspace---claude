"""Read-only source audit and immutable chronological, whole-burst partitioning."""
import csv, hashlib, json, statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SENSORS = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def nearest_cut(bursts, target):
    counts=[]; n=0
    for b in bursts[:-1]: n+=len(b); counts.append(n)
    return min(range(1,len(bursts)), key=lambda i:(abs(counts[i-1]-target),i))

def prepare():
    out=ROOT/'configs'/'frozen'
    if out.exists(): raise SystemExit('Frozen split already exists; refuse to overwrite.')
    allrows=[]; burstrows=[]; diagnostics=[]; jumps=[]; removed=[]
    for file,y in [('press_data_normal.csv',0),('outlier_data.csv',1)]:
        path=ROOT/'data'/file; raw=path.read_bytes()
        with path.open(encoding='utf-8-sig',newline='') as f: original=list(csv.DictReader(f))
        missing={k:sum(r[k].strip().lower() in {'','nan','na','null','none','n/a'} for r in original) for k in original[0]}
        failures={s:0 for s in SENSORS}; time_fail=0
        for r in original:
            for s in SENSORS:
                try: float(r[s])
                except ValueError: failures[s]+=1
            try: datetime.fromisoformat(r['TimeStamp'])
            except ValueError: time_fail+=1
        assert not any(missing.values()) and not any(failures.values()) and time_fail==0
        seen={}; rows=[]; bursts=[]; previous=None
        for i,r in enumerate(original,1):
            key=tuple(r[k] for k in ['TimeStamp']+SENSORS+['Equipment_state'])
            if key in seen:
                removed.append(dict(file=file,removed_source_row=i,kept_source_row=seen[key],TimeStamp=r['TimeStamp'])); continue
            seen[key]=i; t=datetime.fromisoformat(r['TimeStamp'])
            gap=(t-previous).total_seconds() if previous else None
            assert gap is None or gap>0, 'nonpositive time needs explicit investigation'
            if previous is None or gap>0.5: bursts.append([])
            row=dict(row_id=f'{y}:{i}',source_file=file,source_row=i,csv_line=i+1,source_index=r[''],TimeStamp=r['TimeStamp'],**{s:float(r[s]) for s in SENSORS},label=int(r['Equipment_state']),burst_id=f'{y}:{len(bursts)}',burst_pos=len(bursts[-1])+1,gap_seconds=gap)
            assert row['label']==y
            bursts[-1].append(row); rows.append(row)
            if gap is not None: jumps.append(dict(file=file,previous_row=rows[-2]['source_row'],next_row=i,gap_seconds=gap))
            previous=t
        if y==0:
            c1=nearest_cut(bursts,len(rows)*.6); c2=nearest_cut(bursts,len(rows)*.8)
            parts={'train':bursts[:c1],'dev':bursts[c1:c2],'test':bursts[c2:]}
        else:
            c=nearest_cut(bursts,len(rows)*.4); parts={'dev':bursts[:c],'test':bursts[c:]}
        dev=parts.pop('dev'); c=nearest_cut(dev,sum(map(len,dev))*.5)
        parts['calibration']=dev[:c]; parts['selection']=dev[c:]
        for split,bs in parts.items():
            early_cut=nearest_cut(bs,sum(map(len,bs))*.85) if split=='train' else None
            for bi,b in enumerate(bs):
                for r in b:
                    r['split']=split; r['train_role']=('fit' if bi<early_cut else 'early_stop') if split=='train' else ''
                    r['burst_length']=len(b)
                burstrows.append(dict(file=file,label=y,burst_id=b[0]['burst_id'],split=split,rows=len(b),start=b[0]['TimeStamp'],end=b[-1]['TimeStamp'],duration_seconds=(datetime.fromisoformat(b[-1]['TimeStamp'])-datetime.fromisoformat(b[0]['TimeStamp'])).total_seconds(),source_row_start=b[0]['source_row'],source_row_end=b[-1]['source_row']))
        allrows+=rows
        gaps=[j['gap_seconds'] for j in jumps if j['file']==file]
        diagnostics.append(dict(file=file,sha256=hashlib.sha256(raw).hexdigest(),original_rows=len(original),analysis_rows=len(rows),removed_duplicates=len(original)-len(rows),missing=missing,numeric_failures=failures,timestamp_failures=time_fail,full_row_duplicates=len(original)-len({tuple(r.values()) for r in original}),state_counts=dict(Counter(r['Equipment_state'] for r in original)),bursts=len(bursts),short_bursts={str(w):sum(len(b)<w for b in bursts) for w in [5,10,20]},burst_length_summary=dict(min=min(map(len,bursts)),median=statistics.median(map(len,bursts)),max=max(map(len,bursts))),gap_summary=dict(min=min(gaps),median=statistics.median(gaps),max=max(gaps),nonpositive=sum(g<=0 for g in gaps),outside_tolerance=sum(not .095<=g<=.105 for g in gaps),above_05=sum(g>.5 for g in gaps))))
    allrows.sort(key=lambda r:(r['label'],r['source_row']))
    summary=[]
    for label in [0,1]:
        for split in ['train','calibration','selection','test']:
            rs=[r for r in allrows if r['label']==label and r['split']==split]
            if rs: summary.append(dict(label=label,split=split,rows=len(rs),bursts=len({r['burst_id'] for r in rs}),start=rs[0]['TimeStamp'],end=rs[-1]['TimeStamp']))
    sensitivity=[]
    train=[r for r in allrows if r['split']=='train']
    for threshold in [.2,.5,1.]:
        lengths=[]
        for r in train:
            if not lengths or r['gap_seconds']>threshold: lengths.append(0)
            lengths[-1]+=1
        sensitivity.append(dict(threshold=threshold,bursts=len(lengths),lengths=lengths))
    dump(out/'audit.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files=diagnostics,prior_full_data_summary_seen=True,prior_checks_match=True,burst_sensitivity_normal_train=sensitivity))
    table(out/'rows.csv',allrows); table(out/'bursts.csv',burstrows); table(out/'split_summary.csv',summary); table(out/'time_gaps.csv',jumps); table(out/'removed_duplicates.csv',removed)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Sensitivity:',[(x['threshold'],x['bursts']) for x in sensitivity])
    print('Frozen rows SHA256',hashlib.sha256((out/'rows.csv').read_bytes()).hexdigest())

if __name__=='__main__': prepare()

from pathlib import Path
import json,re
R=Path(__file__).parent
p=Path(json.loads((R/'session_inventory.json').read_text())[0]['path']);out=[];users=[]
def shell_lines(s):
 lines=s.splitlines();clean=[];end=None
 for line in lines:
  if end:
   if line.strip()==end:end=None
   continue
  m=re.search(r"<<-?\s*['\"]?([A-Za-z_][\w]*)['\"]?",line)
  if m:
   end=m[1]
   # Keep invocation that actually executes stdin, but never dump its Python body.
   if re.search(r'(?:python|python3|\.venv/bin/python)\s+-\s*<<',line):clean.append(line)
   continue
  clean.append(line)
 return clean
for n,line in enumerate(p.open(),1):
 x=json.loads(line);time=x.get('timestamp','');y=x.get('payload',{})
 if time>='2026-10-04T13:59:57':continue
 if y.get('type')=='custom_tool_call':
  inp=y.get('input','')
  for m in re.finditer(r'(?:"cmd"|\bcmd)\s*:\s*("(?:[^"\\]|\\.)*")',inp):
   try:cmd=json.loads(m[1])
   except Exception:continue
   for s in shell_lines(cmd):
    if re.search(r'(?:python[^ ]*|\.venv/bin/python|bash|RUN_EXPERIMENT|RUN_REVIEW|RUN_.*sh)',s) and not re.match(r'\s*(?:cat|rg|sed|head|tail|ls|find|cp|chmod|echo|for|if|export|print|text)',s):
     # Safe structural command references only: no raw environment/auth material.
     scripts=re.findall(r'(?:[\w./-]+\.py|[\w./-]+\.sh)',s)
     if scripts or 'python - <<' in s:
      out.append(dict(session_line=n,timestamp=time,call_id=y.get('call_id'),scripts=scripts,mode='inline_python_body_not_exported' if '<<' in s else 'script_invocation',action=re.search(r'\b(select|evaluate|develop|final|prepare)\b',s)[0] if re.search(r'\b(select|evaluate|develop|final|prepare)\b',s) else '',safe_command=re.sub(r'\s*>.*','',s).strip() if scripts and not any(k in s.lower() for k in ['token','auth','secret','password']) else None))
 if y.get('type')=='message' and y.get('role')=='user':
  texts=[z.get('text','') for z in y.get('content',[]) if isinstance(z,dict)];t='\n'.join(texts)
  if any(k in t for k in ['PCA','R5','G1']):
   snippets=[line.strip() for line in t.splitlines() if any(k in line for k in ['FN 0','FN0','S2','FP 6','FP6','R5_A0','G1','미사용','151','선정 G1'])]
   users.append(dict(session_line=n,timestamp=time,relevant_excerpt=snippets[:16]))
(R/'session_commands.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));(R/'conversation_direction_evidence.json').write_text(json.dumps(users,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))

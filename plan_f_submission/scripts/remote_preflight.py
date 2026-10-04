import importlib.util, subprocess, sys, json
from pathlib import Path
if not importlib.util.find_spec('pypdf'):
    subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'pypdf==6.1.1'])
from pypdf import PdfReader
r = PdfReader('/content/guidebook.pdf')
text = '\n'.join(f'\n=== PDF PAGE {i+1} ===\n'+(p.extract_text() or '') for i,p in enumerate(r.pages))
Path('/content/guidebook.txt').write_text(text)
print('PDF pages', len(r.pages))
print(text)
print('PACKAGES')
for name in ['numpy','pandas','sklearn','torch','matplotlib','scipy','joblib']:
    try:
        m=__import__(name); print(name, m.__version__)
    except ImportError: print(name, 'MISSING')

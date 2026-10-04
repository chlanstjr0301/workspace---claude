"""Read-only integrity and structural checks; Python standard library only."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/experiment_20261003_01'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify():
    manifest = json.loads((ROOT / 'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    for relative, digest in manifest.items():
        path = ROOT / relative
        assert path.is_file(), f'Missing file: {relative}'
        assert sha256(path) == digest, f'Changed file: {relative}'
    audit = json.loads((ROOT / 'configs/frozen/audit.json').read_text(encoding='utf-8'))
    for entry in audit['files']:
        assert sha256(ROOT / 'data' / entry['file']) == entry['sha256']
    original = json.loads((RUN / 'artifact_hashes.json').read_text(encoding='utf-8'))
    for relative, digest in original.items():
        assert sha256(RUN / relative) == digest, f'Original result changed: {relative}'
    evidence = json.loads((ROOT / 'paper/evidence_map.json').read_text(encoding='utf-8'))
    for table in evidence:
        assert (ROOT / table['generated_table']).is_file()
        for source in table['source_files']:
            assert (ROOT / source).is_file() or (RUN / source).is_file(), source
    pyfiles = list((ROOT / 'src').glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
    for path in pyfiles:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    with zipfile.ZipFile(ROOT / 'paper/논문_한국어.docx') as docx:
        assert docx.testzip() is None
    assert (ROOT / 'paper/논문_한국어.pdf').read_bytes().startswith(b'%PDF-')
    required = ['agent/AGENT_PROMPT_KO.txt', 'agent/QUICK_START_KO.txt',
                'paper/manuscript.json', 'paper/manuscript.typ',
                'requirements.lock.txt', 'scripts/run_all.py']
    assert all((ROOT / name).is_file() for name in required)
    assert not any(part in {'.aws', '.codex', '.git', '.config', '.venv', '.paper_venv'}
                   for relative in manifest for part in Path(relative).parts)
    result = dict(status='PASS', package_files=len(manifest), original_csv_files=len(audit['files']),
                  original_result_artifacts=len(original), evidence_tables=len(evidence),
                  parsed_python_files=len(pyfiles), writes_performed=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    verify()

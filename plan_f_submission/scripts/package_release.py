"""Build a new ZIP release without changing data or original experiment results."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'PACKAGE_SHA256.json'
ARCHIVE = ROOT.parent / 'hydraulic_paper_code_data_v1.zip'
IGNORED = {'__pycache__', '.git', '.aws', '.codex', '.config', '.venv', '.paper_venv'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    # This is a release operation. A revised release needs new paths/version.
    assert not MANIFEST.exists(), 'Manifest already exists; use a new release folder.'
    assert not ARCHIVE.exists(), 'Archive already exists; do not overwrite a release.'
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and not (set(p.relative_to(ROOT).parts) & IGNORED))
    assert not any(p.is_symlink() for p in files), 'Do not include external symlink targets.'
    manifest = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    with MANIFEST.open('x', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/verify_package.py')],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    checks = json.loads(result.stdout)
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files + [MANIFEST]:
            archive.write(path, arcname=f'{ROOT.name}/{path.relative_to(ROOT).as_posix()}')
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files) + 1
        for relative, expected in manifest.items():
            with archive.open(f'{ROOT.name}/{relative}') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, relative
    checks.update(archive=ARCHIVE.name, archive_bytes=ARCHIVE.stat().st_size,
                  archive_sha256=digest(ARCHIVE), zip_crc_and_entry_hashes='PASS')
    with ARCHIVE.with_suffix('.zip.sha256').open('x', encoding='utf-8') as f:
        f.write(checks['archive_sha256'] + '  ' + ARCHIVE.name + '\n')
    with ARCHIVE.with_suffix('.verification.json').open('x', encoding='utf-8') as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

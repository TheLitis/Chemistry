"""Read-only CASMI environment/data audit; never outputs credential values."""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import urllib.error
import urllib.request

SLUG = 'enveda-CASMI26-molecule-id-mass-spectra'
DATA_SUFFIXES = {'.parquet', '.csv', '.tsv', '.mgf', '.msp', '.jsonl', '.ndjson', '.zip', '.gz', '.7z'}


def version(package: str):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def manifest(root: Path, limit: int = 200) -> dict:
    result = {'root': str(root), 'exists': root.exists(), 'files': []}
    if not root.is_dir():
        return result
    try:
        for folder, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (Path(folder)/d).is_symlink())
            for name in sorted(files):
                file = Path(folder)/name
                if file.suffix.lower() not in DATA_SUFFIXES or file.is_symlink():
                    continue
                result['files'].append({'path': str(file.relative_to(root)).replace('\\', '/'), 'bytes': file.stat().st_size})
                if len(result['files']) >= limit:
                    result['truncated'] = True
                    return result
    except OSError as exc:
        result['error_type'] = type(exc).__name__
    return result


def public_probe(url: str) -> dict:
    result = {'url': url, 'authenticated': False}
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Chemistry-CASMI26-audit/1.0'})
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read(4 * 1024 * 1024)
            result.update(status=response.status, bytes_read=len(body), content_type=response.headers.get('Content-Type'), sha256=hashlib.sha256(body).hexdigest())
            if 'json' in str(result['content_type']):
                payload = json.loads(body)
                result['public_response'] = payload
            else:
                text = body.decode('utf-8', errors='replace')
                start = text.lower().find('<title>')
                end = text.lower().find('</title>', start)
                result['title'] = text[start+7:end][:200] if start >= 0 and end > start else None
    except urllib.error.HTTPError as exc:
        result['status'] = exc.code
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        result['reason_type'] = type(getattr(exc, 'reason', None)).__name__
    return result


def main() -> int:
    state = Path(os.environ.get('CHEMISTRY_STATE_ROOT', r'C:\ProgramData\ChemistryRunner'))
    output = Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    config = {}
    config_file = state/'machine.json'
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding='utf-8-sig'))
    data_root = Path(config.get('dataRoot', str(state/'data')))
    roots = [data_root/'casmi26', Path(r'C:\Users\loval\Chemistry\data'), Path(r'C:\Users\loval\Chemistry\work\casmi26\data')]
    auth_dirs = [state/'kaggle', state/'.kaggle', Path.home()/'.kaggle', Path(r'C:\Users\loval\.kaggle')]
    if os.environ.get('KAGGLE_CONFIG_DIR'):
        auth_dirs.insert(0, Path(os.environ['KAGGLE_CONFIG_DIR']))
    auth_files = []
    for root in auth_dirs:
        for name in ('kaggle.json', 'access_token'):
            file = root/name
            try:
                if file.is_file():
                    with file.open('rb'):
                        pass
                    auth_files.append({'path': str(file), 'readable': True})
            except OSError:
                auth_files.append({'path': str(file), 'readable': False})
    packages = {p: version(p) for p in ('numpy', 'rdkit', 'pyarrow', 'torch', 'kaggle', 'pytest')}
    report = {'request_id': os.environ.get('CHEMISTRY_REQUEST_ID'), 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'machine': platform.node(), 'runner': os.environ.get('RUNNER_NAME'), 'python': platform.python_version(),
              'data': [manifest(p) for p in roots], 'packages': packages,
              'kaggle_auth': {'files': auth_files, 'environment_token': bool(os.environ.get('KAGGLE_API_TOKEN')),
                             'environment_legacy_pair': bool(os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'))},
              'disk_free_gib': round(shutil.disk_usage(state).free/2**30, 2)}
    if packages['torch']:
        import torch
        report['cuda'] = {'available': torch.cuda.is_available(), 'build': torch.version.cuda}
        if torch.cuda.is_available():
            report['cuda'].update(name=torch.cuda.get_device_name(0), vram_bytes=torch.cuda.get_device_properties(0).total_memory)
    downloads = Path(r'C:\Users\loval\Downloads')
    found = []
    try:
        for pattern in ('*CASMI*', '*casmi*', '*enveda*'):
            for f in downloads.glob(pattern):
                if f.is_file() and not f.is_symlink() and f.suffix.lower() in DATA_SUFFIXES:
                    item = {'name': f.name, 'bytes': f.stat().st_size}
                    if item not in found:
                        found.append(item)
    except OSError:
        pass
    report['casmi_downloads'] = found
    report['network'] = [public_probe(url) for url in (
        f'https://www.kaggle.com/competitions/{SLUG}',
        f'https://www.kaggle.com/api/v1/competitions/data/list/{SLUG}',
        'https://www.kaggle.com/api/v1/competitions/list?search=CASMI',
    )]
    output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=True, indent=2)
    (output/'casmi-audit.json').write_text(text+'\n', encoding='utf-8')
    print('CASMI_AUDIT_BEGIN\n'+text+'\nCASMI_AUDIT_END', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

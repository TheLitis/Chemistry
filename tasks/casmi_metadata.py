"""Read public CASMI pages in a fresh anonymous browser; never use user cookies."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('casmi_prepare', root/'tasks/casmi_prepare.py')
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    state = Path(os.environ['CHEMISTRY_STATE_ROOT'])
    out = Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    out.mkdir(parents=True, exist_ok=True)
    python = state/'envs/casmi26/python.exe'
    env = prepare.kaggle_environment(state, os.environ)
    probe = subprocess.run([str(python), '-c', 'from kaggle.cli import main; main()', 'competitions',
                            'files', prepare.SLUG, '--page-size', '200', '-v'], env=env,
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    message = (probe.stdout + probe.stderr).decode('utf-8', errors='replace').lower()
    auth_status = {'exit_code': probe.returncode,
                   'missing_credentials_message': any(x in message for x in ('could not find kaggle.json',
                       'kaggle_api_token', 'authenticate', 'credentials', 'kaggle login')),
                   'http_401': '401' in message, 'http_403': '403' in message}
    print('KAGGLE_ACCESS ' + json.dumps(auth_status, ensure_ascii=True), flush=True)
    edge = next((p for p in (Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'),
                            Path(r'C:\Program Files\Microsoft\Edge\Application\msedge.exe')) if p.exists()), None)
    pages = []
    if edge:
        for name, suffix in (('evaluation', 'overview/evaluation'), ('rules', 'rules'), ('data', 'data')):
            url = f'https://www.kaggle.com/competitions/{prepare.SLUG}/{suffix}'
            item = {'page': name, 'url': url, 'authenticated': False}
            with tempfile.TemporaryDirectory(prefix='casmi-public-', dir=state/'cache',
                                               ignore_cleanup_errors=True) as profile:
                try:
                    result = subprocess.run([str(edge), '--headless=new', '--disable-gpu', '--no-first-run',
                        '--no-default-browser-check', f'--user-data-dir={profile}', '--dump-dom',
                        '--virtual-time-budget=30000', url], stdin=subprocess.DEVNULL,
                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=75)
                    text = prepare.visible_text(result.stdout)
                    item.update(exit_code=result.returncode, text=text[:80000])
                    (out/f'public-{name}.txt').write_text(text, encoding='utf-8')
                except subprocess.TimeoutExpired:
                    item['error'] = 'browser_timeout'
            pages.append(item)
            print('CASMI_PUBLIC_PAGE ' + json.dumps(item, ensure_ascii=True), flush=True)
    report = {'kaggle_access': auth_status, 'pages': pages, 'official_score': None}
    (out/'casmi-metadata.json').write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding='utf-8')


if __name__ == '__main__':
    main()

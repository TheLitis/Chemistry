"""Diagnose Kaggle access-token discovery without printing the token.

Compares the current staged-file path with the documented KAGGLE_API_TOKEN
environment path. No submission or rule-changing action is performed.
"""
from __future__ import annotations
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    state = Path(os.environ['CHEMISTRY_STATE_ROOT'])
    output = Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']); output.mkdir(parents=True, exist_ok=True)
    python = state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve() != python.resolve():
        return subprocess.call([str(python), str(Path(__file__).resolve())], env={**os.environ, 'PYTHONUTF8':'1'})

    prepare = load('casmi_prepare', repo/'tasks/casmi_prepare.py')
    token_path = state/'kaggle'/'access_token'
    token = token_path.read_text(encoding='utf-8-sig').strip()
    if len(token) < 20 or any(c.isspace() for c in token):
        raise RuntimeError('Staged token has an unexpected shape')

    cli = [str(python), '-c', 'from kaggle.cli import main; main()']
    slug = prepare.SLUG
    base = prepare.kaggle_environment(state, dict(os.environ))
    base['PYTHONUTF8']='1'; base['PYTHONIOENCODING']='utf-8'

    direct = dict(base)
    direct['KAGGLE_API_TOKEN'] = token
    # Ensure no legacy partial pair can interfere with the token path.
    direct.pop('KAGGLE_USERNAME', None); direct.pop('KAGGLE_KEY', None)

    result = prepare.run(cli+['competitions','files',slug,'--page-size','3','-v','-q'],
                         env=direct, timeout=90, log=output/'direct-token.log', expose=True)
    report = {
        'request_id': os.environ.get('CHEMISTRY_REQUEST_ID'),
        'kaggle_version': importlib.metadata.version('kaggle'),
        'token_file_present': token_path.is_file(),
        'token_length': len(token),
        'token_value_logged': False,
        'used_documented_env_path': True,
        'competition_files_exit_code': result['exit_code'],
        'competition_files_ok': result['exit_code'] == 0,
        'submission_made': False,
        'official_score': None,
    }
    text = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False)
    (output/'kaggle-diag.json').write_text(text+'\n', encoding='utf-8')
    print('KAGGLE_DIAG_BEGIN\n'+text+'\nKAGGLE_DIAG_END', flush=True)
    token = None
    return 0 if report['competition_files_ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

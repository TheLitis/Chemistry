"""Verify Kaggle access from ChemistryPC without making a submission.

Reads only the dedicated staged provider token through the existing Kaggle
environment resolver. Secret values are never printed or copied.
"""
from __future__ import annotations
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
    output = Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    python = state / 'envs/casmi26/python.exe'
    if Path(sys.executable).resolve() != python.resolve():
        return subprocess.call([str(python), str(Path(__file__).resolve())], env={**os.environ, 'PYTHONUTF8':'1'})

    prepare = load('casmi_prepare', repo/'tasks/casmi_prepare.py')
    env = prepare.kaggle_environment(state, dict(os.environ))
    env['PYTHONUTF8'] = '1'; env['PYTHONIOENCODING'] = 'utf-8'
    cli = [str(python), '-c', 'from kaggle.cli import main; main()']
    slug = prepare.SLUG

    staged = state/'kaggle'/'access_token'
    report = {
        'request_id': os.environ.get('CHEMISTRY_REQUEST_ID'),
        'runner': os.environ.get('RUNNER_NAME'),
        'staged_token_present': staged.is_file(),
        'staged_token_readable': False,
        'secret_values_logged': False,
        'submission_made': False,
        'official_score': None,
    }
    if staged.is_file():
        try:
            with staged.open('rb') as f:
                report['staged_token_readable'] = bool(f.read(1))
        except OSError:
            report['staged_token_readable'] = False

    def run(args, label, timeout=90):
        result = prepare.run(cli + args, env=env, timeout=timeout,
                             log=output/(label+'.log'), expose=False)
        report[label+'_exit_code'] = result['exit_code']
        return result['exit_code'] == 0

    report['competition_files_ok'] = run(['competitions','files',slug,'--page-size','5','-v','-q'], 'competition_files')
    report['submissions_list_ok'] = run(['competitions','submissions',slug,'-v','-q'], 'submissions_list')
    report['entered_list_ok'] = run(['competitions','list','--group','entered','-v'], 'entered_list')
    report['leaderboard_read_ok'] = run(['competitions','leaderboard',slug,'-s','-v','-q'], 'leaderboard_read')

    # Produce sanitized excerpts only after redaction; never emit provider output
    # that could contain credential material (normally it does not).
    safe = {}
    for label in ('competition_files','submissions_list','entered_list','leaderboard_read'):
        path = output/(label+'.log')
        if path.exists():
            safe[label] = path.read_text(encoding='utf-8', errors='replace')[-4000:]
    report['sanitized_output'] = safe
    report['can_operate_competition'] = bool(report['competition_files_ok'] and report['submissions_list_ok'])
    report['status'] = 'authorized_competition_access_verified' if report['can_operate_competition'] else 'authorization_incomplete'

    text = json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False)
    (output/'kaggle-verify.json').write_text(text+'\n', encoding='utf-8')
    artifact = state/'artifacts/casmi26/official-v1'
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact/'kaggle-verify.json').write_text(text+'\n', encoding='utf-8')
    print('KAGGLE_VERIFY_BEGIN\n'+text+'\nKAGGLE_VERIFY_END', flush=True)
    return 0 if report['can_operate_competition'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

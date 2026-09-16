"""Resume the existing sealed R06 tournament; no service, ACL or Kaggle writes."""
from __future__ import annotations
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    state = Path(os.environ['CHEMISTRY_STATE_ROOT'])
    python = state / 'envs/casmi26/python.exe'
    if Path(sys.executable).resolve() != python.resolve():
        return subprocess.call([str(python), str(Path(__file__).resolve())], env={**os.environ,
            'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'OPENBLAS_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '4', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    repo = Path(__file__).resolve().parents[1]
    out = Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    out.mkdir(parents=True, exist_ok=True)
    identity = subprocess.run(['whoami'], capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    report = {'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'identity': identity,
              'runner': os.environ.get('RUNNER_NAME'), 'commit': os.environ.get('GITHUB_SHA'),
              'new_submissions': 0, 'permissions_changed': False, 'stages_requested': ['screen_reuse', 'refine', 'audit']}
    protocol = state / 'artifacts/casmi26/research-r06/protocol.json'
    screen = protocol.with_name('screen-results.json')
    if not protocol.is_file() or not screen.is_file():
        raise RuntimeError('The previously sealed protocol and completed screening are required')
    previous = json.loads(screen.read_text(encoding='utf-8'))
    if previous.get('status') != 'completed' or len(previous.get('advance', [])) != 2:
        raise RuntimeError('Screening has no verified pair of finalists')
    report['screen_finalists'] = previous['advance']
    report['screen_completed'] = len(previous.get('results', {}))
    report['screen_failures'] = previous.get('failures', {})
    (out / 'resume-preflight.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('R06_RESUME_PREFLIGHT\n' + json.dumps(report, indent=2), flush=True)
    execution = subprocess.run([str(python), str(repo / 'tasks/casmi_architecture_research.py'), '--stage', 'all'],
                               stdin=subprocess.DEVNULL, timeout=10800)
    report['research_exit_code'] = execution.returncode
    if execution.returncode == 0:
        final = json.loads(protocol.with_name('report.json').read_text(encoding='utf-8'))
        report['research_status'] = final.get('status')
        report['selected'] = final.get('selection', {}).get('configuration')
        report['independent_audit'] = final.get('audit')
    spec = importlib.util.spec_from_file_location('casmi_prepare', repo / 'tasks/casmi_prepare.py')
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    env = prep.kaggle_environment(state, dict(os.environ))
    report['kaggle_reads'] = {}
    for label, arguments in (
        ('history', ['competitions', 'submissions', prep.SLUG, '--format', 'json']),
        ('limits', ['competitions', 'submission-limits', prep.SLUG, '--json']),
    ):
        result = subprocess.run([str(python), '-c', 'from kaggle.cli import main;main()'] + arguments,
                                env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=90)
        record = {'exit_code': result.returncode}
        if result.returncode == 0:
            try:
                record['data'] = json.loads(result.stdout)
            except ValueError:
                record['error'] = 'Non-JSON response; no score inferred'
        else:
            record['error'] = prep.redact(result.stderr, env)[:1000]
        report['kaggle_reads'][label] = record
    (out / 'resume-result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('R06_RESUME_RESULT_BEGIN\n' + json.dumps(report, indent=2) + '\nR06_RESUME_RESULT_END', flush=True)
    return execution.returncode


if __name__ == '__main__':
    raise SystemExit(main())

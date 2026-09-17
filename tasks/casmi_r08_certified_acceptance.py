"""Complete independent acceptance of preserved R08 results; no model or Kaggle writes."""
from __future__ import annotations
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    state = Path(os.environ['CHEMISTRY_STATE_ROOT'])
    python = state / 'envs/casmi26/python.exe'
    if Path(sys.executable).resolve() != python.resolve():
        return subprocess.call([str(python), str(Path(__file__).resolve())],
            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'})
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / 'tasks'))
    sys.path.insert(0, str(repo / 'work/casmi26'))
    from casmi26.production import sha256, write_json
    import casmi_r08_exact_verify as independent
    import casmi_r08_completion as original
    root = state / 'artifacts/casmi26/research-r08/exact-screen-v1'
    source = root.parent / 'forward-v1'
    out = Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    out.mkdir(parents=True, exist_ok=True)
    verified = independent.verify(root, source)
    write_json(out / 'independent-exact-verification.json', verified)
    write_json(root / 'independent-exact-verification.json', verified)
    gate = original.promotion_gate(json.loads((source / 'report.json').read_text()))
    helper = load('python_paths', repo / 'tasks/r08_python_command.py')
    forward = load('forward_env', repo / 'tasks/casmi_r08_forward.py')
    fiora = source / ('fiora-' + forward.SOURCE)
    paths = [source / 'deps', fiora, repo / 'work/casmi26', repo / 'tasks']
    clean = forward.env_clean()
    clean['FIORA_TEST_MODEL'] = str(fiora / 'fiora/resources/models/fiora_OS_v1.0.0.pt')
    tests = subprocess.run(helper.python_command(python, paths,
        'import pytest,sys;raise SystemExit(pytest.main(sys.argv[1:]))',
        ['-q', str(repo / 'work/casmi26/tests')]), env=clean, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    (out / 'tests.log').write_text(tests.stdout + '\n' + tests.stderr, encoding='utf-8')
    print(tests.stdout, flush=True)
    if tests.returncode:
        raise RuntimeError('Tests failed; no acceptance claim')
    champion = state / 'artifacts/casmi26/final-r07-v1/bundle'
    expected = {'model.npz': '4a05a97b65276df6558e4c156f2129b5463b758f4ea730b0f1f93e99acf055a6',
                'r07-bundle.json': '195e2a73b7d25ce570c178b2f1fb0603d2f8cf47a82e1050b4a8b18ab540baf6'}
    for name, digest in expected.items():
        if sha256(champion / name) != digest:
            raise ValueError('Frozen R07 champion changed: ' + name)
    prep = load('prepare', repo / 'tasks/casmi_prepare.py')
    reader = load('kaggle_reader', repo / 'tasks/casmi_r07_submission.py')
    env = prep.kaggle_environment(state, dict(os.environ))
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    query = subprocess.run([str(python), '-c', reader.API_READ], env=env,
        stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
    snapshot = {'read_only': True, 'exit_code': query.returncode}
    if query.returncode == 0:
        snapshot['data'] = json.loads(query.stdout)
    else:
        snapshot['error'] = prep.redact(query.stderr, env)[:1000]
    write_json(out / 'kaggle-status.json', snapshot)
    for name in ('protocol.json', 'report.json', 'audit-ranks.json', 'certificates.json.gz', 'features.json.gz'):
        shutil.copy2(root / name, out / name)
    result = {'status': 'completed', 'commit': os.environ.get('GITHUB_SHA'),
        'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'verification': {k: v for k, v in verified.items() if k != 'metrics'},
        'tests': tests.stdout.strip(), 'original_promotion_gate': gate,
        'promotion_gate_relaxed': False, 'r07_champion_unchanged': True,
        'kaggle': snapshot, 'new_training': False, 'new_simulations': 0,
        'new_submissions': 0, 'new_uploads': 0,
        'scope': 'Numerical fixed-score certificate and stored experiment verification, not unseen-molecule accuracy'}
    write_json(out / 'acceptance.json', result)
    write_json(root / 'acceptance.json', result)
    subprocess.run(['git', '-c', 'safe.directory=' + str(repo), '-C', str(repo), 'archive',
        '--format=zip', '--output=' + str(out / 'source.zip'), 'HEAD'], check=True, timeout=60)
    print('R08_CERTIFIED_ACCEPTANCE_BEGIN\n' + json.dumps(result, indent=2) + '\nR08_CERTIFIED_ACCEPTANCE_END', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

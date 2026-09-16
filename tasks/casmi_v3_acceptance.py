"""Verify the unified V3 entry point on the full visible data; no Kaggle writes."""
from __future__ import annotations
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def verify_equivalent_prediction(expected: dict, actual: dict) -> bool:
    for key in ('submission_sha256', 'model_sha256', 'train_sha256', 'test_sha256',
                'prediction_count', 'test_spectra', 'model_format'):
        if key not in expected or key not in actual or expected[key] != actual[key]:
            raise ValueError('Unified entry point changed ' + key)
    return True


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    state = Path(os.environ['CHEMISTRY_STATE_ROOT'])
    py = state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve() != py.resolve():
        return subprocess.call([str(py), str(Path(__file__).resolve())], env={**os.environ,
            'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4'})
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo/'work/casmi26'))
    from casmi26.pipeline_v3 import verify_bundle
    from casmi26.notebook_v3 import build_notebook
    from casmi26.production import sha256, write_json
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    root = state/'artifacts/casmi26/highres-v3'
    out = Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']); out.mkdir(parents=True, exist_ok=True)
    release = root/'release-verified'; release.mkdir(parents=True, exist_ok=True)
    bundle = root/'bundle'
    started = time.monotonic()
    checks = subprocess.run([str(py), '-m', 'pytest', '-q', str(repo/'work/casmi26/tests')],
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
    (out/'tests.log').write_text(checks.stdout+'\n'+checks.stderr, encoding='utf-8')
    print(checks.stdout, flush=True)
    if checks.returncode: raise RuntimeError('Tests failed; acceptance stopped before inference')
    manifest = verify_bundle(bundle)
    old_model = state/'artifacts/casmi26/official-v1/model.npz'
    if sha256(old_model) != '4d79a6d4f24060cd0f7c09bebefbced2c9e1672cb1cdf5f7831cd5a19c03b55d':
        raise RuntimeError('Best officially evaluated model changed')
    staged = load('staged', repo/'tasks/casmi_staged.py')
    data = staged.find_dataset(state/'data/external')
    output = release/'submission.csv'
    command = [str(py), str(repo/'predict.py'), '--test', str(data/'test.parquet'),
        '--train', str(data/'train.parquet'), '--bundle', str(bundle), '--output', str(output)]
    if (data/'sample_submission.csv').is_file():
        command += ['--sample-submission', str(data/'sample_submission.csv')]
    print('V3_ROOT_ENTRYPOINT_START', flush=True)
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=1800)
    (out/'inference.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
    if result.returncode: raise RuntimeError('Root inference failed; inspect inference.log')
    current = json.loads(output.with_suffix('.csv.report.json').read_text())
    previous = json.loads((root/'notebook-output/submission.csv.report.json').read_text())
    verify_equivalent_prediction(previous, current)
    verify = load('v3verify', repo/'tasks/casmi_v3_verify.py')
    integrity = verify.verify_prediction(output, output.with_suffix('.csv.report.json'))
    notebook = build_notebook(release/'casmi26-v3.ipynb')
    archive = release/'casmi-v3-release.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for file in sorted(bundle.rglob('*')):
            if file.is_file(): z.write(file, 'bundle/'+file.relative_to(bundle).as_posix())
        z.write(repo/'predict.py', 'predict.py')
        for file in sorted((repo/'work/casmi26/casmi26').glob('*.py')):
            z.write(file, file.relative_to(repo).as_posix())
        z.write(notebook, notebook.name)
        z.write(repo/'work/casmi26/requirements.txt', 'requirements.txt')
    prep = load('prepare', repo/'tasks/casmi_prepare.py')
    env = prep.kaggle_environment(state, dict(os.environ))
    reads = {}
    for name, args in (
        ('history', ['competitions','submissions',prep.SLUG,'--format','json']),
        ('limits', ['competitions','submission-limits',prep.SLUG,'--json']),
    ):
        r = subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args, env=env,
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
        reads[name] = {'exit_code': r.returncode}
        if r.returncode == 0: reads[name]['data'] = json.loads(r.stdout)
        else: reads[name]['error'] = prep.redact(r.stderr, env)[:1000]
    diagnostic = root/'spectrum-budget/report.json'
    if diagnostic.is_file(): shutil.copy2(diagnostic, out/'spectrum-budget.json')
    report = {'status':'completed','commit':os.environ.get('GITHUB_SHA'),
        'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'tests':{'exit_code':checks.returncode,'output':checks.stdout.strip()},
        'actual_entrypoint_command':command,'output':integrity,'unchanged_predictions':True,
        'model_sha256':manifest['files']['model.npz'],'original_champion_unchanged':True,
        'input_features':manifest['feature_dim'],'head_sizes':manifest['head_sizes'],
        'release_directory':str(release),'archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha256(archive)},
        'notebook_sha256':sha256(notebook),'notebook_rebuilt_not_rerun_this_pass':True,
        'kaggle_reads':reads,'new_submissions':0,'new_uploads':0,'new_training':False,
        'official_score_for_v3':None,'candidate_not_champion':True,'seconds':time.monotonic()-started}
    write_json(out/'acceptance.json',report);write_json(release/'acceptance.json',report)
    shutil.copy2(notebook,out/notebook.name)
    subprocess.run(['git','-C',str(repo),'archive','--format=zip','--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
    print('V3_ACCEPTANCE_BEGIN\n'+json.dumps(report,indent=2)+'\nV3_ACCEPTANCE_END',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

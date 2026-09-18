"""Full visible-data parity of the selected-feature backend. No Kaggle writes.

The submitted R08B package, notebooks, weights and prediction cache stay intact.
A read-only SQLite backup supplies identical simulations for the new execution.
"""
from __future__ import annotations
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time


def verify_equivalence(expected, actual):
    fields=('format','submission_sha256','train_sha256','test_sha256','model_sha256',
            'bundle_manifest_sha256','prediction_count','test_spectra','empty_candidate_rows','test_labels_used')
    for key in fields:
        if key not in expected or key not in actual or expected[key]!=actual[key]:
            raise ValueError('Changed output contract: '+key)
    forward=actual.get('forward',{})
    if (forward.get('weight')!=.25 or forward.get('feature')!='cosine_nearest' or
        forward.get('all_top25_numerically_certified') is not True or
        forward.get('scoring_backend')!='nearest-only'):
        raise ValueError('Changed forward scoring contract')
    return True


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,
            'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
    repo=Path(__file__).resolve().parents[1];sys.path[:0]=[str(repo/'tasks'),str(repo/'work/casmi26')]
    from casmi26.production import sha256,write_json
    import casmi_r08_forward as runtime
    import r08_python_command as imports
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    art=state/'artifacts/casmi26';original=art/'candidate-r08b-20260918'
    root=art/'r08b-fast-parity-20260918';root.mkdir(parents=True,exist_ok=True)
    built=json.loads((original/'build-status.json').read_text(encoding='utf-8-sig'))
    if built['status']!='candidate_built_and_locally_executed':raise RuntimeError('Missing completed R08B')
    expected=built['prediction'];bundle=original/'package/bundle'
    frozen_files=[original/'package/r08b-package.json',original/'submission.csv',bundle/'model.npz',bundle/'r07-bundle.json']
    before={str(p):sha256(p) for p in frozen_files}
    fiora_root=art/'research-r08/forward-v1'
    public=fiora_root/'fiora-e19ef82c9a6cb9dbac92bce23e914008f1aeb44e';deps=fiora_root/'deps'
    paths=[deps,public,repo/'work/casmi26',repo/'tasks'];env=runtime.env_clean()
    env['FIORA_TEST_MODEL']=str(public/'fiora/resources/models/fiora_OS_v1.0.0.pt')
    tests=subprocess.run(imports.python_command(py,paths,'import pytest,sys;raise SystemExit(pytest.main(sys.argv[1:]))',
        ['-q',repo/'work/casmi26/tests']),env=env,stdin=subprocess.DEVNULL,capture_output=True,
        text=True,encoding='utf-8',errors='replace',timeout=600)
    (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8');print(tests.stdout,flush=True)
    if tests.returncode:raise RuntimeError('Tests failed before full parity execution')
    reader=load('status_reader',repo/'tasks/casmi_r08b_submission.py')
    prep=load('prepare',repo/'tasks/casmi_prepare.py');shared=load('shared',repo/'tasks/casmi_r07_submission.py')
    kenv=prep.kaggle_environment(state,dict(os.environ));kenv.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    def status(label):
        proc=subprocess.run([str(py),'-c',shared.API_READ],env=kenv,stdin=subprocess.DEVNULL,capture_output=True,
            text=True,encoding='utf-8',errors='replace',timeout=90)
        if proc.returncode:
            write_json(out/(label+'.json'),{'read_failed':True,'exit_code':proc.returncode});return None
        value=json.loads(proc.stdout);reader.validate_history(value)
        write_json(out/(label+'.json'),value)
        return value
    initial_status=status('kaggle-before')
    source_cache=original/'forward-cache.sqlite';cache=root/'forward-cache.sqlite'
    if not cache.exists():
        if not source_cache.is_file():raise RuntimeError('Original simulations are missing')
        with sqlite3.connect(source_cache.resolve().as_uri()+'?mode=ro',uri=True) as source:
            with sqlite3.connect(cache) as destination:source.backup(destination)
    inventory=json.loads((art/'research-r07/inventory.json').read_text());train=Path(inventory['train_path'])
    test=train.parent/'test.parquet';output=root/'submission.csv'
    args=['--test',test,'--train',train,'--bundle',bundle,'--output',output,
        '--model-path',public/'fiora/resources/models/fiora_OS_v1.0.0.pt',
        '--workers','8','--cache',cache,'--device','cpu','--nearest-only']
    template=train.parent/'sample_submission.csv'
    if template.exists():args+=['--sample-submission',template]
    print('R08B_FAST_FULL_PARITY_START',flush=True);started=time.monotonic()
    result=subprocess.run(imports.python_command(py,paths,
        'import sys;from casmi26.forward_release import main;raise SystemExit(main(sys.argv[1:]))',args),
        env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=7200)
    (out/'inference.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Fast full-data execution failed; inspect inference.log')
    actual=json.loads(output.with_suffix('.csv.report.json').read_text());verify_equivalence(expected,actual)
    if sha256(output)!=expected['submission_sha256']:raise RuntimeError('CSV bytes changed')
    if before!={str(p):sha256(p) for p in frozen_files}:raise RuntimeError('Frozen R08B changed')
    final_status=status('kaggle-after')
    report={'status':'full_visible_output_identical','commit':os.environ.get('GITHUB_SHA'),
        'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,
        'tests':tests.stdout.strip(),'prediction':{k:v for k,v in actual.items() if k!='details'},
        'submitted_package_unchanged':True,'original_cache_access':'read-only SQLite backup',
        'new_training':False,'new_submissions':0,'new_kaggle_uploads':0,
        'kaggle':final_status or initial_status,'output_directory':str(root),
        'timing_note':'Warm simulation cache; do not compare as end-to-end speedup against the cold R08B run.',
        'quality_note':'Exact output parity, not a new accuracy improvement or official score.'}
    write_json(root/'acceptance.json',report);write_json(out/'acceptance.json',report)
    for p in [output,output.with_suffix('.csv.report.json')]:shutil.copy2(p,out/p.name)
    subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip',
                    '--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
    print('R08B_FAST_ACCEPTANCE_BEGIN\n'+json.dumps(report,indent=2)+'\nR08B_FAST_ACCEPTANCE_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

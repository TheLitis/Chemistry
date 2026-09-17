"""Run the packaged R07 candidate independently and check live Kaggle status.

Only Kaggle reads are performed. No upload, submission, or rule acceptance.
"""
from __future__ import annotations
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

CONTRACT=('format','prediction_count','test_spectra','submission_sha256','bundle_manifest_sha256',
          'model_sha256','test_sha256','train_sha256','selection','test_labels_used','empty_candidate_rows')


def same_output(expected,actual):
    for key in CONTRACT:
        if key not in expected or key not in actual or expected[key]!=actual[key]:
            raise ValueError('Standalone output differs: '+key)
    return True


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,
            'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    from casmi26.r07_release import verify_bundle
    from casmi26.notebook_r07 import build_notebook
    from casmi26.metric import structure_key
    root=state/'artifacts/casmi26/final-r07-v1'
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    release=json.loads((root/'release.json').read_text())
    archive=Path(release['archive']['path'])
    if sha256(archive)!=release['archive']['sha256']:raise RuntimeError('Release archive changed')
    checks=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=360)
    (out/'tests.log').write_text(checks.stdout+'\n'+checks.stderr,encoding='utf-8')
    print(checks.stdout,flush=True)
    if checks.returncode:raise RuntimeError('Acceptance tests failed')
    started=time.monotonic()
    folder=Path(tempfile.mkdtemp(prefix='archive-check-',dir=root))
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            name=Path(item.filename)
            if name.is_absolute() or '..' in name.parts or ':' in item.filename:
                raise RuntimeError('Unsafe archive path')
        z.extractall(folder)
    manifest=verify_bundle(folder/'bundle')
    data=load('staged',repo/'tasks/casmi_staged.py').find_dataset(state/'data/external')
    env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    command=[str(py),str(folder/'predict_r07.py'),'--test',str(data/'test.parquet'),'--train',str(data/'train.parquet'),
        '--bundle',str(folder/'bundle'),'--output',str(folder/'submission.csv'),'--workers','4']
    if (data/'sample_submission.csv').is_file():command+=['--sample-submission',str(data/'sample_submission.csv')]
    print('R07_ARCHIVE_INFERENCE_START',flush=True)
    result=subprocess.run(command,cwd=folder,env=env,stdin=subprocess.DEVNULL,
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=2400)
    (out/'standalone.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Standalone archive failed; inspect standalone.log')
    actual=json.loads((folder/'submission.csv.report.json').read_text())
    expected=json.loads((root/'submission.csv.report.json').read_text())
    same_output(expected,actual)
    if sha256(folder/'submission.csv')!=expected['submission_sha256']:raise RuntimeError('Standalone CSV hash differs')
    import csv
    with (folder/'submission.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        keys=[structure_key(g) for g in row['smiles'].split(';')]
        if not 1<=len(keys)<=25 or None in keys or len(set(keys))!=len(keys):raise RuntimeError('Invalid structural list')
    book=build_notebook(root/'casmi26-r07.ipynb');shutil.copy2(book,out/book.name)
    notebook=json.loads(book.read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<R07-notebook>','exec')
    prep=load('prepare',repo/'tasks/casmi_prepare.py');kenv=prep.kaggle_environment(state,dict(os.environ))
    access={'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()','competitions','submissions',prep.SLUG,'--format','json'],
        env=kenv,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
    access['history_exit_code']=r.returncode
    if r.returncode==0:
        history=json.loads(r.stdout)
        access['history']=history
        budget=load('publishing',repo/'tasks/casmi_kaggle_publish.py').budget_status(history,dt.datetime.now(dt.timezone.utc),5)
        access['internal_budget']=budget
    else:access['history_error']=prep.redact(r.stderr,kenv)[:600]
    r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()','competitions','submission-limits',prep.SLUG,'--json'],
        env=kenv,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
    access['limits_exit_code']=r.returncode
    if r.returncode==0:access['limits']=json.loads(r.stdout)
    else:access['limits_error']=prep.redact(r.stderr,kenv)[:600]
    # The legacy CLI summary omits scoring errors. A separate API read retains them.
    probe="""import json
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
rows=api.competition_submissions('enveda-CASMI26-molecule-id-mass-spectra') or []
print(json.dumps([{'ref':r.ref,'status':str(r.status),'public_score':r.public_score,'error_description':getattr(r,'error_description',None)} for r in rows]))
"""
    r=subprocess.run([str(py),'-c',probe],env=kenv,stdin=subprocess.DEVNULL,capture_output=True,text=True,
                      encoding='utf-8',errors='replace',timeout=90)
    if r.returncode==0:access['unprojected_history']=json.loads(r.stdout)
    else:access['unprojected_history_error']=prep.redact(r.stderr,kenv)[:600]
    report={'status':'standalone_archive_verified','commit':os.environ.get('GITHUB_SHA'),
        'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'tests':checks.stdout.strip(),
        'archive_sha256':sha256(archive),'extracted_directory':str(folder),'same_output_contract':True,
        'submission_sha256':actual['submission_sha256'],'model_sha256':actual['model_sha256'],
        'bundle_manifest_sha256':actual['bundle_manifest_sha256'],'prediction_count':actual['prediction_count'],
        'test_spectra':actual['test_spectra'],'total_guesses':sum(len(r['smiles'].split(';')) for r in rows),
        'all_structures_valid_and_distinct':True,'mass_incompatible_fallbacks':actual['mass_incompatible_fallbacks'],
        'actual_inference_seconds':actual['seconds'],'notebook_sha256':sha256(book),
        'notebook_built_and_compiled':True,'notebook_executed_on_full_data_here':False,
        'new_training':False,'new_submissions':0,'new_kaggle_uploads':0,'official_score_for_r07':None,
        'kaggle_reads':access,'seconds':time.monotonic()-started,'champion_changed':False}
    write_json(root/'archive-acceptance.json',report);write_json(out/'acceptance.json',report)
    shutil.copy2(folder/'submission.csv',out/'submission.csv')
    shutil.copy2(folder/'submission.csv.report.json',out/'submission.csv.report.json')
    subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip','--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=90)
    print('R07_ACCEPTANCE_BEGIN\n'+json.dumps(report,indent=2)+'\nR07_ACCEPTANCE_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

"""Verify completed v3 artifacts and current score/quota; no model or Kaggle writes."""
from __future__ import annotations
import csv
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def verify_prediction(path, report_path):
    from casmi26.production import sha256
    from casmi26.metric import distinct_guesses
    path=Path(path);report=json.loads(Path(report_path).read_text())
    digest=sha256(path)
    if digest!=report['submission_sha256']:raise ValueError('Submission hash mismatch')
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['molecule_id','smiles']:raise ValueError('Unexpected submission columns')
        rows=list(reader)
    ids=[r['molecule_id'] for r in rows]
    if not rows or len(rows)!=report['prediction_count'] or len(set(ids))!=len(ids) or any(not k or not k.strip() for k in ids):
        raise ValueError('Submission ID/row count mismatch')
    guesses=0
    for row in rows:
        values=(row['smiles'] or '').split(';')
        if not 1<=len(values)<=25 or len(distinct_guesses(values,25))!=len(values):
            raise ValueError('Invalid or duplicate molecular guesses')
        guesses+=len(values)
    return {'rows':len(rows),'guesses':guesses,'sha256':digest,'all_guesses_valid_and_distinct':True}


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.pipeline_v3 import verify_bundle
    from casmi26.production import sha256,write_json
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    root=state/'artifacts/casmi26/highres-v3';experiment=state/'artifacts/casmi26/research-v3'
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    run=json.loads((root/'run-report.json').read_text())
    if run['status']!='completed':raise RuntimeError('The full training/delivery task did not complete')
    validation=json.loads((experiment/'validation-v3.json').read_text())
    plan=json.loads((experiment/'protocol.json').read_text())
    old=state/'artifacts/casmi26/official-v1'
    if sha256(old/'model.npz')!='4d79a6d4f24060cd0f7c09bebefbced2c9e1672cb1cdf5f7831cd5a19c03b55d':
        raise RuntimeError('Original best-evaluated model changed unexpectedly')
    if sha256(old/'holdout-model.npz')!=plan['warmstart_sha256']:raise RuntimeError('Holdout source model changed')
    for name,digest in validation['model_hashes'].items():
        if sha256(experiment/name)!=digest:raise RuntimeError('Evaluated model changed: '+name)
    results={'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'verification_commit':os.environ.get('GITHUB_SHA'),
       'training_commit':run['commit'],'original_model_unchanged':True,'validation_models_unchanged':True,
       'new_submissions':0,'new_uploads':0,'validation':validation,'training_run':{k:v for k,v in run.items() if k not in ('prepare','validation','refit')}}
    if validation['eligible_for_candidate_bundle']:
        manifest=verify_bundle(root/'bundle')
        results['bundle']={k:v for k,v in manifest.items() if k not in ('files','training_identity')}
        results['output']=verify_prediction(root/'notebook-output/submission.csv',root/'notebook-output/submission.csv.report.json')
        archive=root/'casmi-v3-delivery.zip'
        if sha256(archive)!=run['delivery']['sha256']:raise RuntimeError('Delivery archive hash mismatch')
        results['archive']={'path':str(archive),'bytes':archive.stat().st_size,'sha256':run['delivery']['sha256']}
        staging=root/'kaggle-staging';staging.mkdir(exist_ok=True)
        shutil.copy2(root/'casmi26-v3.ipynb',staging/'casmi26-v3.ipynb')
        write_json(staging/'kernel-metadata.json',{'id':'thelindortis/casmi26-highres-v3','title':'CASMI26 Highres V3',
            'code_file':'casmi26-v3.ipynb','language':'python','kernel_type':'notebook','is_private':'true',
            'enable_gpu':'false','enable_internet':'false','dataset_sources':['thelindortis/casmi26-highres-v3-assets'],
            'competition_sources':['enveda-CASMI26-molecule-id-mass-spectra'],'kernel_sources':[],'model_sources':[]})
        results['staged_not_uploaded']=str(staging)
    spec=importlib.util.spec_from_file_location('prepare',repo/'tasks/casmi_prepare.py')
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep);env=prep.kaggle_environment(state,dict(os.environ))
    results['kaggle_reads']={}
    for name,args in (('history',['competitions','submissions','enveda-CASMI26-molecule-id-mass-spectra','--format','json']),
                       ('limits',['competitions','submission-limits','enveda-CASMI26-molecule-id-mass-spectra','--json'])):
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args,env=env,stdin=subprocess.DEVNULL,
               capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        value={'exit_code':r.returncode}
        if r.returncode==0:value['data']=json.loads(r.stdout)
        else:value['error']=prep.redact(r.stderr,env)[:1000]
        results['kaggle_reads'][name]=value
    tests=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,
                        text=True,encoding='utf-8',errors='replace',timeout=300)
    results['tests']={'exit_code':tests.returncode,'output':tests.stdout.strip()}
    (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8')
    for name in ('validation-v3.json','selection-before-audit.json','protocol.json','audit-ranks.json'):
        shutil.copy2(experiment/name,out/name)
    subprocess.run(['git','-C',str(repo),'archive','--format=zip','--output='+str(out/'v3-source.zip'),'HEAD'],check=True,timeout=60)
    write_json(out/'verified-v3.json',results);write_json(root/'verified-v3.json',results)
    compact={k:v for k,v in results.items() if k!='validation'}
    compact['audit']=validation['audit'];compact['selected']=validation['selection']['variant']
    print('VERIFIED_V3_BEGIN\n'+json.dumps(compact,indent=2)+'\nVERIFIED_V3_END',flush=True)
    return 0 if tests.returncode==0 else 1


if __name__=='__main__':raise SystemExit(main())

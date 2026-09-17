"""Read-only R08 inventory. No model training, uploads, submissions or secret output."""
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
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    prep=load('prepare_r08',repo/'tasks/casmi_prepare.py')
    submission=load('submission_r08',repo/'tasks/casmi_r07_submission.py')
    env=prep.kaggle_environment(state,dict(os.environ))
    r=subprocess.run([str(python),'-c',submission.API_READ],env=env,stdin=subprocess.DEVNULL,
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
    report={'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'commit':os.environ.get('GITHUB_SHA'),
        'new_submissions':0,'new_uploads':0,'test_data_read':False,'models_changed':False,
        'kaggle_read_exit_code':r.returncode}
    if r.returncode==0:report['kaggle']=json.loads(r.stdout)
    else:report['kaggle_error']=prep.redact(r.stderr,env)[:1000]
    study=state/'artifacts/casmi26/research-r07/full-system-v1'
    report['study_files']=[{'name':p.name,'bytes':p.stat().st_size} for p in sorted(study.iterdir()) if p.is_file()]
    report['environments']=[str(p) for p in (state/'envs').glob('*/python.exe')]
    for name in ('protocol.json','report.json','selection-before-audit.json','v1-independent.npz','external-manifest.json'):
        p=study/name
        if p.exists():shutil.copy2(p,out/name)
    target=state/'artifacts/casmi26/research-r07/np-examples.parquet'
    if target.exists():shutil.copy2(target,out/target.name)
    report['public_material']=[{'name':p.name,'bytes':p.stat().st_size} for p in (state/'artifacts/casmi26/public-model-probe-20260917').glob('*') if p.is_file()]
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],
            capture_output=True,text=True,timeout=15)
        report['gpu']=r.stdout.strip();report['gpu_query_exit_code']=r.returncode
    except (OSError,subprocess.TimeoutExpired) as exc:report['gpu_query_error']=type(exc).__name__
    write_json(out/'r08-preflight.json',report)
    root=state/'artifacts/casmi26/research-r08';root.mkdir(parents=True,exist_ok=True);write_json(root/'preflight.json',report)
    subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip','--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
    print('R08_PREFLIGHT_BEGIN\n'+json.dumps(report,indent=2)+'\nR08_PREFLIGHT_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

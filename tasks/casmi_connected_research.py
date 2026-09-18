"""Run graph-connected isomer diagnostics, preserving the scored R08B package."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,
            'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    repo=Path(__file__).resolve().parents[1];sys.path[:0]=[str(repo/'tasks'),str(repo/'work/casmi26')]
    from casmi26.connected_experiment import run_study,digest,dump
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    art=state/'artifacts/casmi26';source=art/'research-r08b/fixed-confirmation-v1'
    expected={'evidence.json.gz':'217153e52d50112f47c51de90806691fd1c6dd67c2b45ed6625a9c5d7ef0c629',
              'certificates.json.gz':'8b37cefafaa018bd6028938d311274320410db454f4601d703f1a258ecb40449'}
    for name,h in expected.items():
        if digest(source/name)!=h:raise RuntimeError('Fixed research input changed: '+name)
    import casmi_r08_forward as runtime
    import r08_python_command as imports
    public=art/'research-r08/forward-v1/fiora-e19ef82c9a6cb9dbac92bce23e914008f1aeb44e'
    deps=public.parent/'deps';paths=[deps,public,repo/'work/casmi26',repo/'tasks']
    env=runtime.env_clean();env['FIORA_TEST_MODEL']=str(public/'fiora/resources/models/fiora_OS_v1.0.0.pt')
    tests=subprocess.run(imports.python_command(py,paths,'import pytest,sys;raise SystemExit(pytest.main(sys.argv[1:]))',
        ['-q',repo/'work/casmi26/tests']),env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,
        encoding='utf-8',errors='replace',timeout=600)
    (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8');print(tests.stdout,flush=True)
    if tests.returncode:raise RuntimeError('Tests failed before connected-fragment study')
    root=art/'research-r10-connected/diagnostic-v1';root.mkdir(parents=True,exist_ok=True)
    lock=root/'study.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    frozen_root=art/'candidate-r08b-20260918'
    frozen=[frozen_root/'package/r08b-package.json',frozen_root/'package/bundle/model.npz',frozen_root/'submission.csv']
    before={str(p):digest(p) for p in frozen}
    try:
        report=run_study(source/'evidence.json.gz',source/'certificates.json.gz',root,workers=4)
        if before!={str(p):digest(p) for p in frozen}:raise RuntimeError('Frozen R08B bytes changed')
        for name in ('protocol.json','report.json','results.json.gz'):shutil.copy2(root/name,out/name)
        sub=out/'kaggle-read';sub.mkdir(exist_ok=True)
        status=subprocess.run([str(py),str(repo/'tasks/casmi_r08b_submission.py'),'--stage','status'],
            env={**os.environ,'CHEMISTRY_REQUEST_OUTPUT':str(sub)},stdin=subprocess.DEVNULL,
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        summary={'status':'completed','commit':os.environ.get('GITHUB_SHA'),
            'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'tests':tests.stdout.strip(),
            'report_sha256':digest(root/'report.json'),'results_sha256':digest(root/'results.json.gz'),
            'frozen_package_unchanged':True,'quality_claim':'exploratory_reused_cohorts',
            'baseline':'complete_R08B_certified_top25','new_training':False,'new_submissions':0,
            'kaggle_status_reader_exit_code':status.returncode,'output':str(root)}
        dump(out/'execution.json',summary)
        subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip',
                        '--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
        print('CONNECTED_RESEARCH_COMPLETE\n'+json.dumps(summary,indent=2),flush=True)
        return 0
    finally:lock.unlink()


if __name__=='__main__':raise SystemExit(main())

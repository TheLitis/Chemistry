"""Run fixed elemental-evidence diagnostics, never update the submitted model."""
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
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.formula_experiment import run_study,digest,dump
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    art=state/'artifacts/casmi26'
    source=art/'research-r08b/fixed-confirmation-v1/evidence.json.gz'
    if digest(source)!='217153e52d50112f47c51de90806691fd1c6dd67c2b45ed6625a9c5d7ef0c629':
        raise RuntimeError('Research input changed; no reinterpretation of another cohort')
    tests=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests/test_formula_evidence.py'),
        str(repo/'work/casmi26/tests/test_formula_experiment.py')],capture_output=True,text=True,
        encoding='utf-8',errors='replace',timeout=120)
    (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8')
    if tests.returncode:raise RuntimeError('Formula tests failed')
    root=art/'research-r09-formula/diagnostic-v1';root.mkdir(parents=True,exist_ok=True)
    frozen=art/'candidate-r08b-20260918/package/r08b-package.json'
    before=digest(frozen)
    report=run_study(source,root,workers=4)
    if digest(frozen)!=before:raise RuntimeError('Frozen submitted package changed')
    for name in ('protocol.json','report.json','results.json.gz'):shutil.copy2(root/name,out/name)
    prior=art/'r08b-fast-parity-20260918/acceptance.json'
    if prior.is_file():shutil.copy2(prior,out/'prior-fast-acceptance.json')
    sub=out/'kaggle-read';sub.mkdir(exist_ok=True)
    status=subprocess.run([str(py),str(repo/'tasks/casmi_r08b_submission.py'),'--stage','status'],
        env={**os.environ,'CHEMISTRY_REQUEST_OUTPUT':str(sub)},capture_output=True,text=True,
        encoding='utf-8',errors='replace',timeout=180)
    # The existing status reader redacts its own provider errors; don't print its raw stderr.
    summary={'status':'completed','commit':os.environ.get('GITHUB_SHA'),
        'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'tests':tests.stdout.strip(),
        'experiment_report_sha256':digest(root/'report.json'),'frozen_package_unchanged':True,
        'quality_claim':'exploratory_reused_cohorts','new_training':False,'new_submissions':0,
        'kaggle_status_reader_exit_code':status.returncode,'output':str(root)}
    dump(out/'execution.json',summary)
    subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip',
        '--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
    print('FORMULA_EXPERIMENT_COMPLETE\n'+json.dumps(summary,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

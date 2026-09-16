"""Poll the already accepted R06 submission until Kaggle exposes publicScore.

Read-only: never uploads datasets, pushes kernels, or creates submissions.
"""
from __future__ import annotations
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SLUG='enveda-CASMI26-molecule-id-mass-spectra'
REF=56278642
DESCRIPTION='CASMI26 R06 - two-seed MassSet hybrid rank ensemble; all spectra'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];prep=load('prep',repo/'tasks/casmi_prepare.py')
    env=prep.kaggle_environment(state,dict(os.environ));env['PYTHONIOENCODING']='utf-8'
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    result={'submission_ref':REF,'description':DESCRIPTION,'new_submissions':0,'read_only':True,'public_score':None,
            'checked_utc':None,'polls':0,'status':None}
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()','competitions','submissions',SLUG,'--format','json'],
                         env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        if r.returncode:
            raise RuntimeError('Read-only Kaggle history query failed: '+prep.redact(r.stderr,env)[:800])
        rows=json.loads(r.stdout) if r.stdout.strip() else []
        matches=[x for x in rows if int(x.get('ref',-1))==REF]
        if len(matches)!=1:raise RuntimeError('Accepted R06 submission is not uniquely present in history')
        row=matches[0]
        if row.get('description')!=DESCRIPTION:raise RuntimeError('Submission ref points to a different description')
        result['polls']+=1;result['checked_utc']=dt.datetime.now(dt.timezone.utc).isoformat();result['row']=row
        result['status']=str(row.get('status','')).split('.')[-1].upper()
        raw=row.get('publicScore')
        if raw not in (None,''):
            score=float(raw)
            if not 0<=score<=1:raise ValueError('Invalid public MRR score')
            result['public_score']=score;break
        time.sleep(20)
    result['score_available']=result['public_score'] is not None
    (out/'r06-score-poll.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('R06_SCORE_POLL_BEGIN\n'+json.dumps(result,indent=2)+'\nR06_SCORE_POLL_END',flush=True)
    return 0 if result['score_available'] else 3

if __name__=='__main__':raise SystemExit(main())

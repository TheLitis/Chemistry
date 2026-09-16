"""Read-only reconciliation/status polling for the already-attempted R06.1 submission."""
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
DESCRIPTION='CASMI26 R06.1 - MassSet rank ensemble + scorer-safe mass fallback'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def utc(value):
    d=dt.datetime.fromisoformat(str(value).replace('Z','+00:00'))
    return d.replace(tzinfo=dt.timezone.utc) if d.tzinfo is None else d.astimezone(dt.timezone.utc)


def scalar(row,name,default=None):
    return getattr(row,name,default)


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import write_json
    prep=load('prep',repo/'tasks/casmi_prepare.py');env=prep.kaggle_environment(state,dict(os.environ))
    root=state/'artifacts/casmi26/kaggle-r06-v1';journal_path=root/'publish-journal.json';journal=json.loads(journal_path.read_text())
    if not journal.get('submission_v3_attempted'):
        raise RuntimeError('No journaled v3 write attempt to reconcile')
    command_rc=journal.get('submission_v3_command_exit_code')
    if command_rc is not None and command_rc!=0:
        raise RuntimeError('The journal records a failed v3 submission command')
    attempted=utc(journal['submission_v3_attempted_utc'])
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    old=os.environ.get('KAGGLE_API_TOKEN');os.environ['KAGGLE_API_TOKEN']=env['KAGGLE_API_TOKEN']
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        final=None
        for attempt in range(180):
            rows=api.competition_submissions(SLUG) or []
            matches=[]
            for row in rows:
                if str(scalar(row,'description',''))!=DESCRIPTION:continue
                try:date=utc(scalar(row,'date'))
                except Exception:continue
                if abs((date-attempted).total_seconds())<=600:matches.append(row)
            if len(matches)>1:raise RuntimeError('Ambiguous R06.1 submission history')
            if matches:
                row=matches[0];ref=int(scalar(row,'ref'));status=str(scalar(row,'status','')).split('.')[-1].upper()
                public=str(scalar(row,'public_score','') or '').strip()
                error=str(scalar(row,'error_description','') or '').strip()
                final={'ref':ref,'status':status,'public_score':float(public) if public else None,'error_description':error,
                       'file_name':str(scalar(row,'file_name','')),'date':str(scalar(row,'date','')),'attempts':attempt+1}
                journal['submission_v3_ref']=ref;write_json(journal_path,journal)
                if public or error or status in ('ERROR','FAILED','CANCELLED','CANCELED'):break
            time.sleep(10)
    finally:
        if old is None:os.environ.pop('KAGGLE_API_TOKEN',None)
        else:os.environ['KAGGLE_API_TOKEN']=old
    if final is None:raise RuntimeError('R06.1 submission was not found in account history')
    journal.update(submission_v3_ref=final['ref'],submission_v3_status=final['status'],submission_v3_public_score=final['public_score'],
                   submission_v3_error_description=final['error_description'],submission_v3_checked_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(journal_path,journal)
    r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()','competitions','submission-limits',SLUG,'--json'],
                     env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
    limits=json.loads(r.stdout) if r.returncode==0 else None
    result={'submission':final,'limits':limits,'new_submissions':0,'read_only':True,
            'scored_successfully':final['public_score'] is not None and not final['error_description']}
    write_json(out/'r06-v3-status.json',result)
    print('R06_V3_STATUS_BEGIN\n'+json.dumps(result,indent=2)+'\nR06_V3_STATUS_END',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())

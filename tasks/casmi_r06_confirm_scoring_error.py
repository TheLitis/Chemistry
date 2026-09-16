"""Confirm the already-observed R06 scorer failure in the local write journal.

Read-only against Kaggle: never uploads, pushes kernels, or submits. The only write
is the local runner journal gate consumed by the separately verified v3 workflow.
"""
from __future__ import annotations
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import sys

SLUG='enveda-CASMI26-molecule-id-mass-spectra'
OLD_REF=56278642


def confirm_scoring_error(row):
    try:ref=int(getattr(row,'ref',row.get('ref') if isinstance(row,dict) else -1))
    except (TypeError,ValueError):return None
    if ref!=OLD_REF:return None
    if isinstance(row,dict):raw=row.get('error_description') or row.get('errorDescription') or row.get('error')
    else:raw=getattr(row,'error_description',None) or getattr(row,'errorDescription',None) or getattr(row,'error',None)
    text=str(raw or '').strip()
    low=text.lower()
    if 'incorrect format' not in low:return None
    if not any(term in low for term in ('empty value','wrong number of rows','wrong number of columns','invalid submission value','incorrect data type')):return None
    return text


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return os.spawnve(os.P_WAIT,str(py),[str(py),str(Path(__file__).resolve())],{**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import write_json
    prep=load('prep',repo/'tasks/casmi_prepare.py')
    env=prep.kaggle_environment(state,dict(os.environ))
    root=state/'artifacts/casmi26/kaggle-r06-v1';journal_path=root/'publish-journal.json'
    journal=json.loads(journal_path.read_text(encoding='utf-8'))
    if journal.get('submission_ref')!=OLD_REF or not journal.get('submission_accepted'):
        raise RuntimeError('Journal does not identify the accepted old R06 submission')
    old=os.environ.get('KAGGLE_API_TOKEN');os.environ['KAGGLE_API_TOKEN']=env['KAGGLE_API_TOKEN']
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate();rows=api.competition_submissions(SLUG) or []
    finally:
        if old is None:os.environ.pop('KAGGLE_API_TOKEN',None)
        else:os.environ['KAGGLE_API_TOKEN']=old
    matches=[row for row in rows if str(getattr(row,'ref',None))==str(OLD_REF)]
    if len(matches)!=1:raise RuntimeError('Old R06 submission was not uniquely returned by Kaggle')
    error=confirm_scoring_error(matches[0])
    if not error:raise RuntimeError('Kaggle did not return the expected scoring-format error')
    journal.update(scoring_error_confirmed=True,old_scoring_error=error,
                   scoring_error_confirmed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(journal_path,journal)
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    result={'submission_ref':OLD_REF,'scoring_error_confirmed':True,'error_description':error,
            'new_submissions':0,'kaggle_writes':0,'checked_utc':journal['scoring_error_confirmed_utc']}
    write_json(out/'r06-scoring-confirmation.json',result)
    print('R06_SCORING_CONFIRM_BEGIN\n'+json.dumps(result,indent=2)+'\nR06_SCORING_CONFIRM_END',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())

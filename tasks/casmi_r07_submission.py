"""At-most-once submission of the already validated private R07 notebook.

No dataset upload, kernel push, parameter search, or blind submission retry.
All versions' completed, failed and pending attempts count toward a shared
rolling-24-hour cap of two. A code-COMPLETE state alone is not a scored result.
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SLUG='enveda-CASMI26-molecule-id-mass-spectra'
KERNEL='casmi26-r07-timstof-hybrid'
DESCRIPTION='CASMI26 R07 - locked timsTOF hybrid, full refit, offline COCONUT 2026-08'
BUNDLE='195e2a73b7d25ce570c178b2f1fb0603d2f8cf47a82e1050b4a8b18ab540baf6'
MODEL='4a05a97b65276df6558e4c156f2129b5463b758f4ea730b0f1f93e99acf055a6'
VISIBLE='d36059b2eb15a3b60cdde1bce28a678282e55c634507f75982bf22c56f482a1b'
NOTEBOOK='34c68113f42edcfdeaa0fccf8d2499f379e07e68a79b8798c14b35e5b95ae436'
UTC=dt.timezone.utc


def utc(value):
    d=value if isinstance(value,dt.datetime) else dt.datetime.fromisoformat(str(value).replace('Z','+00:00'))
    return d.replace(tzinfo=UTC) if d.tzinfo is None else d.astimezone(UTC)


def classify(row):
    state=str(row.get('status','')).split('.')[-1].upper()
    error=str(row.get('error_description') or row.get('errorDescription') or '').strip()
    raw=row.get('public_score',row.get('publicScore'))
    result={'state':'pending','terminal':False,'public_score':None,'error_description':error}
    if error or state in ('ERROR','FAILED','CANCELLED','CANCELED'):
        result.update(state='scoring_error',terminal=True);return result
    if raw is not None and str(raw).strip():
        try:value=float(raw)
        except (TypeError,ValueError):value=float('nan')
        if not math.isfinite(value) or not 0<=value<=1:
            result.update(state='invalid_score',terminal=False);return result
        if state=='COMPLETE':result.update(state='scored',terminal=True,public_score=value)
    elif state=='COMPLETE':result['state']='complete_without_score'
    return result


def budget_decision(history,limits,now):
    now=utc(now);dates=[];pending=[];seen=set()
    for row in history:
        ref=str(row['ref'])
        if ref in seen:raise ValueError('Duplicate submission ref in API history')
        seen.add(ref);date=utc(row['date'])
        if date>now+dt.timedelta(minutes=5):raise ValueError('Submission time is unexpectedly in the future')
        if date>now-dt.timedelta(hours=24):dates.append(date)
        if not classify(row)['terminal']:pending.append(ref)
    try:remaining=int(limits.get('numAllowedNow',0))
    except (TypeError,ValueError):remaining=0
    if remaining<0:raise ValueError('Negative official remaining allowance')
    dates.sort()
    next_window=dates[-2]+dt.timedelta(hours=24) if len(dates)>=2 else now
    return {'attempts_last_24h':len(dates),'internal_rolling_24h_cap':2,
            'official_remaining':remaining,'pending_or_ambiguous_refs':pending,
            'may_submit':len(dates)<2 and not pending and remaining>=1,
            'next_internal_window_utc':next_window.isoformat()}


def reconcile(history,journal):
    if not journal.get('attempted'):return None
    description=journal.get('description')
    if description!=DESCRIPTION:raise ValueError('Journal refers to another candidate')
    attempt=utc(journal['attempted_utc']);retained=journal.get('submission_ref');matches=[]
    for row in history:
        if retained is not None:
            if str(row.get('ref'))!=str(retained):continue
            if row.get('description')!=DESCRIPTION:raise ValueError('Retained ref changed description')
            matches.append(row)
        elif row.get('description')==DESCRIPTION and -60<=(utc(row['date'])-attempt).total_seconds()<=900:
            matches.append(row)
    if len(matches)>1:raise ValueError('Ambiguous submission reconciliation')
    return matches[0] if matches else None


def submit_arguments(kernel,version):
    if kernel!='thelindortis/'+KERNEL or version!=1:raise ValueError('Wrong immutable notebook target')
    return ['competitions','submit',SLUG,'-k',kernel,'-v','1','-f','submission.csv','-m',DESCRIPTION]


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


API_READ="""import json
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
h=api.competition_submissions('enveda-CASMI26-molecule-id-mass-spectra',page_size=100) or []
l=api.competition_get_submission_limits('enveda-CASMI26-molecule-id-mass-spectra')
print(json.dumps({'history':[{'ref':r.ref,'date':r.date.isoformat(),'description':r.description,
'public_score':r.public_score,'status':str(r.status),'error_description':getattr(r,'error_description',None)} for r in h],
'limits':{'numToday':getattr(l,'num_today',0),'numTotal':getattr(l,'num_total',0),'numAllowedNow':getattr(l,'num_allowed_now',0)}}))
"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['check','status','submit'],required=True);args=parser.parse_args()
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    prep=load('prepare',repo/'tasks/casmi_prepare.py');env=prep.kaggle_environment(state,dict(os.environ))
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    root=state/'artifacts/casmi26/kaggle-r07-v1';release=state/'artifacts/casmi26/final-r07-v1'
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    lock=root/'submission.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    journal_path=root/'submission-journal.json'
    journal=json.loads(journal_path.read_text()) if journal_path.is_file() else {}
    report={'stage':args.stage,'new_submissions':0,'new_uploads':0,'official_score':None,'commit':os.environ.get('GITHUB_SHA')}
    def save():write_json(journal_path,journal)
    def read_api():
        r=subprocess.run([str(python),'-c',API_READ],env=env,stdin=subprocess.DEVNULL,
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        if r.returncode:
            (out/'read-error.log').write_text(prep.redact(r.stderr,env),encoding='utf-8')
            raise RuntimeError('Kaggle account read failed; no write attempted')
        value=json.loads(r.stdout)
        if not isinstance(value.get('history'),list) or not isinstance(value.get('limits'),dict):raise ValueError('Invalid Kaggle response')
        return value
    try:
        if args.stage=='check':
            r=subprocess.run([str(python),'-m','pytest','-q',str(repo/'work/casmi26/tests')],
                capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=360)
            (out/'tests.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
            report['tests']=r.stdout.strip()
            if r.returncode:raise RuntimeError('Tests failed before submission readiness check')
        preview=json.loads((root/'preview-journal.json').read_text())
        acceptance=json.loads((release/'archive-acceptance.json').read_text())
        if not preview.get('preview_verified') or preview.get('bundle_manifest_sha256')!=BUNDLE or preview.get('notebook_sha256')!=NOTEBOOK:
            raise ValueError('Private Kaggle preview is not the frozen candidate')
        if preview.get('verified_submission_sha256')!=VISIBLE or acceptance.get('submission_sha256')!=VISIBLE:
            raise ValueError('Preview and standalone acceptance differ')
        if acceptance.get('bundle_manifest_sha256')!=BUNDLE or acceptance.get('model_sha256')!=MODEL:
            raise ValueError('Standalone acceptance refers to different weights')
        if sha256(root/'verified-output/submission.csv')!=VISIBLE or sha256(release/'bundle/r07-bundle.json')!=BUNDLE:
            raise ValueError('Verified candidate output or manifest was modified')
        command=submit_arguments(preview['kernel'],preview['kernel_version'])
        current=read_api();now=dt.datetime.now(UTC);budget=budget_decision(current['history'],current['limits'],now)
        report.update(checked_utc=now.isoformat(),budget=budget,limits=current['limits'],kernel=preview['kernel'],kernel_version=1)
        if journal.get('attempted'):
            row=reconcile(current['history'],journal)
            report.update(status='attempt_already_exists_no_resubmit',submission=row)
            if row:
                journal.update(submission_ref=int(row['ref']),classification=classify(row),last_checked_utc=now.isoformat());save()
                report.update(classification=journal['classification'],official_score=journal['classification']['public_score'])
            else:report['status']='attempt_not_yet_reconciled_no_resubmit'
        elif args.stage!='submit' or not budget['may_submit']:
            report['status']='ready_for_one_submission' if budget['may_submit'] else 'waiting_for_shared_submission_budget'
        else:
            journal.update(attempted=True,attempted_utc=now.isoformat(),description=DESCRIPTION,
                kernel=preview['kernel'],kernel_version=1,bundle_manifest_sha256=BUNDLE,notebook_sha256=NOTEBOOK)
            save()
            report['new_submissions']=1
            try:
                r=subprocess.run([str(python),'-c','from kaggle.cli import main;main()']+command,env=env,
                    stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
                text=prep.redact(r.stdout+'\n'+r.stderr,env)
                text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
                (out/'submit.log').write_text(text,encoding='utf-8')
                journal['command_exit_code']=r.returncode
            except subprocess.TimeoutExpired:
                journal['command_timeout']=True
            save()
            report['status']='attempt_recorded_not_yet_scored'
            for i in range(6):
                current=read_api();row=reconcile(current['history'],journal)
                if row:
                    state_=classify(row);journal.update(submission_ref=int(row['ref']),classification=state_,last_checked_utc=dt.datetime.now(UTC).isoformat());save()
                    report.update(submission=row,classification=state_,official_score=state_['public_score']);break
                if i<5:time.sleep(5)
            report['limits']=current['limits']
        report['journal']=journal
        write_json(out/'r07-submission-status.json',report);write_json(root/'submission-status.json',report)
        print('R07_SUBMISSION_STATUS_BEGIN\n'+json.dumps(report,indent=2)+'\nR07_SUBMISSION_STATUS_END',flush=True)
        return 0
    finally:
        lock.unlink()


if __name__=='__main__':raise SystemExit(main())

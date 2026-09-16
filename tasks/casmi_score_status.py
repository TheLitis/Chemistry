"""Read/reconcile the already accepted CASMI submission; never submit or upload."""
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

DESCRIPTION='CASMI26 baseline v1 - trained catalog + spectral matching; first official evaluation'


def utc(value):
    value=dt.datetime.fromisoformat(str(value).replace('Z','+00:00'))
    return value.replace(tzinfo=dt.timezone.utc) if value.tzinfo is None else value.astimezone(dt.timezone.utc)


def verified_submission(rows, journal, ref):
    matches=[row for row in rows if str(row.get('ref'))==str(ref)]
    if len(matches)!=1 or not journal.get('submission_accepted'):
        raise ValueError('Known accepted submission not uniquely confirmed in account history')
    row=matches[0]
    if row.get('description')!=DESCRIPTION or row.get('fileName')!='submission.csv':
        raise ValueError('Submission identity differs from the authorized notebook')
    if abs((utc(row['date'])-utc(journal['attempted_utc'])).total_seconds())>120:
        raise ValueError('Submission time does not match local write journal')
    return row


def score_summary(row):
    state=str(row.get('status','')).split('.')[-1].upper()
    terminal=state in ('COMPLETE','ERROR','CANCELLED','CANCELED','FAILED')
    def score(name):
        raw=row.get(name)
        if raw is None or str(raw).strip()=='':return None
        value=float(raw)
        if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid MRR score in server response')
        return value if state=='COMPLETE' else None
    return {'submission_ref':int(row['ref']),'status':state,'terminal':terminal,
            'public_score':score('publicScore'),'private_score':score('privateScore')}


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ref',type=int,required=True);a=p.parse_args()
    if a.ref<=0:raise ValueError('Positive submission reference required')
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],
                               env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1]
    prep=load('prep',repo/'tasks/casmi_prepare.py');pub=load('pub',repo/'tasks/casmi_kaggle_publish.py')
    env=prep.kaggle_environment(state,dict(os.environ));env['PYTHONIOENCODING']='utf-8'
    root=state/'artifacts/casmi26/kaggle-v1';out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    journal=json.loads((root/'publish-journal.json').read_text())
    def run(label,args,check=True):
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args,
            env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        text=prep.redact(r.stdout+'\n'+r.stderr,env)
        text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
        (out/(label+'.txt')).write_text(text,encoding='utf-8')
        if check and r.returncode:raise RuntimeError(label+' failed; no score inferred')
        return text,r.returncode
    history,_=run('history',['competitions','submissions',pub.SLUG,'-v'])
    rows=pub.parse_history(history);row=verified_submission(rows,journal,a.ref)
    journal['submission_ref']=a.ref;pub.write_json(root/'publish-journal.json',journal)
    # Installed CLI 2.2.4 exposes scores via submissions, not the newer singular command.
    limits_text,limits_rc=run('live-limits',['competitions','submission-limits',pub.SLUG,'--json'],check=False)
    live_limits=json.loads(limits_text) if limits_rc==0 else None
    rank_text,rank_rc=run('rank',['competitions','list','--group','entered','--search','CASMI','--format','json'],check=False)
    rank=None;teams=None
    if rank_rc==0:
        for item in json.loads(rank_text):
            if pub.SLUG in item.get('ref',''):
                rank=item.get('userRank') or None;teams=item.get('teamCount')
    tests=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],
        stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
    print(tests.stdout,flush=True)
    (out/'tests.txt').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8')
    result=score_summary(row)
    result.update(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),commit=os.environ.get('GITHUB_SHA'),
                  request_id=os.environ.get('CHEMISTRY_REQUEST_ID'),kernel=journal['kernel'],kernel_version=journal['kernel_version'],
                  submission_created_this_run=False,total_account_submissions=len(rows),
                  budget=pub.budget_status(rows,dt.datetime.now(dt.timezone.utc),5),
                  public_rank=rank,competition_teams=teams,history_entry=row,
                  live_limits_command_exit_code=limits_rc,live_submission_limits=live_limits,
                  tests_exit_code=tests.returncode,tests=tests.stdout.strip())
    pub.write_json(root/'official-score.json',result);pub.write_json(out/'official-score.json',result)
    print('OFFICIAL_SCORE_BEGIN\n'+json.dumps(result,indent=2)+'\nOFFICIAL_SCORE_END',flush=True)
    return 0 if tests.returncode==0 else 1


if __name__=='__main__':raise SystemExit(main())

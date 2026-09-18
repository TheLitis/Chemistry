"""Resume only the approved R08B delivery stages, preserving the frozen model.

No model training or candidate build. Each call takes at most three monotonic
stages, stops on a pending preview/budget, and never repeats a submission write.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PINS = {
    'preview': ('casmi_r08b_preview.py', 'c7a545bcafb295965c628a4afef1d3b2583216fb'),
    'submission': ('casmi_r08b_submission.py', 'c377b95ab719ab906412c920ad3ab708716c109a'),
}


def normalized_blob(path):
    data=Path(path).read_text(encoding='utf-8-sig').encode('utf-8')
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def next_stage(build,preview,journal):
    if journal.get('attempted'):return ('submission','status')
    if build is None:return None
    if build.get('status')!='candidate_built_and_locally_executed' or build.get('gate',{}).get('eligible') is not True:
        raise ValueError('Candidate build is not accepted')
    for attempted,completed in (('dataset_attempted','dataset_created'),('kernel_attempted','kernel_pushed')):
        if preview.get(attempted) and not preview.get(completed):
            raise ValueError('An ambiguous '+attempted+' cannot be retried')
    if preview.get('kernel_pushed') and (type(preview.get('kernel_version')) is not int or preview['kernel_version']!=1):
        raise ValueError('Unexpected notebook version')
    if not preview.get('assets_staged'):return ('preview','stage')
    if not preview.get('kernel_pushed'):return ('preview','publish')
    if not preview.get('preview_verified'):return ('preview','verify')
    return ('submission','check')


def may_follow_with_submit(report):
    return bool(report.get('status')=='ready_for_one_submission' and
                report.get('budget',{}).get('may_submit') is True and
                not report.get('journal',{}).get('attempted'))


def read_if_exists(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else None


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve())],env={**os.environ,
            'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1]
    art=state/'artifacts/casmi26';root=art/'kaggle-r08b-v1';root.mkdir(parents=True,exist_ok=True)
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    lock=root/'continuation.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    result={'status':'started','commit':os.environ.get('GITHUB_SHA'),'steps':[],
            'new_training':False,'new_candidate_builds':0,'new_submissions':0,'official_score':None}
    def run(kind,stage):
        name,pin=PINS[kind];script=repo/'tasks'/name
        if normalized_blob(script)!=pin:raise ValueError('Approved helper changed: '+name)
        folder=out/('step-'+str(len(result['steps'])+1)+'-'+kind+'-'+stage);folder.mkdir()
        env={**os.environ,'CHEMISTRY_REQUEST_OUTPUT':str(folder),'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'}
        timeout=4000 if (kind,stage)==('preview','stage') else 2400 if stage=='publish' else 1200
        proc=subprocess.run([str(python),str(script),'--stage',stage],env=env,stdin=subprocess.DEVNULL,
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        (folder/'stdout.log').write_text(proc.stdout,encoding='utf-8')
        (folder/'stderr.log').write_text(proc.stderr,encoding='utf-8')
        if proc.returncode:raise RuntimeError('Approved stage failed: '+kind+'/'+stage+'; inspect step logs')
        target=folder/('preview-status.json' if kind=='preview' else 'submission-status.json')
        report=read_if_exists(target)
        if not isinstance(report,dict):raise ValueError('Stage result was not recorded')
        result['steps'].append({'helper':name,'stage':stage,'blob_sha':pin,'report':report})
        result['new_submissions']+=int(report.get('new_submissions',0))
        result['status']=report['status'];result['official_score']=report.get('official_score')
        if report.get('submission') is not None:result['submission']=report['submission']
        if report.get('classification') is not None:result['classification']=report['classification']
        return report
    try:
        for _ in range(3):
            build=read_if_exists(art/'candidate-r08b-20260918/build-status.json')
            preview=read_if_exists(root/'preview-journal.json') or {}
            journal=read_if_exists(root/'submission-journal.json') or {}
            planned=next_stage(build,preview,journal)
            if planned is None:
                result['status']='waiting_for_existing_candidate_build';break
            report=run(*planned)
            if planned[0]=='submission':
                if planned[1]=='check' and may_follow_with_submit(report):
                    run('submission','submit')
                break
            if report['status']=='preview_pending':break
        result['checked_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        write(out/'continuation-status.json',result);write(root/'continuation-status.json',result)
        print('R08B_CONTINUATION_BEGIN\n'+json.dumps(result,indent=2)+'\nR08B_CONTINUATION_END',flush=True)
        return 0
    except Exception as exc:
        result.update(status='blocked',error_type=type(exc).__name__,error=str(exc),
                      checked_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write(out/'continuation-status.json',result);write(root/'continuation-status.json',result)
        print('R08B_CONTINUATION_BLOCKED '+type(exc).__name__,flush=True)
        return 2
    finally:lock.unlink()


if __name__=='__main__':raise SystemExit(main())

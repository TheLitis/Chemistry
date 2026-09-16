"""Read-only diagnostics for the one accepted submission and installed CLI."""
import contextlib
import importlib.util
import io
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1'})
    repo=Path(__file__).resolve().parents[1];prep=load('prep',repo/'tasks/casmi_prepare.py')
    env=prep.kaggle_environment(state,dict(os.environ));env['PYTHONIOENCODING']='utf-8'
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);root=state/'artifacts/casmi26/kaggle-v1'
    journal=json.loads((root/'publish-journal.json').read_text());ref=int(journal['submission_ref'])
    report={'submission_ref':ref,'new_submissions':0,'reads':{}}
    for label,args in [('logs_help',['competitions','logs','--help']),
                       ('limits_help',['competitions','submission-limits','--help']),
                       ('history',['competitions','submissions',prep.SLUG,'--format','json'])]:
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args,env=env,stdin=subprocess.DEVNULL,
                         capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        text=prep.redact(r.stdout+'\n'+r.stderr,env)
        report['reads'][label]={'exit_code':r.returncode,'text':text[:16000]}
    os.environ.update(env)
    captured=io.StringIO()
    try:
        with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
            from kaggle import api
            fn=getattr(api,'competition_submissions',None)
            if fn:
                report['submission_method_signature']=str(inspect.signature(fn))
                entries=fn(prep.SLUG)
                safe=[]
                for item in entries:
                    data=item.to_dict() if hasattr(item,'to_dict') else vars(item)
                    safe.append({k:v for k,v in data.items() if k.lower().replace('_','') in
                                 ('ref','id','date','filename','status','description','publicscore','privatescore',
                                  'errordescription','errormessage','error','submissionid','submittedat')})
                report['own_submission_details']=safe
            report['related_read_methods']={name:str(inspect.signature(getattr(api,name))) for name in dir(api)
                         if callable(getattr(api,name)) and (name.startswith('competition_') and any(s in name for s in ('logs','limits','submission_status')))}
    except Exception as exc:report['detail_error']=type(exc).__name__+': '+prep.redact(str(exc),env)[:1200]
    text=prep.redact(json.dumps(report,indent=2,default=str),env)
    text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
    (out/'scoring-diagnostics.json').write_text(text,encoding='utf-8')
    (root/'scoring-diagnostics.json').write_text(text,encoding='utf-8')
    print('SCORING_DIAGNOSTICS_BEGIN\n'+text+'\nSCORING_DIAGNOSTICS_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

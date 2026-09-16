"""Read-only Kaggle preflight plus local metadata initialization; no uploads."""
from __future__ import annotations
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1'})
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('prepare',repo/'tasks/casmi_prepare.py')
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    env=prep.kaggle_environment(state,dict(os.environ));env['PYTHONIOENCODING']='utf-8'
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    publish=state/'artifacts/casmi26/kaggle-v1';publish.mkdir(parents=True,exist_ok=True)
    report={'commit':os.environ.get('GITHUB_SHA'),'submission_made':False,'uploads_made':False,'checks':{}}
    def run(label,args,timeout=90):
        cmd=[str(py),'-c','from kaggle.cli import main;main()']+args
        result=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        text=prep.redact(result.stdout+'\n'+result.stderr,env)
        (out/(label+'.txt')).write_text(text,encoding='utf-8')
        report['checks'][label]={'exit_code':result.returncode,'text':text[:24000]}
        print(label.upper()+'_BEGIN\n'+text[:24000]+'\n'+label.upper()+'_END',flush=True)
        return result.returncode
    run('submissions',['competitions','submissions',prep.SLUG,'-v'])
    run('entered',['competitions','list','--group','entered','--search','CASMI','--format','json'])
    run('mine_notebooks',['kernels','list','--mine','--competition',prep.SLUG,'-v'])
    run('mine_datasets',['datasets','list','--mine','--search','casmi','-v'])
    init=publish/'init';init.mkdir(exist_ok=True)
    run('kernel_init',['kernels','init','-p',str(init)])
    metadata=init/'kernel-metadata.json'
    if metadata.exists():report['kernel_init_metadata']=json.loads(metadata.read_text(encoding='utf-8'))
    # Only provider API method signatures and competition metadata are exposed.
    os.environ.update(env)
    try:
        from kaggle import api
        names=['competition_list','competition_submissions','competition_submit','competition_submit_code','kernels_push','kernels_status','kernels_pull','dataset_create_new','dataset_status']
        report['api_signatures']={name:str(inspect.signature(getattr(api,name))) for name in names if hasattr(api,name)}
        competitions=api.competitions_list(group='entered',search='CASMI') if hasattr(api,'competitions_list') else api.competition_list(group='entered',search='CASMI')
        report['competition_metadata']=[]
        for obj in competitions:
            values=obj.to_dict() if hasattr(obj,'to_dict') else vars(obj)
            report['competition_metadata'].append({k:v for k,v in values.items() if not any(x in k.lower() for x in ('token','secret','key','password'))})
    except Exception as exc:report['metadata_error']=prep.redact(str(exc),env)[:1500]
    # Inspect public rules in a fresh anonymous browser, not the user's profile.
    edge=next((p for p in (Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'),Path(r'C:\Program Files\Microsoft\Edge\Application\msedge.exe')) if p.exists()),None)
    if edge:
        with tempfile.TemporaryDirectory(prefix='casmi-submit-public-',dir=state/'cache',ignore_cleanup_errors=True) as profile:
            try:
                r=subprocess.run([str(edge),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',f'--user-data-dir={profile}','--dump-dom','--virtual-time-budget=20000',f'https://www.kaggle.com/competitions/{prep.SLUG}/rules'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=65)
                text=prep.visible_text(r.stdout);(out/'public-rules.txt').write_text(text,encoding='utf-8')
                lines=text.splitlines();indexes=[i for i,s in enumerate(lines) if any(k in s.lower() for k in ('submission','entries per','daily'))]
                report['rules_excerpts']=['\n'.join(lines[max(0,i-1):i+4]) for i in indexes][:30]
            except subprocess.TimeoutExpired:report['rules_error']='public_page_timeout'
    text=prep.redact(json.dumps(report,indent=2,default=str),env)
    (publish/'preflight.json').write_text(text,encoding='utf-8');(out/'preflight.json').write_text(text,encoding='utf-8')
    print('SUBMIT_PREFLIGHT_BEGIN\n'+text+'\nSUBMIT_PREFLIGHT_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

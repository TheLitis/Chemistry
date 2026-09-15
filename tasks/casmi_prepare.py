"""Prepare an isolated CASMI environment and read public competition pages.

No rule acceptance, submission, account creation, browser-profile access or
credential extraction. --download uses only already provisioned Kaggle access.
"""
from __future__ import annotations
import argparse
import datetime as dt
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

SLUG = 'enveda-CASMI26-molecule-id-mass-spectra'


def visible_text(html: str) -> str:
    class Reader(HTMLParser):
        def __init__(self):
            super().__init__(); self.hidden=0; self.parts=[]
        def handle_starttag(self,tag,attrs):
            if tag in ('script','style','noscript'): self.hidden+=1
        def handle_endtag(self,tag):
            if tag in ('script','style','noscript') and self.hidden: self.hidden-=1
        def handle_data(self,data):
            if not self.hidden and data.strip(): self.parts.append(data.strip())
    reader=Reader();reader.feed(html);return '\n'.join(reader.parts)


def redact(text: str, env: dict) -> str:
    for key in ('KAGGLE_KEY','KAGGLE_API_TOKEN'):
        value=env.get(key)
        if value: text=text.replace(value,'[REDACTED]')
    return re.sub(r'KGAT_[A-Za-z0-9_\-]+','[REDACTED]',text)


def kaggle_environment(state: Path, inherited: dict) -> dict:
    env=dict(inherited)
    if env.get('KAGGLE_API_TOKEN') or (env.get('KAGGLE_USERNAME') and env.get('KAGGLE_KEY')):
        return env
    roots=[]
    if env.get('KAGGLE_CONFIG_DIR'): roots.append(Path(env['KAGGLE_CONFIG_DIR']))
    roots += [state/'kaggle',state/'.kaggle',Path.home()/'.kaggle',Path(r'C:\Users\loval\.kaggle')]
    for root in roots:
        for name in ('access_token','kaggle.json'):
            file=root/name
            try:
                if file.is_file():
                    with file.open('rb'): pass
                    env['KAGGLE_CONFIG_DIR']=str(root)
                    return env
            except OSError: pass
    return env


def run(args: list[str], *, env: dict, timeout: int, log: Path, expose: bool=True) -> dict:
    try:
        result=subprocess.run(args,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,
                              encoding='utf-8',errors='replace',timeout=timeout)
        content=redact(result.stdout+'\n'+result.stderr,env)
        if expose:
            log.write_text(content,encoding='utf-8');print(content[-14000:],flush=True)
        return {'exit_code':result.returncode, 'output':content if expose else None}
    except subprocess.TimeoutExpired:
        return {'exit_code':124,'error':'timeout'}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    state=Path(os.environ.get('CHEMISTRY_STATE_ROOT',r'C:\ProgramData\ChemistryRunner'))
    output=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
    env=kaggle_environment(state,os.environ)
    env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
    venv=state/'envs'/'casmi26'; python=venv/'Scripts'/'python.exe'
    report={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'request_id':os.environ.get('CHEMISTRY_REQUEST_ID'),
            'official_score':None,'real_data_predictions':False,'environment':str(venv)}
    if not python.exists():
        venv.parent.mkdir(parents=True,exist_ok=True)
        result=run([sys.executable,'-m','venv','--system-site-packages',str(venv)],env=env,timeout=180,log=output/'venv.log')
        if result['exit_code']: raise RuntimeError('Isolated environment creation failed')
    install=run([str(python),'-m','pip','install','--disable-pip-version-check','-r',str(repo/'work/casmi26/requirements.txt'),
                 'kaggle==2.2.4','pytest'],env=env,timeout=900,log=output/'casmi-install.log')
    report['dependencies_exit_code']=install['exit_code']
    if install['exit_code']: raise RuntimeError('CASMI dependencies failed to install')
    test=run([str(python),'-m','pytest','-q',str(repo/'work/casmi26/tests')],env=env,timeout=180,log=output/'casmi-tests.log')
    report['tests_exit_code']=test['exit_code']
    cli=venv/'Scripts'/'kaggle.exe'
    listed=run([str(cli),'competitions','files',SLUG,'--page-size','200','-v'],env=env,timeout=60,
               log=output/'kaggle-files.log',expose=False)
    report['kaggle_files_exit_code']=listed['exit_code']
    if listed['exit_code']==0:
        safe=run([str(cli),'competitions','files',SLUG,'--page-size','200','-v'],env=env,timeout=60,
                 log=output/'kaggle-files.csv',expose=True)
        report['kaggle_files_confirmed']=safe['exit_code']==0
    if args.download and listed['exit_code']==0:
        if shutil.disk_usage(state).free < 30*2**30:
            raise RuntimeError('Less than 30 GiB free; not starting download')
        data=state/'data'/'casmi26'/'raw';data.mkdir(parents=True,exist_ok=True)
        downloaded=run([str(cli),'competitions','download',SLUG,'-p',str(data)],env=env,timeout=1800,
                       log=output/'kaggle-download.log',expose=False)
        report['download_exit_code']=downloaded['exit_code']
    edge=next((p for p in (Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'),
                              Path(r'C:\Program Files\Microsoft\Edge\Application\msedge.exe')) if p.exists()),None)
    report['public_pages']=[]
    if edge:
        cache=state/'cache';cache.mkdir(parents=True,exist_ok=True)
        for name,suffix in (('evaluation','overview/evaluation'),('rules','rules'),('data','data')):
            url=f'https://www.kaggle.com/competitions/{SLUG}/{suffix}'
            item={'name':name,'url':url,'authenticated':False}
            with tempfile.TemporaryDirectory(prefix='casmi-public-',dir=cache,ignore_cleanup_errors=True) as profile:
                try:
                    result=subprocess.run([str(edge),'--headless=new','--disable-gpu','--no-first-run',
                        '--no-default-browser-check',f'--user-data-dir={profile}','--dump-dom',
                        '--virtual-time-budget=20000',url],stdin=subprocess.DEVNULL,capture_output=True,
                        text=True,encoding='utf-8',errors='replace',timeout=60)
                    text=visible_text(result.stdout)
                    item.update(exit_code=result.returncode,text=text[:60000])
                    (output/f'public-{name}.txt').write_text(text,encoding='utf-8')
                    print(f'PUBLIC_{name.upper()}_BEGIN\n{text[:50000]}\nPUBLIC_{name.upper()}_END',flush=True)
                except subprocess.TimeoutExpired:
                    item['error']='anonymous_browser_timeout'
            report['public_pages'].append(item)
    report['status']='ready_for_data' if listed['exit_code']==0 else 'blocked_kaggle_authorization_or_access'
    (output/'casmi-prepare.json').write_text(json.dumps(report,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    print('CASMI_PREPARE_SUMMARY '+json.dumps({k:v for k,v in report.items() if k!='public_pages'}),flush=True)
    return 0 if test['exit_code']==0 else 1


if __name__=='__main__':
    raise SystemExit(main())

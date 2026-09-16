"""Read-only diagnostics for the already-pushed failed R06 Kaggle kernel v1."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

KERNEL='thelindortis/casmi26-r06-massset-rank-ensemble'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];prep=load('prep',repo/'tasks/casmi_prepare.py')
    env=prep.kaggle_environment(state,dict(os.environ));out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    report={'kernel':KERNEL,'version':1,'writes':0,'commands':{}}
    def call(name,args):
        folder=out/name;shutil.rmtree(folder,ignore_errors=True);folder.mkdir()
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args,env=env,stdin=subprocess.DEVNULL,
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        text=prep.redact(r.stdout+'\n'+r.stderr,env)
        text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
        (out/(name+'.log')).write_text(text,encoding='utf-8')
        report['commands'][name]={'exit_code':r.returncode,'text':text[-5000:]}
        return folder,r.returncode
    pull,rc=call('pull',['kernels','pull',KERNEL+'/1','-p',str(out/'pull'),'-m'])
    output,orc=call('output',['kernels','output',KERNEL,'-p',str(out/'output'),'-o','--page-size','200'])
    _,frc=call('files',['kernels','files',KERNEL,'--page-size','200','-v'])
    # Extract notebook cell errors without echoing arbitrary full outputs into logs.
    notebooks=list((out/'pull').glob('*.ipynb'))
    errors=[]
    if notebooks:
        nb=json.loads(notebooks[0].read_text(encoding='utf-8'))
        for ci,cell in enumerate(nb.get('cells',[])):
            for item in cell.get('outputs',[]):
                if item.get('output_type')=='error':
                    errors.append({'cell':ci,'ename':item.get('ename'),'evalue':item.get('evalue'),
                        'traceback':item.get('traceback',[])})
                elif item.get('output_type')=='stream':
                    text=''.join(item.get('text',[]))
                    if 'Traceback' in text or 'Error' in text or 'ERROR' in text:
                        errors.append({'cell':ci,'stream_tail':text[-12000:]})
    report['notebooks']=[p.name for p in notebooks];report['extracted_errors']=errors
    report['output_files']=[str(p.relative_to(out/'output')) for p in (out/'output').rglob('*') if p.is_file()]
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('R06_KERNEL_DIAG_BEGIN\n'+json.dumps(report,indent=2,ensure_ascii=False)+'\nR06_KERNEL_DIAG_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

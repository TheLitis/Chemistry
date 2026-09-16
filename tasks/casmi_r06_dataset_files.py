"""Read-only inspection of the already-created private R06 Kaggle dataset."""
from __future__ import annotations
import importlib.util,json,os
from pathlib import Path
import subprocess,sys

DATASET='thelindortis/casmi26-r06-assets-v1'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];prep=load('prep',repo/'tasks/casmi_prepare.py');env=prep.kaggle_environment(state,dict(os.environ))
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    result={'dataset':DATASET,'writes':0,'commands':{}}
    for name,args in (
        ('files',['datasets','files',DATASET,'--page-size','200','-v']),
        ('metadata',['datasets','metadata',DATASET,'-p',str(out/'metadata')]),
    ):
        (out/'metadata').mkdir(exist_ok=True)
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
        text=prep.redact(r.stdout+'\n'+r.stderr,env)
        result['commands'][name]={'exit_code':r.returncode,'text':text[-12000:]}
    meta=out/'metadata/dataset-metadata.json'
    if meta.exists():
        data=json.loads(meta.read_text(encoding='utf-8'));info=data.get('info',data)
        result['is_private']=info.get('isPrivate');result['title']=info.get('title')
    (out/'r06-dataset-files.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('R06_DATASET_FILES_BEGIN\n'+json.dumps(result,indent=2,ensure_ascii=False)+'\nR06_DATASET_FILES_END',flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())

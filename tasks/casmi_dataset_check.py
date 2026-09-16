"""Read only the dataset just created by this project; no publication or retry."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),__file__],env={**os.environ,'PYTHONUTF8':'1'})
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('prep',repo/'tasks/casmi_prepare.py');prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    env=prep.kaggle_environment(state,dict(os.environ));out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    root=state/'artifacts/casmi26/kaggle-v1';journal=json.loads((root/'publish-journal.json').read_text())
    results={};metadata=root/'remote-metadata';metadata.mkdir(exist_ok=True)
    for name,args in [('status',['datasets','status',journal['dataset'],'--format','json']),
                      ('files',['datasets','files',journal['dataset'],'-v','--page-size','200']),
                      ('mine',['datasets','list','--mine','--search','CASMI26','--format','json']),
                      ('metadata',['datasets','metadata',journal['dataset'],'-p',str(metadata)])]:
        result=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args,env=env,stdin=subprocess.DEVNULL,
                              capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        text=prep.redact(result.stdout+'\n'+result.stderr,env)
        results[name]={'exit_code':result.returncode,'text':text[:22000]}
    if (metadata/'dataset-metadata.json').exists():results['downloaded_metadata']=json.loads((metadata/'dataset-metadata.json').read_text())
    text=json.dumps(results,indent=2);(out/'dataset-check.json').write_text(text);(root/'dataset-check.json').write_text(text)
    print('DATASET_CHECK_BEGIN\n'+text+'\nDATASET_CHECK_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

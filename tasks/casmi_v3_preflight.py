"""Read-only scoped inventory before v3 integration. No uploads or submissions."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('prepare',repo/'tasks/casmi_prepare.py')
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    env=prep.kaggle_environment(state,dict(os.environ))
    result={'commit':os.environ.get('GITHUB_SHA'),'new_submissions':0,'reads':{},'research':{},'arrays':{}}
    for name,args in (
        ('history',['competitions','submissions','enveda-CASMI26-molecule-id-mass-spectra','--format','json']),
        ('limits',['competitions','submission-limits','enveda-CASMI26-molecule-id-mass-spectra','--json']),
    ):
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args,env=env,stdin=subprocess.DEVNULL,
                         capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        result['reads'][name]={'exit_code':r.returncode,'text':prep.redact(r.stdout+'\n'+r.stderr,env)}
    import numpy as np
    cache=state/'cache/casmi26/official-v1'
    for name in ('features.npy','counts.npy','fingerprints.npy','observed.npy'):
        a=np.load(cache/name,mmap_mode='r',allow_pickle=False)
        result['arrays'][name]={'shape':list(a.shape),'dtype':str(a.dtype)}
    for name in ('research-r01','research-r02','research-r03','research-catalog','kaggle-v2'):
        root=state/'artifacts/casmi26'/name
        result['research'][name]={'exists':root.exists(),'files':sorted(p.name for p in root.glob('*.json'))}
        for path in root.glob('*.json'):
            if path.name in ('report.json','official-score.json','publish-journal.json'):
                data=json.loads(path.read_text(encoding='utf-8-sig'))
                result['research'][name][path.name]={k:v for k,v in data.items() if k in ('status','selection','selected','audit','recommend_gate','kernel','kernel_version','submission_ref','submission_attempted','submission_accepted','attempted_utc','public_score','official_score')}
    import torch
    result['compute']={'torch':torch.__version__,'cuda':torch.cuda.is_available()}
    if torch.cuda.is_available():result['compute']['gpu']=torch.cuda.get_device_name(0)
    # Archive only tracked project source, not credentials or local data.
    subprocess.run(['git','-C',str(repo),'archive','--format=zip','--output='+str(out/'source-snapshot.zip'),'HEAD'],check=True,timeout=60)
    (out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    # Keep large calibration tables in the artifact, not in the user-facing log.
    for r in result['research'].values():
        report=r.get('report.json',{})
        if 'selection' in report:report['selection']={k:v for k,v in report['selection'].items() if k!='calibration'}
    print('V3_PREFLIGHT_BEGIN\n'+json.dumps(result,indent=2)+'\nV3_PREFLIGHT_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

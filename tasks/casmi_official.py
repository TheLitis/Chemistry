"""Run official-data preparation/training/prediction in the isolated PC runtime.

No account changes, permission changes, rule acceptance or auto-submission.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main(argv=None):
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'}
        return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env=env)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','fit','predict','all'),default='all')
    parser.add_argument('--epochs',type=int,default=30)
    args=parser.parse_args(argv)
    repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26 import production
    spec=importlib.util.spec_from_file_location('staged',repo/'tasks/casmi_staged.py')
    staged=importlib.util.module_from_spec(spec);spec.loader.exec_module(staged)
    data=staged.find_dataset(state/'data/external')
    cache=state/'cache/casmi26/official-v1';artifacts=state/'artifacts/casmi26/official-v1'
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])/'official';out.mkdir(parents=True,exist_ok=True)
    report={'request_id':os.environ.get('CHEMISTRY_REQUEST_ID'),'commit':os.environ.get('GITHUB_SHA'),
            'stage':args.stage,'data':str(data),'artifacts':str(artifacts),'official_score':None,'submission_made':False}
    started=time.monotonic()
    try:
        tests=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],
                             stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
        (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8')
        print(tests.stdout[-16000:],flush=True)
        if tests.returncode:raise RuntimeError('Pipeline tests failed before the official-data run')
        report['tests_passed']=True
        production.main(['--data',str(data),'--cache',str(cache),'--artifacts',str(artifacts),
                         '--stage',args.stage,'--epochs',str(args.epochs),'--workers','8'])
        report['status']='stage_completed'
    except Exception as exc:
        report['status']='failed';report['error']=str(exc)[:2000]
        raise
    finally:
        report['seconds']=time.monotonic()-started
        for name,folder in [('prepared.json',cache),('validation.json',artifacts),
                            ('model-manifest.json',artifacts),('prediction-report.json',artifacts)]:
            path=folder/name
            if path.exists():
                d=json.loads(path.read_text(encoding='utf-8'))
                report[name]={k:v for k,v in d.items() if k not in ('details','broad_mass_candidate_counts')}
                if 'broad_mass_candidate_counts' in d:
                    values=d['broad_mass_candidate_counts'].values()
                    report[name]['mass_candidate_coverage']=sum(v>0 for v in values)
                shutil.copy2(path,out/name)
        for name in ('submission.csv','model.npz'):
            path=artifacts/name
            if path.exists() and report['status']=='stage_completed':shutil.copy2(path,out/name)
        production.write_json(out/'stage-report.json',report)
        production.write_json(artifacts/(args.stage+'-run.json'),report)
        print('OFFICIAL_STAGE_REPORT_BEGIN\n'+json.dumps(report,indent=2,allow_nan=False)+'\nOFFICIAL_STAGE_REPORT_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

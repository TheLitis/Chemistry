"""Full v3 implementation run. No Kaggle writes; previous champion is preserved."""
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
import zipfile


def dataset_path(repo, root):
    spec=importlib.util.spec_from_file_location('casmi_staged_v3',Path(repo)/'tasks/casmi_staged.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.find_dataset(root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('all','prepare','fit','refit','deliver','tests'),default='all')
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--state',type=Path)
    a=p.parse_args();state=a.state or Path(os.environ.get('CHEMISTRY_STATE_ROOT',r'C:\ProgramData\ChemistryRunner'))
    py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],
           env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.cache_v3 import prepare_cache
    from casmi26.pipeline_v3 import fit_experiment,refit_bundle,verify_bundle
    from casmi26.notebook_v3 import build_notebook
    from casmi26.production import sha256,write_json
    from casmi26.metric import require_official_rdkit
    require_official_rdkit();out=Path(os.environ.get('CHEMISTRY_REQUEST_OUTPUT',state/'artifacts/casmi26/manual-v3'))
    out.mkdir(parents=True,exist_ok=True)
    tests=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
    print(tests.stdout,flush=True);(out/'v3-tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8')
    if tests.returncode:raise RuntimeError('Tests failed before any v3 training')
    if a.stage=='tests':return 0
    data=dataset_path(repo,state/'data/external');base=state/'cache/casmi26/official-v1'
    cache=state/'cache/casmi26/highres-v3';old=state/'artifacts/casmi26/official-v1'
    experiment=state/'artifacts/casmi26/research-v3';bundle=state/'artifacts/casmi26/highres-v3/bundle'
    started=time.monotonic();report={'commit':os.environ.get('GITHUB_SHA'),'stage':a.stage,'new_kaggle_submissions':0,'official_score':None}
    try:
        if a.stage in ('all','prepare'):report['prepare']=prepare_cache(data/'train.parquet',base,cache)
        if a.stage in ('all','fit'):report['validation']=fit_experiment(cache,old,experiment,epochs=a.epochs)
        if a.stage in ('all','refit'):report['refit']=refit_bundle(cache,old,experiment,bundle)
        if a.stage in ('all','deliver') and (bundle/'v3-bundle.json').is_file():
            verify_bundle(bundle)
            wheels=old/'bundle/wheels'
            if not wheels.is_dir():raise RuntimeError('Pinned offline wheel set absent')
            shutil.copytree(wheels,bundle/'wheels',dirs_exist_ok=True)
            notebook=build_notebook(bundle.parent/'casmi26-v3.ipynb')
            workspace=bundle.parent/'notebook-output';workspace.mkdir(parents=True,exist_ok=True)
            source=json.loads(notebook.read_text());scope={'__name__':'__main__'}
            env_keys=('CASMI_INPUT_ROOT','CASMI_BUNDLE_DIR','CASMI_WORK_ROOT');prior={k:os.environ.get(k) for k in env_keys}
            os.environ.update(CASMI_INPUT_ROOT=str(data),CASMI_BUNDLE_DIR=str(bundle),CASMI_WORK_ROOT=str(workspace))
            try:
                for cell in source['cells']:
                    if cell['cell_type']=='code':exec(compile(''.join(cell['source']),str(notebook),'exec'),scope)
            finally:
                for k,v in prior.items():
                    if v is None:os.environ.pop(k,None)
                    else:os.environ[k]=v
            pred=json.loads((workspace/'submission.csv.report.json').read_text())
            if pred['submission_sha256']!=sha256(workspace/'submission.csv'):raise RuntimeError('Submission output hash differs')
            report['inference']={k:v for k,v in pred.items() if k!='details'}
            archive=bundle.parent/'casmi-v3-delivery.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
                for file in sorted(bundle.rglob('*')):
                    if file.is_file():z.write(file,'bundle/'+str(file.relative_to(bundle)).replace('\\','/'))
                z.write(notebook,notebook.name)
                z.write(experiment/'validation-v3.json','validation-v3.json')
            report['delivery']={'archive':str(archive),'sha256':sha256(archive),'notebook':str(notebook),
                'submission':str(workspace/'submission.csv'),'candidate_not_champion':True}
            shutil.copy2(notebook,out/notebook.name)
        report['status']='completed'
    except Exception as exc:
        report['status']='failed';report['error']=str(exc);raise
    finally:
        report['seconds']=time.monotonic()-started
        for name in ('validation-v3.json','selection-before-audit.json','protocol.json','audit-ranks.json'):
            path=experiment/name
            if path.exists():shutil.copy2(path,out/name)
        write_json(out/'v3-run.json',report);write_json(state/'artifacts/casmi26/highres-v3/run-report.json',report)
        compact={k:v for k,v in report.items() if k not in ('prepare','validation','refit')}
        if 'validation' in report:
            v=report['validation'];compact['validation']={k:v[k] for k in ('selection','audit','selected_vs_frozen','eligible_for_candidate_bundle')}
        if 'refit' in report:compact['refit_status']=report['refit']['status']
        print('V3_RUN_BEGIN\n'+json.dumps(compact,indent=2)+'\nV3_RUN_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

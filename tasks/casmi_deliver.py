"""Assemble and execute the trained notebook on official inputs; no silent submission.

Original data stays read-only. Assets and logs remain in the private project.
Kaggle account settings, permissions and browser sessions are not modified.
"""
from __future__ import annotations
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


def execute_notebook(source: Path, destination: Path) -> dict:
    book=json.loads(Path(source).read_text(encoding='utf-8'));namespace={'__name__':'__main__'};count=0
    started=time.monotonic()
    try:
        for cell in book['cells']:
            if cell['cell_type']!='code':continue
            count+=1;cell['execution_count']=count;cell['outputs']=[]
            stdout,stderr=io.StringIO(),io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                    exec(compile(''.join(cell['source']),str(source)+f':cell-{count}','exec'),namespace)
            except BaseException as exc:
                cell['outputs'].append({'output_type':'error','ename':type(exc).__name__,
                    'evalue':str(exc),'traceback':traceback.format_exception(type(exc),exc,exc.__traceback__)})
                raise
            finally:
                for name,stream in [('stdout',stdout),('stderr',stderr)]:
                    text=stream.getvalue()
                    if text:
                        cell['outputs'].insert(0,{'output_type':'stream','name':name,'text':text})
                        print(text[-12000:],flush=True)
    finally:
        Path(destination).write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return {'code_cells_executed':count,'seconds':time.monotonic()-started,
            'execution':'sequential code cells in isolated Windows Python; not a Kaggle Linux run'}


def wheel_arguments(folder: Path,python_version: str) -> list[str]:
    if python_version not in ('312','313'):raise ValueError('Unsupported wheel Python version')
    return ['-m','pip','download','--disable-pip-version-check','--dest',str(folder),
        '--platform','manylinux_2_28_x86_64','--platform','manylinux_2_17_x86_64',
        '--platform','manylinux2014_x86_64','--implementation','cp','--python-version',python_version,
        '--abi','cp'+python_version,'--only-binary=:all:',
        'numpy==2.3.5','rdkit==2026.3.3','pyarrow==21.0.0']


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve())],
            env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26 import portable,production,submission_notebook
    from casmi26.metric import structure_key,require_official_rdkit
    require_official_rdkit()
    staged=load('staged',repo/'tasks/casmi_staged.py');data=staged.find_dataset(state/'data/external')
    prepare=load('prepare',repo/'tasks/casmi_prepare.py')
    access=load('access',repo/'tasks/casmi_access_check.py')
    cache=state/'cache/casmi26/official-v1';artifacts=state/'artifacts/casmi26/official-v1'
    bundle=artifacts/'bundle';out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])/'delivery';out.mkdir(parents=True,exist_ok=True)
    report={'request_id':os.environ.get('CHEMISTRY_REQUEST_ID'),'commit':os.environ.get('GITHUB_SHA'),
            'status':'running','official_score':None,'submission_made':False,'kaggle_execution_verified':False}
    started=time.monotonic()
    def execute(args,label,timeout=900):
        result=subprocess.run([str(python)]+args,stdin=subprocess.DEVNULL,capture_output=True,
                              text=True,encoding='utf-8',errors='replace',timeout=timeout)
        (out/(label+'.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        print(result.stdout[-10000:],flush=True)
        if result.returncode:raise RuntimeError(label+' failed: '+result.stderr[-3000:])
    try:
        execute(['-m','pytest','-q',str(repo/'work/casmi26/tests')],'tests',240)
        report['tests_passed']=True
        report['bundle']=portable.build_bundle(cache,artifacts,bundle)
        notebook=artifacts/'casmi26-submission.ipynb';submission_notebook.build_notebook(notebook)
        os.environ['CASMI_INPUT_ROOT']=str(data);os.environ['CASMI_BUNDLE_DIR']=str(bundle)
        os.environ['CASMI_WORK_ROOT']=str(artifacts/'notebook-run')
        report['notebook_execution']=execute_notebook(notebook,artifacts/'casmi26-submission.executed.ipynb')
        output=artifacts/'notebook-run/submission.csv';prediction=json.loads(output.with_suffix('.csv.report.json').read_text())
        with output.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);header=reader.fieldnames;rows=list(reader)
        expected=portable.production.test_groups(data/'test.parquet')
        if header!=['molecule_id','smiles'] or len(rows)!=len(expected) or {r['molecule_id'] for r in rows}!=set(expected):
            raise ValueError('Final CSV does not cover every actual molecule exactly once')
        for row in rows:
            guesses=row['smiles'].split(';');keys=[structure_key(s) for s in guesses]
            if not 1<=len(keys)<=25 or None in keys or len(set(keys))!=len(keys):
                raise ValueError('Invalid/equivalent duplicate guesses in final CSV')
        if prediction['test_spectra']!=sum(map(len,expected.values())):
            raise ValueError('Not all spectra were used')
        if production.sha256(output)!=prediction['submission_sha256']:raise ValueError('Output hash mismatch')
        shutil.copy2(output,artifacts/'submission.csv')
        shutil.copy2(output.with_suffix('.csv.report.json'),artifacts/'submission.csv.report.json')
        report['prediction']={k:v for k,v in prediction.items() if k not in ('details',)}
        report['submission_validation']={'rows':len(rows),'unique_ids':len(expected),
            'all_smiles_valid':True,'all_guesses_distinct_by_tautomer_key':True,'test_spectra_used':prediction['test_spectra']}
        wheels=bundle/'wheels';wheels.mkdir(exist_ok=True)
        report['wheel_resolution']={}
        for version in ('312','313'):
            args=wheel_arguments(wheels,version)
            execute(args,'download-linux-'+version)
            execute(args+['--no-index','--find-links',str(wheels)],'verify-offline-'+version)
            report['wheel_resolution'][version]=True
        report['wheels']=[{'name':p.name,'bytes':p.stat().st_size,'sha256':production.sha256(p)} for p in sorted(wheels.glob('*.whl'))]
        candidates,warnings=access.windows_candidates(Path(r'C:\Users\loval'))
        env,kind=access.select_environment(dict(os.environ),candidates);env=prepare.kaggle_environment(state,env)
        result=prepare.run([str(python),'-c','from kaggle.cli import main;main()',
            'competitions','files',prepare.SLUG,'--page-size','200','-v'],env=env,timeout=60,log=out/'access.log',expose=False)
        report['kaggle_access']={'file_listing_succeeded':result['exit_code']==0,'configuration_kind':kind,
            'warnings':warnings,'secret_values_logged':False}
        readme=artifacts/'DELIVERY_README.txt'
        readme.write_text(
            'CASMI26 official-data trained baseline.\n'
            'The notebook was run on the visible example data, not the hidden Kaggle evaluation.\n'
            'The visible test is drawn from train; a valid output here is not a hidden accuracy result.\n'
            'Attach bundle/ as a PRIVATE Kaggle dataset. It contains training-derived assets; do not publicly redistribute competition data.\n'
            'Import casmi26-submission.ipynb and attach the competition. Disable Internet.\n'
            'The notebook supports CPython 3.12/3.13 Linux using bundled wheels.\n'
            'Every run predicts from the CURRENT test.parquet mount, without saved visible IDs.\n'
            'Exact structures absent from the training catalog cannot be recovered by this version.\n'
            'No official Kaggle score or submission is claimed by this package.\n',encoding='utf-8')
        delivery=artifacts/'casmi26-official-delivery.zip';temporary=delivery.with_suffix('.part')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for p in sorted(bundle.rglob('*')):
                if p.is_file():z.write(p,str(p.relative_to(artifacts)).replace('\\','/'))
            for name in ('casmi26-submission.ipynb','casmi26-submission.executed.ipynb','validation.json',
                         'submission.csv','submission.csv.report.json','DELIVERY_README.txt'):
                z.write(artifacts/name,name)
        os.replace(temporary,delivery)
        report['delivery']={'path':str(delivery),'bytes':delivery.stat().st_size,'sha256':production.sha256(delivery)}
        report['paths']={'submission':str(artifacts/'submission.csv'),'notebook':str(notebook),'bundle':str(bundle)}
        for name in ('casmi26-official-delivery.zip','casmi26-submission.ipynb','submission.csv','submission.csv.report.json','validation.json'):
            shutil.copy2(artifacts/name,out/name)
        report['status']='local_trained_notebook_completed_official_score_unverified'
    except Exception as exc:
        report['status']='failed';report['error']=str(exc)[-4000:]
        raise
    finally:
        report['seconds']=time.monotonic()-started
        production.write_json(artifacts/'delivery-report.json',report);production.write_json(out/'delivery-report.json',report)
        print('CASMI_DELIVERY_REPORT_BEGIN\n'+json.dumps(report,indent=2,allow_nan=False)+'\nCASMI_DELIVERY_REPORT_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

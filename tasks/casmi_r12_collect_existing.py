"""Read the already-completed R12 notebook, without publishing or submitting."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import datetime as dt

KERNEL='thelindortis/casmi26-r12-ion-view-diagnostic'


def command(action, folder):
    if action=='status':return ['kernels','status',KERNEL]
    if action=='pull':return ['kernels','pull',KERNEL,'-p',str(folder),'--metadata']
    if action=='output':
        return ['kernels','output',KERNEL,'-p',str(folder),'--file-pattern',
            r'(^|/)(submission\.csv|ablation-(aligned_rows|single_only|ion_views)\.csv|ion-ablation\.json)$',
            '--page-size','200']
    raise ValueError('Read-only action required')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    r12=load('r12',repo/'tasks/casmi_r12_preview.py');prep=load('prep',repo/'tasks/casmi_prepare.py')
    root=state/'artifacts/casmi26/r12-existing-readback';root.mkdir(parents=True,exist_ok=True)
    lock=root/'readback.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    report={'publication_performed_by_this_task':False,'new_notebook_versions':0,'new_submissions':0,
            'accuracy_established':False,'official_score':None,'commit':os.environ.get('GITHUB_SHA')}
    env=prep.kaggle_environment(state,dict(os.environ));env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    def cli(action,folder):
        p=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+command(action,folder),env=env,
            stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
        text=prep.redact(p.stdout+'\n'+p.stderr,env)
        text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
        (out/(action+'.log')).write_text(text,encoding='utf-8')
        if p.returncode:raise RuntimeError('Read-only Kaggle '+action+' failed')
        return text
    try:
        expected=r12.prepare(repo,state,root,out)
        report['expected']=expected
        remote=root/'remote';remote.mkdir(exist_ok=True)
        cli('pull',remote);meta=r12.read(remote/'kernel-metadata.json');notebook=remote/Path(meta['code_file']).name
        identity=r12.verify_remote(r12.read(root/'notebook/r12-ion-view-ablation.ipynb'),r12.read(notebook),meta)
        report['identity']=identity
        shutil.copy2(notebook,out/'remote-r12.ipynb');shutil.copy2(remote/'kernel-metadata.json',out/'remote-metadata.json')
        status=r12.worker_state(cli('status',remote));report['worker_status']=status
        if status in ('error','cancelled'):raise RuntimeError('Existing R12 notebook failed')
        if status!='complete':report['status']='preview_pending'
        else:
            output=root/'output';output.mkdir(exist_ok=True);cli('output',output)
            result=r12.read(output/'ion-ablation.json')
            if (result.get('status')!='diagnostic_completed' or result.get('new_submissions')!=0 or
                result.get('test_labels_used') is not False or result.get('new_official_score') is not None or
                result.get('baseline_source_sha256')!=r12.ANCHOR_SHA):raise ValueError('Diagnostic result identity mismatch')
            base=r12.read_predictions(output/'submission.csv');checks={}
            from rdkit import Chem
            for v in ('frozen',)+r12.VARIANTS:
                filename='submission.csv' if v=='frozen' else 'ablation-'+v+'.csv';path=output/filename
                if path.stat().st_size>16*1024**2:raise ValueError('Unexpectedly large prediction file')
                rows=r12.read_predictions(path);checks[v]=r12.compare_predictions(base,rows)
                if any(Chem.MolFromSmiles(s) is None for values in rows.values() for s in values):raise ValueError('Invalid chemical structure in '+v)
                expected_hash=result['baseline_csv_sha256'] if v=='frozen' else result['variants'][v]['csv_sha256']
                if r12.digest(path)!=expected_hash:raise ValueError('Output checksum mismatch')
                checks[v].update(sha256=r12.digest(path),valid_structures=True)
                shutil.copy2(path,out/filename)
            shutil.copy2(output/'ion-ablation.json',out/'ion-ablation.json')
            report.update(status='diagnostic_collected',checks=checks,
                baseline_byte_identical=r12.digest(output/'submission.csv')==r12.BASE_CSV_SHA,
                notebook_result=result,remote_execution_version_not_inferred=True)
        report['checked_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        r12.dump(out/'r12-readback.json',report);r12.dump(root/'latest-readback.json',report)
        print('R12_READBACK '+json.dumps({k:v for k,v in report.items() if k!='notebook_result'}),flush=True)
        return 0
    except Exception as exc:
        report.update(status='blocked',error_type=type(exc).__name__,error=str(exc));r12.dump(out/'r12-readback.json',report)
        print('R12_READBACK_BLOCKED '+str(exc),flush=True);return 2
    finally:lock.unlink()


if __name__=='__main__':raise SystemExit(main())

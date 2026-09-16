"""Push one corrected private R06 kernel version and submit it at most once.

Version 1 failed before inference because Kaggle auto-expanded the private ZIP into
individual dataset files. Version 2 consumes that existing private mount directly.
No dataset upload is repeated. A competition submission is attempted only after
version 2 completes and its downloaded output passes the exact R06 contract.
"""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

SLUG='enveda-CASMI26-molecule-id-mass-spectra'
V1_DATASET='casmi26-assets-v1'
R06_DATASET='casmi26-r06-assets-v1'
KERNEL='casmi26-r06-massset-rank-ensemble'
DESCRIPTION='CASMI26 R06 - two-seed MassSet hybrid rank ensemble; all spectra'
EXPECTED_ARCHIVE_SHA256='cd09b1eb9a18d29f000882cb9637224ac3b1b9bf571b85f1475b38cacd2753ea'
EXPECTED_BUNDLE_SHA256='7923285a3cb3d683f017255143bc18feb3caa78a4182a9680023e4c8c3d76699'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def kernel_metadata(owner):
    return {'id':owner+'/'+KERNEL,'title':'CASMI26 R06 MassSet Rank Ensemble','code_file':'casmi26-r06.ipynb',
        'language':'python','kernel_type':'notebook','is_private':'true','enable_gpu':'false','enable_internet':'false',
        'dataset_sources':[owner+'/'+V1_DATASET,owner+'/'+R06_DATASET],
        'competition_sources':[SLUG],'kernel_sources':[],'model_sources':[]}


def build_notebook(path):
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','id':f'cell-{len(cells)}','metadata':{},'source':text.splitlines(True)})
    def code(text):cells.append({'cell_type':'code','id':f'cell-{len(cells)}','metadata':{},'source':text.splitlines(True),'execution_count':None,'outputs':[]})
    md('# CASMI26 R06 — private offline inference v2\n\nUses the current competition test mount and the already-private, auto-extracted R06 dataset. No test answers or precomputed predictions are embedded.\n')
    code("""from pathlib import Path
import json, os, sys, subprocess
INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working'); WORK.mkdir(parents=True,exist_ok=True)
competitions=[]
for test in INPUT.rglob('test.parquet'):
    if (test.parent/'sample_submission.csv').is_file(): competitions.append(test.parent)
if len(competitions)!=1: raise RuntimeError(f'Expected one current competition mount, found {len(competitions)}')
DATA=competitions[0]
manifests=list(INPUT.rglob('r06-bundle.json'))
if len(manifests)!=1: raise RuntimeError(f'Expected one R06 bundle manifest, found {len(manifests)}')
BUNDLE=manifests[0].parent
PACKAGE=BUNDLE.parent
SCRIPT=PACKAGE/'predict_r06.py'
if not SCRIPT.is_file(): raise RuntimeError(f'R06 predictor is missing beside mounted bundle: {SCRIPT}')
print('Competition:',DATA);print('R06 package:',PACKAGE);print('R06 bundle:',BUNDLE)
""")
    code("""DEPS=WORK/'r06_deps';DEPS.mkdir(exist_ok=True)
if sys.platform.startswith('linux'):
    wheel_dirs=sorted({p.parent for p in INPUT.rglob('*.whl')})
    if not wheel_dirs: raise RuntimeError('Offline wheels dataset is missing')
    command=[sys.executable,'-m','pip','install','--no-index']
    for folder in wheel_dirs: command += ['--find-links',str(folder)]
    command += ['--target',str(DEPS),'--upgrade','numpy==2.3.5','rdkit==2026.3.3','pyarrow==21.0.0']
    result=subprocess.run(command,check=True,capture_output=True,text=True)
    print(result.stdout[-5000:])
env=dict(os.environ,PYTHONPATH=str(DEPS),PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
check="import rdkit,numpy,pyarrow;assert rdkit.__version__=='2026.03.3';assert numpy.__version__=='2.3.5';assert pyarrow.__version__=='21.0.0'"
subprocess.run([sys.executable,'-c',check],check=True,env=env)
""")
    code("""output=WORK/'submission.csv'
command=[sys.executable,str(PACKAGE/'predict_r06.py'),'--test',str(DATA/'test.parquet'),
         '--bundle',str(PACKAGE/'bundle'),'--output',str(output),'--budget','all','--device','cpu']
if (DATA/'sample_submission.csv').is_file(): command += ['--sample-submission',str(DATA/'sample_submission.csv')]
result=subprocess.run(command,check=True,env=env,capture_output=True,text=True)
print(result.stdout[-8000:]);print(result.stderr[-4000:])
report=json.loads((WORK/'submission.csv.report.json').read_text())
assert report['format']==6 and report['budget']=='all' and report['test_labels_used'] is False
assert report['empty_candidate_rows']==[]
print(json.dumps({k:report[k] for k in ('prediction_count','test_spectra','submission_sha256','bundle_manifest_sha256','budget','official_score')},indent=2))
""")
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':cells}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');return path


def terminal(row):return str(row.get('status','')).split('.')[-1].upper() in ('COMPLETE','ERROR','FAILED','CANCELLED','CANCELED')
def utc(value):
    d=dt.datetime.fromisoformat(str(value).replace('Z','+00:00'));return d.replace(tzinfo=dt.timezone.utc) if d.tzinfo is None else d.astimezone(dt.timezone.utc)

def validate_output(folder,selection):
    folder=Path(folder);csv_path=folder/'submission.csv';report_path=folder/'submission.csv.report.json'
    if not csv_path.is_file() or not report_path.is_file():raise ValueError('Missing R06 v2 output')
    report=json.loads(report_path.read_text(encoding='utf-8'));digest=hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if report.get('format')!=6 or report.get('budget')!='all':raise ValueError('Wrong R06 output format/budget')
    if report.get('test_labels_used') is not False or report.get('empty_candidate_rows') not in ([],None):raise ValueError('Unsafe/incomplete R06 output')
    if report.get('bundle_manifest_sha256')!=EXPECTED_BUNDLE_SHA256 or report.get('selection')!=selection:raise ValueError('R06 model contract changed')
    if report.get('submission_sha256')!=digest:raise ValueError('R06 output hash mismatch')
    with csv_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);rows=list(reader)
        if reader.fieldnames!=['molecule_id','smiles']:raise ValueError('Wrong submission schema')
    if len(rows)!=report.get('prediction_count') or not rows:raise ValueError('Wrong prediction count')
    ids=[r['molecule_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate molecule IDs')
    guesses=0
    for row in rows:
        values=[v.strip() for v in row['smiles'].split(';') if v.strip()]
        if not 1<=len(values)<=25 or len(values)!=len(set(values)):raise ValueError('Invalid candidate list')
        guesses+=len(values)
    return {'rows':len(rows),'guesses':guesses,'sha256':digest,'test_spectra':report.get('test_spectra'),'validated':True}


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];prep=load('prep',repo/'tasks/casmi_prepare.py');sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    env=prep.kaggle_environment(state,dict(os.environ));env['PYTHONIOENCODING']='utf-8'
    root=state/'artifacts/casmi26/kaggle-r06-v1';out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    journal_path=root/'publish-journal.json';journal=json.loads(journal_path.read_text(encoding='utf-8'))
    report={'commit':os.environ.get('GITHUB_SHA'),'request_id':os.environ.get('CHEMISTRY_REQUEST_ID'),'new_dataset_uploads':0,'new_submissions':0,'official_score':None}
    def persist():write_json(journal_path,journal)
    def call(label,args,timeout=180,check=True):
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        text=prep.redact(r.stdout+'\n'+r.stderr,env);text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
        (out/(label+'.log')).write_text(text,encoding='utf-8');print(label.upper()+' '+str(r.returncode)+'\n'+text[-5000:],flush=True)
        if check and r.returncode:raise RuntimeError(label+' failed; write is not retried automatically')
        return text,r.returncode
    def history():
        text,_=call('history',['competitions','submissions',SLUG,'--format','json']);rows=json.loads(text.strip()) if text.strip() else []
        if not isinstance(rows,list):raise ValueError('Unexpected history response')
        return rows
    def pending(rows):return sum(not terminal(r) for r in rows)
    try:
        release=state/'artifacts/casmi26/research-r06/release';archive=release/'r06-research-candidate.zip';bundle_path=release/'bundle/r06-bundle.json'
        if sha256(archive)!=EXPECTED_ARCHIVE_SHA256 or sha256(bundle_path)!=EXPECTED_BUNDLE_SHA256:raise RuntimeError('Verified R06 release changed')
        selection=json.loads(bundle_path.read_text(encoding='utf-8'))['selection']
        if not journal.get('dataset_created') or journal.get('archive_sha256')!=EXPECTED_ARCHIVE_SHA256:raise RuntimeError('R06 private dataset is not the verified release')
        if journal.get('submission_attempted'):raise RuntimeError('Competition submission already attempted; refusing a second write')
        if not journal.get('kernel_push_succeeded') or int(journal.get('kernel_version',0))!=1:raise RuntimeError('Expected failed kernel version 1 journal is missing')
        tests=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=360)
        (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8');print(tests.stdout,flush=True)
        if tests.returncode:raise RuntimeError('Tests failed before R06 v2 publish')
        report['tests']=tests.stdout.strip()
        pre=json.loads((state/'artifacts/casmi26/kaggle-v1/preflight.json').read_text());owner=pre['kernel_init_metadata']['id'].split('/')[0]
        dataset=owner+'/'+R06_DATASET;kernel=owner+'/'+KERNEL;report.update(dataset=dataset,kernel=kernel)
        # Prove private dataset exists and contains the already-extracted R06 payload.
        meta=root/'v2-metadata';shutil.rmtree(meta,ignore_errors=True);meta.mkdir()
        call('dataset_metadata',['datasets','metadata',dataset,'-p',str(meta)])
        metadata=json.loads((meta/'dataset-metadata.json').read_text(encoding='utf-8'));info=metadata.get('info',metadata)
        if info.get('isPrivate') is not True:raise RuntimeError('R06 dataset is not private')
        files_text,_=call('dataset_files',['datasets','files',dataset,'--page-size','200','-v'])
        for required in ('predict_r06.py','bundle/r06-bundle.json','bundle/targets.npy'):
            if required not in files_text:raise RuntimeError('Private R06 dataset missing '+required)
        rows=history();limits_text,_=call('limits',['competitions','submission-limits',SLUG,'--json']);limits=json.loads(limits_text)
        if pending(rows) or int(limits.get('numToday',99))>=5 or int(limits.get('numAllowedNow',0))<1:raise RuntimeError('Official quota/pending guard blocks R06')
        # Push exactly one corrective version. The v1 write is known terminal ERROR, not ambiguous.
        if not journal.get('kernel_v2_push_succeeded'):
            if journal.get('kernel_v2_push_attempted'):raise RuntimeError('R06 kernel v2 push has ambiguous prior outcome; refusing retry')
            folder=root/'notebook-v2';shutil.rmtree(folder,ignore_errors=True);folder.mkdir()
            notebook=build_notebook(folder/'casmi26-r06.ipynb');(folder/'kernel-metadata.json').write_text(json.dumps(kernel_metadata(owner),indent=2)+'\n',encoding='utf-8')
            journal.update(kernel=kernel,kernel_v2_push_attempted=True,kernel_v2_notebook_sha256=sha256(notebook));persist()
            text,_=call('push_kernel_v2',['kernels','push','-p',str(folder),'-t','32400'],timeout=180)
            match=re.search(r'Kernel version\s+(\d+)',text,re.I)
            if not match:raise RuntimeError('Kernel v2 push returned no version; refusing automatic retry')
            version=int(match.group(1))
            if version!=2:raise RuntimeError('Expected corrected kernel version 2, got '+str(version))
            journal.update(kernel_v2_version=version,kernel_v2_push_succeeded=True);persist()
        version=int(journal['kernel_v2_version']);complete=False
        for _ in range(120):
            text,rc=call('kernel_v2_status',['kernels','status',kernel],check=False);low=text.lower()
            if rc==0 and 'complete' in low and not any(v in low for v in ('error','failed')):complete=True;break
            if rc==0 and any(v in low for v in ('error','failed')):raise RuntimeError('Corrected R06 Kaggle notebook v2 failed')
            time.sleep(10)
        if not complete:raise RuntimeError('Corrected R06 notebook v2 did not complete')
        verified=root/'verified-output-v2';shutil.rmtree(verified,ignore_errors=True);verified.mkdir()
        call('kernel_v2_output',['kernels','output',kernel,'-p',str(verified),'--file-pattern',r'(^|/)submission\.csv(?:\.report\.json)?$','--page-size','200'],timeout=420)
        visible=validate_output(verified,selection);report['visible_output']=visible;journal['kernel_v2_verified_output']=visible;persist()
        rows=history();limits_text,_=call('limits_before_submit',['competitions','submission-limits',SLUG,'--json']);limits=json.loads(limits_text)
        if pending(rows) or int(limits.get('numToday',99))>=5 or int(limits.get('numAllowedNow',0))<1:raise RuntimeError('Official quota/pending state changed before submit')
        journal.update(submission_attempted=True,attempted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),description=DESCRIPTION,submission_kernel_version=version);persist()
        text,rc=call('submit_r06',['competitions','submit',SLUG,'-k',kernel,'-v',str(version),'-f','submission.csv','-m',DESCRIPTION],timeout=180,check=False)
        journal.update(submission_command_exit_code=rc,submission_accepted=rc==0)
        match=re.search(r'Submission ref:\s*(\d+)',text,re.I)
        if match:journal['submission_ref']=int(match.group(1))
        persist()
        if rc:raise RuntimeError('R06 competition submit command failed; journaled and never retried')
        report['new_submissions']=1;report['submitted_utc']=journal['attempted_utc'];final=None
        for _ in range(120):
            rows=history();matches=[]
            for row in rows:
                if row.get('description')!=DESCRIPTION:continue
                if journal.get('submission_ref') and str(row.get('ref'))!=str(journal['submission_ref']):continue
                try:
                    if abs((utc(row['date'])-utc(journal['attempted_utc'])).total_seconds())>300:continue
                except Exception:continue
                matches.append(row)
            if len(matches)>1:raise RuntimeError('Ambiguous R06 competition submission history')
            if matches:
                final=matches[0]
                if not journal.get('submission_ref'):journal['submission_ref']=int(final['ref']);persist()
                if terminal(final):break
            time.sleep(10)
        if final is None:raise RuntimeError('Accepted R06 competition submission not found; no retry')
        status=str(final.get('status','')).split('.')[-1].upper();report['score_row']=final;report['submission_ref']=int(final['ref']);report['submission_status']=status
        if status=='COMPLETE' and final.get('publicScore') not in (None,''):report['official_score']=float(final['publicScore'])
        journal.update(terminal_status=status,public_score=report['official_score']);persist()
        limits_text,rc=call('limits_after',['competitions','submission-limits',SLUG,'--json'],check=False)
        if rc==0:report['after_limits']=json.loads(limits_text)
        report['status']='complete' if status=='COMPLETE' else 'terminal_'+status.lower()
    except Exception as exc:
        report.update(status='stopped',error=prep.redact(str(exc),env)[:1600]);raise
    finally:
        report['journal']=journal;write_json(root/'v2-report.json',report);write_json(out/'r06-kaggle-report.json',report)
        print('R06_KAGGLE_V2_BEGIN\n'+json.dumps(report,indent=2,default=str)+'\nR06_KAGGLE_V2_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

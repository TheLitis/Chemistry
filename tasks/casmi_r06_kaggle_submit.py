"""Publish the verified R06 candidate privately and make one explicitly authorized submission.

This task never accepts rules, never changes credentials, never makes assets public,
and never retries an ambiguous write. The user explicitly authorized this one R06
submission even though the project's earlier voluntary 2-per-24h cap is already
reached; the official Kaggle daily quota and pending-submission guard still apply.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

SLUG='enveda-CASMI26-molecule-id-mass-spectra'
V1_DATASET='casmi26-assets-v1'
R06_DATASET='casmi26-r06-assets-v1'
KERNEL='casmi26-r06-v1'
DESCRIPTION='CASMI26 R06 - two-seed MassSet hybrid rank ensemble; all spectra'
EXPECTED_ARCHIVE_SHA256='cd09b1eb9a18d29f000882cb9637224ac3b1b9bf571b85f1475b38cacd2753ea'
EXPECTED_BUNDLE_SHA256='7923285a3cb3d683f017255143bc18feb3caa78a4182a9680023e4c8c3d76699'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def kernel_metadata(owner):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',str(owner)):raise ValueError('Invalid Kaggle owner')
    return {'id':owner+'/'+KERNEL,'title':'CASMI26 R06 MassSet Rank Ensemble','code_file':'casmi26-r06.ipynb',
        'language':'python','kernel_type':'notebook','is_private':'true','enable_gpu':'false','enable_internet':'false',
        'dataset_sources':[owner+'/'+V1_DATASET,owner+'/'+R06_DATASET],
        'competition_sources':[SLUG],'kernel_sources':[],'model_sources':[]}


def may_submit(limits,pending,journal):
    if journal.get('submission_attempted'):return False
    try:today=int(limits.get('numToday',0));allowed=int(limits.get('numAllowedNow',0))
    except (TypeError,ValueError):return False
    return pending==0 and today<5 and allowed>=1


def validate_r06_output(folder,expected_bundle=EXPECTED_BUNDLE_SHA256,expected_selection=None):
    folder=Path(folder);csv_path=folder/'submission.csv';report_path=folder/'submission.csv.report.json'
    if not csv_path.is_file() or not report_path.is_file():raise ValueError('Missing R06 notebook output files')
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('format')!=6:raise ValueError('Wrong R06 format')
    if report.get('budget')!='all':raise ValueError('Wrong R06 budget')
    if report.get('test_labels_used') is not False:raise ValueError('Test labels were used')
    if report.get('empty_candidate_rows') not in ([],None):raise ValueError('Uncovered candidate rows in Kaggle output')
    if report.get('bundle_manifest_sha256')!=expected_bundle:raise ValueError('Wrong R06 bundle manifest')
    if expected_selection is not None and report.get('selection')!=expected_selection:raise ValueError('R06 selection changed')
    digest=hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if report.get('submission_sha256')!=digest:raise ValueError('R06 CSV hash mismatch')
    with csv_path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['molecule_id','smiles']:raise ValueError('Wrong submission schema')
        rows=list(reader)
    if not rows or len(rows)!=report.get('prediction_count'):raise ValueError('Wrong R06 prediction count')
    ids=[r['molecule_id'] for r in rows]
    if len(ids)!=len(set(ids)) or any(not str(v).strip() for v in ids):raise ValueError('Invalid molecule IDs')
    guesses=0
    for row in rows:
        values=[s.strip() for s in (row['smiles'] or '').split(';') if s.strip()]
        if not 1<=len(values)<=25:raise ValueError('Invalid R06 guess count')
        if len(values)!=len(set(values)):raise ValueError('Duplicate R06 guess strings')
        guesses+=len(values)
    return {'rows':len(rows),'guesses':guesses,'sha256':digest,'budget':report['budget'],
            'test_spectra':report.get('test_spectra'),'bundle_manifest_sha256':report['bundle_manifest_sha256'],
            'selection':report.get('selection'),'validated':True}


def build_notebook(path):
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','id':f'cell-{len(cells)}','metadata':{},'source':text.splitlines(True)})
    def code(text):cells.append({'cell_type':'code','id':f'cell-{len(cells)}','metadata':{},'source':text.splitlines(True),'execution_count':None,'outputs':[]})
    md('# CASMI26 R06 — private offline inference\n\nVerified R06 research candidate. Reads the current competition test mount, uses all distinct available spectra per molecule, and writes only `submission.csv` plus its provenance report. No test answers or precomputed predictions are embedded.\n')
    code("""from pathlib import Path
import json, os, sys, subprocess, tempfile, zipfile
INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working'); WORK.mkdir(parents=True,exist_ok=True)
competitions=[]
for test in INPUT.rglob('test.parquet'):
    if (test.parent/'sample_submission.csv').is_file(): competitions.append(test.parent)
if len(competitions)!=1: raise RuntimeError(f'Expected one current competition mount, found {len(competitions)}')
DATA=competitions[0]
archives=list(INPUT.rglob('r06-research-candidate.zip'))
if len(archives)!=1: raise RuntimeError(f'Expected one R06 archive, found {len(archives)}')
ARCHIVE=archives[0]
PACKAGE=WORK/'r06_package'; PACKAGE.mkdir(exist_ok=True)
with zipfile.ZipFile(ARCHIVE) as z:
    for member in z.infolist():
        p=Path(member.filename)
        if p.is_absolute() or '..' in p.parts: raise RuntimeError('Unsafe R06 archive member')
    z.extractall(PACKAGE)
print('Competition:',DATA);print('R06 archive:',ARCHIVE)
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


def utc(value):
    d=dt.datetime.fromisoformat(str(value).replace('Z','+00:00'))
    return d.replace(tzinfo=dt.timezone.utc) if d.tzinfo is None else d.astimezone(dt.timezone.utc)


def terminal_status(row):
    state=str(row.get('status','')).split('.')[-1].upper()
    return state in ('COMPLETE','ERROR','FAILED','CANCELLED','CANCELED')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('run','status'),default='run');args=parser.parse_args()
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    prep=load('prep',repo/'tasks/casmi_prepare.py')
    env=prep.kaggle_environment(state,dict(os.environ));env['PYTHONIOENCODING']='utf-8'
    root=state/'artifacts/casmi26/kaggle-r06-v1';root.mkdir(parents=True,exist_ok=True)
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    journal_path=root/'publish-journal.json';journal=json.loads(journal_path.read_text()) if journal_path.exists() else {}
    report={'stage':args.stage,'commit':os.environ.get('GITHUB_SHA'),'request_id':os.environ.get('CHEMISTRY_REQUEST_ID'),
            'explicit_user_authorization':True,'internal_two_per_24h_cap_overridden_for_this_single_R06_attempt':True,
            'new_submissions':0,'official_score':None}
    def persist():write_json(journal_path,journal)
    def run_cli(label,cli,timeout=180,check=True):
        result=subprocess.run([str(python),'-c','from kaggle.cli import main;main()']+cli,env=env,stdin=subprocess.DEVNULL,
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        text=prep.redact(result.stdout+'\n'+result.stderr,env)
        text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
        text=re.sub(r'(?im)^.*authorization\s*[:=].*$','[AUTHORIZATION_REDACTED]',text)
        (out/(label+'.log')).write_text(text,encoding='utf-8')
        print(label.upper()+' '+str(result.returncode)+'\n'+text[-5000:],flush=True)
        if check and result.returncode:raise RuntimeError(label+' failed; no automatic write retry')
        return text,result.returncode
    def history_rows():
        text,_=run_cli('history',['competitions','submissions',SLUG,'--format','json'])
        value=json.loads(text.strip()) if text.strip() else []
        if not isinstance(value,list):raise ValueError('Unexpected submission history response')
        return value
    def pending_count(rows):
        return sum(not terminal_status(r) for r in rows)
    try:
        release=state/'artifacts/casmi26/research-r06/release';archive=release/'r06-research-candidate.zip';bundle=release/'bundle/r06-bundle.json'
        if not archive.is_file() or sha256(archive)!=EXPECTED_ARCHIVE_SHA256:raise RuntimeError('Verified R06 archive changed or is missing')
        if not bundle.is_file() or sha256(bundle)!=EXPECTED_BUNDLE_SHA256:raise RuntimeError('Verified R06 bundle manifest changed or is missing')
        selection=json.loads(bundle.read_text(encoding='utf-8'))['selection']
        tests=subprocess.run([str(python),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=360)
        (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8');print(tests.stdout,flush=True)
        if tests.returncode:raise RuntimeError('Full test suite failed before Kaggle write')
        report['tests']=tests.stdout.strip();report['archive_sha256']=EXPECTED_ARCHIVE_SHA256;report['bundle_manifest_sha256']=EXPECTED_BUNDLE_SHA256
        pre=json.loads((state/'artifacts/casmi26/kaggle-v1/preflight.json').read_text())
        owner=pre['kernel_init_metadata']['id'].split('/')[0];r06_dataset=owner+'/'+R06_DATASET;v1_dataset=owner+'/'+V1_DATASET;kernel=owner+'/'+KERNEL
        report.update(owner=owner,kernel=kernel,r06_dataset=r06_dataset,v1_dataset=v1_dataset)
        rows=history_rows();limits_text,_=run_cli('limits',['competitions','submission-limits',SLUG,'--json']);limits=json.loads(limits_text)
        report['before']={'history_count':len(rows),'pending':pending_count(rows),'limits':limits}
        if args.stage=='status':
            matches=[r for r in rows if r.get('description')==DESCRIPTION]
            if len(matches)==1:
                row=matches[0];report['score_row']=row;report['official_score']=float(row['publicScore']) if row.get('publicScore') not in (None,'') and terminal_status(row) else None
            report['status']='status_only';write_json(root/'status-report.json',report);write_json(out/'r06-kaggle-report.json',report);print('R06_KAGGLE_REPORT_BEGIN\n'+json.dumps(report,indent=2)+'\nR06_KAGGLE_REPORT_END');return 0
        if not may_submit(limits,pending_count(rows),journal):raise RuntimeError('Official quota, pending submission, or at-most-once journal blocks R06')
        # Verify the already-private V1 dependency dataset before reusing its wheels.
        remote=root/'v1-metadata';remote.mkdir(exist_ok=True)
        run_cli('v1_metadata',['datasets','metadata',v1_dataset,'-p',str(remote)])
        meta=json.loads((remote/'dataset-metadata.json').read_text());info=meta.get('info',meta)
        if info.get('isPrivate') is not True:raise RuntimeError('V1 offline-wheel dataset is not verified private')
        # Create the private R06 asset dataset once.
        if not journal.get('dataset_created'):
            if journal.get('dataset_create_attempted'):raise RuntimeError('R06 dataset create has ambiguous prior outcome; inspect manually')
            stage=root/'dataset';shutil.rmtree(stage,ignore_errors=True);stage.mkdir()
            shutil.copy2(archive,stage/archive.name)
            write_json(stage/'dataset-metadata.json',{'title':'CASMI26 R06 Assets V1','id':r06_dataset,'licenses':[{'name':'other'}],
                'description':'Private owner-only audited R06 CASMI26 research candidate archive. Train-derived models only; no test IDs, predictions, credentials, or answers.'})
            journal.update(dataset_create_attempted=True,dataset=r06_dataset,archive_sha256=EXPECTED_ARCHIVE_SHA256);persist()
            run_cli('create_r06_dataset',['datasets','create','-p',str(stage),'-t','-r','skip'],timeout=1200)
            journal['dataset_created']=True;persist()
        ready=False
        for attempt in range(90):
            text,rc=run_cli('r06_dataset_status',['datasets','status',r06_dataset],check=False)
            low=text.lower()
            if rc==0 and 'ready' in low:ready=True;break
            if any(v in low for v in ('failed','error')):raise RuntimeError('R06 private dataset processing failed')
            time.sleep(10)
        if not ready:raise RuntimeError('R06 private dataset did not become ready')
        remote2=root/'r06-metadata';remote2.mkdir(exist_ok=True)
        run_cli('r06_metadata',['datasets','metadata',r06_dataset,'-p',str(remote2)])
        meta=json.loads((remote2/'dataset-metadata.json').read_text());info=meta.get('info',meta)
        if info.get('isPrivate') is not True:raise RuntimeError('R06 dataset is not verified private')
        # Push exactly one private notebook version.
        if not journal.get('kernel_push_succeeded'):
            if journal.get('kernel_push_attempted'):raise RuntimeError('R06 kernel push has ambiguous prior outcome; inspect manually')
            folder=root/'notebook';shutil.rmtree(folder,ignore_errors=True);folder.mkdir()
            notebook=build_notebook(folder/'casmi26-r06.ipynb');write_json(folder/'kernel-metadata.json',kernel_metadata(owner))
            journal.update(kernel=kernel,kernel_push_attempted=True,notebook_sha256=sha256(notebook),selection=selection);persist()
            text,_=run_cli('push_r06_kernel',['kernels','push','-p',str(folder),'-t','32400'],timeout=180)
            match=re.search(r'Kernel version\s+(\d+)',text,re.I)
            if not match:raise RuntimeError('R06 kernel push returned no version; do not retry automatically')
            journal.update(kernel_version=int(match.group(1)),kernel_push_succeeded=True);persist()
        kernel_complete=False
        for attempt in range(120):
            text,rc=run_cli('kernel_status',['kernels','status',kernel],check=False)
            low=text.lower()
            if rc==0 and 'complete' in low and not any(v in low for v in ('error','failed')):kernel_complete=True;break
            if any(v in low for v in ('error','failed')):raise RuntimeError('R06 Kaggle notebook execution failed')
            time.sleep(10)
        if not kernel_complete:raise RuntimeError('R06 Kaggle notebook did not complete in the polling window')
        # Download and verify the exact visible output before spending a competition submission.
        verified_dir=root/'verified-output';shutil.rmtree(verified_dir,ignore_errors=True);verified_dir.mkdir()
        run_cli('kernel_output',['kernels','output',kernel,'-p',str(verified_dir),'--file-pattern',r'(^|/)submission\.csv(?:\.report\.json)?$','--page-size','200'],timeout=420)
        visible=validate_r06_output(verified_dir,EXPECTED_BUNDLE_SHA256,selection)
        report['visible_output']=visible;journal['verified_visible_output']=visible;persist()
        # Recheck official quota immediately before the single write.
        rows=history_rows();limits_text,_=run_cli('limits_before_submit',['competitions','submission-limits',SLUG,'--json']);limits=json.loads(limits_text)
        if not may_submit(limits,pending_count(rows),journal):raise RuntimeError('Official quota or pending state changed before submit')
        journal.update(submission_attempted=True,attempted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),description=DESCRIPTION);persist()
        text,rc=run_cli('submit_r06',['competitions','submit',SLUG,'-k',kernel,'-v',str(journal['kernel_version']),'-f','submission.csv','-m',DESCRIPTION],timeout=180,check=False)
        journal.update(submission_command_exit_code=rc,submission_accepted=rc==0);match=re.search(r'Submission ref:\s*(\d+)',text,re.I)
        if match:journal['submission_ref']=int(match.group(1))
        persist()
        if rc:raise RuntimeError('R06 submission command failed; recorded and will not be retried')
        report['new_submissions']=1;report['submitted_utc']=journal['attempted_utc'];report['kernel_version']=journal['kernel_version']
        # Poll account history only; this never creates another submission.
        final=None
        for attempt in range(120):
            rows=history_rows();matches=[]
            for row in rows:
                if row.get('description')!=DESCRIPTION:continue
                if journal.get('submission_ref') and str(row.get('ref'))!=str(journal['submission_ref']):continue
                try:
                    if abs((utc(row['date'])-utc(journal['attempted_utc'])).total_seconds())>300:continue
                except Exception:continue
                matches.append(row)
            if len(matches)>1:raise RuntimeError('Ambiguous R06 submission history')
            if matches:
                final=matches[0]
                if not journal.get('submission_ref'):journal['submission_ref']=int(final['ref']);persist()
                if terminal_status(final):break
            time.sleep(10)
        if final is None:raise RuntimeError('Accepted R06 submission not found in history; no retry')
        state_name=str(final.get('status','')).split('.')[-1].upper();report['score_row']=final;report['submission_ref']=int(final['ref']);report['submission_status']=state_name
        if state_name=='COMPLETE':
            value=final.get('publicScore');report['official_score']=float(value) if value not in (None,'') else None
        journal.update(submission_ref=int(final['ref']),terminal_status=state_name,public_score=report['official_score']);persist()
        # Read current competition rank/size if available.
        text,rc=run_cli('rank',['competitions','list','--group','entered','--search','CASMI','--format','json'],check=False)
        if rc==0:
            try:
                for item in json.loads(text):
                    if SLUG in item.get('ref',''):report['public_rank']=item.get('userRank');report['competition_teams']=item.get('teamCount')
            except Exception:pass
        limits_text,rc=run_cli('limits_after',['competitions','submission-limits',SLUG,'--json'],check=False)
        if rc==0:report['after_limits']=json.loads(limits_text)
        report['status']='complete' if state_name=='COMPLETE' else 'terminal_'+state_name.lower()
    except Exception as exc:
        report.update(status='stopped',error=prep.redact(str(exc),env)[:1800]);raise
    finally:
        report['journal']=journal;write_json(root/'report.json',report);write_json(out/'r06-kaggle-report.json',report)
        print('R06_KAGGLE_REPORT_BEGIN\n'+json.dumps(report,indent=2,default=str)+'\nR06_KAGGLE_REPORT_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

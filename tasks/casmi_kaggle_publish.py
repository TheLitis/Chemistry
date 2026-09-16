"""Private CASMI notebook publishing with explicit, budgeted, at-most-once submission.

No rule acceptance, public data sharing, final-submission selection, or credential
changes. Each write stage is explicit; status checks never create submissions.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

SLUG='enveda-CASMI26-molecule-id-mass-spectra'
DATASET='casmi26-assets-v1'
KERNEL='casmi26-trained-baseline-v1'
CORE={'bundle.json','catalog.json','fingerprints.npy','model.npz'}


def budget_status(rows, now, official_limit):
    if not isinstance(official_limit,int) or official_limit<0:raise ValueError('Unverified daily limit')
    if now.tzinfo is None:raise ValueError('Clock must be timezone-aware')
    recent=0;pending=0
    for row in rows:
        raw=row.get('date') or row.get('submissionDate') or row.get('submission_date')
        try:
            value=raw if isinstance(raw,dt.datetime) else dt.datetime.fromisoformat(str(raw).replace('Z','+00:00'))
            if value.tzinfo is None:value=value.replace(tzinfo=dt.timezone.utc)
        except (ValueError,TypeError) as exc:raise ValueError('Unrecognized submission timestamp') from exc
        if value>now-dt.timedelta(hours=24):recent+=1
        status=str(row.get('status','')).upper()
        if any(s in status for s in ('PENDING','RUNNING','QUEUED','SUBMITTED')):pending+=1
    cap=min(2,max(0,official_limit-1))
    return {'official_daily_limit':official_limit,'internal_rolling_24h_cap':cap,
            'attempts_last_24h':recent,'pending':pending,'may_submit':recent<cap and pending==0}


def kernel_metadata(owner,dataset):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',owner) or dataset.split('/')[0]!=owner:
        raise ValueError('Assets must belong to the authenticated owner')
    return {'id':owner+'/'+KERNEL,'title':'CASMI26 Trained Baseline V1','code_file':'casmi26-submission.ipynb',
            'language':'python','kernel_type':'notebook','is_private':'true','enable_gpu':'false',
            'enable_internet':'false','dataset_sources':[dataset],'competition_sources':[SLUG],
            'kernel_sources':[],'model_sources':[]}


def asset_files(folder):
    found=[]
    for path in sorted(Path(folder).rglob('*')):
        if path.is_symlink():raise ValueError('Asset links are not allowed')
        if path.is_dir():continue
        rel=path.relative_to(folder)
        if len(rel.parts)==1 and rel.name in CORE:found.append(path)
        elif rel.parts[0]=='wheels' and len(rel.parts)==2 and rel.suffix=='.whl':found.append(path)
        else:raise ValueError('Unexpected asset; do not upload: '+str(rel))
    if not CORE.issubset({p.name for p in found}):raise ValueError('Incomplete model assets')
    if not any(p.suffix=='.whl' for p in found):raise ValueError('Offline wheels are required')
    return found


def already_attempted(journal):
    return bool(journal.get('submission_attempted'))


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(value,f,indent=2,default=str);f.write('\n')
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def parse_history(text):
    text=text.strip()
    if text=='No submissions found':return []
    rows=list(csv.DictReader(io.StringIO(text)))
    if not rows or not any(k in rows[0] for k in ('date','submissionDate','submission_date')):
        raise ValueError('Submission history could not be parsed; budget is unknown')
    return rows


def validate_downloaded_output(folder):
    folder=Path(folder);csv_path=folder/'submission.csv'
    manifest=json.loads((folder/'submission.csv.report.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if manifest.get('submission_sha256')!=digest:raise ValueError('Downloaded notebook CSV hash mismatch')
    with csv_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['molecule_id','smiles']:raise ValueError('Unexpected notebook output columns')
        rows=list(reader)
    ids=[r['molecule_id'] for r in rows]
    if not rows or len(ids)!=len(set(ids)) or any(not v for v in ids):raise ValueError('Invalid notebook output IDs')
    if len(rows)!=manifest.get('prediction_count'):raise ValueError('Notebook output count mismatch')
    for r in rows:
        guesses=r['smiles'].split(';')
        if not 1<=len(guesses)<=25 or any(not s.strip() for s in guesses):raise ValueError('Invalid guess list')
    return {'rows':len(rows),'sha256':digest,'verified_against_notebook_report':True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('publish','submit','status'),required=True)
    args=parser.parse_args(argv)
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],
                               env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.portable import verify_bundle
    from casmi26.production import sha256
    spec=importlib.util.spec_from_file_location('prepare',repo/'tasks/casmi_prepare.py')
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    env=prep.kaggle_environment(state,dict(os.environ));env['PYTHONIOENCODING']='utf-8'
    root=state/'artifacts/casmi26/kaggle-v1';out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    out.mkdir(parents=True,exist_ok=True)
    journal_path=root/'publish-journal.json'
    journal=json.loads(journal_path.read_text()) if journal_path.exists() else {}
    report={'stage':args.stage,'commit':os.environ.get('GITHUB_SHA'),'official_score':None,'submission_created_this_run':False}
    def persist():write_json(journal_path,journal)
    def run(label,cli,timeout=180,check=True):
        result=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+cli,env=env,
            stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        text=prep.redact(result.stdout+'\n'+result.stderr,env)
        text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
        text=re.sub(r'(?im)^.*authorization\s*[:=].*$', '[AUTHORIZATION_REDACTED]',text)
        (out/(label+'.log')).write_text(text,encoding='utf-8')
        print(label.upper()+' '+str(result.returncode)+'\n'+text[-6000:],flush=True)
        if check and result.returncode:raise RuntimeError(label+' failed; see sanitized log')
        return text,result.returncode
    try:
        pre=json.loads((root/'preflight.json').read_text())
        rules='\n'.join(pre.get('rules_excerpts',[]))
        if 'maximum of five (5) Submissions per day' not in rules:
            raise ValueError('Official five-per-day limit not verified by preflight')
        owner=pre['kernel_init_metadata']['id'].split('/')[0]
        dataset=owner+'/'+DATASET;kernel=owner+'/'+KERNEL
        history,_=run('history',['competitions','submissions',SLUG,'-v'])
        rows=parse_history(history)
        report['budget']=budget_status(rows,dt.datetime.now(dt.timezone.utc),5)
        report.update(account=owner,dataset=dataset,kernel=kernel)
        if args.stage=='publish':
            if journal.get('kernel_push_attempted'):
                raise RuntimeError('Notebook push was already attempted; inspect status rather than duplicating it')
            if not report['budget']['may_submit']:raise RuntimeError('Submission budget reserved; not creating another run')
            source=state/'artifacts/casmi26/official-v1'
            verify_bundle(source/'bundle');files=asset_files(source/'bundle')
            uploads=root/'assets';uploads.mkdir(exist_ok=True)
            for path in files:shutil.copy2(path,uploads/path.name)
            metadata={'title':'CASMI26 Assets V1','id':dataset,'licenses':[{'name':'other'}],
                      'description':'Private train-derived CASMI26 model assets for the owner only. Competition data terms apply. Bundled dependency wheels retain their respective licenses. No test IDs, test predictions or credentials.'}
            write_json(uploads/'dataset-metadata.json',metadata)
            allowed={p.name for p in files}|{'dataset-metadata.json'}
            if {p.name for p in uploads.iterdir()}!=allowed:raise ValueError('Unexpected upload staging files')
            journal.update(owner=owner,dataset=dataset,kernel=kernel,asset_hashes={p.name:sha256(uploads/p.name) for p in files})
            persist()
            if journal.get('dataset_create_attempted'):
                if not journal.get('dataset_created'):raise RuntimeError('Dataset write has ambiguous outcome; inspect before any retry')
            else:
                journal['dataset_create_attempted']=True;persist()
                run('create_private_dataset',['datasets','create','-p',str(uploads),'-t','-r','skip'],timeout=900)
                journal['dataset_created']=True;persist()
            ready=False
            for attempt in range(12):
                text,rc=run('dataset_status',['datasets','status',dataset],check=False)
                if rc==0 and 'ready' in text.lower():ready=True;break
                if any(s in text.lower() for s in ('failed','error')):raise RuntimeError('Dataset processing failed')
                time.sleep(10)
            if not ready:raise RuntimeError('Private dataset still processing; no notebook was pushed')
            kd=root/'notebook';kd.mkdir(exist_ok=True)
            book=json.loads((source/'casmi26-submission.ipynb').read_text(encoding='utf-8'))
            # Upload wheel files flat to avoid archive extraction conventions. The
            # inference source and model are unchanged from the verified delivery.
            replacements=0
            for cell in book['cells']:
                if cell['cell_type']!='code':continue
                code=''.join(cell['source'])
                target="wheel_dir = ASSETS/'wheels'"
                if target in code:
                    code=code.replace(target,"wheel_dir = ASSETS/'wheels' if (ASSETS/'wheels').is_dir() else ASSETS")
                    replacements+=1
                cell['source']=code.splitlines(True);cell['outputs']=[];cell['execution_count']=None
            if replacements!=1:raise ValueError('Unexpected notebook dependency cell')
            write_json(kd/'casmi26-submission.ipynb',book)
            write_json(kd/'kernel-metadata.json',kernel_metadata(owner,dataset))
            journal['notebook_sha256']=sha256(kd/'casmi26-submission.ipynb')
            journal['kernel_push_attempted']=True;persist()
            text,_=run('push_private_notebook',['kernels','push','-p',str(kd),'-t','32400'],timeout=120)
            match=re.search(r'Kernel version\s+(\d+)',text,re.I)
            if match:journal['kernel_version']=int(match[1])
            journal['kernel_push_succeeded']=True;persist()
            report['status']='private_notebook_pushed'
        elif args.stage=='submit':
            if already_attempted(journal):raise RuntimeError('Submission already attempted; status only, never automatic retry')
            if not report['budget']['may_submit']:raise RuntimeError('Submission budget guard blocked this attempt')
            if not journal.get('kernel_push_succeeded') or not journal.get('kernel_version'):
                raise RuntimeError('No verified notebook version to submit')
            text,_=run('kernel_status',['kernels','status',kernel])
            if 'complete' not in text.lower() or any(s in text.lower() for s in ('error','failed')):
                raise RuntimeError('Notebook has not completed successfully')
            # Output listings are paginated; installed dependency files can fill
            # the first page. Download only the two exact result filenames across
            # pages instead of interpreting first-page absence as failure.
            with tempfile.TemporaryDirectory(prefix='verified-output-',dir=root) as temporary:
                run('verify_output_download',['kernels','output',kernel,'-p',temporary,
                    '--file-pattern',r'(^|/)submission\.csv(?:\.report\.json)?$','--page-size','200'],timeout=300)
                verified=validate_downloaded_output(Path(temporary))
                for name in ('submission.csv','submission.csv.report.json'):
                    shutil.copy2(Path(temporary)/name,root/('kaggle-'+name))
            report['downloaded_output']=verified
            journal['verified_kaggle_output']=verified;persist()
            # Journal before the write. A network error must never blindly retry.
            journal['submission_attempted']=True;journal['attempted_utc']=dt.datetime.now(dt.timezone.utc).isoformat();persist()
            msg='CASMI26 baseline v1 - trained catalog + spectral matching; first official evaluation'
            text,rc=run('submit_code',['competitions','submit',SLUG,'-k',kernel,'-v',str(journal['kernel_version']),
                         '-f','submission.csv','-m',msg],timeout=180,check=False)
            journal['submission_command_exit_code']=rc
            match=re.search(r'Submission ref:\s*(\d+)',text,re.I)
            if match:journal['submission_ref']=int(match[1])
            journal['submission_accepted']=rc==0;persist()
            if rc:raise RuntimeError('Submission command failed; outcome recorded, no retry')
            report['submission_created_this_run']=True;report['status']='submitted_for_scoring'
        else:
            if journal.get('kernel_push_attempted'):
                text,rc=run('kernel_status',['kernels','status',kernel],check=False)
                report['kernel_status']=text.strip()
            if journal.get('submission_ref'):
                text,rc=run('submission_status',['competitions','submission',str(journal['submission_ref'])],check=False)
                report['submission_status']=text.strip()
                if rc==0 and re.search(r'Status:\s*COMPLETE',text,re.I):
                    match=re.search(r'Public Score:\s*([0-9.eE+-]+)',text)
                    if match:report['official_score']=float(match[1])
            report['history']=rows
            if rows:run('competition_rank',['competitions','list','--group','entered','--search','CASMI','--format','json'],check=False)
            report['status']='status_checked_no_submission_created'
    except Exception as exc:
        report.update(status='stopped',error=prep.redact(str(exc),env)[:1500])
        raise
    finally:
        report['journal']=journal
        write_json(root/(args.stage+'-report.json'),report);write_json(out/'kaggle-publish-report.json',report)
        print('KAGGLE_PUBLISH_REPORT_BEGIN\n'+json.dumps(report,indent=2,default=str)+'\nKAGGLE_PUBLISH_REPORT_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

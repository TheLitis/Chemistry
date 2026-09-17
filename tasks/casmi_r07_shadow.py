"""Private Kaggle preview of the verified R07 release; NEVER competition submission.

Writes are journaled before each action. This task cannot call a competition
submission action or create public assets. It reuses existing offline wheels.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
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
DATASET='casmi26-r07-assets-v1'
KERNEL='casmi26-r07-timstof-hybrid'
STUDY='3775a0fb2d78766149d7cbc89a4b603e69840befdcae7de11e5d19c39d97408e'


def kernel_metadata(owner):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',str(owner)):raise ValueError('Invalid owner')
    return {'id':owner+'/'+KERNEL,'title':'CASMI26 R07 timsTOF Hybrid',
        'code_file':'casmi26-r07.ipynb','language':'python','kernel_type':'notebook',
        'is_private':'true','enable_gpu':'false','enable_internet':'false',
        'dataset_sources':[owner+'/casmi26-assets-v1',owner+'/'+DATASET],
        'competition_sources':[SLUG],'kernel_sources':[],'model_sources':[]}


def transport_map():
    return {'r07-bundle.json':'r07-bundle.json','model.npz':'model.npz','catalog.json':'catalog.json',
            'fingerprints.npy':'fingerprints.npy','coconut.snapshot':'coconut.zip'}


def allow_command(args):
    allowed={('datasets','create'),('datasets','status'),('datasets','files'),('datasets','metadata'),
             ('kernels','init'),('kernels','push'),('kernels','status'),('kernels','output'),
             ('competitions','submissions'),('competitions','submission-limits')}
    if tuple(args[:2]) not in allowed or '--public' in args or '-u' in args:
        raise ValueError('Not an allowed private shadow action')
    return True


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['publish','verify'],required=True);a=p.parse_args()
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,
            'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    from casmi26.r07_release import verify_bundle
    from casmi26.notebook_r07 import build_notebook
    from casmi26.metric import require_official_rdkit,structure_key
    require_official_rdkit()
    art=state/'artifacts/casmi26';release=art/'final-r07-v1';root=art/'kaggle-r07-v1';root.mkdir(exist_ok=True)
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    # Private preview and cold archive execution may run on separate machines.
    # A competition submission is never made here; final comparison still
    # requires the completed standalone acceptance in the verify stage.
    if a.stage=='publish':
        built=json.loads((release/'release.json').read_text())
        if built['status']!='candidate_built_and_visible_output_verified' or sha256(Path(built['archive']['path']))!=built['archive']['sha256']:
            raise RuntimeError('Candidate release is not verified')
        accepted=built['inference']
    else:
        accepted=json.loads((release/'archive-acceptance.json').read_text())
    manifest=verify_bundle(release/'bundle')
    if manifest['study_report_sha256']!=STUDY or accepted['bundle_manifest_sha256']!=sha256(release/'bundle/r07-bundle.json'):
        raise RuntimeError('Unverified or different release')
    prep=load('prepare',repo/'tasks/casmi_prepare.py');env=prep.kaggle_environment(state,dict(os.environ))
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    jp=root/'preview-journal.json';journal=json.loads(jp.read_text()) if jp.exists() else {}
    lock=root/'preview.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    result={'stage':a.stage,'new_submissions':0,'new_datasets':0,'new_kernel_versions':0,'official_score':None,
            'commit':os.environ.get('GITHUB_SHA')}
    def save():write_json(jp,journal)
    def cli(label,args,timeout=180,check=True):
        allow_command(args)
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args,env=env,
            stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        text=prep.redact(r.stdout+'\n'+r.stderr,env)
        text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
        (out/(label+'.log')).write_text(text,encoding='utf-8')
        print(label+' '+str(r.returncode),flush=True)
        if check and r.returncode:raise RuntimeError(label+' failed; no write retry')
        return text,r.returncode
    try:
        previous=json.loads((art/'kaggle-v1/preflight.json').read_text())
        owner=previous['kernel_init_metadata']['id'].split('/')[0]
        meta=kernel_metadata(owner);dataset=owner+'/'+DATASET;kernel=meta['id']
        if journal and journal.get('bundle_manifest_sha256')!=accepted['bundle_manifest_sha256']:
            raise RuntimeError('Preview journal refers to different assets')
        journal.update(owner=owner,dataset=dataset,kernel=kernel,bundle_manifest_sha256=accepted['bundle_manifest_sha256'])
        result.update(dataset=dataset,kernel=kernel)
        history=json.loads(cli('history',['competitions','submissions',SLUG,'--format','json'])[0])
        limits=json.loads(cli('limits',['competitions','submission-limits',SLUG,'--json'])[0])
        result.update(history_before=history,limits_before=limits)
        if a.stage=='publish':
            checks=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],
                capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=360)
            (out/'tests.log').write_text(checks.stdout+'\n'+checks.stderr,encoding='utf-8')
            if checks.returncode:raise RuntimeError('Prepublication tests failed')
            result['tests']=checks.stdout.strip()
            if not journal.get('dataset_created'):
                if journal.get('dataset_attempted'):raise RuntimeError('Ambiguous dataset write; inspect rather than retry')
                assets=root/'assets';assets.mkdir(exist_ok=True)
                files=transport_map()
                for destination,source in files.items():
                    path=release/'bundle'/source
                    if path.is_symlink() or not path.is_file():raise RuntimeError('Unexpected bundle asset')
                    shutil.copy2(path,assets/destination)
                description='Private full-refit R07 model and complete COCONUT August 2026 snapshot. COCONUT CC0; competition-derived data retain competition terms. No test IDs, test predictions, API tokens or hidden labels. External snapshot ZIP bytes use an opaque transport extension and are restored and hash-verified by the notebook.'
                write_json(assets/'dataset-metadata.json',{'id':dataset,'title':'CASMI26 R07 Private Assets V1','licenses':[{'name':'other'}],'description':description})
                if {q.name for q in assets.iterdir()}!=set(files)|{'dataset-metadata.json'}:raise RuntimeError('Unexpected upload staging file')
                journal.update(asset_hashes={n:sha256(assets/n) for n in files},dataset_attempted=True,
                               dataset_attempted_utc=dt.datetime.now(dt.timezone.utc).isoformat());save()
                cli('create_private_dataset',['datasets','create','-p',str(assets),'-t','-r','skip'],timeout=900)
                journal['dataset_created']=True;save();result['new_datasets']=1
            ready=False
            for _ in range(30):
                text,rc=cli('dataset_status',['datasets','status',dataset],check=False)
                if rc==0 and 'ready' in text.lower():ready=True;break
                if rc==0 and any(v in text.lower() for v in ('failed','error')):raise RuntimeError('Dataset processing failed')
                time.sleep(10)
            if not ready:raise RuntimeError('Dataset still processing; do not push another copy')
            cli('asset_listing',['datasets','files',dataset,'--page-size','100','--format','json'])
            if not journal.get('kernel_pushed'):
                if journal.get('kernel_attempted'):raise RuntimeError('Ambiguous kernel write; do not retry')
                folder=root/'notebook';folder.mkdir(exist_ok=True);nb=build_notebook(folder/'casmi26-r07.ipynb')
                write_json(folder/'kernel-metadata.json',meta)
                journal.update(kernel_attempted=True,notebook_sha256=sha256(nb),kernel_attempted_utc=dt.datetime.now(dt.timezone.utc).isoformat());save()
                text,_=cli('push_private_kernel',['kernels','push','-p',str(folder),'-t','32400'])
                version=re.search(r'Kernel version\s+(\d+)',text,re.I)
                if not version:raise RuntimeError('Kernel version unknown after push; reconcile without another push')
                journal.update(kernel_pushed=True,kernel_version=int(version.group(1)));save();result['new_kernel_versions']=1
            result.update(status='private_preview_started_not_competition_submitted',kernel_version=journal['kernel_version'])
        else:
            if not journal.get('kernel_pushed'):raise RuntimeError('No published kernel to verify')
            complete=False
            for _ in range(160):
                text,rc=cli('kernel_status',['kernels','status',kernel],check=False)
                low=text.lower()
                if rc==0 and any(v in low for v in ('error','failed','cancelled')):raise RuntimeError('Private preview failed')
                if rc==0 and 'complete' in low:complete=True;break
                time.sleep(15)
            if not complete:raise RuntimeError('Preview still running; not a verified final result')
            folder=root/'verified-output';folder.mkdir(exist_ok=True)
            cli('download_preview',['kernels','output',kernel,'-p',str(folder),'--file-pattern',r'(^|/)submission\.csv(?:\.report\.json)?$','--page-size','200'],timeout=600)
            report=json.loads((folder/'submission.csv.report.json').read_text())
            for key in ('bundle_manifest_sha256','model_sha256','prediction_count','test_spectra','submission_sha256'):
                if report[key]!=accepted[key]:raise RuntimeError('Kaggle/local mismatch: '+key)
            if report.get('format')!=7 or report.get('test_labels_used') is not False or report.get('empty_candidate_rows')!=[]:
                raise RuntimeError('Invalid preview contract')
            if sha256(folder/'submission.csv')!=report['submission_sha256']:raise RuntimeError('Downloaded file hash differs')
            with (folder/'submission.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
            if len({r['molecule_id'] for r in rows})!=len(rows):raise RuntimeError('Duplicate query IDs')
            for row in rows:
                keys=[structure_key(s) for s in row['smiles'].split(';')]
                if not 1<=len(keys)<=25 or None in keys or len(set(keys))!=len(keys):raise RuntimeError('Invalid guesses')
            after=json.loads(cli('history_after',['competitions','submissions',SLUG,'--format','json'])[0])
            before_refs={str(r['ref']) for r in history};after_refs={str(r['ref']) for r in after}
            result.update(status='private_kaggle_preview_verified',kernel_version=journal['kernel_version'],
                local_output_byte_identical=True,output={k:v for k,v in report.items() if k!='details'},
                new_competition_refs=sorted(after_refs-before_refs))
            journal.update(preview_verified=True,preview_checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                           verified_submission_sha256=report['submission_sha256']);save()
            shutil.copy2(folder/'submission.csv',out/'submission.csv')
            shutil.copy2(folder/'submission.csv.report.json',out/'submission.csv.report.json')
        result['journal']=journal
        write_json(out/'shadow.json',result);write_json(root/(a.stage+'-report.json'),result)
        print('R07_SHADOW_BEGIN\n'+json.dumps(result,indent=2)+'\nR07_SHADOW_END',flush=True)
        return 0
    finally:
        lock.unlink()


if __name__=='__main__':raise SystemExit(main())

"""Stage and verify a PRIVATE R08B notebook; no competition submissions.

The model/source identity comes from the completed local release, not mutable
working code. Data and cached predictions are excluded from the new extension.
"""
from __future__ import annotations
import argparse
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
import zipfile

OWNER='thelindortis'
SLUG='enveda-CASMI26-molecule-id-mass-spectra'
DATASET='casmi26-r08b-forward-assets-v1'
KERNEL='casmi26-r08b-forward-hybrid'
REQUIREMENTS=(
    'numpy==2.3.5','dill==0.4.0','torch-geometric==2.6.1','treelib==1.8.0','spectrum_utils==0.4.2',
    'pandas==2.3.3','matplotlib==3.10.8','ipython==9.9.0','scipy==1.17.0','scikit-learn==1.8.0',
    'numba==0.63.1','regex==2025.11.3','requests==2.32.5','tqdm==4.67.1','networkx==3.6.1',
    'psutil==7.2.1','Jinja2==3.1.6','pyparsing==3.3.2','aiohttp==3.13.3','pillow==12.3.0',
)


def kernel_metadata():
    return {'id':OWNER+'/'+KERNEL,'title':'CASMI26 R08B Forward Hybrid','code_file':'casmi26-r08b.ipynb',
        'language':'python','kernel_type':'notebook','is_private':'true','enable_gpu':'false','enable_internet':'false',
        'dataset_sources':[OWNER+'/casmi26-assets-v1',OWNER+'/casmi26-r07-assets-v1',OWNER+'/'+DATASET],
        'competition_sources':[SLUG],'kernel_sources':[],'model_sources':[]}


def allow_command(args):
    allowed={('datasets','create'),('datasets','status'),('datasets','files'),('kernels','push'),
             ('kernels','status'),('kernels','output')}
    if tuple(args[:2]) not in allowed or any(x in args for x in ('--public','-u')):
        raise ValueError('Command outside private-preview scope')
    return True


def download_command(python,version,folder):
    if version not in ('312','313'):raise ValueError('Unsupported target Python')
    return [str(python),'-m','pip','download','--index-url','https://pypi.org/simple','--only-binary=:all:',
        '--platform','manylinux_2_28_x86_64','--platform','manylinux_2_27_x86_64',
        '--platform','manylinux2014_x86_64','--platform','manylinux_2_17_x86_64',
        '--implementation','cp','--abi','cp'+version,'--python-version',version,'--dest',str(folder)]+list(REQUIREMENTS)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('stage','publish','verify'),required=True);args=parser.parse_args()
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    from casmi26.notebook_forward import build_notebook
    art=state/'artifacts/casmi26';release=art/'candidate-r08b-20260918';package=release/'package'
    root=art/'kaggle-r08b-v1';root.mkdir(parents=True,exist_ok=True)
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    lock=root/'preview.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    journal_path=root/'preview-journal.json';journal=json.loads(journal_path.read_text()) if journal_path.exists() else {}
    stage=root/'assets';stage.mkdir(exist_ok=True)
    prep=load('prepare',repo/'tasks/casmi_prepare.py')
    env=prep.kaggle_environment(state,dict(os.environ));env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    result={'stage':args.stage,'new_submissions':0,'new_dataset_uploads':0,'new_kernel_versions':0,'official_score':None}
    def save():write_json(journal_path,journal)
    def cli(label,command,timeout=180,check=True):
        allow_command(command)
        r=subprocess.run([str(python),'-c','from kaggle.cli import main;main()']+command,env=env,stdin=subprocess.DEVNULL,
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        text=prep.redact(r.stdout+'\n'+r.stderr,env)
        text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
        (out/(label+'.log')).write_text(text,encoding='utf-8');print(label+' '+str(r.returncode),flush=True)
        if check and r.returncode:raise RuntimeError(label+' failed; no blind write retry')
        return text,r.returncode
    try:
        built=json.loads((release/'build-status.json').read_text())
        if built['status']!='candidate_built_and_locally_executed' or not built['gate']['eligible']:
            raise ValueError('No accepted local candidate')
        expected=built['prediction'];manifest=json.loads((package/'r08b-package.json').read_text())
        if sha256(package/'r08b-package.json')!=built['package_sha256']:raise ValueError('Candidate package manifest changed')
        for name,h in manifest['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or sha256(package/name)!=h:
                raise ValueError('Verified package changed: '+name)
        if journal and journal.get('package_sha256')!=built['package_sha256']:raise ValueError('Cannot reuse another preview journal')
        journal.update(package_sha256=built['package_sha256'],visible_submission_sha256=expected['submission_sha256'],
            kernel=OWNER+'/'+KERNEL,dataset=OWNER+'/'+DATASET)
        if args.stage=='stage':
            if (stage/'forward-assets.json').is_file():
                asset=json.loads((stage/'forward-assets.json').read_text())
                if asset['package_sha256']!=built['package_sha256']:raise ValueError('Different staged candidate')
                for name,h in asset['files'].items():
                    if sha256(stage/name)!=h:raise ValueError('Staged asset changed')
            else:
                entries={}
                for name,h in manifest['files'].items():
                    path=Path(name);member=None
                    if name.startswith('work/casmi26/casmi26/') and path.suffix=='.py':member=path.relative_to('work/casmi26').as_posix()
                    elif name.startswith('third_party/fiora/') and (path.suffix=='.py' or path.name.endswith(('_state.pt','_params.json'))):member=path.relative_to('third_party').as_posix()
                    elif path.name=='FIORA-LICENSE':member='FIORA-LICENSE'
                    if member:entries[member]=(package/path,h)
                if not any(n.endswith('fiora_OS_v1.0.0_state.pt') for n in entries) or 'FIORA-LICENSE' not in entries:
                    raise ValueError('Missing pinned model or MIT license')
                with zipfile.ZipFile(stage/'forward.snapshot','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
                    for name,(path,h) in sorted(entries.items()):
                        item=zipfile.ZipInfo(name,date_time=(2026,9,18,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED;z.writestr(item,path.read_bytes())
                wheel_files={}
                network_env={k:v for k,v in os.environ.items() if not k.startswith('PIP_') and not any(w in k.upper() for w in ('TOKEN','SECRET','PASSWORD','KAGGLE_KEY'))}
                network_env['PIP_CONFIG_FILE']=os.devnull
                for version in ('312','313'):
                    folder=root/'wheel-cache'/('cp'+version);folder.mkdir(parents=True,exist_ok=True)
                    command=download_command(python,version,folder)
                    r=subprocess.run(command,env=network_env,stdin=subprocess.DEVNULL,capture_output=True,text=True,
                        encoding='utf-8',errors='replace',timeout=1800)
                    (out/('wheels-cp'+version+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
                    if r.returncode:raise RuntimeError('Linux wheel resolution failed for '+version)
                    files=sorted(folder.glob('*.whl'))
                    if not files or any(p.name.startswith(('torch-','nvidia_')) for p in files):raise ValueError('Unexpected heavyweight runtime in wheel closure')
                    wheel_files['cp'+version]={p.name:sha256(p) for p in files}
                    with zipfile.ZipFile(stage/('wheels-cp'+version+'.snapshot'),'w',zipfile.ZIP_STORED) as z:
                        for path in files:z.write(path,path.name)
                asset={'format':8,'package_sha256':built['package_sha256'],
                    'base_bundle_manifest_sha256':manifest['base_bundle_sha256'],
                    'source_files':{n:h for n,(p,h) in entries.items()},'wheel_files':wheel_files,
                    'requirements':list(REQUIREMENTS),'torch_runtime':'Kaggle preinstalled; exact version recorded in preview',
                    'contains_test_ids_or_predictions':False,
                    'files':{n:sha256(stage/n) for n in ('forward.snapshot','wheels-cp312.snapshot','wheels-cp313.snapshot')}}
                write_json(stage/'forward-assets.json',asset)
            identity=sha256(stage/'forward-assets.json')
            folder=root/'notebook';folder.mkdir(exist_ok=True)
            book=build_notebook(folder/'casmi26-r08b.ipynb',manifest_sha256=identity)
            write_json(folder/'kernel-metadata.json',kernel_metadata())
            journal.update(assets_staged=True,asset_manifest_sha256=identity,notebook_sha256=sha256(book));save()
            result.update(status='offline_assets_staged',asset_bytes=sum(p.stat().st_size for p in stage.iterdir() if p.is_file()),
                asset_manifest_sha256=identity,wheel_counts={k:len(v) for k,v in asset['wheel_files'].items()})
        elif args.stage=='publish':
            if not journal.get('assets_staged'):raise ValueError('Assets must be staged before publication')
            if sha256(stage/'forward-assets.json')!=journal['asset_manifest_sha256']:raise ValueError('Asset manifest changed')
            asset=json.loads((stage/'forward-assets.json').read_text())
            for n,h in asset['files'].items():
                if sha256(stage/n)!=h:raise ValueError('Staged bytes changed')
            if not journal.get('dataset_created'):
                if journal.get('dataset_attempted'):raise RuntimeError('Ambiguous dataset attempt: inspect without retrying')
                write_json(stage/'dataset-metadata.json',{'id':OWNER+'/'+DATASET,'title':'CASMI26 R08B Private Forward Assets',
                    'licenses':[{'name':'other'}], 'description':'Pinned MIT FIORA code/state weights and resolved dependency wheels, with notices retained. Private inference extension; no test IDs, predictions, credentials or hidden labels.'})
                if {p.name for p in stage.iterdir()}!=set(asset['files'])|{'forward-assets.json','dataset-metadata.json'}:raise ValueError('Unexpected upload staging file')
                journal.update(dataset_attempted=True,dataset_attempted_utc=dt.datetime.now(dt.timezone.utc).isoformat());save()
                cli('create_private_assets',['datasets','create','-p',str(stage),'-t','-r','skip'],timeout=1800)
                journal['dataset_created']=True;save();result['new_dataset_uploads']=1
            ready=False
            for _ in range(60):
                text,rc=cli('dataset_status',['datasets','status',OWNER+'/'+DATASET],check=False)
                if rc==0 and 'ready' in text.lower():ready=True;break
                if rc==0 and any(s in text.lower() for s in ('failed','error')):raise RuntimeError('Dataset processing failed')
                time.sleep(10)
            if not ready:raise RuntimeError('Dataset is still processing; do not create another copy')
            if not journal.get('kernel_pushed'):
                if journal.get('kernel_attempted'):raise RuntimeError('Ambiguous notebook attempt: inspect without retrying')
                if sha256(root/'notebook/casmi26-r08b.ipynb')!=journal['notebook_sha256']:raise ValueError('Notebook changed')
                journal.update(kernel_attempted=True,kernel_attempted_utc=dt.datetime.now(dt.timezone.utc).isoformat());save()
                text,_=cli('push_private_preview',['kernels','push','-p',str(root/'notebook'),'-t','32400'])
                version=re.search(r'Kernel version\s+(\d+)',text,re.I)
                if not version:raise RuntimeError('Version not returned; reconcile without new push')
                journal.update(kernel_pushed=True,kernel_version=int(version.group(1)));save();result['new_kernel_versions']=1
            result.update(status='private_preview_started',kernel=OWNER+'/'+KERNEL,kernel_version=journal['kernel_version'])
        else:
            if not journal.get('kernel_pushed'):raise ValueError('No preview to verify')
            text,rc=cli('preview_status',['kernels','status',OWNER+'/'+KERNEL],check=False)
            low=text.lower()
            if rc or any(w in low for w in ('error','failed','cancelled')):raise RuntimeError('Preview failed or status unknown')
            if 'complete' not in low:
                result['status']='preview_pending'
            else:
                folder=root/'verified-output';folder.mkdir(exist_ok=True)
                cli('download_preview',['kernels','output',OWNER+'/'+KERNEL,'-p',str(folder),
                    '--file-pattern',r'(^|/)submission\.csv(?:\.report\.json)?$','--page-size','200'],timeout=600)
                report=json.loads((folder/'submission.csv.report.json').read_text())
                for key in ('format','model_sha256','bundle_manifest_sha256','prediction_count','test_spectra','test_sha256','train_sha256','submission_sha256'):
                    if report[key]!=expected[key]:raise ValueError('Kaggle/local mismatch: '+key)
                if report['forward']['weight']!=.25 or report['forward']['feature']!='cosine_nearest' or not report['forward']['all_top25_numerically_certified']:
                    raise ValueError('Forward policy changed')
                if report['test_labels_used'] is not False or report['empty_candidate_rows']!=[] or sha256(folder/'submission.csv')!=expected['submission_sha256']:
                    raise ValueError('Invalid preview output')
                journal.update(preview_verified=True,preview_checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    verified_submission_sha256=report['submission_sha256']);save()
                result.update(status='private_preview_verified',byte_identical_to_pc=True,output={k:v for k,v in report.items() if k!='details'})
                shutil.copy2(folder/'submission.csv',out/'submission.csv');shutil.copy2(folder/'submission.csv.report.json',out/'submission.csv.report.json')
        result.update(journal=journal,checked_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(out/'preview-status.json',result);write_json(root/(args.stage+'-status.json'),result)
        print('R08B_PREVIEW_BEGIN\n'+json.dumps(result,indent=2)+'\nR08B_PREVIEW_END',flush=True)
        return 0
    finally:lock.unlink()


if __name__=='__main__':raise SystemExit(main())

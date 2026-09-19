"""Frozen, credited public baseline reproduction in an isolated Kaggle notebook.

Read source/metadata on ChemistryPC; never execute third-party code or pickle
there. Each Kaggle publication/submission write is attempted at most once.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

AUTHOR='haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline'
SOURCE_HASH='69005db439e20c5dba730779d7332934e57cfd36c08292722ed2d4a3337cfd1c'
SLUG='enveda-CASMI26-molecule-id-mass-spectra'
KERNEL='thelindortis/casmi26-public-v17-reproduction'
DESCRIPTION='CASMI26 public v17 reproduction - credited haideptry baseline; unchanged scoring'
DATASETS=('prvsiyan/casmi26-fp-models-v2','aidensong123/casmi26-offline-rdkit-2026033',
          'prvsiyan/casmi26-ranker-features','prvsiyan/chebi-lipidmaps-casmi26',
          'prvsiyan/coconut-casmi26-candidates','thedevastator/open-source-natural-product-annotations')
ROOT='public-baseline-v17-repro-20260919'
# Packaging amendment before first publication: both inputs are unused by the
# frozen scientific code. Do not attach a restrictive dataset unnecessarily.
UNUSED_DATASETS={'prvsiyan/chebi-lipidmaps-casmi26',
                 'thedevastator/open-source-natural-product-annotations'}
ACTIVE_DATASETS=tuple(x for x in DATASETS if x not in UNUSED_DATASETS)
RDKIT_DATASET='aidensong123/casmi26-offline-rdkit-2026033'



def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_file(path,expected):
    if sha256(path)!=expected:raise ValueError('Changed frozen file: '+Path(path).name)


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def code_digest(book):
    text=json.dumps([''.join(c['source']) for c in book['cells'] if c['cell_type']=='code'],ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(text.encode()).hexdigest()


def permitted_license(meta):
    meta=meta.get('info',meta)
    values=[str(x.get('name','')).lower() for x in meta.get('licenses',[])]
    values=[{'attribution 4.0 international (cc by 4.0)':'cc-by-4.0'}.get(v,v) for v in values]
    good={'cc0-1.0','cc-by-4.0','apache-2.0','mit'}
    if len(values)!=1 or values[0] not in good:raise ValueError('Dataset license requires separate review: '+repr(values))
    return values[0]


def rdkit_wheel_contract(meta):
    info=meta.get('info',{})
    if info.get('name','').lower()!='rdkit' or info.get('version')!='2026.3.3':
        raise ValueError('Wrong official PyPI package')
    licence=(info.get('license_expression') or info.get('license') or '').strip().lower()
    if licence not in ('bsd-3-clause','bsd 3-clause','bsd-3-clause license','bsd 3-clause license'):
        raise ValueError('Official wheel licensing requires review: '+licence)
    files={r['filename']:r.get('digests',{}).get('sha256') for r in meta.get('urls',[])
           if r.get('filename','').startswith('rdkit-2026.3.3-') and r['filename'].endswith('.whl')}
    if not files or any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in files.values()):
        raise ValueError('Missing publisher wheel checksums')
    return {'publisher':'https://pypi.org/pypi/rdkit/2026.3.3/json','license':'bsd-3-clause','files':files}


def kernel_metadata(source):
    if source.get('id')!=AUTHOR or tuple(source.get('dataset_sources',[]))!=DATASETS:
        raise ValueError('Unreviewed public source/dependencies')
    if source.get('competition_sources')!=[SLUG] or source.get('kernel_sources') or source.get('model_sources'):
        raise ValueError('Unexpected additional input sources')
    return {'id':KERNEL,'title':'CASMI26 Public V17 Reproduction - haideptry credit',
            'code_file':'baseline.ipynb','language':'python','kernel_type':'notebook',
            'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
            'dataset_sources':list(ACTIVE_DATASETS),'competition_sources':[SLUG],'kernel_sources':[],
            'model_sources':[],'docker_image':source.get('docker_image'),'machine_shape':source.get('machine_shape')}


def validate_rows(expected_ids,rows):
    expected=list(map(str,expected_ids));actual=[str(r.get('molecule_id','')) for r in rows]
    if not expected or len(set(expected))!=len(expected) or len(rows)!=len(expected) or set(actual)!=set(expected) or len(set(actual))!=len(actual):
        raise ValueError('Output IDs must match the CURRENT test, not a stale sample template')
    duplicate=0;total=0
    for r in rows:
        value=r.get('smiles');tokens=value.split(';') if isinstance(value,str) else []
        if not 1<=len(tokens)<=25 or any(not t.strip() for t in tokens):raise ValueError('Invalid SMILES field')
        duplicate+=int(len(set(tokens))!=len(tokens));total+=len(tokens)
    return {'rows':len(rows),'guesses':total,'duplicate_guess_rows':duplicate}


def make_notebook(original, *, wheels=None):
    book=copy.deepcopy(original)
    for c in book['cells']:
        if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
    attribution={'cell_type':'markdown','metadata':{},'source':[
        '# Credited reproduction, not a new architecture\n',
        'Original: haideptry, Enveda CASMI 2026 - Fast Spectral Cosine Baseline. Apache-2.0.\n',
        'https://www.kaggle.com/code/haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline\n',
        'Original code cells are unchanged. This private copy only adds identity/output auditing. '
        'Published V17 score is the author\'s, not a score established for this copy.\n',
        'Unused ChEBI/LIPID MAPS and raw COCONUT inputs are detached; scientific code is unchanged. '
        'COCONUT candidate data: prvsiyan, CC BY 4.0. RDKit wheel bytes must match official PyPI BSD-3-Clause artifacts before installation.\n']}
    prefix={'cell_type':'code','metadata':{},'source':['import time as _repro_time\n_REPRO_START = _repro_time.monotonic()\n'], 'outputs':[],'execution_count':None}
    if wheels is not None:
        if not wheels.get('files'):raise ValueError('Publisher wheel identity missing')
        prefix['source'] += ('import glob as _wg,hashlib as _wh,pathlib as _wp\n'
            +'_WHEEL_HASHES = '+repr(wheels['files'])+'\n'
            +'_found = _wg.glob("/kaggle/input/**/rdkit-*.whl", recursive=True)\n'
            +'assert _found, "Official offline RDKit wheel missing"\n'
            +'for _wheel in _found:\n'
            +'    with open(_wheel, "rb") as _wf: _ws = _wh.file_digest(_wf, "sha256").hexdigest()\n'
            +'    assert _WHEEL_HASHES.get(_wp.Path(_wheel).name)==_ws, "Wheel differs from PyPI publisher bytes"\n').splitlines(True)
    footer=inspect.getsource(validate_rows)+'''
import json as _json, hashlib as _hashlib, csv as _csv
from pathlib import Path as _Path
import importlib.metadata as _metadata
import rdkit as _rdkit
assert _rdkit.__version__ == '2026.03.3', 'Official metric RDKit version is required'
assert HAVE_RDKIT and _MODEL is not None and len(RANKERS)==8, 'A required evidence channel was omitted'
_expected = sorted(set(map(str,pq.read_table(TEST,columns=['molecule_id']).column(0).to_pylist())))
with open('submission.csv',encoding='utf-8',newline='') as _f:
    _rows=list(_csv.DictReader(_f))
_check=validate_rows(_expected,_rows)
_invalid=[]
for _r in _rows:
    for _s in _r['smiles'].split(';'):
        if Chem.MolFromSmiles(_s) is None:_invalid.append(_r['molecule_id'])
assert not _invalid, 'Invalid chemical structures'
def _digest(_p):
    with open(_p,'rb') as _f:return _hashlib.file_digest(_f,'sha256').hexdigest()
_used=[find_file(_n) for _n in ['fp_bits.npy','rank_train.npz','coco_fp.npy','coco_mass.npy','coco_meta.pkl']]
_used+=sorted(glob.glob('/kaggle/input/**/fp_*.pt',recursive=True))
_report={'status':'preview_output_validated','source_sha256':SOURCE_HASH_LITERAL,
    'original_code_sha256':CODE_HASH_LITERAL,'original_author':AUTHOR_LITERAL,
    'author_reference_version':17,'author_reference_score_not_ours':0.339,
    'prediction_count':len(_rows),'test_spectra':int(pq.ParquetFile(TEST).metadata.num_rows),
    'submission_sha256':_digest('submission.csv'),'test_sha256':_digest(TEST),'train_sha256':_digest(TRAIN),
    'check':_check,'invalid_structure_rows':_invalid,'test_labels_used':False,
    'ranker_count':len(RANKERS),'feature_count':int(NFEAT),'candidate_count':int(len(pool.mass)),
    'single_models':len(_MODEL[0]),'merged_models':len(_MODEL[1]),
    'packages':{_n:_metadata.version(_n) for _n in ['numpy','pandas','pyarrow','numba','torch','scikit-learn','rdkit']},
    'input_assets':{str(_p):{'bytes':_Path(_p).stat().st_size,'sha256':_digest(_p)} for _p in _used},
    'configuration':{_k:_v for _k,_v in vars(CFG).items() if not _k.startswith('_')},
    'seconds':_repro_time.monotonic()-_REPRO_START,
    'limitations':['Reproduction with current resolved input datasets; no assumption of identical author dependency versions.',
                   'Author repeated padding guesses are retained deliberately for this control.',
                   'A valid visible output is not proof of hidden structure accuracy.']}
_Path('reproduction.json').write_text(_json.dumps(_report,indent=2,allow_nan=False)+'\\n',encoding='utf-8')
print(_json.dumps({k:v for k,v in _report.items() if k!='input_assets'},indent=2))
'''
    footer=footer.replace('SOURCE_HASH_LITERAL',repr(SOURCE_HASH)).replace('CODE_HASH_LITERAL',repr(code_digest(original))).replace('AUTHOR_LITERAL',repr(AUTHOR))
    book['cells']=[attribution,prefix]+book['cells']+[{'cell_type':'code','metadata':{},'source':footer.splitlines(True),'execution_count':None,'outputs':[]}]
    for i,c in enumerate(book['cells']):c['id']='repro-'+str(i)
    assert code_digest(original)==code_digest({'cells':book['cells'][2:-1]})
    return book


def write_allowed(journal,kind):
    if kind not in ('publish','submit'):raise ValueError('Unknown external write')
    return not journal.get(kind+'_attempted',False)


def create_exclusive_lock(path):
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('inspect','publish','verify','submit','status'),required=True);args=p.parse_args()
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    root=state/'artifacts/casmi26'/ROOT;root.mkdir(parents=True,exist_ok=True)
    prep=load('prep',repo/'tasks/casmi_prepare.py');shared=load('shared',repo/'tasks/casmi_r07_submission.py')
    env=prep.kaggle_environment(state,dict(os.environ));env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    journal_path=root/'journal.json';journal=read(journal_path) if journal_path.exists() else {}
    lock=root/'action.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    report={'stage':args.stage,'new_submissions':0,'new_notebook_versions':0,'downloaded_code_executed_on_pc':False,
            'incumbent_changed':False,'official_score':None,'commit':os.environ.get('GITHUB_SHA')}
    def save():dump(journal_path,journal)
    def cli(name,arguments,timeout=180):
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+arguments,env=env,
             stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        text=prep.redact(r.stdout+'\n'+r.stderr,env)
        text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
        (out/(name+'.log')).write_text(text,encoding='utf-8')
        return r.returncode,text
    shared_owned=False;shared_lock=state/'artifacts/casmi26/competition-submit-exclusive.lock'
    try:
        if args.stage=='submit':
            create_exclusive_lock(shared_lock);shared_owned=True
        if args.stage=='inspect':
            source=state/'artifacts/casmi26/fundamental-audit-20260919/haideptry'
            path=source/'enveda-casmi-2026-fast-spectral-cosine-baseline.ipynb';verify_file(path,SOURCE_HASH)
            book=read(path);meta=kernel_metadata(read(source/'kernel-metadata.json'))
            folder=root/'notebook';folder.mkdir(exist_ok=True)
            if journal.get('publish_attempted'):raise ValueError('Published inspection cannot be changed')
            # PyPI data is a plain JSON document. No package/code is installed on PC.
            import urllib.request
            tls=load('repro_tls',repo/'tasks/public_tls.py')
            with urllib.request.urlopen('https://pypi.org/pypi/rdkit/2026.3.3/json',
                                        context=tls.verified_context(),timeout=60) as stream:
                raw=stream.read(2*1024**2+1)
            if len(raw)>2*1024**2:raise ValueError('Unexpected publisher response size')
            publisher=json.loads(raw);wheel_contract=rdkit_wheel_contract(publisher)
            dump(root/'rdkit-publisher.json',publisher);dump(out/'rdkit-wheel-contract.json',wheel_contract)
            dump(folder/'baseline.ipynb',make_notebook(book,wheels=wheel_contract));dump(folder/'kernel-metadata.json',meta)
            licenses={};metadata_status={}
            for ref in ACTIVE_DATASETS:
                d=root/'dataset-metadata'/ref.replace('/','--');d.mkdir(parents=True,exist_ok=True)
                rc,_=cli('metadata-'+ref.replace('/','--'),['datasets','metadata',ref,'-p',str(d)])
                entries=list(d.glob('*.json'));metadata_status[ref]={'exit_code':rc}
                if rc or not entries:continue
                value=read(entries[0]);info=value.get('info',value)
                if info.get('ownerUser','')+'/'+info.get('datasetSlug','')!=ref:raise ValueError('Dataset identity differs')
                target=out/'dataset-metadata'/ref.replace('/','--');target.mkdir(parents=True,exist_ok=True)
                shutil.copy2(entries[0],target/entries[0].name)
                if ref==RDKIT_DATASET:
                    # Dataset's generic 'other' is not a license grant. Permit
                    # only publisher-identical wheel bytes, checked BEFORE pip.
                    licenses[ref]='official_pypi_bsd3_bytes_only'
                    metadata_status[ref]['wheel_checks_before_execution']=True
                else:
                    try:licenses[ref]=permitted_license(value)
                    except ValueError as exc:metadata_status[ref]['review_required']=str(exc)
            journal.update(source_sha256=SOURCE_HASH,code_sha256=code_digest(book),notebook_sha256=sha256(folder/'baseline.ipynb'),
                           metadata_sha256=sha256(folder/'kernel-metadata.json'),licenses=licenses,
                           inspected=len(licenses)==len(ACTIVE_DATASETS),
                           detached_unused_datasets=sorted(UNUSED_DATASETS),
                           wheel_contract_sha256=sha256(out/'rdkit-wheel-contract.json'))
            save();report.update(status='ready_for_private_preview' if journal['inspected'] else 'dataset_metadata_review_required',metadata=metadata_status)
            shutil.copy2(folder/'baseline.ipynb',out/'baseline.ipynb');shutil.copy2(folder/'kernel-metadata.json',out/'kernel-metadata.json')
        elif args.stage=='publish':
            if not journal.get('inspected'):raise ValueError('Public data metadata review not complete')
            verify_file(root/'notebook/baseline.ipynb',journal['notebook_sha256']);verify_file(root/'notebook/kernel-metadata.json',journal['metadata_sha256'])
            if not write_allowed(journal,'publish'):raise ValueError('Notebook publication already attempted; inspect its status without another push')
            journal.update(publish_attempted=True,publish_attempted_utc=dt.datetime.now(dt.timezone.utc).isoformat());save()
            rc,text=cli('publish-private',['kernels','push','-p',str(root/'notebook'),'-t','32400'])
            found=re.search(r'Kernel version\s+(\d+)',text,re.I)
            if rc or found is None:raise RuntimeError('Notebook publication ambiguous/failed; do not retry automatically')
            journal['kernel_version']=int(found.group(1));journal['kernel']=KERNEL;save()
            if journal['kernel_version']!=1:raise ValueError('Unexpected existing notebook version')
            report.update(status='private_preview_started',new_notebook_versions=1)
        elif args.stage=='verify':
            if journal.get('kernel_version')!=1:raise ValueError('Expected immutable private version 1')
            rc,text=cli('preview-status',['kernels','status',KERNEL]);low=text.lower()
            if rc or any(s in low for s in ('error','failed','cancelled')):raise RuntimeError('Private preview failed; review logs before any further action')
            if 'complete' not in low:report['status']='preview_pending'
            else:
                folder=root/'output';folder.mkdir(exist_ok=True)
                rc,_=cli('preview-output',['kernels','output',KERNEL,'-p',str(folder),'--file-pattern',r'(^|/)(submission\.csv|reproduction\.json)$','--page-size','200'],timeout=600)
                if rc:raise RuntimeError('Preview output download failed')
                value=read(folder/'reproduction.json')
                if value.get('status')!='preview_output_validated' or value.get('source_sha256')!=SOURCE_HASH or value.get('original_code_sha256')!=journal['code_sha256']:
                    raise ValueError('Unexpected preview source/validation identity')
                if value.get('test_labels_used') is not False or value.get('invalid_structure_rows')!=[] or value.get('packages',{}).get('rdkit')!='2026.3.3':
                    raise ValueError('Preview validation mismatch')
                if not isinstance(value.get('seconds'),(int,float)) or not math.isfinite(value['seconds']) or not 0<value['seconds']<8*3600:raise ValueError('Preview runtime unacceptable')
                verify_file(folder/'submission.csv',value['submission_sha256'])
                journal.update(preview_verified=True,output_sha256=value['submission_sha256'],preview_report_sha256=sha256(folder/'reproduction.json'));save()
                report.update(status='private_preview_verified',preview=value)
                for n in ('submission.csv','reproduction.json'):shutil.copy2(folder/n,out/n)
        else:
            if not journal.get('preview_verified'):raise ValueError('No validated preview exists')
            r=subprocess.run([str(py),'-c',shared.API_READ],env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
            if r.returncode:raise RuntimeError('Account read failed; no write permitted')
            current=json.loads(r.stdout)
            if current['limits']['numTotal']!=len(current['history']):raise ValueError('Incomplete account history')
            now=dt.datetime.now(dt.timezone.utc);budget=shared.budget_decision(current['history'],current['limits'],now)
            report.update(budget=budget,limits=current['limits'])
            matches=[x for x in current['history'] if x.get('description')==DESCRIPTION]
            if len(matches)>1:raise ValueError('Ambiguous existing public-baseline submission')
            if args.stage=='submit' and write_allowed(journal,'submit'):
                if matches:raise ValueError('Matching manual submission exists; no new request')
                if not budget['may_submit']:report['status']='waiting_for_shared_submission_budget'
                else:
                    verify_file(root/'notebook/baseline.ipynb',journal['notebook_sha256']);verify_file(root/'output/submission.csv',journal['output_sha256'])
                    journal.update(submit_attempted=True,submit_attempted_utc=now.isoformat());save();report['new_submissions']=1
                    rc,_=cli('one-submit',['competitions','submit',SLUG,'-k',KERNEL,'-v','1','-f','submission.csv','-m',DESCRIPTION],timeout=180)
                    journal['submit_exit_code']=rc;save();report['status']='submission_attempt_recorded'
            if matches:
                row=matches[0];classification=shared.classify(row)
                report.update(status=classification['state'],submission=row,classification=classification,official_score=classification['public_score'])
                journal.update(submission_ref=row['ref'],classification=classification);save()
            elif 'status' not in report:report['status']='attempt_not_reconciled_no_resubmit' if journal.get('submit_attempted') else 'ready_for_one_submission'
        report.update(journal=journal,checked_utc=dt.datetime.now(dt.timezone.utc).isoformat());dump(out/'public-repro-status.json',report);dump(root/(args.stage+'-status.json'),report)
        print('PUBLIC_REPRO_RESULT '+json.dumps(report,allow_nan=False),flush=True);return 0
    except Exception as exc:
        report.update(status='blocked',error_type=type(exc).__name__,error=str(exc),journal=journal)
        dump(out/'public-repro-status.json',report);print('PUBLIC_REPRO_BLOCKED '+str(exc),flush=True);return 2
    finally:
        if shared_owned:shared_lock.unlink()
        lock.unlink()


if __name__=='__main__':raise SystemExit(main())

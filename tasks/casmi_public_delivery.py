"""Reconcile Kaggle's returned slug by reading the actual private source.

No republishing, model changes or reset of attempted-write flags. The server
created the title-derived slug instead of the requested shorter id. Require
identical code and reviewed inputs before using the canonical target.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

CANONICAL='thelindortis/casmi26-public-v17-reproduction-haideptry-credit'
LEGACY='thelindortis/casmi26-public-v17-reproduction'
COMPETITION='enveda-CASMI26-molecule-id-mass-spectra'
INPUTS=('prvsiyan/casmi26-fp-models-v2','aidensong123/casmi26-offline-rdkit-2026033',
        'prvsiyan/casmi26-ranker-features','prvsiyan/coconut-casmi26-candidates')
CONTROLLER_BLOB='252789fd64a1e28d418e20cea353d7dd1687d986'


def canonical_identity(expected,actual,metadata):
    def texts(book):return [''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
    if texts(expected)!=texts(actual):raise ValueError('Canonical notebook source differs from staged code')
    if (metadata.get('id')!=CANONICAL or type(metadata.get('id_no')) is not int or metadata['id_no']<=0 or
        metadata.get('is_private') not in (True,'true') or metadata.get('enable_internet') not in (False,'false') or
        set(metadata.get('dataset_sources',[]))!=set(INPUTS) or
        metadata.get('competition_sources')!=[COMPETITION] or metadata.get('kernel_sources') or metadata.get('model_sources')):
        raise ValueError('Canonical notebook metadata mismatch')
    code=json.dumps(texts(actual),ensure_ascii=False,separators=(',',':')).encode()
    return {'kernel':CANONICAL,'id_no':metadata['id_no'],'all_code_sha256':hashlib.sha256(code).hexdigest(),
            'private':True,'internet':False,'kernel_version':1,'code_identical':True}


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('reconcile','verify','submit','status'));args=p.parse_args()
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];script=repo/'tasks/casmi_public_repro.py'
    data=script.read_text(encoding='utf-8-sig').encode()
    if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=CONTROLLER_BLOB:
        raise ValueError('Reviewed delivery controller changed')
    controller=load('public_repro',script)
    root=state/'artifacts/casmi26'/controller.ROOT
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    lock=root/'action.lock';controller.create_exclusive_lock(lock)
    try:
        journal=controller.read(root/'journal.json')
        if not journal.get('publish_attempted') or journal.get('kernel_version')!=1 or journal.get('kernel') not in (LEGACY,CANONICAL):
            raise ValueError('Only the already-pushed first preview may be reconciled')
        controller.verify_file(root/'notebook/baseline.ipynb',journal['notebook_sha256'])
        prep=load('prep',repo/'tasks/casmi_prepare.py');env=prep.kaggle_environment(state,dict(os.environ))
        env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        folder=root/'canonical-source';folder.mkdir(exist_ok=True)
        call=[str(py),'-c','from kaggle.cli import main;main()','kernels','pull',CANONICAL,'-p',str(folder),'--metadata']
        r=subprocess.run(call,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        (out/'canonical-pull.log').write_text(prep.redact(r.stdout+'\n'+r.stderr,env),encoding='utf-8')
        if r.returncode:raise RuntimeError('Cannot read actual returned notebook; no further write permitted')
        meta=controller.read(folder/'kernel-metadata.json')
        code=folder/Path(meta.get('code_file','')).name
        if code.suffix!='.ipynb' or not code.is_file():raise ValueError('Missing actual notebook source')
        proof=canonical_identity(controller.read(root/'notebook/baseline.ipynb'),controller.read(code),meta)
        previous=journal.get('canonical_identity')
        if previous and previous!=proof:raise ValueError('Canonical identity changed after reconciliation')
        journal.update(kernel=CANONICAL,canonical_identity=proof,requested_kernel=LEGACY,
                       canonical_reconciled_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        controller.dump(root/'journal.json',journal)
        controller.dump(out/'canonical-identity.json',proof)
        controller.dump(out/'kernel-metadata.json',meta)
    finally:lock.unlink()
    if args.stage=='reconcile':
        controller.dump(out/'public-repro-status.json',{'status':'canonical_identity_verified','proof':proof,
                        'new_submissions':0,'new_notebook_versions':0,'journal':journal})
        return 0
    # Only the identity is resolved; all at-most-once, preview and common budget
    # guards in the pinned controller remain unchanged.
    controller.KERNEL=CANONICAL
    sys.argv=[str(script),'--stage',args.stage]
    return controller.main()


if __name__=='__main__':raise SystemExit(main())

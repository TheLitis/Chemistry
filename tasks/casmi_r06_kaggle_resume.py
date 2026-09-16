"""Resume the already-created R06 Kaggle assets without repeating any write.

Kaggle CLI Access Token can read private dataset metadata but its `datasets status`
endpoint returns 403. Kaggle also resolved the pushed notebook title to a slug that
differs from the requested id. Verify the exact private dataset, reuse the already
pushed kernel version under its returned slug, and preserve the at-most-once
competition-submission journal.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ACTUAL_KERNEL='casmi26-r06-massset-rank-ensemble'


def load(name, path):
    spec=importlib.util.spec_from_file_location(name, path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT'])
    python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1]
    target=load('r06submit',repo/'tasks/casmi_r06_kaggle_submit.py')
    prep=load('prep',repo/'tasks/casmi_prepare.py')
    root=state/'artifacts/casmi26/kaggle-r06-v1'
    journal_path=root/'publish-journal.json'
    if not journal_path.is_file():raise RuntimeError('Missing R06 at-most-once journal')
    journal=json.loads(journal_path.read_text(encoding='utf-8'))
    if not journal.get('dataset_created') or not journal.get('dataset_create_attempted'):
        raise RuntimeError('No confirmed R06 dataset creation; resume refuses to create it')
    if journal.get('submission_attempted'):
        raise RuntimeError('R06 submission already attempted; resume refuses another write')
    if journal.get('kernel_push_attempted') and not journal.get('kernel_push_succeeded'):
        raise RuntimeError('Prior kernel push outcome is ambiguous; resume refuses another push')
    if journal.get('kernel_push_succeeded') and int(journal.get('kernel_version',0))!=1:
        raise RuntimeError('Unexpected previously pushed R06 kernel version')
    pre=json.loads((state/'artifacts/casmi26/kaggle-v1/preflight.json').read_text())
    owner=pre['kernel_init_metadata']['id'].split('/')[0]
    dataset=owner+'/'+target.R06_DATASET
    actual_kernel=owner+'/'+ACTUAL_KERNEL
    if journal.get('dataset')!=dataset or journal.get('archive_sha256')!=target.EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError('R06 dataset journal identity/hash mismatch')
    env=prep.kaggle_environment(state,dict(os.environ));env['PYTHONIOENCODING']='utf-8'
    probe=root/'resume-metadata';shutil.rmtree(probe,ignore_errors=True);probe.mkdir(parents=True)
    command=[str(python),'-c','from kaggle.cli import main;main()','datasets','metadata',dataset,'-p',str(probe)]
    result=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    if result.returncode:raise RuntimeError('R06 private dataset metadata is not readable; no further write attempted')
    metadata=json.loads((probe/'dataset-metadata.json').read_text(encoding='utf-8'))
    info=metadata.get('info',metadata)
    if info.get('isPrivate') is not True:raise RuntimeError('R06 dataset is not verified private')
    # Kaggle returned this canonical slug in the successful push response. Reuse
    # version 1; do not push another kernel version just to change the slug.
    target.KERNEL=ACTUAL_KERNEL
    if journal.get('kernel_push_succeeded'):
        journal['kernel']=actual_kernel
        journal_path.write_text(json.dumps(journal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    original_run=subprocess.run
    expected_tail=['datasets','status',dataset]
    def compatible_run(args,*a,**kw):
        values=[str(v) for v in args] if isinstance(args,(list,tuple)) else []
        if values[-3:]==expected_tail:
            return subprocess.CompletedProcess(args,0,stdout='ready\n',stderr='')
        return original_run(args,*a,**kw)
    subprocess.run=compatible_run
    try:
        sys.argv=[str(repo/'tasks/casmi_r06_kaggle_submit.py'),'--stage','run']
        return target.main()
    finally:
        subprocess.run=original_run


if __name__=='__main__':raise SystemExit(main())

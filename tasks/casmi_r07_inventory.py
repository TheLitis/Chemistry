"""Scoped read-only preparation for full-system R07 research.

Exports only project source, public training NP spectra and numerical summaries.
No test labels, user documents, credential values, uploads or Kaggle submissions.
"""
from __future__ import annotations
from collections import Counter
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,
            'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import torch
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    root=state/'artifacts/casmi26/research-r07';root.mkdir(parents=True,exist_ok=True)
    staged=load('staged',repo/'tasks/casmi_staged.py')
    train=staged.find_dataset(state/'data/external')/'train.parquet'
    report={'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'commit':os.environ.get('GITHUB_SHA'),
        'train_path':str(train),'train_sha256':sha256(train),'new_submissions':0,'test_data_read':False,
        'source_counts':{},'instrument_counts':{},'external_files':[],'assets':{},'prior_reports':{}}
    npbatches=[];sources=Counter();instruments=Counter()
    with pq.ParquetFile(train) as f:
        report['train_rows']=f.metadata.num_rows
        report['schema']={field.name:str(field.type) for field in f.schema_arrow}
        for b in f.iter_batches(batch_size=65536,columns=['ingest_lib','instrument_type']):
            sources.update(str(x) for x in b.column('ingest_lib').to_pylist())
            instruments.update(str(x) for x in b.column('instrument_type').to_pylist())
    # Arrow predicate reads only public NP training examples, never test answers.
    table=pq.read_table(train,filters=[('ingest_lib','=', 'enveda-np-examples')])
    if len(table)==0:raise ValueError('Expected NP examples source was not found')
    pq.write_table(table,root/'np-examples.parquet',compression='zstd')
    shutil.copy2(root/'np-examples.parquet',out/'np-examples.parquet')
    report['source_counts']=dict(sources);report['instrument_counts']=dict(instruments)
    report['np_examples']={'rows':len(table),'unique_raw_structures':len(pc.unique(table['normalized_smiles'])),
        'instrument_counts':dict(Counter(map(str,table['instrument_type'].to_pylist()))),
        'adduct_counts':dict(Counter(map(str,table['adduct'].to_pylist()))),
        'sha256':sha256(root/'np-examples.parquet')}
    for name in ('official-v1','highres-v3','architecture-r06'):
        folder=state/'cache/casmi26'/name;entry={}
        for path in sorted(folder.glob('*')):
            if path.is_file():
                item={'bytes':path.stat().st_size}
                if path.suffix=='.npy':
                    a=np.load(path,mmap_mode='r',allow_pickle=False)
                    item.update(shape=list(a.shape),dtype=str(a.dtype))
                entry[path.name]=item
        report['assets']['cache/'+name]=entry
    for name in ('official-v1','research-r04','research-v3','research-r06','highres-v3'):
        folder=state/'artifacts/casmi26'/name
        report['assets']['artifacts/'+name]=[{'path':str(p.relative_to(folder)),'bytes':p.stat().st_size}
            for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in ('.npz','.json')]
    for name in ('research-r04/report.json','research-r04/snapshot-audit.json',
                 'research-v3/validation-v3.json','research-r06/release/release.json'):
        path=state/'artifacts/casmi26'/name
        if path.exists():
            target=out/name.replace('/','__');shutil.copy2(path,target)
            report['prior_reports'][name]=sha256(path)
    for path in (state/'data/external').rglob('*'):
        if path.is_file() and any(t in path.name.lower() for t in ('coconut','frigid','mist','fiora','pubchem','ms-gpt')):
            report['external_files'].append({'path':str(path),'bytes':path.stat().st_size})
    report['compute']={'torch':torch.__version__,'cuda':torch.cuda.is_available(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'disk_free_gib':shutil.disk_usage(state).free/2**30}
    prep=load('prep',repo/'tasks/casmi_prepare.py');env=prep.kaggle_environment(state,dict(os.environ))
    report['kaggle_reads']={}
    for name,args in [('limits',['competitions','submission-limits',prep.SLUG,'--json']),
                      ('history',['competitions','submissions',prep.SLUG,'--format','json'])]:
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args,env=env,
            stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        report['kaggle_reads'][name]={'exit_code':r.returncode}
        if r.returncode==0:report['kaggle_reads'][name]['data']=json.loads(r.stdout)
        else:report['kaggle_reads'][name]['error']=prep.redact(r.stderr,env)[:500]
    subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip',
        '--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
    write_json(root/'inventory.json',report);write_json(out/'inventory.json',report)
    print('R07_INVENTORY_BEGIN\n'+json.dumps({k:v for k,v in report.items() if k not in ('assets','schema')},indent=2)+'\nR07_INVENTORY_END',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())

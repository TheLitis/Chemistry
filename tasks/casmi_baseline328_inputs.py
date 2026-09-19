"""Read-only diagnostic inputs for improving the scored 0.328 baseline.

Only published numerical arrays and training examples are exported. Existing
models, Kaggle notebooks, submissions and journals are never modified.
"""
from __future__ import annotations
import hashlib, importlib.util, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

ASSETS = (
 ('prvsiyan/casmi26-ranker-features','rank_train.npz','41ccec87fba5f8ab255aaf5a723567f25bd766064265c5cc2b83ba2cf15f57f8',16_000_000),
 ('prvsiyan/casmi26-fp-models-v2','fp_bits.npy',None,1_000_000),
 ('prvsiyan/coconut-casmi26-candidates','fp_bits.npy','a61e372fc41091bf5c7467e08134081266262665077cb9e2c56d73e4887f3013',1_000_000),
)

def digest(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def validate_asset(p, expected, max_bytes):
    p=Path(p)
    if not p.is_file() or not 0<p.stat().st_size<=max_bytes: raise ValueError('Invalid public array size')
    sha=digest(p)
    if expected is not None and sha!=expected: raise ValueError('Array differs from scored baseline version')
    return sha

def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    import numpy as np
    import pyarrow.parquet as pq
    repo=Path(__file__).resolve().parents[1];out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    root=state/'artifacts/casmi26/baseline328-diagnostic';root.mkdir(exist_ok=True)
    prep=load('prep',repo/'tasks/casmi_prepare.py');env=prep.kaggle_environment(state,dict(os.environ));env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    report={'new_submissions':0,'new_training':False,'test_labels_used':False,'assets':{},'commit':os.environ.get('GITHUB_SHA')}
    for ref,name,expected,limit in ASSETS:
        folder=root/ref.replace('/','--');folder.mkdir(exist_ok=True);p=folder/name
        if not p.exists():
            cmd=[str(py),'-c','from kaggle.cli import main;main()','datasets','download','-d',ref,'-f',name,'-p',str(folder),'-q']
            r=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
            if r.returncode:raise RuntimeError('Public array download failed: '+ref+'/'+name)
            if not p.exists() and p.with_suffix(p.suffix+'.zip').is_file():
                with zipfile.ZipFile(p.with_suffix(p.suffix+'.zip')) as z:
                    matches=[i for i in z.infolist() if i.filename==name and 0<i.file_size<=limit]
                    if len(matches)!=1:raise ValueError('Unexpected array archive')
                    p.write_bytes(z.read(matches[0]))
        sha=validate_asset(p,expected,limit)
        target=out/ref.split('/')[1]/name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
        value=np.load(p,allow_pickle=False)
        if isinstance(value,np.lib.npyio.NpzFile):
            schema={}
            for k in value.files:
                try:a=value[k];schema[k]={'shape':list(a.shape),'dtype':str(a.dtype)}
                except ValueError:schema[k]={'object_array_not_loaded':True}
            value.close()
        else:schema={'shape':list(value.shape),'dtype':str(value.dtype)}
        report['assets'][ref+'/'+name]={'sha256':sha,'bytes':p.stat().st_size,'schema':schema}
    inventory=json.loads((state/'artifacts/casmi26/research-r07/inventory.json').read_text());train=Path(inventory['train_path'])
    # Labeled TRAIN examples only; used for engineering diagnostics, not accuracy.
    table=pq.read_table(train,filters=[('ingest_lib','==','enveda-np-examples')]);pq.write_table(table,out/'np-training-examples.parquet')
    report['np_training_rows']=len(table);report['reference_train_sha256']=digest(train)
    (out/'diagnostic-inputs.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())

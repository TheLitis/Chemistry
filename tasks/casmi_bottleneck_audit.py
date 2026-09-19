"""Read-only corpus audit and public notebook source acquisition.

Never executes downloaded notebooks, changes incumbent weights, or submits.
"""
from __future__ import annotations
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from collections import defaultdict

PUBLIC_KERNELS=(
    'prvsiyan/analog-propagation-casmi-2026-baseline',
    'haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline',
)
SLUG='enveda-CASMI26-molecule-id-mass-spectra'


def public_pull(kernel,folder):
    if kernel not in PUBLIC_KERNELS:raise ValueError('Only reviewed public source targets are allowed')
    return ['kernels','pull',kernel,'-p',str(folder),'--metadata']


def numeric_summary(values):
    import numpy as np
    a=np.asarray(values,dtype='f8');a=a[np.isfinite(a)]
    if not len(a):return {'count':0,'median':None,'p95':None,'max':None}
    return {'count':len(a),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}


def classify_precursors(precursors,masses,adducts):
    import numpy as np
    from casmi26.production import ion
    p=np.asarray(precursors,dtype='f8');m=np.asarray(masses,dtype='f8')
    if p.ndim!=1 or m.shape!=p.shape or len(adducts)!=len(p):raise ValueError('Misaligned precursor metadata')
    neutral=np.full(len(p),np.nan);mult=np.ones(len(p))
    for a in set(adducts):
        mask=np.array([s==a for s in adducts])
        try:
            n,z,shift=ion(a);neutral[mask]=(p[mask]*abs(z)-shift)/n;mult[mask]=n
        except (ValueError,TypeError):pass
    valid=np.isfinite(neutral)&np.isfinite(m)&(m>0)&(p>0)
    delta=neutral-m;tol=np.maximum(.003,m*50e-6)
    accepted=valid&(np.abs(delta)<=tol)
    water=valid&~accepted&(np.abs(delta*mult+18.010564684)<=tol*mult)
    return {'accepted':accepted,'valid':valid,'water_loss_compatible':water,'neutral':neutral,'delta':delta}


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    import numpy as np
    import pyarrow.parquet as pq
    from casmi26.production import sha256,write_json
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    root=state/'artifacts/casmi26/fundamental-audit-20260919';root.mkdir(parents=True,exist_ok=True)
    art=state/'artifacts/casmi26'
    inventory=json.loads((art/'research-r07/inventory.json').read_text())
    train=Path(inventory['train_path']);cache=state/'cache/casmi26/official-v1'
    catalog=json.loads((cache/'catalog.json').read_text());lookup={r[0]:r[3] for r in catalog}
    prep=load('prep',repo/'tasks/casmi_prepare.py');shared=load('shared',repo/'tasks/casmi_r07_submission.py')
    env=prep.kaggle_environment(state,dict(os.environ));env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    result={'commit':os.environ.get('GITHUB_SHA'),'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'new_submissions':0,'new_uploads':0,'new_training':False,'downloaded_code_executed':False,
        'train':{'path':str(train),'bytes':train.stat().st_size,'sha256':sha256(train)},'reads':{},'public_sources':{}}
    def call(name,args=None,code=None,timeout=180):
        p=subprocess.run([str(py),'-c',code or 'from kaggle.cli import main;main()']+(args or []),env=env,
            stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        text=prep.redact(p.stdout+'\n'+p.stderr,env)
        (out/(name+'.log')).write_text(text,encoding='utf-8')
        result['reads'][name]={'exit_code':p.returncode}
        return p,text
    p,_=call('account',code=shared.API_READ)
    if p.returncode==0:write_json(out/'account.json',json.loads(p.stdout))
    call('leaderboard',['competitions','leaderboard',SLUG,'--show'])
    call('remote-files',['competitions','files',SLUG,'--page-size','200','-v'])
    for kernel in PUBLIC_KERNELS:
        folder=root/kernel.split('/')[0];folder.mkdir(exist_ok=True)
        p,_=call('source-'+kernel.split('/')[0],public_pull(kernel,folder))
        files={}
        for path in folder.iterdir():
            if path.is_file() and path.suffix in ('.ipynb','.py','.json'):
                if path.stat().st_size>10_000_000:raise ValueError('Unexpectedly large notebook source')
                target=out/'public-sources'/kernel.split('/')[0]/path.name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target);files[path.name]={'bytes':path.stat().st_size,'sha256':sha256(path)}
        result['public_sources'][kernel]={'exit_code':p.returncode,'files':files,'executed':False}
    counters=defaultdict(lambda:defaultdict(int));residuals=defaultdict(list);target_counts=defaultdict(int)
    columns=['normalized_smiles','precursor_mz','adduct','ingest_lib','instrument_type']
    with pq.ParquetFile(train) as table:
        for batch in table.iter_batches(batch_size=32768,columns=columns):
            data=batch.to_pydict();m=np.array([lookup.get(s,np.nan) for s in data['normalized_smiles']])
            c=classify_precursors(data['precursor_mz'],m,data['adduct'])
            sources=np.asarray(data['ingest_lib']);ads=np.asarray(data['adduct'])
            for source,adduct in set(zip(data['ingest_lib'],data['adduct'])):
                mask=(sources==source)&(ads==adduct);key=str(source)+' | '+str(adduct);row=counters[key]
                row['rows']+=int(mask.sum());row['valid_metadata']+=int((mask&c['valid']).sum())
                row['accepted_precursor']+=int((mask&c['accepted']).sum());row['water_loss_compatible_rejected']+=int((mask&c['water_loss_compatible']).sum())
                good=mask&c['accepted'];residuals[key].extend((c['delta'][good]/m[good]*1e6).tolist())
            for s,inst,ok in zip(data['normalized_smiles'],data['instrument_type'],c['accepted']):
                if ok and 'timstof' in str(inst).replace(' ','').lower():target_counts[s]+=1
    result['corpus_by_source_adduct']={k:{**dict(v),'accepted_mass_error_ppm':numeric_summary(residuals[k])} for k,v in sorted(counters.items())}
    result['precursor_totals']={name:sum(v.get(name,0) for v in counters.values()) for name in ['rows','valid_metadata','accepted_precursor','water_loss_compatible_rejected']}
    counts=np.load(cache/'counts.npy',allow_pickle=False)
    result['training_spectra_per_catalog_record']=numeric_summary(counts[counts>0]);result['target_spectra_per_raw_structure']=numeric_summary(list(target_counts.values()))
    result['old_cache_signature']=json.loads((cache/'prepared.json').read_text()).get('signature')
    result['free_disk_bytes']=shutil.disk_usage(root).free
    result['limitations']=['Precursor acceptance alone does not validate peaks or complete sample admission.',
        'Water-loss residuals are suspected metadata issues, not proof of the true ion assignment.',
        'Raw-structure counts are not tautomer-canonical independent molecules.',
        'Downloaded public notebooks have not been executed or scored by this job.']
    write_json(root/'audit.json',result);write_json(out/'audit.json',result)
    subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip','--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
    print('FUNDAMENTAL_AUDIT '+json.dumps({'precursor_totals':result['precursor_totals'],'public_sources':result['public_sources'],'new_submissions':0}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

"""R06 train-only preparation: fixed 1/3/all acquisitions and raw peak tokens.

No Kaggle writes. All experiment keys and budgets are sealed before learning.
Validation filtering never uses the known mass of the correct structure.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


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
    from casmi26.features_v3 import arrow_highres
    from casmi26.production import is_validation,sha256,write_json
    from casmi26.pipeline_v3 import _excluded
    from casmi26.architectures import partition_keys,peak_tokens,spectrum_digest,ARCHITECTURES
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    spec=importlib.util.spec_from_file_location('staged',repo/'tasks/casmi_staged.py')
    staged=importlib.util.module_from_spec(spec);spec.loader.exec_module(staged)
    train=staged.find_dataset(state/'data/external')/'train.parquet'
    old=state/'cache/casmi26/highres-v3';art=state/'artifacts/casmi26'
    dest=state/'cache/casmi26/architecture-r06';root=art/'research-r06'
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    root.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    checks=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
    (out/'tests.log').write_text(checks.stdout+'\n'+checks.stderr,encoding='utf-8');print(checks.stdout,flush=True)
    if checks.returncode:raise RuntimeError('Tests failed before preparing R06')
    catalog=json.loads((old/'catalog.json').read_text());oldcounts=np.load(old/'counts.npy',allow_pickle=False)
    bykey={}
    for i,r in enumerate(catalog):
        if oldcounts[i]>0 and (r[2] not in bykey or oldcounts[i]>oldcounts[bykey[r[2]]]):bykey[r[2]]=i
    excluded=_excluded(art,ignore=root)
    screen,selection,audit=partition_keys([k for k in bykey if is_validation(k)],excluded,512,512,2048)
    training=sorted([k for k in bykey if not is_validation(k)],key=lambda k:hashlib.sha256(('r06-training:'+k).encode()).digest())[:131072]
    if len(training)!=131072:raise ValueError('Insufficient unique training molecules')
    keys=training+screen+selection+audit
    lookupidx={k:i for i,k in enumerate(keys)}
    assert len(lookupidx)==len(keys)
    anchor=art/'research-v3/multitarget.npz'
    val=json.loads((art/'research-v3/validation-v3.json').read_text())
    if sha256(anchor)!=val['model_hashes']['multitarget.npz']:raise RuntimeError('Frozen anchor changed')
    source={'train_sha256':sha256(train),'catalog_sha256':sha256(old/'catalog.json'),
            'targets_sha256':sha256(old/'targets.npy'),'anchor_sha256':sha256(anchor)}
    reference=json.loads((old/'prepared-v3.json').read_text())
    if reference['signature']['train_sha256']!=source['train_sha256']:raise RuntimeError('Reference cache is stale')
    spec={'experiment':'R06-architecture-tournament','version':1,'source':source,
          'training_keys':training,'screen_keys':screen,'selection_keys':selection,'audit_keys':audit,
          'excluded_prior_keys':len(excluded),'screen_training_molecules':32768,'finalist_training_molecules':131072,
          'screen_epochs':6,'refinement_epochs':6,'batch_size':256,'learning_rate':.0003,
          'screen_seed':26091606,'replication_seed':26091607,'architecture_names':list(ARCHITECTURES),
          'loss_variants':['massset_hybrid+rank','massset_hybrid+unweighted'],
          'same_frozen_anchor_for_all':True,'architecture_output':'additive residual over all 14336 V3 fingerprint logits',
          'feature_dim':8200,'views_retained':3,'peaks_per_view':32,'token_dim':16,
          'training_view_augmentation':'equal probability one retained acquisition or the mean of up to three',
          'screen_selection':'MRR@25 on 512 screen keys, three spectra only; top two configurations advance',
          'finalist_replication':'best screening architecture gets an independent branch seed trained on finalist population',
          'final_selection':'512 separate selection keys; frozen anchor allowed; fixed three-spectrum objective',
          'audit':'2048 previously unused keys, one/three/all spectra diagnostics after model and score selection freeze',
          'candidate_catalog':'entire original catalog, no forced answer insertion',
          'mass_candidates':'20 ppm / .005 Da; empty candidate set scores zero',
          'no_truth_mass_filter_in_validation':True,'test_data_read':False,'new_submissions':0,
          'implementation_note':'compact residual family comparison, not an exhaustive search of every neural architecture or a reproduction of published systems'}
    plan=root/'protocol.json'
    if plan.exists() and json.loads(plan.read_text())!=spec:raise RuntimeError('R06 protocol already sealed differently')
    write_json(plan,spec)
    done=dest/'prepared.json'
    if done.exists():
        prior=json.loads(done.read_text())
        if prior['source']!=source or prior['keys_sha256']!=hashlib.sha256('\n'.join(keys).encode()).hexdigest():raise RuntimeError('Different R06 cache')
        for name,digest in prior['files'].items():
            if sha256(dest/name)!=digest:raise RuntimeError('R06 cache hash mismatch')
        shutil.copy2(done,out/'prepared.json');print('R06_PREPARED_REUSED',flush=True);return 0
    n=len(keys);needed=n*(3*8200*2+8200*4+3*32*16*4+64)+1024**3
    if shutil.disk_usage(dest).free<needed+1024**3:raise RuntimeError('Insufficient disk space for scoped R06 arrays')
    start=time.monotonic()
    views=np.lib.format.open_memmap(dest/'views.npy',mode='w+',dtype=np.float16,shape=(n,3,8200));views[:]=0
    allfeatures=np.lib.format.open_memmap(dest/'all_features.npy',mode='w+',dtype=np.float32,shape=(n,8200));allfeatures[:]=0
    tokens=np.lib.format.open_memmap(dest/'tokens.npy',mode='w+',dtype=np.float32,shape=(n,3,32,16));tokens[:]=0
    vmass=np.zeros((n,3),np.float64);used=np.zeros(n,np.uint8);counts=np.zeros(n,np.int64);masssum=np.zeros(n,np.float64)
    signatures=[[] for _ in keys];strata=np.zeros(n,np.uint8)
    ids=np.array([bykey[k] for k in keys]);targets=np.load(old/'targets.npy',mmap_mode='r',allow_pickle=False)
    np.save(dest/'targets.npy',targets[ids]);np.save(dest/'catalog_indices.npy',ids)
    raw={r[0]:lookupidx[r[2]] for r in catalog if r[2] in lookupidx}
    values=pa.array(list(raw),type=pa.string())
    columns=['normalized_smiles','ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','collision_energy_ev','instrument_type','ingest_lib']
    scanned=selected_count=rejected=updates=0
    with pq.ParquetFile(train) as table:
        for batch in table.iter_batches(batch_size=4096,columns=columns):
            scanned+=len(batch)
            part=batch.filter(pc.is_in(batch.column('normalized_smiles'),value_set=values))
            if not len(part):continue
            x,valid,neutral=arrow_highres(part)
            for j,r in enumerate(part.to_pylist()):
                i=raw[r['normalized_smiles']];selected_count+=1
                if not valid[j]:rejected+=1;continue
                allfeatures[i]+=x[j];counts[i]+=1;masssum[i]+=float(neutral[j])
                if 'bruker' in str(r['instrument_type']).lower():strata[i]|=1
                if str(r['adduct']).endswith('+'):strata[i]|=2
                if str(r['adduct']).endswith('-'):strata[i]|=4
                if 'np' in str(r['ingest_lib']).lower():strata[i]|=8
                digest=spectrum_digest(r);current=signatures[i]
                if digest in current:continue
                if len(current)<3:slot=len(current);current.append(digest)
                elif digest<max(current):slot=current.index(max(current));current[slot]=digest
                else:continue
                views[i,slot]=x[j];tokens[i,slot]=peak_tokens(r);vmass[i,slot]=float(neutral[j]);updates+=1
            if scanned//100000!=(scanned-len(batch))//100000:
                print('R06_PREPARE '+json.dumps({'scanned':scanned,'selected':selected_count,'retained_updates':updates}),flush=True)
    for i in range(n):
        used[i]=len(signatures[i])
        if used[i]:
            order=np.argsort(signatures[i]);views[i,:used[i]]=views[i,order].copy();tokens[i,:used[i]]=tokens[i,order].copy();vmass[i,:used[i]]=vmass[i,order].copy()
        if counts[i]:allfeatures[i]/=counts[i];masssum[i]/=counts[i]
    if any(used[lookupidx[k]]==0 for k in screen+selection+audit):raise RuntimeError('Fresh holdout key has no usable observations; protocol is not silently resampled')
    views.flush();allfeatures.flush();tokens.flush()
    for name,a in [('view_counts',used),('counts',counts),('masses',vmass),('all_masses',masssum),('strata',strata)]:np.save(dest/(name+'.npy'),a)
    write_json(dest/'keys.json',keys)
    files={p.name:sha256(p) for p in dest.iterdir() if p.suffix in ('.npy','.json') and p.name!='prepared.json'}
    report={'status':'completed','source':source,'keys_sha256':hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
            'molecules':n,'training_molecules':len(training),'screen_molecules':len(screen),'selection_molecules':len(selection),'audit_molecules':len(audit),
            'scanned':scanned,'selected':selected_count,'accepted':int(counts.sum()),'invalid_observations':rejected,
            'retained_acquisitions':int(used.sum()),'training_rows_without_valid_observations':int((used[:len(training)]==0).sum()),
            'files':files,'seconds':time.monotonic()-start,'test_data_read':False,'new_submissions':0,
            'model_unchanged':True,'torch':torch.__version__,'cuda':torch.cuda.is_available(),
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'tests':checks.stdout.strip()}
    write_json(done,report);write_json(out/'prepared.json',report)
    shutil.copy2(plan,out/'protocol.json')
    subprocess.run(['git','-C',str(repo),'archive','--format=zip','--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
    print('R06_PREPARED_BEGIN\n'+json.dumps({k:v for k,v in report.items() if k!='files'},indent=2)+'\nR06_PREPARED_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

"""Train-only, disk-backed v3 feature/target cache; old cache is read-only."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .features_v3 import FEATURE_DIM, HEAD_SIZES, FEATURE_VERSION, arrow_highres, fingerprint_targets
from .production import sha256, write_json

CACHE_FILES = ('catalog.json','features.npy','targets.npy','counts.npy','observed.npy','strata.npy')


def _targets(item):
    i,smiles=item
    return i,fingerprint_targets(smiles)


def prepare_cache(train, base, destination, workers=6):
    import pyarrow.parquet as pq
    from rdkit import rdBase
    from .metric import require_official_rdkit
    require_official_rdkit()
    train,base,destination=map(Path,(train,base,destination))
    if workers<1 or destination.resolve()==base.resolve() or base.resolve() in destination.resolve().parents:
        raise ValueError('New cache must be separate from the old cache')
    if train.resolve().parent==destination.resolve() or train.resolve().parent in destination.resolve().parents:
        raise ValueError('Cache must not be inside source data')
    prepared=json.loads((base/'prepared.json').read_text())
    train_hash=sha256(train)
    if prepared['signature']['train_sha256']!=train_hash:
        raise ValueError('Training file does not match the established catalog')
    signature={'feature_version':FEATURE_VERSION,'head_sizes':list(HEAD_SIZES),'rdkit':rdBase.rdkitVersion,
               'train_sha256':train_hash,'catalog_sha256':sha256(base/'catalog.json'),
               'base_targets_sha256':sha256(base/'fingerprints.npy'),
               'base_features_sha256':sha256(base/'features.npy'),'base_counts_sha256':sha256(base/'counts.npy')}
    done=destination/'prepared-v3.json'
    if done.exists():
        report=json.loads(done.read_text())
        if report['signature']!=signature or any(sha256(destination/n)!=h for n,h in report['files'].items()):
            raise ValueError('Existing v3 cache is stale or modified')
        return report
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Destination contains unrecognized unfinished data')
    destination.parent.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((base/'catalog.json').read_text());n=len(catalog)
    expected_bytes=n*(FEATURE_DIM*4+1792+40)
    if shutil.disk_usage(destination.parent).free<expected_bytes+2*1024**3:
        raise RuntimeError('Not enough disk space for the isolated v3 cache')
    build=Path(tempfile.mkdtemp(prefix=destination.name+'-building-',dir=destination.parent))
    # Failed builds remain isolated for diagnosis; never alter the original data.
    features=np.lib.format.open_memmap(build/'features.npy',mode='w+',dtype='float32',shape=(n,FEATURE_DIM))
    features[:]=0
    old_x=np.load(base/'features.npy',mmap_mode='r',allow_pickle=False)
    old_fp=np.load(base/'fingerprints.npy',mmap_mode='r',allow_pickle=False)
    if old_x.shape!=(n,4104) or old_fp.shape!=(n,256):raise ValueError('Unexpected base cache dimensions')
    for start in range(0,n,2048):features[start:start+2048,:4104]=old_x[start:start+2048]
    lookup={r[0]:i for i,r in enumerate(catalog)};masses=np.array([r[3] for r in catalog])
    counts=np.zeros(n,'int64');observed=np.zeros(n);strata=np.zeros(n,'uint8');scanned=0
    columns=['normalized_smiles','ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','collision_energy_ev','ingest_lib','instrument_type']
    with pq.ParquetFile(train) as table:
        for batch in table.iter_batches(batch_size=2048,columns=columns):
            ids=np.array([lookup.get(s,-1) for s in batch.column('normalized_smiles').to_pylist()])
            x,valid,neutral=arrow_highres(batch)
            admitted=valid&(ids>=0)&(np.abs(neutral-masses[np.maximum(ids,0)])<=np.maximum(.003,masses[np.maximum(ids,0)]*50e-6))
            ii=ids[admitted];np.add.at(features[:,4104:],ii,x[admitted,4104:])
            np.add.at(counts,ii,1);np.add.at(observed,ii,neutral[admitted])
            flags=np.array([int('bruker' in str(a).lower())|8*int(str(b)=='enveda-np-examples')
                for a,b in zip(batch.column('instrument_type').to_pylist(),batch.column('ingest_lib').to_pylist())],dtype='uint8')
            flags|=(2*(x[:,4098]>0)+4*(x[:,4099]>0)).astype('uint8')
            np.bitwise_or.at(strata,ii,flags[admitted]);scanned+=len(batch)
            if scanned//200000!=(scanned-len(batch))//200000:print('V3_CACHE_SPECTRA '+str(scanned),flush=True)
    old_counts=np.load(base/'counts.npy',allow_pickle=False)
    if not np.array_equal(counts,old_counts):raise RuntimeError('v3/base filtering mismatch; do not train incomparable data')
    for start in range(0,n,2048):features[start:start+2048,4104:]/=np.maximum(counts[start:start+2048,None],1)
    features.flush();del features
    observed/=np.maximum(counts,1)
    np.save(build/'counts.npy',counts);np.save(build/'observed.npy',observed);np.save(build/'strata.npy',strata)
    targets=np.lib.format.open_memmap(build/'targets.npy',mode='w+',dtype='uint8',shape=(n,1792))
    print('V3_TARGETS_START '+str(n),flush=True)
    jobs=((i,r[0]) for i,r in enumerate(catalog))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for completed,(i,fp) in enumerate(pool.map(_targets,jobs,chunksize=128),1):
            if not np.array_equal(fp[:256],old_fp[i]):raise RuntimeError('Existing Morgan target changed for row '+str(i))
            targets[i]=fp
            if completed%25000==0:print('V3_TARGETS '+str(completed),flush=True)
    targets.flush();del targets
    shutil.copy2(base/'catalog.json',build/'catalog.json')
    report={'format':3,'signature':signature,'scanned_spectra':scanned,'accepted_spectra':int(counts.sum()),
            'catalog_rows':n,'training_eligible_rows':int((counts>0).sum()),'features':FEATURE_DIM,
            'target_bits':sum(HEAD_SIZES),'files':{p:sha256(build/p) for p in CACHE_FILES},
            'test_data_read':False,'official_score':None}
    write_json(build/'prepared-v3.json',report)
    if destination.exists():destination.rmdir() # Only an empty directory is removed.
    build.replace(destination)
    return report

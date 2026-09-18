"""Offline implementation of the calibration-selected R07 complete-system recipe.

This is known-structure catalog retrieval, not a de novo generator. The full
external snapshot is bundled; it is never restricted to visible test IDs.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import io
import csv
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from .production import sha256, mass_candidates, molecule_record, write_json
from .model_v3 import MultiFingerprintModel
from .target_domain import mass_prior, hybrid_scores

FILES = ('model.npz', 'catalog.json', 'fingerprints.npy', 'coconut.zip')


def source_identity(root, paths):
    root=Path(root)
    items=[(name,hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()) for name in paths]
    return hashlib.sha256(json.dumps(items,separators=(',',':')).encode()).hexdigest()


def validate_selection(config, offset):
    required={'kind','spectral_weight','mass_weight','external_penalty'}
    if not isinstance(config,dict) or set(config)!=required or config['kind']!='v1_tanimoto':
        raise ValueError('Only the sealed R07 V1-fingerprint recipe is supported')
    for key in required-{'kind'}:
        v=config[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0:
            raise ValueError('Invalid selection weight: '+key)
    if isinstance(offset,bool) or not isinstance(offset,(int,float)) or not np.isfinite(offset) or abs(offset)>1000:
        raise ValueError('Invalid mass offset')
    return config


def selected_scores(logits, bits, spectral, masses, observed, external, config, offset):
    validate_selection(config,offset)
    z=np.asarray(logits);f=np.asarray(bits)
    if z.ndim!=1 or f.ndim!=2 or f.shape[1]!=len(z) or not np.isfinite(z).all():
        raise ValueError('Invalid neural score dimensions')
    if np.any((f!=0)&(f!=1)):raise ValueError('Fingerprints must be binary')
    ext=np.asarray(external,dtype=np.float64)
    if ext.shape!=(len(f),) or np.any((ext!=0)&(ext!=1)):raise ValueError('Invalid external-source indicators')
    p=1/(1+np.exp(-np.clip(z,-40,40)))
    dots=f@p
    neural=dots/np.maximum(p.sum()+f.sum(1)-dots,1e-8)
    mass=mass_prior(masses,observed,offset_ppm=offset,scale_ppm=5.)
    return hybrid_scores(neural,spectral,mass,spectral_weight=config['spectral_weight'],
                         mass_weight=config['mass_weight'])-config['external_penalty']*ext


def verify_bundle(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'r07-bundle.json').read_text(encoding='utf-8'))
    if manifest.get('format')!=7 or manifest.get('algorithm')!='r07-selected-v1-hybrid':
        raise ValueError('Unknown R07 bundle')
    if manifest.get('feature_version')!='official-corpus-v1' or manifest.get('mass_scale_ppm')!=5.:
        raise ValueError('R07 feature/mass contract changed')
    validate_selection(manifest.get('selection'),manifest.get('mass_offset_ppm'))
    if manifest.get('contains_test_ids_or_predictions') is not False or set(manifest.get('files',{}))!=set(FILES):
        raise ValueError('Incomplete R07 provenance')
    for name,digest in manifest['files'].items():
        if sha256(folder/name)!=digest:raise ValueError('Bundle hash mismatch: '+name)
    model=MultiFingerprintModel(folder/'model.npz')
    if model.feature_dim!=4104 or model.head_sizes!=(2048,):raise ValueError('Incorrect model for chosen recipe')
    rows=json.loads((folder/'catalog.json').read_text());fp=np.load(folder/'fingerprints.npy',allow_pickle=False,mmap_mode='r')
    if not rows or fp.shape!=(len(rows),256) or fp.dtype!=np.uint8:raise ValueError('Catalog/fingerprint alignment differs')
    if any(len(r)!=4 or not r[1] or not r[2] or not np.isfinite(r[3]) or r[3]<=0 for r in rows):
        raise ValueError('Invalid structural catalog')
    if any(rows[i][3]>rows[i+1][3] for i in range(len(rows)-1)):raise ValueError('Catalog is not mass sorted')
    return manifest


def _external_record(payload):
    identifier,smiles,hint=payload
    r=molecule_record(smiles)
    if r[2] is None:return None
    fp=np.frombuffer(bytes.fromhex(r[4]),dtype=np.uint8).copy()
    return ['coconut:'+identifier,r[1],r[2],r[3]],fp


def external_candidates(archive,observed,workers=1):
    from .catalog_candidates import MassWindows,candidate_rows
    if not 1<=workers<=16:raise ValueError('Use 1-16 external catalog workers')
    windows=MassWindows(observed,ppm=50,da=.02,padding=.001)
    counter=Counter();pending=list(candidate_rows(Path(archive),windows,counter));accepted=[]
    if workers==1:
        iterator=map(_external_record,pending)
        for r in iterator:
            if r is not None and windows.contains(r[0][3]):accepted.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for r in pool.map(_external_record,pending,chunksize=32):
                if r is not None and windows.contains(r[0][3]):accepted.append(r)
    accepted.sort(key=lambda r:(r[0][3],r[0][2],r[0][0]))
    rows=[r[0] for r in accepted]
    fp=np.stack([r[1] for r in accepted]) if accepted else np.empty((0,256),np.uint8)
    return rows,fp,{'mass_selected':len(pending),'accepted':len(rows),'unique_keys':len({r[2] for r in rows}),'scan':dict(counter)}


def infer(test,train,bundle,output,*,template=None,workers=4,ranking_adapter=None):
    import pyarrow.parquet as pq
    from .production import test_groups, reference_score
    from .portable import reference_library,output_order
    from .metric import require_official_rdkit
    require_official_rdkit()
    test,train,bundle,output=map(Path,(test,train,bundle,output))
    sidecar=output.with_suffix(output.suffix+'.report.json')
    inputs={test.resolve(),train.resolve(),(bundle/'r07-bundle.json').resolve()}|{(bundle/n).resolve() for n in FILES}
    if template:inputs.add(Path(template).resolve())
    if output.resolve() in inputs or sidecar.resolve() in inputs or bundle.resolve() in output.resolve().parents:
        raise ValueError('Output would overwrite input or bundle')
    if test.resolve()==train.resolve():raise ValueError('Reference/query overlap')
    started=time.monotonic();manifest=verify_bundle(bundle)
    train_hash=sha256(train);test_hash=sha256(test)
    if test_hash==train_hash:raise ValueError('Reference/query content overlap')
    if train_hash!=manifest['train_sha256']:raise ValueError('Reference corpus hash changed')
    with pq.ParquetFile(test) as pf:
        for batch in pf.iter_batches(columns=['molecule_id']):
            if any(cid is None or not str(cid).strip() for cid in batch.column('molecule_id').to_pylist()):
                raise ValueError('Missing molecule ID')
    groups=test_groups(test)
    if not groups:raise ValueError('No query compounds')
    order,order_source=output_order(template,groups)
    observed={cid:float(np.median([q['mass'] for q in rows])) for cid,rows in groups.items()}
    catalog=json.loads((bundle/'catalog.json').read_text(encoding='utf-8'))
    masses=np.array([r[3] for r in catalog]);packed=np.load(bundle/'fingerprints.npy',mmap_mode='r',allow_pickle=False)
    model=MultiFingerprintModel(bundle/'model.npz')
    erows,efp,external=external_candidates(bundle/'coconut.zip',list(observed.values()),workers)
    emasses=np.array([r[3] for r in erows]);selected={};chosen=set();fallback=[]
    for cid,mass in observed.items():
        oi=mass_candidates(masses,mass,50,.02);ei=mass_candidates(emasses,mass,50,.02)
        mode='mass_compatible_50ppm_0.02Da'
        if len(oi)+len(ei)==0:
            oi=np.sort(np.argsort(np.abs(masses-mass),kind='stable')[:64]);mode='nearest_mass_no_compatible_structure';fallback.append(cid)
        selected[cid]=(oi,ei,mode);chosen.update(map(int,oi))
    raw,reference_counts=reference_library(train,catalog,sorted(chosen));library=defaultdict(list)
    for i,rows in raw.items():library[catalog[i][2]].extend(rows)
    del raw
    predictions=[];details={}
    for cid in order:
        queries=groups[cid];oi,ei,mode=selected[cid];mass=observed[cid]
        rows=[catalog[i] for i in oi]+[erows[i] for i in ei]
        fp=np.concatenate((packed[oi],efp[ei]),axis=0)
        flags=np.concatenate((np.zeros(len(oi)),np.ones(len(ei))))
        by_key={}
        for i,r in enumerate(rows):
            if r[2] not in by_key:by_key[r[2]]=i
        indices=np.array(list(by_key.values()),dtype=np.int64)
        rows=[rows[i] for i in indices];fp=fp[indices];flags=flags[indices]
        if not rows:raise ValueError('No valid structural guesses')
        spectral=np.array([float(np.mean([max((reference_score(q,r) for r in library[row[2]]),default=0.) for q in queries])) for row in rows])
        z=model.logits(np.mean([q['x'] for q in queries],axis=0))
        score=selected_scores(z,np.unpackbits(fp,axis=1),spectral,np.array([r[3] for r in rows]),mass,
                              flags,manifest['selection'],manifest['mass_offset_ppm'])
        ranked=np.argsort(-score,kind='stable')[:25]
        extension=None
        if ranking_adapter is not None:
            proposal=ranking_adapter(cid,rows,queries,score.copy())
            if not isinstance(proposal,dict):raise ValueError('Invalid ranking adapter result')
            indices_=proposal.get('top_indices');extension=proposal.get('metadata',{})
            if not isinstance(indices_,list) or len(indices_)!=min(25,len(rows)) or not all(type(i) is int and 0<=i<len(rows) for i in indices_):
                raise ValueError('Invalid ranking indices')
            if len(set(indices_))!=len(indices_) or not isinstance(extension,dict):raise ValueError('Invalid ranking duplicates/metadata')
            json.dumps(extension,allow_nan=False)
            ranked=np.asarray(indices_,dtype=np.int64)
        guesses=[rows[i][1] for i in ranked]
        predictions.append({'molecule_id':cid,'smiles':';'.join(guesses)})
        details[cid]={'spectra_used':len(queries),'candidates':len(rows),'external_only_candidates':int(flags.sum()),
            'mode':mode,'guesses':len(guesses),'selected_key':rows[ranked[0]][2],
            'top1_mass_error_da':float(rows[ranked[0]][3]-mass),'top1_external_only':bool(flags[ranked[0]])}
        if extension is not None:details[cid]['ranking_extension']=extension
        if len(predictions)%50==0:print('R07_INFER '+str(len(predictions))+'/'+str(len(order)),flush=True)
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=['molecule_id','smiles'],lineterminator='\n')
    writer.writeheader();writer.writerows(predictions);text=buf.getvalue()
    report={'status':'predictions_generated','format':7,'selection':manifest['selection'],'mass_offset_ppm':manifest['mass_offset_ppm'],
        'test_spectra':sum(map(len,groups.values())),'prediction_count':len(predictions),'test_sha256':test_hash,'train_sha256':train_hash,
        'submission_sha256':hashlib.sha256(text.encode()).hexdigest(),'bundle_manifest_sha256':sha256(bundle/'r07-bundle.json'),
        'model_sha256':manifest['files']['model.npz'],'weights_kind':manifest['weights_kind'],
        'external':external,'reference_counts':reference_counts,'details':details,'empty_candidate_rows':[],
        'mass_incompatible_fallbacks':fallback,'order_source':order_source,'test_labels_used':False,'official_score':None,
        'seconds':time.monotonic()-started,'candidate_not_champion':True,
        'limitations':['Catalog search, not de novo generation.','The 250-molecule natural-product audit does not prove hidden performance.',
                       'Any full-data refit is distinct from the leakage-controlled validation model.',
                       'Mass fallback produces a valid file, not proof of correct chemistry.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=output.parent,suffix='.csv.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:f.write(text)
        write_json(sidecar,report);os.replace(name,output)
    finally:
        if os.path.exists(name):os.unlink(name)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('test','train','bundle','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--sample-submission',type=Path);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(argv)
    result=infer(a.test,a.train,a.bundle,a.output,template=a.sample_submission,workers=a.workers)
    print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

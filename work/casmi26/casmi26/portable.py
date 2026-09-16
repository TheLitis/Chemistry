"""Offline inference on arbitrary (including hidden-rerun) MS/MS input files.

The bundle contains train-derived structures/fingerprints/weights only, never
visible-test identifiers, predictions, or a query-specific spectral cache.
Reference spectra are selected afresh from the supplied training file.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from . import production
from .production import sha256,write_json

ASSET_NAMES=('catalog.json','fingerprints.npy','model.npz')


def verify_bundle(folder: Path) -> dict:
    folder=Path(folder)
    manifest=json.loads((folder/'bundle.json').read_text(encoding='utf-8'))
    if manifest.get('format')!=1 or set(manifest.get('files',{}))!=set(ASSET_NAMES):
        raise ValueError('Unsupported/incomplete inference bundle')
    for name in ASSET_NAMES:
        if sha256(folder/name)!=manifest['files'][name]:
            raise ValueError('Inference bundle hash mismatch: '+name)
    return manifest


def build_bundle(cache: Path,artifacts: Path,destination: Path) -> dict:
    cache,artifacts,destination=map(Path,(cache,artifacts,destination))
    trained=json.loads((artifacts/'model-manifest.json').read_text())
    prepared=json.loads((cache/'prepared.json').read_text())
    if trained['checkpoint_sha256']!=sha256(artifacts/'model.npz') or trained['source']!=prepared['signature']:
        raise ValueError('Model and prepared corpus do not agree')
    destination.mkdir(parents=True,exist_ok=True)
    for name in ASSET_NAMES:
        source=(artifacts if name=='model.npz' else cache)/name
        if source.resolve()==(destination/name).resolve():raise ValueError('Bundle would overwrite source')
        shutil.copy2(source,destination/name)
    manifest={'format':1,'version':production.VERSION,
              'source':{k:v for k,v in trained['source'].items() if k!='test_sha256'},
              'files':{name:sha256(destination/name) for name in ASSET_NAMES},
              'contains_test_ids_or_predictions':False,'official_score':None}
    write_json(destination/'bundle.json',manifest)
    return verify_bundle(destination)


def select_candidates(masses: np.ndarray,mass: float):
    if not len(masses) or not np.isfinite(mass) or mass<=0:
        raise ValueError('Invalid query mass or empty catalog')
    idx=production.mass_candidates(masses,mass)
    if len(idx):return idx,'mass_compatible'
    idx=production.mass_candidates(masses,mass,50.,.02)
    if len(idx):return idx,'expanded_mass_window'
    # Still return a genuine model prediction for every input. This is an
    # explicitly mass-incompatible last-resort catalog guess, NOT successful
    # structure recovery or an arbitrary constant such as carbon/ethanol.
    near=np.argsort(np.abs(masses-mass),kind='stable')[:64]
    return np.sort(near),'nearest_mass_no_compatible_structure'


def select_distinct(scores,catalog,k=25):
    if not 1<=k<=25:raise ValueError('Need 1-25 guesses')
    result=[];seen=set()
    for _,idx,_,_ in scores:
        key=catalog[idx][2]
        if key in seen:continue
        seen.add(key);result.append(catalog[idx][1])
        if len(result)==k:break
    return result


def output_order(template: Path | None,groups: dict):
    if not groups:raise ValueError('No test molecules')
    if template is not None:
        with Path(template).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if reader.fieldnames!=['molecule_id','smiles']:raise ValueError('Unexpected submission header')
            ids=[r['molecule_id'] for r in reader]
        if len(ids)==len(set(ids)) and set(ids)==set(groups):return ids,'matching_template_ids'
    # Kaggle may replace test but keep the visible sample template. Always use
    # CURRENT test IDs; never pad, leak, or hardcode the original 400 IDs.
    return sorted(groups),'current_test_ids'


def reference_library(train,catalog,chosen):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    lookup={catalog[i][0]:int(i) for i in chosen};values=pa.array(list(lookup),type=pa.string())
    columns=['normalized_smiles','ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','collision_energy_ev']
    library=defaultdict(list);counts=Counter()
    with pq.ParquetFile(train) as table:
        for batch in table.iter_batches(batch_size=4096,columns=columns):
            counts['scanned']+=len(batch)
            selected=batch.filter(pc.is_in(batch.column('normalized_smiles'),value_set=values))
            if not len(selected):continue
            _,valid,neutral=production.arrow_features(selected)
            for j,r in enumerate(selected.to_pylist()):
                idx=lookup[r['normalized_smiles']];mass=catalog[idx][3]
                if not valid[j] or abs(neutral[j]-mass)>max(.003,mass*50e-6):
                    counts['rejected']+=1;continue
                p=np.column_stack((r['ms2_mzs'],r['ms2_normalized_intensities']));p=p[p[:,1]>0]
                p[:,1]/=p[:,1].sum()
                library[idx].append({'peaks':p,'precursor':r['precursor_mz'],'adduct':r['adduct']})
                counts['accepted']+=1
    return library,dict(counts)


def infer(test: Path,train: Path,bundle: Path,output: Path,template: Path | None=None) -> dict:
    from .learning import FingerprintRanker
    test,train,bundle,output=map(Path,(test,train,bundle,output))
    manifest=verify_bundle(bundle)
    if manifest.get('version')!=production.VERSION:raise ValueError('Feature/model version mismatch')
    if manifest['source']['train_sha256']!=sha256(train):raise ValueError('Reference train hash differs from trained corpus')
    forbidden={test.resolve(),train.resolve()}|{(bundle/n).resolve() for n in ASSET_NAMES+('bundle.json',)}
    if template:forbidden.add(Path(template).resolve())
    sidecar=output.with_suffix(output.suffix+'.report.json')
    if output.resolve() in forbidden or sidecar.resolve() in forbidden or bundle.resolve() in output.resolve().parents:
        raise ValueError('Output would overwrite input or bundle')
    started=time.monotonic();groups=production.test_groups(test)
    catalog=json.loads((bundle/'catalog.json').read_text(encoding='utf-8'))
    masses=np.array([r[3] for r in catalog])
    if not np.isfinite(masses).all() or np.any(np.diff(masses)<0):raise ValueError('Invalid or unsorted catalog masses')
    packed=np.load(bundle/'fingerprints.npy',allow_pickle=False)
    if packed.dtype!=np.uint8 or packed.shape!=(len(catalog),256):raise ValueError('Malformed fingerprint catalog')
    selections={};chosen=set()
    for cid,gg in groups.items():
        mass=float(np.median([q['mass'] for q in gg]));idx,mode=select_candidates(masses,mass)
        selections[cid]=(idx,mode,mass);chosen.update(map(int,idx))
    library,reference_counts=reference_library(train,catalog,sorted(chosen))
    model=FingerprintRanker(bundle/'model.npz');expected,order_source=output_order(template,groups)
    predictions=[];details={};fallback=[]
    for cid in expected:
        gg=groups[cid];idx,mode,mass=selections[cid]
        p=model.posterior_features(np.mean([q['x'] for q in gg],axis=0))
        fps=np.unpackbits(packed[idx],axis=1).astype(np.float32)
        dots=fps@p;neural=dots/np.maximum(1e-8,p.sum()+fps.sum(axis=1)-dots);scores=[]
        for j,i in enumerate(idx):
            ss=float(np.mean([max((production.reference_score(q,r) for r in library.get(int(i),[])),default=0.) for q in gg]))
            scores.append((.75*ss+.25*float(neural[j]),int(i),ss,float(neural[j])))
        scores.sort(key=lambda s:(-s[0],catalog[s[1]][2],catalog[s[1]][1]))
        guesses=select_distinct(scores,catalog)
        if not guesses:raise ValueError('No valid structural predictions for '+cid)
        predictions.append({'molecule_id':cid,'smiles':';'.join(guesses)})
        if mode=='nearest_mass_no_compatible_structure':fallback.append(cid)
        details[cid]={'spectra_used':len(gg),'mass':mass,'candidate_count':len(idx),'candidate_mode':mode,
                      'returned_structures':len(guesses),'reference_score':scores[0][2],'neural_score':scores[0][3],
                      'top1_mass_error_da':float(catalog[scores[0][1]][3]-mass)}
        if len(predictions)%25==0:print('PORTABLE_PREDICT '+str(len(predictions)),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=output.parent,suffix='.csv')
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['molecule_id','smiles']);writer.writeheader();writer.writerows(predictions)
        os.replace(name,output)
    finally:
        if os.path.exists(name):os.unlink(name)
    report={'status':'predictions_generated','version':production.VERSION,'prediction_count':len(predictions),
            'test_spectra':sum(map(len,groups.values())),'test_sha256':sha256(test),'train_sha256':manifest['source']['train_sha256'],
            'model_sha256':manifest['files']['model.npz'],'submission_sha256':sha256(output),'order_source':order_source,
            'reference_counts':reference_counts,'mass_incompatible_fallbacks':fallback,'details':details,
            'seconds':time.monotonic()-started,'official_score':None,'test_labels_used':False,
            'limitations':['Catalog retrieval only: absent true molecular graphs cannot be recovered.',
                           'Fallbacks without mass-compatible catalog candidates are explicitly marked.',
                           'Scores are uncalibrated. A complete CSV is not an official score.']}
    write_json(sidecar,report);return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train',type=Path);p.add_argument('--bundle',type=Path)
    p.add_argument('--sample-submission',type=Path)
    a=p.parse_args(argv)
    from .metric import require_official_rdkit
    require_official_rdkit()
    test=a.test/'test.parquet' if a.test.is_dir() else a.test
    train=a.train or test.parent/'train.parquet'
    bundle=a.bundle or Path(os.environ.get('CASMI26_BUNDLE',r'C:\ProgramData\ChemistryRunner\artifacts\casmi26\official-v1\bundle'))
    template=a.sample_submission or (test.parent/'sample_submission.csv' if (test.parent/'sample_submission.csv').is_file() else None)
    result=infer(test,train,bundle,a.output,template)
    print('INFERENCE_SUMMARY '+json.dumps({k:v for k,v in result.items() if k!='details'},allow_nan=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

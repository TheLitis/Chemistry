"""R04: public-catalog recovery and distractor cost on unused train holdouts.

No test spectra or hidden labels. No fitting or threshold selection. The actual
public snapshot can restore an excised answer; the script never inserts it.
"""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def molecule(payload):
    from casmi26.production import molecule_record
    identifier,smiles,hint=payload
    record=molecule_record(smiles)
    if record[2] is None:return None
    return ['coconut:'+identifier,record[1],record[2],record[3],record[4]],abs(record[3]-hint)


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    import numpy as np
    import pyarrow.parquet as pq
    from rdkit import rdBase
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import is_validation,mass_candidates,sha256,write_json
    from casmi26.learning import FingerprintRanker
    from casmi26.ranking import neural_scores,paired_effect
    from casmi26.catalog_candidates import MassWindows,candidate_rows,distinct_candidates
    from casmi26.metric import require_official_rdkit
    require_official_rdkit();r01=load('r01',repo/'tasks/casmi_rank_research.py')
    cache=state/'cache/casmi26/official-v1';art=state/'artifacts/casmi26/official-v1'
    dest=state/'artifacts/casmi26/research-r04';out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'report.json').exists():raise RuntimeError('R04 already evaluated; do not reuse its audit')
    start=time.monotonic()
    check=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
    print(check.stdout,flush=True)
    if check.returncode:raise RuntimeError('Tests failed before R04')
    snapshot=json.loads((dest/'snapshot-audit.json').read_text())
    archive=state/'data/external/coconut-2026-08/coconut_csv_lite-08-2026.zip'
    if sha256(archive)!=snapshot['archive_sha256']:raise RuntimeError('External snapshot changed')
    catalog=json.loads((cache/'catalog.json').read_text());counts=np.load(cache/'counts.npy')
    packed=np.load(cache/'fingerprints.npy',mmap_mode='r');x=np.load(cache/'features.npy',mmap_mode='r')
    observed=np.load(cache/'observed.npy');masses=np.array([r[3] for r in catalog])
    used={r['key'] for r in json.loads((art/'validation.json').read_text())['details']}
    for name in ('research-r01','research-r02','research-r03'):
        root=state/'artifacts/casmi26'/name
        used.update(json.loads((root/'calibration-keys.json').read_text()))
        used.update(r['key'] for r in json.loads((root/'audit-ranks.json').read_text()))
    representatives={}
    for i,r in enumerate(catalog):
        if counts[i]>0 and is_validation(r[2]) and r[2] not in used:
            if r[2] not in representatives or counts[i]>counts[representatives[r[2]]]:representatives[r[2]]=i
    mixed=sorted(representatives,key=lambda k:hashlib.sha256(('public-catalog-r04:'+k).encode()).digest())[:800]
    if len(mixed)!=800:raise RuntimeError('Need 800 fresh query keys')
    npkeys=set();lookup={r[0]:r[2] for r in catalog}
    train=state/'data/external/enveda-CASMI26-molecule-id-mass-spectra/train.parquet'
    with pq.ParquetFile(train) as table:
        for batch in table.iter_batches(batch_size=65536,columns=['ingest_lib','normalized_smiles']):
            for lib,smi in zip(batch.column('ingest_lib').to_pylist(),batch.column('normalized_smiles').to_pylist()):
                if lib=='enveda-np-examples' and smi in lookup:npkeys.add(lookup[smi])
    npfresh=sorted(npkeys & set(representatives));keys=mixed+[k for k in npfresh if k not in set(mixed)]
    keyset=set(keys)
    if any(not is_validation(k) for k in keys) or keyset & used:raise RuntimeError('Query/model split leakage')
    query_masses=[float(observed[representatives[k]]) for k in keys]
    selection={'keys':keys,'mixed_keys':mixed,'np_keys':npfresh,'previously_examined_excluded':len(used),
               'query_keys_sha256':hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
               'selection_uses_external_coverage':False,'tuning_performed':False}
    write_json(dest/'selection-before-evaluation.json',selection)
    counters=Counter();windows=MassWindows(query_masses)
    public=list(candidate_rows(archive,windows,counters))
    print('R04_PUBLIC_MASS_SELECTED '+str(len(public)),flush=True)
    if len(public)>250000:raise RuntimeError('More than 250k candidates require a separately budgeted build; none truncated')
    external=[];disagreements=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for n,result in enumerate(pool.map(molecule,public,chunksize=64),1):
            if result is None:counters['invalid_structure']+=1
            else:
                record,error=result
                if windows.contains(record[3]):external.append(record)
                else:counters['recomputed_mass_outside_prefilter']+=1
                if error>.0005:disagreements.append(error)
            if n%10000==0:print('R04_CHEMISTRY '+str(n),flush=True)
    del public
    external.sort(key=lambda r:(r[3],r[2],r[0]));emass=np.array([r[3] for r in external])
    counters['valid_external_records']=len(external);counters['external_unique_keys']=len({r[2] for r in external})
    counters['hint_mass_disagreements_gt_0p0005']=len(disagreements)
    write_json(dest/'validated-mass-selected-catalog.json',external)
    modelpath=art/'holdout-model.npz'
    r03=json.loads((state/'artifacts/casmi26/research-r03/report.json').read_text())
    if sha256(modelpath)!=r03['checkpoint_sha256']:raise RuntimeError('Holdout checkpoint differs')
    model=FingerprintRanker(modelpath)
    modes=('original_inclusive','expanded_inclusive','original_excised','expanded_excised')
    ranks={name:[] for name in modes};details=[]
    for n,key in enumerate(keys):
        i=representatives[key];mass=float(observed[i]);p=model.posterior_features(np.asarray(x[i]))
        oi=mass_candidates(masses,mass);ei=mass_candidates(emass,mass) if len(emass) else []
        original=[['official:'+str(j),catalog[j][1],catalog[j][2],catalog[j][3],bytes(packed[j]).hex()] for j in oi]
        added=[external[j] for j in ei]
        excised=[r for r in original if r[2] not in keyset]
        cases=(original,original+added,excised,excised+added)
        d={'key':key,'spectra':int(counts[i]),'np_example':key in npkeys,'external_coverage':key in {r[2] for r in added},
           'candidate_counts':{},'ranks':{}}
        for name,records in zip(modes,cases):
            records=distinct_candidates(records,mass)
            if records:
                fp=np.unpackbits(np.array([np.frombuffer(bytes.fromhex(r[4]),dtype=np.uint8) for r in records]),axis=1)
                score=neural_scores(p,fp,np.array([r[3] for r in records]),mass)
                rank=r01.rank_of(score,[r[2] for r in records],key)
            else:rank=0
            if name=='original_excised' and rank:raise RuntimeError('Removed truth remained in original catalog')
            ranks[name].append(rank);d['candidate_counts'][name]=len(records);d['ranks'][name]=rank
        details.append(d)
        if (n+1)%100==0:print('R04_RANKED '+str(n+1),flush=True)
    def rr(values):
        values=np.asarray(values);return np.where(values>0,1/np.maximum(values,1),0.)
    cohorts={}
    for label,subset in [('mixed_holdout',set(mixed)),('np_example_holdout',set(npfresh))]:
        ii=[i for i,k in enumerate(keys) if k in subset]
        if not ii:cohorts[label]={'molecules':0};continue
        cohorts[label]={'molecules':len(ii),'external_coverage':float(np.mean([details[i]['external_coverage'] for i in ii])),
            'metrics':{name:r01.metrics([ranks[name][i] for i in ii]) for name in modes},
            'distractor_effect':paired_effect(rr(ranks['original_inclusive'])[ii],rr(ranks['expanded_inclusive'])[ii]),
            'recovery_effect':paired_effect(rr(ranks['original_excised'])[ii],rr(ranks['expanded_excised'])[ii])}
    report={'experiment':'R04-external-catalog-transfer','status':'completed','commit':os.environ.get('GITHUB_SHA'),
        'snapshot_release':'2026-08','snapshot_sha256':snapshot['archive_sha256'],'rdkit':rdBase.rdkitVersion,
        'selection':{k:v for k,v in selection.items() if k not in ('keys','mixed_keys','np_keys')},
        'cohorts':cohorts,'external_processing':dict(counters),'checkpoint_sha256':sha256(modelpath),
        'training_query_key_overlap':0,'hyperparameters_changed':False,'ranker':'fixed_R01_after_key_deduplication',
        'true_structures_restored_only_from_actual_external_rows':True,'test_data_read':False,
        'new_training_performed':False,'production_changed':False,'official_score':None,'submission_made':False,
        'tests':check.stdout.strip(),'seconds':time.monotonic()-start,
        'limitations':['Artificial catalog excision measures retrieval from a public source, NOT de novo molecular generation.',
          'Mixed-source holdout is not hidden CASMI; NP-only cohort can be small and unrepresentative.',
          'Only source-mass-compatible rows are chemically processed; incorrect upstream mass hints may reduce coverage.',
          'All query graphs were excluded from model fitting; candidates themselves are allowed at inference.',
          'Audit keys are spent; future tuning must not use these outcomes as a new untouched test.']}
    for path in (dest,out):
        write_json(path/'report.json',report);write_json(path/'audit-ranks.json',details);write_json(path/'calibration-keys.json',[])
    print('RESEARCH_R04_BEGIN\n'+json.dumps(report,indent=2)+'\nRESEARCH_R04_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

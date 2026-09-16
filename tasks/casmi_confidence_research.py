"""R03: paired reference-present/absent experiment, fresh held-out molecules.

Only train-derived data is read. Every molecule is its own paired control.
Identical normalized spectra cannot cross query/reference partitions.
"""
from __future__ import annotations
from collections import Counter,defaultdict
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


def partition(rows):
    from casmi26.ranking import spectrum_signature
    unique={}
    for row in rows:unique.setdefault(spectrum_signature(row),row)
    if len(unique)<2:return None
    rows=list(unique.values())
    source=defaultdict(list)
    for r in rows:source[(r['source'],r['instrument'])].append(r)
    unit='source_instrument'
    if len(source)<2:
        source=defaultdict(list);unit='adduct_energy'
        for r in rows:source[(r['adduct'],str(r['energy']))].append(r)
    if len(source)>=2:
        first=min(source,key=lambda v:hashlib.sha256(('R03-acquisition:'+str(v)).encode()).digest())
        query=source[first];refs=[r for k,rr in source.items() if k!=first for r in rr]
    else:
        unit='distinct_normalized_spectrum';ordered=sorted(unique)
        cut=max(1,len(ordered)//2);query=[unique[k] for k in ordered[:cut]];refs=[unique[k] for k in ordered[cut:]]
    if {spectrum_signature(r) for r in query}&{spectrum_signature(r) for r in refs}:
        raise RuntimeError('Identical spectra leaked across partition')
    return query,refs,unit


def spectral_job(payload):
    import numpy as np
    from casmi26.production import reference_score
    truth,queries,keys,references=payload
    bykey={}
    for key,rows in zip(keys,references):
        if key in bykey:continue
        bykey[key]=float(np.mean([max((reference_score(q,r) for r in rows),default=0.) for q in queries]))
    present=np.array([bykey[k] for k in keys]);absent=present.copy()
    absent[np.array([k==truth for k in keys])]=0.
    return truth,present,absent


def rank_order(order,keys,truth):
    seen=set()
    for i in order:
        key=keys[int(i)]
        if key in seen:continue
        seen.add(key)
        if len(seen)>25:break
        if key==truth:return len(seen)
    return 0


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import is_validation,mass_candidates,arrow_features,sha256,write_json
    from casmi26.portable import reference_library
    from casmi26.learning import FingerprintRanker
    from casmi26.ranking import neural_scores,ranked_indices,paired_effect
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    r01=load('r01',repo/'tasks/casmi_rank_research.py');staged=load('staged',repo/'tasks/casmi_staged.py')
    train=staged.find_dataset(state/'data/external')/'train.parquet'
    cache=state/'cache/casmi26/official-v1';art=state/'artifacts/casmi26/official-v1'
    dest=state/'artifacts/casmi26/research-r03';out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'report.json').exists():raise RuntimeError('R03 already evaluated; audit cannot be retuned')
    start=time.monotonic()
    checks=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
    print(checks.stdout,flush=True)
    if checks.returncode:raise RuntimeError('Tests failed before R03')
    catalog=json.loads((cache/'catalog.json').read_text());counts=np.load(cache/'counts.npy')
    masses=np.array([r[3] for r in catalog]);packed=np.load(cache/'fingerprints.npy',mmap_mode='r')
    excluded={r['key'] for r in json.loads((art/'validation.json').read_text())['details']}
    for name in ('research-r01','research-r02'):
        prev=state/'artifacts/casmi26'/name
        excluded.update(json.loads((prev/'calibration-keys.json').read_text()))
        excluded.update(r['key'] for r in json.loads((prev/'audit-ranks.json').read_text()))
    eligible={r[2] for i,r in enumerate(catalog) if counts[i]>=2 and is_validation(r[2]) and r[2] not in excluded}
    pool=sorted(eligible,key=lambda k:hashlib.sha256(('confidence-r03:'+k).encode()).digest())[:1800]
    lookup={r[0]:(r[2],r[3]) for r in catalog if r[2] in set(pool)}
    values=pa.array(list(lookup),type=pa.string());groups=defaultdict(list);rejected=0
    columns=['normalized_smiles','ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','collision_energy_ev','ingest_lib','instrument_type']
    print('R03_QUERY_SCAN_START',flush=True)
    with pq.ParquetFile(train) as table:
        for batch in table.iter_batches(batch_size=8192,columns=columns):
            selected=batch.filter(pc.is_in(batch.column('normalized_smiles'),value_set=values))
            if not len(selected):continue
            x,valid,neutral=arrow_features(selected)
            for i,r in enumerate(selected.to_pylist()):
                key,mass=lookup[r['normalized_smiles']]
                if not valid[i] or abs(neutral[i]-mass)>max(.003,mass*50e-6):rejected+=1;continue
                peaks=np.column_stack((r['ms2_mzs'],r['ms2_normalized_intensities']));peaks=peaks[peaks[:,1]>0];peaks[:,1]/=peaks[:,1].sum()
                groups[key].append({'x':x[i],'mass':float(neutral[i]),'peaks':peaks,'precursor':r['precursor_mz'],
                    'adduct':r['adduct'],'source':str(r['ingest_lib']),'instrument':str(r['instrument_type']),'energy':r['collision_energy_ev']})
    split={}
    for key in pool:
        result=partition(groups[key])
        if result is not None:split[key]=result
        if len(split)==600:break
    if len(split)!=600:raise RuntimeError('Need 600 unused keys with distinct acquisitions')
    del groups
    keys=list(split);keyset=set(keys);calibration=keys[:200];audit=keys[200:]
    ids={k:mass_candidates(masses,float(np.median([r['mass'] for r in split[k][0]]))) for k in keys}
    if any(not len(v) for v in ids.values()):raise RuntimeError('No mass candidates for fixed holdout')
    chosen=sorted(set(int(i) for vv in ids.values() for i in vv))
    reference_ids=[i for i in chosen if catalog[i][2] not in keyset]
    print('R03_REFERENCE_SCAN_START '+str(len(reference_ids)),flush=True)
    library,refcounts=reference_library(train,catalog,reference_ids)
    bykey=defaultdict(list)
    for i,rows in library.items():bykey[catalog[i][2]].extend(rows)
    del library
    for key in keys:bykey[key]=split[key][1]
    present={};absent={}
    def jobs():
        for key in keys:
            ck=[catalog[i][2] for i in ids[key]]
            yield key,split[key][0],ck,[bykey[k] for k in ck]
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as workers:
        for n,(key,sp,sa) in enumerate(workers.map(spectral_job,jobs(),chunksize=1),1):
            present[key]=sp;absent[key]=sa
            if n%50==0:print('R03_SCORING '+str(n),flush=True)
    modelpath=art/'holdout-model.npz';model=FingerprintRanker(modelpath)
    prior=json.loads((state/'artifacts/casmi26/research-r01/report.json').read_text())
    if sha256(modelpath)!=prior['selection']['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
    scores={};tan={}
    for key in keys:
        q=split[key][0];p=model.posterior_features(np.mean([r['x'] for r in q],axis=0))
        fp=np.unpackbits(packed[ids[key]],axis=1).astype(np.float64)
        observed=float(np.median([r['mass'] for r in q]));scores[key]=neural_scores(p,fp,masses[ids[key]],observed)
        dot=fp@p;tan[key]=dot/np.maximum(p.sum()+fp.sum(1)-dot,1e-8)
    variants={'r01':(None,0.)}
    variants.update({f'gate:{t:g}:{m:g}':(t,m) for t in (.7,.8,.9,.95,.99) for m in (0.,.05,.1,.2)})
    def evaluate(kk,mode,regime):
        rr=[];activated=0
        for key in kk:
            s=(present if regime=='present' else absent)[key];ck=[catalog[i][2] for i in ids[key]]
            if mode=='old_blend':order=np.argsort(-(.75*s+.25*tan[key]),kind='stable')
            else:
                t,m=variants[mode];order,decision=ranked_indices(scores[key],s,ck,threshold=t,margin=m)
                activated+=int(decision['activated'])
            rr.append(rank_order(order,ck,key))
        return rr,{**r01.metrics(rr),'gate_activations':activated}
    cal={name:{reg:evaluate(calibration,name,reg)[1] for reg in ('present','absent')} for name in variants}
    admissible=[name for name,v in cal.items() if v['absent']['mrr_at_25']>=cal['r01']['absent']['mrr_at_25']-.005]
    winner=max(admissible,key=lambda name:(cal[name]['present']['mrr_at_25']+cal[name]['absent']['mrr_at_25'])/2)
    selected={'mode':winner,'threshold':variants[winner][0],'margin':variants[winner][1],'calibration':cal,
              'audit_keys_sha256':hashlib.sha256('\n'.join(audit).encode()).hexdigest(),'chosen_without_audit_labels':True}
    write_json(dest/'selection-before-audit.json',selected)
    def reciprocal(r):
        r=np.asarray(r);return np.where(r>0,1/np.maximum(r,1),0.)
    measured={};raw={}
    for name in ('old_blend','r01',winner):
        if name in measured:continue
        measured[name]={};raw[name]={}
        for reg in ('present','absent'):raw[name][reg],measured[name][reg]=evaluate(audit,name,reg)
        measured[name]['balanced_mrr']=float(np.mean([measured[name][r]['mrr_at_25'] for r in ('present','absent')]))
    def balanced(name):return (reciprocal(raw[name]['present'])+reciprocal(raw[name]['absent']))/2
    effect=paired_effect(balanced('r01'),balanced(winner))
    report={'experiment':'R03-paired-reference-confidence','status':'completed','commit':os.environ.get('GITHUB_SHA'),
        'calibration_molecules':200,'audit_molecules':400,'previous_keys_excluded':len(excluded),
        'training_query_key_overlap':0,'identical_query_reference_spectrum_overlap':0,
        'query_spectra':sum(len(split[k][0]) for k in keys),'same_key_reference_spectra':sum(len(split[k][1]) for k in keys),
        'partition_units':dict(Counter(split[k][2] for k in keys)),
        'query_instruments':dict(Counter(r['instrument'] for k in keys for r in split[k][0])),
        'source_rejections':rejected,'reference_scan':refcounts,'selection':selected,'audit':measured,
        'selected_vs_r01_paired':effect,'selected_vs_old_paired':paired_effect(balanced('old_blend'),balanced(winner)),
        'recommend_gate':effect['ci95'][0]>0 and measured[winner]['absent']['mrr_at_25']>=measured['r01']['absent']['mrr_at_25']-.01,
        'checkpoint_sha256':sha256(modelpath),'production_changed':False,'test_data_read':False,'official_score':None,'submission_made':False,
        'tests':checks.stdout.strip(),'seconds':time.monotonic()-start,
        'limitations':['Known-structure-inclusive catalog; no de novo accuracy claim.',
          'Paired regimes do not estimate their unknown prevalence in hidden CASMI.',
          'Selection favors a balanced two-regime objective subject to a calibration no-reference guard.',
          'Mixed-source acquisition split; not an independent Bruker-only test.',
          'Audit keys are spent and cannot select later variants.']}
    details=[{'key':k,'spectra':len(split[k][0]),'partition':split[k][2],
             'ranks':{name:{reg:raw[name][reg][i] for reg in ('present','absent')} for name in raw}} for i,k in enumerate(audit)]
    for path in (dest,out):
        write_json(path/'report.json',report);write_json(path/'audit-ranks.json',details);write_json(path/'calibration-keys.json',calibration)
    compact={k:v for k,v in report.items() if k!='selection'}
    compact['selected']={k:v for k,v in selected.items() if k!='calibration'}
    print('RESEARCH_R03_BEGIN\n'+json.dumps(compact,indent=2)+'\nRESEARCH_R03_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

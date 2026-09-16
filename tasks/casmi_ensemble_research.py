"""R02: spectral/neural ensemble on fresh structure-disjoint train holdouts.

All spectra of validation keys are removed from the reference library. The
structure-only catalog remains inclusive. No Kaggle/test reads or submissions.
"""
from __future__ import annotations
from collections import defaultdict
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


def fresh_split(keys,excluded,n_cal,n_audit):
    pool=sorted(set(keys)-set(excluded),key=lambda k:hashlib.sha256(('ensemble-r02:'+k).encode()).digest())
    if len(pool)<n_cal+n_audit:raise ValueError('Insufficient unused holdout keys')
    return pool[:n_cal],pool[n_cal:n_cal+n_audit]


def reference_ids(catalog,chosen,heldout):
    return [int(i) for i in chosen if catalog[int(i)][2] not in heldout]


def score_variants(spectral,tanimoto,bernoulli,prior):
    import numpy as np
    def z(v):return (v-np.mean(v))/max(float(np.std(v)),1e-8) if len(v) else v
    r01=z(bernoulli)+prior
    out={'old_blend':.75*spectral+.25*tanimoto,'r01':r01}
    for weight in (.5,1.,2.,4.):out[f'fusion:{weight:g}']=r01+weight*z(spectral)
    return out


def _spectral_job(payload):
    import numpy as np
    from casmi26.production import reference_score
    key,group,refs=payload
    result=[]
    for rows in refs:
        result.append(float(np.mean([max((reference_score(q,r) for r in rows),default=0.) for q in group])))
    return key,np.array(result)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


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
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    r01=load('r01',repo/'tasks/casmi_rank_research.py')
    staged=load('staged',repo/'tasks/casmi_staged.py')
    train=staged.find_dataset(state/'data/external')/'train.parquet'
    cache=state/'cache/casmi26/official-v1';art=state/'artifacts/casmi26/official-v1'
    prev=state/'artifacts/casmi26/research-r01';dest=state/'artifacts/casmi26/research-r02'
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'report.json').exists():raise RuntimeError('R02 audit already used; do not rerun for tuning')
    start=time.monotonic()
    checks=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
    print(checks.stdout,flush=True)
    if checks.returncode:raise RuntimeError('Tests failed before R02')
    catalog=json.loads((cache/'catalog.json').read_text());counts=np.load(cache/'counts.npy')
    masses=np.array([r[3] for r in catalog]);packed=np.load(cache/'fingerprints.npy',mmap_mode='r')
    excluded={d['key'] for d in json.loads((art/'validation.json').read_text())['details']}
    excluded.update(json.loads((prev/'calibration-keys.json').read_text()))
    excluded.update(d['key'] for d in json.loads((prev/'audit-ranks.json').read_text()))
    heldout={r[2] for r in catalog if is_validation(r[2])}
    eligible={r[2] for i,r in enumerate(catalog) if counts[i]>0 and r[2] in heldout}
    calibration,audit=fresh_split(eligible,excluded,160,320);keys=calibration+audit;keyset=set(keys)
    rawlookup={r[0]:(r[2],r[3]) for r in catalog if r[2] in keyset}
    values=pa.array(list(rawlookup),type=pa.string());groups=defaultdict(list);rejected=0
    columns=['normalized_smiles','ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','collision_energy_ev']
    print('R02_QUERY_SCAN_START',flush=True)
    with pq.ParquetFile(train) as table:
        schema=list(table.schema_arrow.names)
        for batch in table.iter_batches(batch_size=8192,columns=columns):
            selected=batch.filter(pc.is_in(batch.column('normalized_smiles'),value_set=values))
            if not len(selected):continue
            x,valid,neutral=arrow_features(selected)
            for i,r in enumerate(selected.to_pylist()):
                key,mass=rawlookup[r['normalized_smiles']]
                if not valid[i] or abs(neutral[i]-mass)>max(.003,mass*50e-6):rejected+=1;continue
                peaks=np.column_stack((r['ms2_mzs'],r['ms2_normalized_intensities']));peaks=peaks[peaks[:,1]>0]
                peaks[:,1]/=peaks[:,1].sum()
                groups[key].append({'x':x[i],'mass':float(neutral[i]),'peaks':peaks,'precursor':r['precursor_mz'],'adduct':r['adduct']})
    if set(groups)!=keyset:raise RuntimeError('Fixed holdout contains missing groups')
    idx={k:mass_candidates(masses,float(np.median([q['mass'] for q in groups[k]]))) for k in keys}
    chosen=sorted(set(int(i) for vv in idx.values() for i in vv));refs=reference_ids(catalog,chosen,heldout)
    assert not({catalog[i][2] for i in refs}&heldout)
    print('R02_REFERENCE_SCAN_START '+str(len(refs)),flush=True)
    library,library_counts=reference_library(train,catalog,refs)
    spectral={}
    def jobs():
        for key in keys:yield key,groups[key],[library.get(int(i),[]) for i in idx[key]]
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for t,(key,scores) in enumerate(pool.map(_spectral_job,jobs(),chunksize=1),1):
            spectral[key]=scores
            if t%40==0:print('R02_SPECTRAL_PROGRESS '+str(t),flush=True)
    modelpath=art/'holdout-model.npz'
    previous=json.loads((prev/'report.json').read_text())
    if sha256(modelpath)!=previous['selection']['checkpoint_sha256']:raise RuntimeError('Holdout checkpoint changed')
    model=FingerprintRanker(modelpath);scores={}
    for key in keys:
        ids=idx[key];p=np.clip(model.posterior_features(np.mean([q['x'] for q in groups[key]],axis=0)),1e-7,1-1e-7)
        f=np.unpackbits(packed[ids],axis=1).astype(np.float64);dots=f@p
        tan=dots/np.maximum(p.sum()+f.sum(1)-dots,1e-8);bern=f@(np.log(p)-np.log1p(-p))
        mass=float(np.median([q['mass'] for q in groups[key]]))
        scores[key]=score_variants(spectral[key],tan,bern,r01.mass_prior(masses[ids],mass))
    names=list(scores[calibration[0]])
    def ranks(kk,name):return [r01.rank_of(scores[k][name],[catalog[i][2] for i in idx[k]],k) for k in kk]
    cal={name:r01.metrics(ranks(calibration,name)) for name in names}
    winner=max(cal,key=lambda name:cal[name]['mrr_at_25'])
    selection={'selected':winner,'calibration':cal,'audit_keys_sha256':hashlib.sha256('\n'.join(audit).encode()).hexdigest(),
               'checkpoint_sha256':sha256(modelpath),'chosen_without_audit_labels':True}
    write_json(dest/'selection-before-audit.json',selection)
    baseline=ranks(audit,'old_blend');selected=ranks(audit,winner);neural=ranks(audit,'r01')
    report={'experiment':'R02-spectral-neural-ensemble','status':'completed','commit':os.environ.get('GITHUB_SHA'),
            'selection':selection,'calibration_molecules':len(calibration),'audit_molecules':len(audit),
            'previously_used_keys_excluded':len(excluded),'training_query_key_overlap':0,'reference_query_key_overlap':0,
            'query_spectra':sum(map(len,groups.values())),'query_rejections':rejected,'reference_scan':library_counts,
            'audit':{'old_blend':r01.metrics(baseline),'selected':r01.metrics(selected),'r01_only':r01.metrics(neural),
                     'paired_bootstrap':r01.paired_interval(baseline,selected)},
            'test_data_read':False,'submission_made':False,'production_changed':False,'official_score':None,
            'limitations':['Inclusive structure catalog, not de novo; unseen structures absent from a real catalog remain unsolved.',
                           'Mixed-source train holdout, not a Bruker-only domain evaluation.',
                           'All holdout-key spectra are excluded from references, so this tests known structures without exact library spectra.',
                           'This audit is now spent and must not select subsequent variants.'],
            'seconds':time.monotonic()-start,'train_schema':schema,'test_result':checks.stdout.strip()}
    details=[{'key':k,'baseline_rank':b,'selected_rank':s,'r01_rank':n,'spectra_used':len(groups[k]),'candidates':len(idx[k])}
             for k,b,s,n in zip(audit,baseline,selected,neural)]
    for path in (dest,out):
        write_json(path/'report.json',report);write_json(path/'audit-ranks.json',details)
        write_json(path/'calibration-keys.json',calibration)
    print('RESEARCH_R02_BEGIN\n'+json.dumps(report,indent=2)+'\nRESEARCH_R02_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

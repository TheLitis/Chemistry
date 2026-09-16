"""Full-corpus feature training and mass-blocked MS/MS retrieval.

All test rows are features only. Validation is structure-key-disjoint but uses
an inclusive structure catalog: this measures known-catalog retrieval, not de
novo discovery or the hidden Kaggle score. All intermediate arrays are numeric.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import argparse
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import tempfile
import time
import numpy as np

BINS=2048
FEATURES=4104
VERSION='official-corpus-v1'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


@lru_cache(maxsize=256)
def ion(adduct):
    """Interpret explicit stoichiometric adducts, rejecting unknown aliases."""
    from rdkit import Chem
    s=str(adduct).replace(' ','').replace('−','-')
    m=re.fullmatch(r'\[(\d*)M((?:[+-].*)?)\](\d*)([+-])',s)
    if not m:raise ValueError('Unsupported adduct: '+s)
    n=int(m[1] or 1);z=int(m[3] or 1)*(1 if m[4]=='+' else -1)
    if n<1 or n>5 or abs(z)>5:raise ValueError('Unsupported ion multiplicity/charge')
    shift=-z*0.000548579909
    pt=Chem.GetPeriodicTable();elements={pt.GetElementSymbol(i) for i in range(1,119)};body=m[2];pos=0
    for part in re.finditer(r'([+-])(\d*)([A-Za-z][A-Za-z0-9]*)',body):
        if part.start()!=pos:raise ValueError('Unparsed ion term')
        pos=part.end();formula=part[3];count=int(part[2] or 1)
        atoms=list(re.finditer(r'([A-Z][a-z]?)(\d*)',formula))
        if ''.join(a[0] for a in atoms)!=formula:raise ValueError('Unknown adduct formula')
        mass=0.
        for a in atoms:
            if a[1] not in elements:raise ValueError('Unknown adduct element')
            try:atomic=pt.GetMostCommonIsotopeMass(a[1])
            except Exception as exc:raise ValueError('Unknown element in adduct') from exc
            if atomic<=0:raise ValueError('Unknown element in adduct')
            mass+=atomic*int(a[2] or 1)
        shift+=(1 if part[1]=='+' else -1)*count*mass
    if pos!=len(body):raise ValueError('Unparsed adduct terms')
    return n,z,shift


def batch_features(mz,intensity,offsets,precursor,adducts,energies):
    """Vectorized peak histograms; exactly this transform is used at inference."""
    mz=np.asarray(mz,dtype=np.float64);intensity=np.asarray(intensity,dtype=np.float64)
    offsets=np.asarray(offsets,dtype=np.int64);precursor=np.asarray(precursor,dtype=np.float64)
    n=len(precursor)
    if len(offsets)!=n+1 or offsets[0]!=0 or offsets[-1]!=len(mz) or len(mz)!=len(intensity) or np.any(np.diff(offsets)<0):
        raise ValueError('Malformed peak offsets')
    if len(adducts)!=n or len(energies)!=n:raise ValueError('Metadata length mismatch')
    row=np.repeat(np.arange(n),np.diff(offsets));neutral=np.zeros(n);charge=np.zeros(n)
    valid=np.isfinite(precursor)&(precursor>0)&(np.diff(offsets)>0)
    for a in set(adducts):
        mask=np.array([v==a for v in adducts])
        try:
            mult,z,shift=ion(a);neutral[mask]=(precursor[mask]*abs(z)-shift)/mult;charge[mask]=z
        except ValueError:valid[mask]=False
    valid&=np.isfinite(neutral)&(neutral>0)
    bad=~np.isfinite(mz)|~np.isfinite(intensity)|(mz<=0)|(intensity<0)
    valid&=np.bincount(row[bad],minlength=n)==0
    valid&=np.bincount(row[(intensity>0)&~bad],minlength=n)>0
    x=np.zeros((n,FEATURES),dtype=np.float32)
    keep=(~bad)&(intensity>0)&(mz<precursor[row]-.01)&valid[row]
    rr=row[keep];ii=intensity[keep];mm=mz[keep]
    for channel,values in enumerate((mm,precursor[rr]-mm)):
        bins=np.floor(values).astype(np.int64);yes=(bins>=0)&(bins<BINS)
        np.add.at(x,(rr[yes],channel*BINS+bins[yes]),np.sqrt(ii[yes]))
        block=x[:,channel*BINS:(channel+1)*BINS]
        block/=np.maximum(np.linalg.norm(block,axis=1,keepdims=True),1e-12)
    x[:,-8]=neutral/1200.;x[:,-7]=np.nan_to_num(precursor)/1200.
    x[:,-6]=charge>0;x[:,-5]=charge<0
    for i,energy in enumerate(energies):
        if energy is None:continue
        if isinstance(energy,(int,float)):energy=[energy]
        if not isinstance(energy,(list,tuple,np.ndarray)):continue
        values=[float(v) for v in energy if v is not None and np.isfinite(float(v))]
        if values:x[i,-4:]=[min(values)/200,max(values)/200,np.mean(values)/200,1]
    x[~valid]=0
    return x,valid,neutral


@lru_cache(maxsize=1)
def chemistry_tools():
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    return rdMolStandardize.TautomerEnumerator(),rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)


def molecule_record(smiles):
    from rdkit import Chem,rdBase
    from rdkit.Chem import rdMolDescriptors
    try:
        with rdBase.BlockLogs():
            mol=Chem.MolFromSmiles(smiles)
            if mol is None or not mol.GetNumAtoms():raise ValueError('Invalid structure')
            for a in mol.GetAtoms():a.SetAtomMapNum(0)
            Chem.RemoveStereochemistry(mol)
            canonical=Chem.MolToSmiles(mol)
            mass=rdMolDescriptors.CalcExactMolWt(mol)
            enum,gen=chemistry_tools();taut=enum.Canonicalize(mol)
            full=Chem.MolToInchiKey(taut)
            if len(full)!=27:raise ValueError('Invalid InChIKey')
            fp=np.packbits(gen.GetFingerprintAsNumPy(taut)).tobytes().hex()
        return smiles,canonical,full[:14],mass,fp
    except Exception:
        return smiles,None,None,None,None


def is_validation(key):
    return int.from_bytes(hashlib.sha256(('official-split-v1:'+key).encode()).digest()[:8],'big')%10==0


def mass_candidates(sorted_masses,mass,ppm=20.,da=.005):
    tol=max(da,mass*ppm*1e-6)
    return np.arange(np.searchsorted(sorted_masses,mass-tol),np.searchsorted(sorted_masses,mass+tol,side='right'))


def fast_cosine(a,b,ppm=20.,da=.01):
    if not len(a) or not len(b):return 0.
    a=a[np.argsort(a[:,0])];b=b[np.argsort(b[:,0])]
    av=np.sqrt(a[:,1]);bv=np.sqrt(b[:,1]);denom=np.linalg.norm(av)*np.linalg.norm(bv)
    if not denom:return 0.
    tol=np.maximum(da,a[:,0]*ppm*1e-6)
    left=np.searchsorted(b[:,0],a[:,0]-tol);right=np.searchsorted(b[:,0],a[:,0]+tol,side='right');counts=right-left
    i=np.repeat(np.arange(len(a)),counts)
    if not len(i):return 0.
    j=np.repeat(left,counts)+np.arange(counts.sum())-np.repeat(np.cumsum(counts)-counts,counts)
    w=av[i]*bv[j]
    if len(np.unique(i))==len(i) and len(np.unique(j))==len(j):return min(1.,float(w.sum()/denom))
    useda=set();usedb=set();total=0.
    for k in np.argsort(-w,kind='stable'):
        if i[k] not in useda and j[k] not in usedb:
            useda.add(i[k]);usedb.add(j[k]);total+=w[k]
    return min(1.,float(total/denom))


def arrow_features(batch):
    mz=batch.column('ms2_mzs');inten=batch.column('ms2_normalized_intensities')
    mo=mz.offsets.to_numpy(zero_copy_only=False);io=inten.offsets.to_numpy(zero_copy_only=False)
    if not np.array_equal(mo-mo[0],io-io[0]):raise ValueError('Peak/intensity list lengths differ')
    mm=mz.values.to_numpy(zero_copy_only=False)[mo[0]:mo[-1]];ii=inten.values.to_numpy(zero_copy_only=False)[io[0]:io[-1]]
    return batch_features(mm,ii,mo-mo[0],batch.column('precursor_mz').to_numpy(zero_copy_only=False),
                          batch.column('adduct').to_pylist(),batch.column('collision_energy_ev').to_pylist())


def test_groups(test):
    import pyarrow.parquet as pq
    # Deliberately select a whitelist; test structure/answer columns cannot enter.
    cols=['molecule_id','ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','collision_energy_ev']
    table=pq.read_table(test,columns=cols).combine_chunks();groups=defaultdict(list)
    for batch in table.to_batches(max_chunksize=8192):
        x,valid,neutral=arrow_features(batch)
        if not valid.all():raise ValueError('Invalid test spectrum; none may be silently dropped')
        for i,r in enumerate(batch.to_pylist()):
            peaks=np.column_stack((r['ms2_mzs'],r['ms2_normalized_intensities']));peaks=peaks[peaks[:,1]>0]
            peaks[:,1]/=peaks[:,1].sum()
            groups[str(r['molecule_id'])].append({'x':x[i],'mass':float(neutral[i]),'peaks':peaks,
                                                'adduct':r['adduct'],'precursor':r['precursor_mz']})
    return groups


def prepare(train,test,cache,workers=8):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from rdkit import rdBase
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    signature={'version':VERSION,'rdkit':rdBase.rdkitVersion,'train_sha256':sha256(train),'test_sha256':sha256(test)}
    done=cache/'prepared.json'
    if done.exists():
        previous=json.loads(done.read_text())
        if previous.get('signature')==signature and all((cache/n).exists() for n in ('catalog.json','features.npy','counts.npy','observed.npy','fingerprints.npy','query-library.parquet')):
            print('PREPARED_CACHE_REUSED',flush=True);return previous
    groups=test_groups(test)
    raw=sorted(s for s in pc.unique(pq.read_table(train,columns=['normalized_smiles'])['normalized_smiles']).to_pylist() if isinstance(s,str) and s)
    print('CATALOG_START '+str(len(raw)),flush=True)
    catalog=[];invalid=0
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,record in enumerate(pool.map(molecule_record,raw,chunksize=64)):
            if record[2] is not None:catalog.append(record)
            else:invalid+=1
            if (i+1)%20000==0:print('CATALOG_PROGRESS '+str(i+1),flush=True)
    if not catalog:raise ValueError('Empty valid molecular catalog')
    catalog.sort(key=lambda r:(r[3],r[2],r[0]))
    n=len(catalog);lookup={r[0]:i for i,r in enumerate(catalog)};masses=np.array([r[3] for r in catalog])
    packed=np.array([np.frombuffer(bytes.fromhex(r[4]),dtype=np.uint8) for r in catalog]);np.save(cache/'fingerprints.npy',packed)
    write_json(cache/'catalog.json',[[r[0],r[1],r[2],r[3]] for r in catalog]);del packed,raw
    features=np.lib.format.open_memmap(cache/'features.npy',mode='w+',dtype=np.float32,shape=(n,FEATURES));features[:]=0
    counts=np.zeros(n,dtype=np.int64);observed=np.zeros(n,dtype=np.float64)
    shortlist=np.zeros(n,dtype=bool);coverage={}
    for cid,gg in groups.items():
        mass=float(np.median([q['mass'] for q in gg]));idx=mass_candidates(masses,mass,50.,.02)
        shortlist[idx]=True;coverage[cid]=len(idx)
    print('QUERY_CATALOG_COVERAGE '+json.dumps({'covered':sum(v>0 for v in coverage.values()),'total':len(groups),'candidate_structures':int(shortlist.sum())}),flush=True)
    seen=accepted=0;rejected=Counter();writer=None
    try:
        with pq.ParquetFile(train) as table:
            for batch in table.iter_batches(batch_size=4096):
                labels=batch.column('normalized_smiles').to_pylist();ids=np.array([lookup.get(s,-1) for s in labels])
                x,valid,neutral=arrow_features(batch)
                correct=np.abs(neutral-masses[np.maximum(ids,0)])<=np.maximum(.003,masses[np.maximum(ids,0)]*50e-6)
                admitted=valid&(ids>=0)&correct
                rejected['invalid_spectrum_or_adduct']+=int((~valid).sum())
                rejected['invalid_structure']+=int((valid&(ids<0)).sum())
                rejected['precursor_inconsistent']+=int((valid&(ids>=0)&~correct).sum())
                np.add.at(features,ids[admitted],x[admitted]);np.add.at(counts,ids[admitted],1)
                np.add.at(observed,ids[admitted],neutral[admitted])
                choose=admitted&shortlist[np.maximum(ids,0)]
                if choose.any():
                    selected=batch.filter(pa.array(choose))
                    if writer is None:writer=pq.ParquetWriter(cache/'query-library.parquet',selected.schema,compression='zstd')
                    writer.write_batch(selected)
                seen+=len(batch);accepted+=int(admitted.sum())
                if seen//100000!=(seen-len(batch))//100000:print('CORPUS_PROGRESS '+json.dumps({'seen':seen,'accepted':accepted}),flush=True)
        if writer is None:raise ValueError('No query-compatible reference spectra')
    finally:
        if writer:writer.close()
    for start in range(0,n,4096):features[start:start+4096]/=np.maximum(counts[start:start+4096,None],1)
    observed/=np.maximum(counts,1);features.flush();np.save(cache/'counts.npy',counts);np.save(cache/'observed.npy',observed)
    report={'signature':signature,'seen_spectra':seen,'accepted_spectra':accepted,'rejected':dict(rejected),
            'catalog_structures':n,'structures_with_features':int((counts>0).sum()),'invalid_raw_structures':invalid,
            'test_molecules':len(groups),'test_spectra':sum(map(len,groups.values())),
            'broad_mass_candidate_counts':coverage,'official_score':None}
    write_json(done,report);return report


def evaluate(modelpath,cache,max_queries=4000):
    from .learning import FingerprintRanker
    catalog=json.loads((cache/'catalog.json').read_text());x=np.load(cache/'features.npy',mmap_mode='r')
    counts=np.load(cache/'counts.npy');observed=np.load(cache/'observed.npy');masses=np.array([r[3] for r in catalog])
    fps=np.unpackbits(np.load(cache/'fingerprints.npy'),axis=1).astype(np.float32)
    eligible=[i for i,r in enumerate(catalog) if counts[i]>0 and is_validation(r[2])]
    eligible.sort(key=lambda i:hashlib.sha256(('evaluate:'+catalog[i][2]).encode()).digest())
    # One molecular connectivity key per evaluation unit.
    seen=set();queries=[]
    for i in eligible:
        if catalog[i][2] not in seen:seen.add(catalog[i][2]);queries.append(i)
        if len(queries)>=max_queries:break
    model=FingerprintRanker(modelpath);ranks=[];baseline=[];details=[]
    def unique_rank(order,truth):
        seen=set();rank=0
        for i in order:
            key=catalog[i][2]
            if key in seen:continue
            seen.add(key);rank+=1
            if rank>25:return None
            if key==truth:return rank
        return None
    for i in queries:
        idx=mass_candidates(masses,observed[i]);p=model.posterior_features(x[i]);f=fps[idx]
        dots=f@p;scores=dots/np.maximum(p.sum()+f.sum(axis=1)-dots,1e-8)
        neural=idx[np.argsort(-scores,kind='stable')];mass_order=idx[np.argsort(np.abs(masses[idx]-observed[i]),kind='stable')]
        r=unique_rank(neural,catalog[i][2]);b=unique_rank(mass_order,catalog[i][2]);ranks.append(r);baseline.append(b)
        details.append({'key':catalog[i][2],'rank':r,'baseline_rank':b,'candidates':len(idx)})
    def summary(rr):
        n=len(rr)
        return {'molecules':n,'mrr_at_25':sum(1/r if r else 0 for r in rr)/n if n else None,
                'top1_accuracy':sum(r==1 for r in rr)/n if n else None,'recall_at_25':sum(r is not None for r in rr)/n if n else None}
    return {'protocol':'Structure-key-disjoint 10% holdout; inclusive structure-only catalog; not de novo and not hidden Kaggle test',
            'validation':summary(ranks),'mass_only_baseline':summary(baseline),
            'ambiguous_validation':summary([r for r,d in zip(ranks,details) if d['candidates']>1]),
            'ambiguous_baseline':summary([r for r,d in zip(baseline,details) if d['candidates']>1]),
            'training_validation_key_overlap':0,'held_out_query_limit':max_queries,'details':details,'official_score':None}


def fit(cache,artifacts,epochs=30,hidden=512):
    from .learning import train_arrays
    catalog=json.loads((cache/'catalog.json').read_text());x=np.load(cache/'features.npy',mmap_mode='r');counts=np.load(cache/'counts.npy')
    fps=np.unpackbits(np.load(cache/'fingerprints.npy'),axis=1).astype(np.float32)
    trainidx=np.array([i for i,r in enumerate(catalog) if counts[i]>0 and not is_validation(r[2])]);allidx=np.flatnonzero(counts>0)
    artifacts.mkdir(parents=True,exist_ok=True)
    print('HOLDOUT_TRAINING_START '+str(len(trainidx)),flush=True)
    trainmeta=train_arrays(x[trainidx],fps[trainidx],artifacts/'holdout-model.npz',epochs=epochs,hidden=hidden,batch_size=256,device='cuda')
    val=evaluate(artifacts/'holdout-model.npz',cache);val['training']=trainmeta
    write_json(artifacts/'validation.json',val)
    print('OFFICIAL_TRAIN_HOLDOUT '+json.dumps({k:v for k,v in val.items() if k not in ('details','training')}),flush=True)
    print('FULL_REFIT_START '+str(len(allidx)),flush=True)
    finalmeta=train_arrays(x[allidx],fps[allidx],artifacts/'model.npz',epochs=epochs,hidden=hidden,batch_size=256,device='cuda')
    write_json(artifacts/'model-manifest.json',{'version':VERSION,'source':json.loads((cache/'prepared.json').read_text())['signature'],
                                             'training':finalmeta,'checkpoint_sha256':sha256(artifacts/'model.npz'),'official_score':None})
    return val


def reference_score(q,r):
    nq,zq,_=ion(q['adduct']);nr,zr,_=ion(r['adduct'])
    if np.sign(zq)!=np.sign(zr):return 0.
    a=q['peaks'];a=a[a[:,0]<q['precursor']-.01];b=r['peaks'];b=b[b[:,0]<r['precursor']-.01]
    direct=fast_cosine(a,b);loss=0.
    if nq==nr and zq==zr:
        loss=fast_cosine(np.column_stack((q['precursor']-a[:,0],a[:,1])),np.column_stack((r['precursor']-b[:,0],b[:,1])))
    return (.85*direct+.15*loss)*(1. if q['adduct']==r['adduct'] else .8)


def predict(test,template,cache,artifacts):
    import pyarrow.parquet as pq
    from .learning import FingerprintRanker
    from .metric import distinct_guesses
    groups=test_groups(test);catalog=json.loads((cache/'catalog.json').read_text());lookup={r[0]:i for i,r in enumerate(catalog)}
    masses=np.array([r[3] for r in catalog]);fps=np.unpackbits(np.load(cache/'fingerprints.npy'),axis=1).astype(np.float32)
    manifest=json.loads((artifacts/'model-manifest.json').read_text())
    signature=json.loads((cache/'prepared.json').read_text())['signature']
    if manifest['source']!=signature or manifest['checkpoint_sha256']!=sha256(artifacts/'model.npz') or signature['test_sha256']!=sha256(test):
        raise ValueError('Stale or modified model/data; refusing mismatched inference')
    model=FingerprintRanker(artifacts/'model.npz');library=defaultdict(list)
    with pq.ParquetFile(cache/'query-library.parquet') as table:
        for batch in table.iter_batches(batch_size=4096):
            for r in batch.to_pylist():
                p=np.column_stack((r['ms2_mzs'],r['ms2_normalized_intensities']));p=p[p[:,1]>0]
                p[:,1]/=p[:,1].sum()
                library[lookup[r['normalized_smiles']]].append({'peaks':p,'precursor':r['precursor_mz'],'adduct':r['adduct']})
    with Path(template).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);header=reader.fieldnames;expected=[r['molecule_id'] for r in reader]
    if header!=['molecule_id','smiles'] or len(expected)!=len(set(expected)) or set(expected)!=set(groups):
        raise ValueError('Official template/test IDs mismatch')
    predictions=[];details={};unresolved=[]
    for cid in expected:
        gg=groups[cid];mass=float(np.median([q['mass'] for q in gg]));idx=mass_candidates(masses,mass)
        expanded=False
        if not len(idx):idx=mass_candidates(masses,mass,50.,.02);expanded=True
        if not len(idx):unresolved.append(cid);continue
        p=model.posterior_features(np.mean([q['x'] for q in gg],axis=0));f=fps[idx];dots=f@p
        ns=dots/np.maximum(p.sum()+f.sum(axis=1)-dots,1e-8);scores=[]
        for j,k in enumerate(idx):
            matches=[max((reference_score(q,r) for r in library.get(int(k),[])),default=0.) for q in gg]
            ss=float(np.mean(matches));scores.append((.75*ss+.25*float(ns[j]),int(k),ss,float(ns[j])))
        scores.sort(key=lambda r:(-r[0],catalog[r[1]][2],catalog[r[1]][1]))
        selected=distinct_guesses([catalog[r[1]][1] for r in scores],25)
        if not selected:unresolved.append(cid);continue
        predictions.append({'molecule_id':cid,'smiles':';'.join(selected)})
        details[cid]={'spectra_used':len(gg),'mass':mass,'candidates':len(idx),'expanded_mass_window':expanded,
                      'top1_reference_score':scores[0][2],'top1_neural_score':scores[0][3],'guesses':len(selected)}
        if len(predictions)%25==0:print('PREDICTION_PROGRESS '+str(len(predictions)),flush=True)
    report={'version':VERSION,'test_molecules':len(groups),'test_spectra':sum(map(len,groups.values())),
            'predicted_molecules':len(predictions),'unresolved':unresolved,'details':details,
            'official_score':None,'test_labels_used':False,'source':json.loads((cache/'prepared.json').read_text())['signature'],
            'model_sha256':sha256(artifacts/'model.npz'),'status':'complete_predictions' if not unresolved else 'incomplete_catalog_coverage'}
    write_json(artifacts/'prediction-report.json',report)
    if unresolved:raise ValueError('No mass-compatible catalog candidates for '+','.join(unresolved))
    fd,name=tempfile.mkstemp(dir=artifacts,suffix='.csv')
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=header);writer.writeheader();writer.writerows(predictions)
        os.replace(name,artifacts/'submission.csv')
    finally:
        if os.path.exists(name):os.unlink(name)
    report['submission_sha256']=sha256(artifacts/'submission.csv');write_json(artifacts/'prediction-report.json',report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--stage',choices=('prepare','fit','predict','all'),default='all');p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--hidden',type=int,default=512);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args(argv)
    from .metric import require_official_rdkit
    require_official_rdkit()
    if a.data.resolve()==a.cache.resolve() or a.data.resolve() in a.cache.resolve().parents or a.data.resolve()==a.artifacts.resolve() or a.data.resolve() in a.artifacts.resolve().parents:
        raise ValueError('Outputs must not be stored inside input data')
    a.artifacts.mkdir(parents=True,exist_ok=True)
    if a.stage in ('prepare','all'):prepare(a.data/'train.parquet',a.data/'test.parquet',a.cache,a.workers)
    if a.stage in ('fit','predict'):
        prepared=json.loads((a.cache/'prepared.json').read_text())
        if prepared['signature']['train_sha256']!=sha256(a.data/'train.parquet') or prepared['signature']['test_sha256']!=sha256(a.data/'test.parquet'):
            raise ValueError('Prepared cache does not match the supplied dataset')
    if a.stage in ('fit','all'):fit(a.cache,a.artifacts,a.epochs,a.hidden)
    if a.stage in ('predict','all'):predict(a.data/'test.parquet',a.data/'sample_submission.csv',a.cache,a.artifacts)
    return 0


if __name__=='__main__':raise SystemExit(main())

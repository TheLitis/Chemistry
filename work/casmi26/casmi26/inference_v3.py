"""Offline high-resolution/multi-fingerprint prediction on current query inputs."""
from __future__ import annotations
from collections import defaultdict
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from .features_v3 import arrow_highres
from .model_v3 import MultiFingerprintModel, candidate_scores
from .pipeline_v3 import verify_bundle
from .production import sha256, write_json


def read_groups(test):
    import pyarrow.parquet as pq
    columns=['molecule_id','ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','collision_energy_ev']
    groups=defaultdict(list)
    with pq.ParquetFile(test) as table:
        for batch in table.iter_batches(batch_size=2048,columns=columns):
            features,valid,neutral=arrow_highres(batch)
            if not valid.all():raise ValueError('Invalid test spectrum: no query is silently dropped')
            for i,row in enumerate(batch.to_pylist()):
                if row['molecule_id'] is None or not str(row['molecule_id']).strip():raise ValueError('Missing molecule ID')
                p=np.column_stack((row['ms2_mzs'],row['ms2_normalized_intensities']));p=p[p[:,1]>0];p[:,1]/=p[:,1].sum()
                groups[str(row['molecule_id'])].append({'x':features[i],'mass':float(neutral[i]),'peaks':p,
                                                     'precursor':row['precursor_mz'],'adduct':row['adduct']})
    if not groups:raise ValueError('No query spectra')
    return groups


def infer(test,train,bundle,output,template=None,*,routing=None):
    from . import portable,production
    from .ranking import ranked_indices
    from .metric import require_official_rdkit
    require_official_rdkit()
    test,train,bundle,output=map(Path,(test,train,bundle,output));sidecar=output.with_suffix(output.suffix+'.report.json')
    if test.resolve() == train.resolve():
        raise ValueError('Reference/query input overlap: test and train must be distinct files')
    manifest=verify_bundle(bundle);mode=routing or manifest['mode']
    if mode not in ('neural','confidence','legacy'):raise ValueError('Unknown routing mode')
    forbidden={test.resolve(),train.resolve(),(bundle/'v3-bundle.json').resolve()}|{(bundle/n).resolve() for n in manifest['files']}
    if template:forbidden.add(Path(template).resolve())
    if output.resolve() in forbidden or sidecar.resolve() in forbidden or bundle.resolve() in output.resolve().parents:
        raise ValueError('Output would overwrite an input or model bundle')
    train_hash, test_hash = sha256(train), sha256(test)
    if train_hash == test_hash:
        raise ValueError('Reference/query content overlap: copied training data cannot be the unknown test')
    if manifest['train_sha256'] != train_hash:raise ValueError('Reference corpus changed')
    started=time.monotonic();groups=read_groups(test)
    catalog=json.loads((bundle/'catalog.json').read_text());masses=np.array([r[3] for r in catalog])
    packed=np.load(bundle/'targets.npy',mmap_mode='r',allow_pickle=False);model=MultiFingerprintModel(bundle/'model.npz')
    selections={};chosen=set()
    for cid,rows in groups.items():
        mass=float(np.median([r['mass'] for r in rows]))
        # Quarantine contradictory group masses rather than silently averaging different compounds.
        if any(abs(r['mass']-mass)>max(.02,mass*50e-6) for r in rows):
            raise ValueError('Inconsistent neutral masses for compound '+cid)
        indices,kind=portable.select_candidates(masses,mass);selections[cid]=(indices,kind,mass);chosen.update(map(int,indices))
    library=defaultdict(list);reference_counts={'scanned':0,'accepted':0}
    if mode!='neural':
        raw,reference_counts=portable.reference_library(train,catalog,sorted(chosen))
        for i,rows in raw.items():library[catalog[i][2]].extend(rows)
        del raw
    expected,order_source=portable.output_order(template,groups);predictions=[];details={};fallback=[]
    for cid in expected:
        rows=groups[cid];indices,kind,mass=selections[cid];keys=[catalog[i][2] for i in indices]
        logits=model.logits(np.mean([r['x'][:model.feature_dim] for r in rows],axis=0))
        n=sum(model.head_sizes);bits=np.unpackbits(packed[indices],axis=1)[:,:n]
        neural=candidate_scores(logits,bits,masses[indices],mass,head_sizes=model.head_sizes,weights=manifest['weights'])
        spectral=np.zeros(len(indices));decision={'activated':False}
        if mode!='neural':
            scores={}
            for k in dict.fromkeys(keys):
                scores[k]=float(np.mean([max((production.reference_score(q,r) for r in library[k]),default=0.) for q in rows]))
            spectral=np.array([scores[k] for k in keys])
        if mode=='legacy':
            p=1/(1+np.exp(-np.clip(logits[:2048],-40,40)));f=bits[:,:2048];dot=f@p
            old=dot/np.maximum(p.sum()+f.sum(1)-dot,1e-8)
            order=np.argsort(-(.75*spectral+.25*old),kind='stable')
        else:
            order,decision=ranked_indices(neural,spectral,keys,
                threshold=manifest.get('gate_threshold',.95) if mode=='confidence' else None,
                margin=manifest.get('gate_margin',.05))
        scored=[(-j,int(indices[i]),float(spectral[i]),float(neural[i])) for j,i in enumerate(order)]
        guesses=portable.select_distinct(scored,catalog)
        if not guesses:raise ValueError('Empty structure list for '+cid)
        predictions.append({'molecule_id':cid,'smiles':';'.join(guesses)})
        if kind=='nearest_mass_no_compatible_structure':fallback.append(cid)
        details[cid]={'spectra_used':len(rows),'candidate_count':len(indices),'candidate_mode':kind,'guesses':len(guesses),
                      'routing':decision,'top1_mass_error_da':float(catalog[scored[0][1]][3]-mass)}
        if len(predictions)%50==0:print('V3_INFERENCE '+str(len(predictions)),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True);fd,name=tempfile.mkstemp(dir=output.parent,suffix='.csv')
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['molecule_id','smiles']);writer.writeheader();writer.writerows(predictions)
        os.replace(name,output)
    finally:
        if os.path.exists(name):os.unlink(name)
    report={'status':'predictions_generated','model_format':3,'model_sha256':manifest['files']['model.npz'],
        'feature_dim':model.feature_dim,'head_sizes':list(model.head_sizes),'head_weights':manifest['weights'],
        'routing':mode,'prediction_count':len(predictions),'test_spectra':sum(map(len,groups.values())),
        'test_sha256':test_hash,'train_sha256':manifest['train_sha256'],'submission_sha256':sha256(output),
        'reference_counts':reference_counts,'mass_incompatible_fallbacks':fallback,'details':details,
        'order_source':order_source,'seconds':time.monotonic()-started,'test_labels_used':False,'official_score':None,
        'limitations':['Catalog retrieval, not de novo generation.','Query feature moments still compress the raw spectrum.',
                       'Scores are not correctness probabilities.','A valid CSV does not establish hidden accuracy.']}
    write_json(sidecar,report);return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('test','train','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sample-submission',type=Path);p.add_argument('--routing',choices=('neural','confidence','legacy'))
    a=p.parse_args(argv);report=infer(a.test,a.train,a.bundle,a.output,a.sample_submission,routing=a.routing)
    print('V3_RESULT '+json.dumps({k:v for k,v in report.items() if k!='details'}),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())

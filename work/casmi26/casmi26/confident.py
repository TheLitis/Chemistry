"""R03-gated catalog inference; original portable baseline remains unchanged."""
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
from .ranking import neural_scores,ranked_indices


def infer(test,train,bundle,output,template=None,*,threshold=.95,margin=.05):
    from . import portable,production
    from .learning import FingerprintRanker
    from .production import sha256,write_json
    test,train,bundle,output=map(Path,(test,train,bundle,output))
    manifest=portable.verify_bundle(bundle)
    if manifest.get('version')!=production.VERSION or manifest['source']['train_sha256']!=sha256(train):
        raise ValueError('Model/corpus provenance mismatch')
    sidecar=output.with_suffix(output.suffix+'.report.json')
    forbidden={test.resolve(),train.resolve()}|{(bundle/n).resolve() for n in portable.ASSET_NAMES+('bundle.json',)}
    if template:forbidden.add(Path(template).resolve())
    if output.resolve() in forbidden or sidecar.resolve() in forbidden or bundle.resolve() in output.resolve().parents:
        raise ValueError('Output would overwrite input or model bundle')
    ranked_indices([0.],[0.],['check'],threshold=threshold,margin=margin)
    start=time.monotonic();groups=production.test_groups(test)
    catalog=json.loads((bundle/'catalog.json').read_text(encoding='utf-8'));masses=np.array([r[3] for r in catalog])
    if not len(masses) or not np.isfinite(masses).all() or np.any(np.diff(masses)<0):raise ValueError('Malformed mass index')
    packed=np.load(bundle/'fingerprints.npy',allow_pickle=False)
    if packed.dtype!=np.uint8 or packed.shape!=(len(catalog),256):raise ValueError('Malformed fingerprints')
    selections={};chosen=set()
    for cid,gg in groups.items():
        mass=float(np.median([q['mass'] for q in gg]));idx,mode=portable.select_candidates(masses,mass)
        selections[cid]=(idx,mode,mass);chosen.update(map(int,idx))
    raw_library,reference_counts=portable.reference_library(train,catalog,sorted(chosen))
    library=defaultdict(list)
    for i,rows in raw_library.items():library[catalog[i][2]].extend(rows)
    del raw_library
    model=FingerprintRanker(bundle/'model.npz');expected,order_source=portable.output_order(template,groups)
    predictions=[];details={};fallback=[];activations=0
    for cid in expected:
        gg=groups[cid];idx,mode,mass=selections[cid];keys=[catalog[i][2] for i in idx]
        p=model.posterior_features(np.mean([q['x'] for q in gg],axis=0))
        fp=np.unpackbits(packed[idx],axis=1)
        ns=neural_scores(p,fp,masses[idx],mass);matches={}
        for key in keys:
            if key in matches:continue
            matches[key]=float(np.mean([max((production.reference_score(q,r) for r in library[key]),default=0.) for q in gg]))
        sp=np.array([matches[k] for k in keys]);order,decision=ranked_indices(ns,sp,keys,threshold=threshold,margin=margin)
        ordered=[(-j,int(idx[i]),float(sp[i]),float(ns[i])) for j,i in enumerate(order)]
        guesses=portable.select_distinct(ordered,catalog)
        if not guesses:raise ValueError('No structural predictions for '+cid)
        predictions.append({'molecule_id':cid,'smiles':';'.join(guesses)})
        activations+=int(decision['activated'])
        if mode=='nearest_mass_no_compatible_structure':fallback.append(cid)
        details[cid]={'spectra_used':len(gg),'mass':mass,'candidate_count':len(idx),'candidate_mode':mode,
            'returned_structures':len(guesses),'gate':decision,'reference_score':ordered[0][2],
            'neural_score':ordered[0][3],'top1_mass_error_da':float(catalog[ordered[0][1]][3]-mass)}
        if len(predictions)%25==0:print('CONFIDENCE_PREDICT '+str(len(predictions)),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True);fd,name=tempfile.mkstemp(dir=output.parent,suffix='.csv')
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['molecule_id','smiles']);writer.writeheader();writer.writerows(predictions)
        os.replace(name,output)
    finally:
        if os.path.exists(name):os.unlink(name)
    report={'status':'predictions_generated','version':production.VERSION,'ranking':'r03-confidence',
        'gate_threshold':threshold,'gate_margin':margin,'gate_activations':activations,
        'prediction_count':len(predictions),'test_spectra':sum(map(len,groups.values())),
        'test_sha256':sha256(test),'train_sha256':manifest['source']['train_sha256'],
        'model_sha256':manifest['files']['model.npz'],'submission_sha256':sha256(output),
        'order_source':order_source,'reference_counts':reference_counts,'mass_incompatible_fallbacks':fallback,
        'details':details,'seconds':time.monotonic()-start,'official_score':None,'test_labels_used':False,
        'limitations':['Known-catalog retrieval; absent graphs cannot be generated.',
           'Gate thresholds were calibrated on train holdouts, not hidden labels.',
           'The gate is not uniformly better for every reference-availability regime.',
           'Similarity/score are not calibrated correctness probabilities.']}
    write_json(sidecar,report);return report


def main(argv=None):
    from .metric import require_official_rdkit
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('test','train','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sample-submission',type=Path)
    p.add_argument('--gate-threshold',type=float,default=.95);p.add_argument('--gate-margin',type=float,default=.05)
    a=p.parse_args(argv);require_official_rdkit()
    report=infer(a.test,a.train,a.bundle,a.output,a.sample_submission,threshold=a.gate_threshold,margin=a.gate_margin)
    print('CONFIDENCE_SUMMARY '+json.dumps({k:v for k,v in report.items() if k!='details'},allow_nan=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

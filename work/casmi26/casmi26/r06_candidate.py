"""Immutable R06 research bundles and offline inference; never access providers.

The archived models are the actually audited holdout-trained models, not a
full-corpus refit. All-acquisition late fusion extends the three-view selection
recipe and is explicitly reported as not independently validated above 3 views.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .production import sha256,write_json,mass_candidates


def _name(value):
    if not isinstance(value,str) or not value or Path(value).name!=value or '/' in value or '\\' in value or ':' in value or value in ('.','..'):
        raise ValueError('Unsafe bundle member')
    return value


def _selection(config):
    if config.get('pooling') not in ('early','late_probability'):raise ValueError('Unknown pooling')
    weights=np.asarray(config.get('weights'),dtype='f8')
    if weights.shape!=(3,) or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('Invalid head weights')
    sigma=config.get('sigma_ppm')
    if sigma is not None and (isinstance(sigma,bool) or not isinstance(sigma,(int,float)) or not math.isfinite(sigma) or sigma<=0):
        raise ValueError('Invalid mass sigma')
    models=config.get('models')
    if not isinstance(models,list) or len(models)>2 or len(set(models))!=len(models):raise ValueError('Invalid selected models')
    for name in models:
        _name(name)
        if not name.endswith('.npz') or name in ('anchor.npz',):raise ValueError('Invalid model member')
    return config


def verify_bundle(folder):
    from .model_v3 import MultiFingerprintModel
    from .architectures import load_model
    folder=Path(folder)
    manifest=json.loads((folder/'r06-bundle.json').read_text(encoding='utf-8'))
    if manifest.get('format')!=6 or manifest.get('feature_version')!='fractional-mass-v3.1':
        raise ValueError('Unknown bundle feature contract')
    config=_selection(manifest['selection'])
    expected={'anchor.npz','catalog.json','targets.npy'}|set(config['models'])|{str(Path(n).with_suffix('.active.npy')) for n in config['models']}
    if set(manifest.get('files',{}))!=expected:raise ValueError('Incomplete bundle')
    for name,digest in manifest['files'].items():
        path=folder/_name(name)
        if path.is_symlink() or sha256(path)!=digest:raise ValueError('Bundle hash mismatch: '+name)
    anchor=MultiFingerprintModel(folder/'anchor.npz')
    if anchor.feature_dim!=8200 or anchor.head_sizes!=(2048,4096,8192):raise ValueError('Wrong anchor shape')
    catalog=json.loads((folder/'catalog.json').read_text(encoding='utf-8'))
    if not catalog or any(len(r)!=4 or not isinstance(r[2],str) or not r[2] or not np.isfinite(r[3]) or r[3]<=0 for r in catalog):
        raise ValueError('Malformed candidate catalog')
    if any(a[3]>b[3] for a,b in zip(catalog,catalog[1:])):raise ValueError('Unsorted catalog')
    packed=np.load(folder/'targets.npy',allow_pickle=False,mmap_mode='r')
    if packed.shape!=(len(catalog),1792) or packed.dtype!=np.uint8:raise ValueError('Misaligned targets')
    for name in config['models']:
        model=load_model(folder/name)
        if model.output_dim!=14336:raise ValueError('Wrong residual target size')
        mask=np.load(folder/Path(name).with_suffix('.active.npy'),allow_pickle=False)
        if mask.shape!=(8200,) or not np.isin(mask,[0,1]).all():raise ValueError('Invalid training mask')
    return manifest


def build_bundle(experiment,cache,anchor,destination):
    experiment,cache,anchor,destination=map(Path,(experiment,cache,anchor,destination))
    report=json.loads((experiment/'report.json').read_text(encoding='utf-8'))
    chosen=json.loads((experiment/'selection-before-audit.json').read_text(encoding='utf-8'))
    plan=json.loads((experiment/'protocol.json').read_text(encoding='utf-8'))
    if report.get('status')!='completed' or not report.get('supported_improvement'):
        raise ValueError('No independently supported improvement; champion is preserved')
    if not chosen.get('selected_before_audit') or report['selection']!={k:v for k,v in chosen.items() if k!='all_calibration_results'}:
        raise ValueError('Selection does not match the sealed audit')
    config=_selection(chosen['configuration'])
    source=plan['source']
    for path,digest in [(anchor,source['anchor_sha256']),(cache/'catalog.json',source['catalog_sha256']),(cache/'targets.npy',source['targets_sha256'])]:
        if sha256(path)!=digest:raise ValueError('Source hash mismatch: '+path.name)
    paths={'anchor.npz':anchor,'catalog.json':cache/'catalog.json','targets.npy':cache/'targets.npy'}
    for name in config['models']:
        trial=experiment/'trials'/name;meta=json.loads(trial.with_suffix('.json').read_text(encoding='utf-8'))
        if sha256(trial)!=chosen['model_hashes'][name] or sha256(trial)!=meta['checkpoint_sha256']:
            raise ValueError('Selected model hash mismatch')
        mask=trial.with_suffix('.active.npy')
        if sha256(mask)!=meta['mask_sha256']:raise ValueError('Selected mask hash mismatch')
        paths[name]=trial;paths[mask.name]=mask
    manifest={'format':6,'feature_version':'fractional-mass-v3.1','selection':config,
        'files':{name:sha256(path) for name,path in paths.items()},'train_sha256':source['train_sha256'],
        'protocol_sha256':sha256(experiment/'protocol.json'),'selection_sha256':sha256(experiment/'selection-before-audit.json'),
        'audit_report_sha256':sha256(experiment/'report.json'),'primary_validation_spectra':3,
        'candidate_not_champion':True,'weights_kind':'unchanged_audited_holdout_models_not_full_refit',
        'contains_test_ids_or_predictions':False,'official_score':None}
    if destination.exists():
        old=verify_bundle(destination)
        if old!=manifest:raise ValueError('Refusing to overwrite a different research bundle')
        return old
    for path in paths.values():
        if path.resolve()==destination.resolve() or path.resolve() in destination.resolve().parents:
            raise ValueError('Unsafe bundle destination')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=destination.name+'.',dir=destination.parent))
    try:
        for name,path in paths.items():shutil.copy2(path,temp/name)
        write_json(temp/'r06-bundle.json',manifest)
        verify_bundle(temp)
        os.replace(temp,destination)
    finally:
        if temp.exists():shutil.rmtree(temp)
    return manifest


def _rows(rows):
    from .architectures import spectrum_digest
    ordered={}
    for row in rows:ordered.setdefault(spectrum_digest(row),row)
    return [ordered[k] for k in sorted(ordered)]


def prepare_group(rows,*,budget=3):
    import pyarrow as pa
    from .features_v3 import arrow_highres
    from .architectures import peak_tokens
    if budget not in (1,3,'all'):raise ValueError('Unsupported spectrum budget')
    ordered=_rows(rows)
    if not ordered:raise ValueError('No spectra in group')
    selected=ordered if budget=='all' else ordered[:budget]
    features,valid,mass=arrow_highres(pa.RecordBatch.from_pylist(selected))
    if not valid.all():raise ValueError('Invalid query spectrum')
    center=float(np.median(mass))
    if np.any(np.abs(mass-center)>max(.02,abs(center)*50e-6)):raise ValueError('Inconsistent compound mass')
    t=np.zeros((96,16),'f4')
    for i,row in enumerate(selected[:3]):t[i*32:(i+1)*32]=peak_tokens(row)
    if budget=='all':x=np.mean(features.astype('f8'),axis=0).astype('f4')
    else:x=np.mean(features.astype('f2').astype('f4'),axis=0)
    return x,t,center,len(selected)


def select_mass_candidates(masses,observed):
    """Always return candidates; strict ranking behavior is unchanged when possible."""
    masses=np.asarray(masses,dtype='f8')
    if masses.ndim!=1 or not len(masses) or not np.isfinite(masses).all():raise ValueError('Invalid candidate masses')
    if not np.isfinite(observed) or observed<=0:raise ValueError('Invalid observed mass')
    idx=mass_candidates(masses,observed)
    if len(idx):return idx,'mass_compatible'
    idx=mass_candidates(masses,observed,50.,.02)
    if len(idx):return idx,'expanded_mass_window'
    near=np.argsort(np.abs(masses-observed),kind='stable')[:min(64,len(masses))]
    return np.sort(near),'nearest_mass_no_compatible_structure'


class Predictor:
    def __init__(self,bundle,device='cpu'):
        import torch
        from .architecture_training import TorchAnchor
        from .architectures import load_model
        self.bundle=Path(bundle);self.manifest=verify_bundle(bundle);self.device=device
        if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        self.anchor=TorchAnchor(self.bundle/'anchor.npz').to(device).eval()
        self.models=[]
        for name in self.manifest['selection']['models']:
            mask=torch.from_numpy(np.load(self.bundle/Path(name).with_suffix('.active.npy'),allow_pickle=False).astype('f4')).to(device)
            self.models.append((load_model(self.bundle/name).to(device).eval(),mask))

    def group(self,rows,budget='all'):
        import torch
        import pyarrow as pa
        from .features_v3 import arrow_highres
        from .architectures import peak_tokens
        ordered=_rows(rows)
        x,t,m,used=prepare_group(ordered,budget=budget)
        config=self.manifest['selection'];components=self.models or [(None,None)]
        def predict(xb,tb,net,mask):
            z=self.anchor(xb)
            return z if net is None else z+net(xb*mask,tb)
        with torch.no_grad():
            if config['pooling']=='early' or budget==1:
                xb=torch.from_numpy(x[None]).to(self.device);tb=torch.from_numpy(t[None]).to(self.device)
                logits=torch.stack([predict(xb,tb,net,mask) for net,mask in components]).mean(0)[0]
            else:
                selected=ordered if budget=='all' else ordered[:budget]
                probabilities=[torch.zeros(14336,device=self.device,dtype=torch.float64) for _ in components]
                for start in range(0,len(selected),64):
                    part=selected[start:start+64]
                    features,valid,_=arrow_highres(pa.RecordBatch.from_pylist(part))
                    if not valid.all():raise ValueError('Invalid late-fusion spectrum')
                    xb=torch.from_numpy(features.astype('f2').astype('f4')).to(self.device)
                    tt=np.zeros((len(part),96,16),'f4')
                    for i,row in enumerate(part):tt[i,:32]=peak_tokens(row)
                    tb=torch.from_numpy(tt).to(self.device)
                    for i,(net,mask) in enumerate(components):
                        probabilities[i]+=predict(xb,tb,net,mask).sigmoid().double().sum(0)
                pooled=[]
                for value in probabilities:
                    p=(value/len(selected)).clamp(1e-7,1-1e-7).float();pooled.append(p.log()-torch.log1p(-p))
                logits=torch.stack(pooled).mean(0)
        return logits.cpu().numpy(),m,{'spectra_received':len(rows),'spectra_used':used,
             'duplicate_spectra_removed':len(rows)-len(ordered),'pooling':config['pooling'],
             'primary_validation_budget_exceeded':used>3,
             'all_view_late_fusion_not_independently_audited':used>3 and config['pooling']=='late_probability'}


def infer(test,bundle,output,*,device='cpu',budget='all',template=None):
    import pyarrow.parquet as pq
    from .architecture_training import fingerprint_basis
    from .metric import require_official_rdkit
    from .portable import output_order
    require_official_rdkit()
    test,bundle,output=map(Path,(test,bundle,output));sidecar=output.with_suffix(output.suffix+'.report.json')
    if test.resolve() in (output.resolve(),sidecar.resolve()) or bundle.resolve() in output.resolve().parents:
        raise ValueError('Output would overwrite input or bundle')
    if template and Path(template).resolve() in (output.resolve(),sidecar.resolve()):raise ValueError('Output would overwrite template')
    manifest=verify_bundle(bundle)
    if sha256(test)==manifest['train_sha256']:raise ValueError('Test content equals reference training corpus')
    columns=['molecule_id','ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','collision_energy_ev']
    groups=defaultdict(list)
    with pq.ParquetFile(test) as f:
        for batch in f.iter_batches(batch_size=2048,columns=columns):
            for row in batch.to_pylist():
                cid=row['molecule_id']
                if cid is None or not str(cid).strip():raise ValueError('Missing compound ID')
                groups[str(cid)].append(row)
    if not groups:raise ValueError('No query spectra')
    order,order_source=output_order(template,groups)
    model=Predictor(bundle,device=device);config=manifest['selection']
    catalog=json.loads((bundle/'catalog.json').read_text());masses=np.array([r[3] for r in catalog])
    packed=np.load(bundle/'targets.npy',allow_pickle=False,mmap_mode='r')
    predictions=[];details={};fallback=[];start=time.monotonic()
    for cid in order:
        z,observed,detail=model.group(groups[cid],budget=budget)
        idx,mode=select_mass_candidates(masses,observed);seen=set();unique=[]
        for i in idx:
            if catalog[i][2] not in seen:seen.add(catalog[i][2]);unique.append(int(i))
        idx=np.asarray(unique,dtype=np.int64)
        if not len(idx):raise RuntimeError('Scorer-safe candidate policy returned no structures')
        bits=np.unpackbits(packed[idx],axis=1);basis=fingerprint_basis(z,bits)
        w=np.asarray(config['weights'],dtype='f8');score=basis@(w/w.sum())
        sigma=config['sigma_ppm']
        if sigma is not None:score-=.5*((masses[idx]-observed)/max(.001,abs(observed)*sigma*1e-6))**2
        ranked=np.argsort(-score,kind='stable')[:25];guesses=[catalog[idx[i]][1] for i in ranked]
        if not guesses:raise RuntimeError('No structural guesses after nonempty candidate selection')
        if mode=='nearest_mass_no_compatible_structure':fallback.append(cid)
        detail.update(candidate_count=len(idx),candidate_mode=mode,mass=observed,guesses=len(guesses))
        details[cid]=detail;predictions.append({'molecule_id':cid,'smiles':';'.join(guesses)})
        if len(predictions)%50==0:print('R06_INFER '+str(len(predictions)),flush=True)
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=['molecule_id','smiles'],lineterminator='\n')
    writer.writeheader();writer.writerows(predictions);text=buf.getvalue()
    report={'status':'predictions_generated','format':6,'prediction_count':len(predictions),
         'test_spectra':sum(map(len,groups.values())), 'budget':budget,
         'test_sha256':sha256(test),'submission_sha256':hashlib.sha256(text.encode()).hexdigest(),
         'bundle_manifest_sha256':sha256(bundle/'r06-bundle.json'),'selection':config,'details':details,
         'empty_candidate_rows':[],'mass_incompatible_fallbacks':fallback,'test_labels_used':False,'official_score':None,
         'order_source':order_source,'candidate_not_champion':True,'weights_kind':manifest['weights_kind'],
         'seconds':time.monotonic()-start,
         'limitations':['Known-catalog retrieval; no de novo generator or external candidate expansion.',
             'Unchanged audited holdout-trained models; not a full-data refit.',
             'All input acquisitions are used in all mode; the token branch retains at most 32 peaks per encoded acquisition.',
             'Early fusion token branch uses at most three hashed acquisitions; all vector features contribute.',
             'Late fusion beyond three views is an inference extension, not a new independent validation result.',
             'Expanded/nearest-mass fallbacks guarantee scorer-safe nonempty rows but do not imply chemical compatibility.']}
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
    for name in ('test','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sample-submission',type=Path);p.add_argument('--device',default='cpu')
    p.add_argument('--budget',choices=('1','3','all'),default='all');a=p.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    r=infer(a.test,a.bundle,a.output,device=a.device,budget=a.budget if a.budget=='all' else int(a.budget),template=a.sample_submission)
    print(json.dumps({k:v for k,v in r.items() if k!='details'},indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

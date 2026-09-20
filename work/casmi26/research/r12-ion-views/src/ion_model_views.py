"""Single-variable ablations for the reviewed public FPNet inference path.

The frozen branch calls the author implementation. Alternative branches keep
its weights, single-spectrum preprocessing, merged-peak algorithm, merged CE
value (25), and ranker unchanged. This module does not load model checkpoints.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from spectral_views import prepare_rows, compatible_groups

VARIANTS=('frozen','aligned_rows','single_only','ion_views')


def _view_rows(sub):
    out=[]
    for pos,r in enumerate(sub.itertuples()):
        mz=np.asarray(r.ms2_mzs,dtype=np.float64);it=np.asarray(r.ms2_normalized_intensities,dtype=np.float64)
        sid=getattr(r,'spectrum_id',None)
        if sid is None:sid=hashlib.sha256(mz.tobytes()+it.tobytes()).hexdigest()
        out.append({'position':pos,'spectrum_id':str(sid),'ms2_mzs':mz,'ms2_normalized_intensities':it,
                    'precursor_mz':float(r.precursor_mz),'adduct':r.adduct,'ionization_mode':r.ionization_mode,
                    'instrument_type':r.instrument_type,'collision_energy_ev':r.collision_energy_ev})
    return out


def _ce(row):
    try:
        v=row['collision_energy_ev']
        values=np.atleast_1d(v) if v is not None else []
        return float(np.mean(values)) if len(values) else 25.
    except (ValueError,TypeError):return 25.


@torch.no_grad()
def aligned_single_logits(base, sub, nets):
    if base['_MODEL'] is None:return None
    dev=base['_MODEL'][2];prepared,_=prepare_rows(_view_rows(sub),base['prep_peaks'])
    if not prepared:return None
    B=len(prepared);N=max(len(x['mz']) for x in prepared)
    mz=np.zeros((B,N),np.float32);it=np.zeros_like(mz);pad=np.ones((B,N),bool)
    rows=[]
    for i,x in enumerate(prepared):
        n=len(x['mz']);mz[i,:n]=x['mz'];it[i,:n]=x['intensity'];pad[i,:n]=False;rows.append(x['row'])
    T=lambda x:torch.as_tensor(x,device=dev)
    ai=base['ADDUCT_IX'];family=base['instr_family']
    args=(T(mz),T(it),T(pad),T(np.array([r['precursor_mz'] for r in rows],np.float32)),
          T(np.array([ai.get(r['adduct'],ai['<unk>']) for r in rows])),
          T(np.array([family(r['instrument_type']) for r in rows])),
          T(np.array([_ce(r) for r in rows],np.float32)),
          T(np.array([1. if r['ionization_mode']=='positive' else -1. for r in rows],np.float32)))
    return np.mean([n(*args).float().mean(0).cpu().numpy() for n in nets],axis=0)


@torch.no_grad()
def model_logits_variant(base, sub, variant='ion_views'):
    if variant not in VARIANTS:raise ValueError('Unknown inference ablation')
    if variant=='frozen':return base['model_logits'](sub)
    if base['_MODEL'] is None:return None
    single,merged,_,_=base['_MODEL'];out=[]
    if single:
        x=aligned_single_logits(base,sub,single)
        if x is not None:out.append(x)
    if merged and variant!='single_only':
        if variant=='aligned_rows':
            mz,it=base['_merge_peaks'](sub);r=next(sub.itertuples())
            x=base['_logits_raw']([(mz,it)],merged,float(np.median(sub.precursor_mz)),r.adduct,r.instrument_type,
                25.,float(np.mean([1. if m=='positive' else -1. for m in sub.ionization_mode])))
            if x is not None:out.append(x)
        else:
            embeddings=[];weights=[]
            for group in compatible_groups(_view_rows(sub)):
                view=sub.iloc[[r['position'] for r in group]]
                mz,it=base['_merge_peaks'](view);r=group[0]
                x=base['_logits_raw']([(mz,it)],merged,float(np.median(view.precursor_mz)),r['adduct'],r['instrument_type'],
                    25.,1. if r['ionization_mode']=='positive' else -1.)
                if x is not None:embeddings.append(x);weights.append(len(group))
            if embeddings:out.append(np.average(np.stack(embeddings),axis=0,weights=weights).astype(np.float32))
    return np.mean(out,axis=0) if out else None

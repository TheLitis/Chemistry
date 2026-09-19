"""Condition-preserving multi-spectrum inference for the public FPNet.

This adapter changes feature extraction, not trained weights. It is an
experimental branch until tested in the full retrieval pipeline. It deliberately
retains the author's prep_peaks transform and the 25 eV merged-model convention,
but never combines different adducts, polarities or instrument families.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import numpy as np
import torch


def energy(row):
    try:
        a=np.asarray(row.collision_energy_ev,dtype=float).reshape(-1)
        return float(a.mean()) if len(a) and np.isfinite(a).all() else 25.
    except (ValueError,TypeError):return 25.


def row_key(row):
    # Used only for stable accumulation, never as a learned input or rank score.
    a=np.column_stack((np.asarray(row.ms2_mzs,dtype='<f8'),np.asarray(row.ms2_normalized_intensities,dtype='<f8')))
    return (str(row.adduct),str(row.instrument_type),str(row.ionization_mode),float(row.precursor_mz),energy(row),
            hashlib.sha256(a.tobytes()).hexdigest())


def aligned_examples(rows,prep_peaks):
    result=[]
    for row in sorted(rows,key=row_key):
        if row.ionization_mode not in ('positive','negative'):raise ValueError('Unknown ionization mode')
        mz,it=prep_peaks(row.ms2_mzs,row.ms2_normalized_intensities,float(row.precursor_mz))
        if len(mz):result.append((row,mz,it))
    return result


def condition_groups(frame,instr_family):
    groups=defaultdict(list)
    rows=list(frame.itertuples(index=False))
    for i,row in enumerate(rows):
        if row.ionization_mode not in ('positive','negative'):raise ValueError('Unknown ionization mode')
        groups[(str(row.adduct),instr_family(row.instrument_type),row.ionization_mode)].append(i)
    result=[]
    for key,indices in sorted(groups.items()):
        order=sorted(indices,key=lambda i:row_key(rows[i]));result.append(frame.iloc[order])
    return result


@torch.no_grad()
def infer_examples(examples,nets,device,adduct_index,instr_family,*,batch_size=16):
    if not examples or not nets:return None
    if batch_size<1:raise ValueError('Invalid inference batch size')
    predictions=[]
    for start in range(0,len(examples),batch_size):
        current=examples[start:start+batch_size];b=len(current);n=max(len(a) for _,a,_ in current)
        mz=np.zeros((b,n),'f4');it=np.zeros((b,n),'f4');pad=np.ones((b,n),bool)
        for i,(row,a,v) in enumerate(current):mz[i,:len(a)]=a;it[i,:len(v)]=v;pad[i,:len(a)]=False
        rows=[r for r,_,_ in current];tensor=lambda x:torch.as_tensor(x,device=device)
        args=(tensor(mz),tensor(it),tensor(pad),tensor(np.array([r.precursor_mz for r in rows],'f4')),
              tensor(np.array([adduct_index.get(r.adduct,adduct_index['<unk>']) for r in rows],'i8')),
              tensor(np.array([instr_family(r.instrument_type) for r in rows],'i8')),
              tensor(np.array([energy(r) for r in rows],'f4')),
              tensor(np.array([1. if r.ionization_mode=='positive' else -1. for r in rows],'f4')))
        logits=torch.stack([net(*args).float() for net in nets]).mean(0)
        if logits.ndim!=2 or logits.shape[0]!=b or not torch.isfinite(logits).all():raise ValueError('Invalid model logits')
        predictions.append(logits.cpu().numpy())
    return np.concatenate(predictions)


@torch.no_grad()
def model_logits(frame, model_bundle, prep_peaks, merge_peaks, adduct_index, instr_family):
    if model_bundle is None:return None
    single,merged,device,nbits=model_bundle;outputs=[]
    if single:
        examples=aligned_examples(list(frame.itertuples(index=False)),prep_peaks)
        logits=infer_examples(examples,single,device,adduct_index,instr_family)
        if logits is not None:outputs.append(logits.mean(0,dtype='f8'))
    if merged:
        values=[];weights=[]
        for group in condition_groups(frame,instr_family):
            mz,it=merge_peaks(group);rows=list(group.itertuples(index=False));prec=float(np.median(group.precursor_mz))
            anchor=rows[0]._replace(precursor_mz=prec,collision_energy_ev=[25.])
            a,v=prep_peaks(mz,it,prec)
            if not len(a):continue
            prediction=infer_examples([(anchor,a,v)],merged,device,adduct_index,instr_family)
            values.append(prediction[0]);weights.append(len(group))
        if values:outputs.append(np.average(np.stack(values).astype('f8'),axis=0,weights=weights))
    if not outputs:return None
    result=np.mean(outputs,axis=0).astype('f4')
    if result.shape!=(nbits,):raise ValueError('Fingerprint dimension mismatch')
    return result

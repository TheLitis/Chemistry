"""Target-domain validation and candidate evidence without answer-dependent filters."""
from __future__ import annotations
import hashlib
import numpy as np
from .production import is_validation


def is_target_instrument(value):
    return 'timstof' in str(value or '').replace(' ', '').lower()


def split_target(keys, previously_evaluated, calibration_size=80):
    keys=set(keys);old=keys & set(previously_evaluated)
    if not 1 <= calibration_size < len(keys) or len(old)>calibration_size:
        raise ValueError('Cannot isolate a nonempty audit under the declared calibration budget')
    order=sorted(keys-old,key=lambda k:hashlib.sha256(('R07-target-split-20260917:'+k).encode()).digest())
    calibration=sorted(old)+order[:calibration_size-len(old)]
    audit=order[calibration_size-len(old):]
    return calibration,audit


def training_rows(catalog,counts,excluded,*,exclude_hash_holdout=True):
    counts=np.asarray(counts);excluded=set(excluded);chosen={}
    if counts.shape!=(len(catalog),):raise ValueError('Counts/catalog alignment differs')
    for i,row in enumerate(catalog):
        key=row[2]
        if counts[i]<=0 or key in excluded or (exclude_hash_holdout and is_validation(key)):continue
        if key not in chosen or counts[i]>counts[chosen[key]]:chosen[key]=i
    return np.asarray(sorted(chosen.values()),dtype=np.int64)


def mass_prior(masses,observed,*,offset_ppm=0.,scale_ppm=5.,floor_da=.001):
    masses=np.asarray(masses,dtype=np.float64)
    if not np.isfinite(masses).all() or not np.isfinite(observed) or observed<=0:
        raise ValueError('Invalid mass input')
    if not np.isfinite([offset_ppm,scale_ppm,floor_da]).all() or scale_ppm<=0 or floor_da<=0:
        raise ValueError('Invalid mass calibration')
    corrected=observed/(1+offset_ppm*1e-6)
    residual=(masses-corrected)/max(floor_da,abs(corrected)*scale_ppm*1e-6)
    # Student-t(3): calibrated finite tail instead of an irreversible narrow gate.
    return -2*np.log1p(residual**2/3)


def fingerprint_evidence(logits,bits,*,head_sizes=(2048,4096,8192)):
    z=np.asarray(logits,dtype=np.float64);bits=np.asarray(bits,dtype=np.float64)
    if z.shape!=(sum(head_sizes),) or bits.ndim!=2 or bits.shape[1]!=len(z):
        raise ValueError('Fingerprint dimensions differ')
    if not np.isfinite(z).all() or not np.isfinite(bits).all() or np.any((bits!=0)&(bits!=1)):
        raise ValueError('Invalid fingerprint input')
    columns=[];start=0
    for size in head_sizes:
        columns.append(bits[:,start:start+size]@np.clip(z[start:start+size],-16.,16.)/np.sqrt(size))
        start+=size
    return np.column_stack(columns)


def hybrid_scores(neural,spectral,mass,*,spectral_weight=2.,mass_weight=.1):
    n,s,m=[np.asarray(x,dtype=np.float64) for x in (neural,spectral,mass)]
    if n.ndim!=1 or n.shape!=s.shape or n.shape!=m.shape or not all(np.isfinite(x).all() for x in (n,s,m)):
        raise ValueError('Invalid candidate scores')
    if not np.isfinite([spectral_weight,mass_weight]).all() or min(spectral_weight,mass_weight)<0:
        raise ValueError('Invalid hybrid weights')
    return n+spectral_weight*s+mass_weight*m


def rank_key(scores,keys,truth):
    scores=np.asarray(scores,dtype=np.float64)
    if scores.shape!=(len(keys),) or not np.isfinite(scores).all():raise ValueError('Invalid rank arrays')
    seen=set()
    for i in np.argsort(-scores,kind='stable'):
        if keys[int(i)] in seen:continue
        seen.add(keys[int(i)])
        if len(seen)>25:break
        if keys[int(i)]==truth:return len(seen)
    return 0


def metrics(ranks):
    r=np.asarray(ranks,dtype=np.int64)
    if r.ndim!=1 or np.any((r<0)|(r>25)):raise ValueError('Invalid reciprocal ranks')
    if not len(r):return {'molecules':0,'mrr_at_25':None,'top1':None,'recall_at_25':None}
    return {'molecules':len(r),'mrr_at_25':float(np.where(r>0,1/np.maximum(r,1),0).mean()),
            'top1':float((r==1).mean()),'recall_at_25':float((r>0).mean())}

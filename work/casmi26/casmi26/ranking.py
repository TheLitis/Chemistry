"""Research-tested candidate ranking. Scores are not calibrated probabilities."""
from __future__ import annotations
import hashlib
import json
import numpy as np


def neural_scores(p, fingerprints, masses, observed):
    p=np.asarray(p,dtype=np.float64); f=np.asarray(fingerprints,dtype=np.float64)
    masses=np.asarray(masses,dtype=np.float64)
    if p.ndim!=1 or f.ndim!=2 or f.shape!=(len(masses),len(p)):
        raise ValueError('Fingerprint/candidate shape mismatch')
    if not np.isfinite(p).all() or not np.isfinite(f).all() or not np.isfinite(masses).all() or not np.isfinite(observed) or observed<=0:
        raise ValueError('Nonfinite/invalid ranking inputs')
    if np.any((p<0)|(p>1)) or np.any((f!=0)&(f!=1)):
        raise ValueError('Require probabilities and binary candidate fingerprints')
    if not len(masses):return np.empty(0,dtype=np.float64)
    p=np.clip(p,1e-7,1-1e-7)
    chemical=f@(np.log(p)-np.log1p(-p))
    chemical=(chemical-chemical.mean())/max(float(chemical.std()),1e-8)
    sigma=max(.001,abs(float(observed))*5e-6)
    return chemical-.5*((masses-observed)/sigma)**2


def ranked_indices(neural, spectral, keys, *, threshold=.9, margin=.1):
    """Promote one unambiguous library match, retain neural order otherwise.

    Confidence is a decision rule, not a calibrated probability. Runner-up
    must have a different tautomer-equivalence key. None disables promotion.
    """
    n=np.asarray(neural,dtype=np.float64);s=np.asarray(spectral,dtype=np.float64)
    if n.ndim!=1 or s.shape!=n.shape or len(keys)!=len(n):
        raise ValueError('Score/key shape mismatch')
    if not np.isfinite(n).all() or not np.isfinite(s).all() or np.any((s<0)|(s>1)):
        raise ValueError('Nonfinite/invalid candidate scores')
    if threshold is not None and (not 0<threshold<=1 or not 0<=margin<=1):
        raise ValueError('Invalid confidence threshold or margin')
    order=np.argsort(-n,kind='stable')
    decision={'activated':False,'best_similarity':0.,'margin':0.}
    if not len(n) or threshold is None:return order,decision
    library=np.argsort(-s,kind='stable');best=int(library[0])
    second=next((float(s[j]) for j in library if keys[j]!=keys[best]),0.)
    gap=float(s[best])-second
    decision.update(best_similarity=float(s[best]),margin=gap)
    if s[best]>=threshold and gap>=margin:
        order=np.concatenate(([best],order[order!=best]))
        decision['activated']=True
    return order,decision


def spectrum_signature(row):
    """Detect identical normalized spectra across sources and input row orders."""
    peaks=np.asarray(row['peaks'],dtype=np.float64)
    if peaks.ndim!=2 or peaks.shape[1]!=2 or not len(peaks) or not np.isfinite(peaks).all() or peaks[:,1].sum()<=0:
        raise ValueError('Malformed spectrum')
    peaks=peaks[np.argsort(peaks[:,0],kind='stable')].copy()
    peaks[:,1]/=peaks[:,1].sum()
    payload=[str(row['adduct']),round(float(row['precursor']),4),
             np.round(peaks[:,0],4).tolist(),np.round(peaks[:,1],6).tolist()]
    return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()


def paired_effect(baseline_rr, selected_rr, repeats=2000):
    a=np.asarray(baseline_rr,dtype=float);b=np.asarray(selected_rr,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all() or repeats<100:
        raise ValueError('Need finite paired molecular reciprocal ranks')
    delta=b-a;rng=np.random.default_rng(26091603)
    samples=[float(delta[rng.integers(len(delta),size=len(delta))].mean()) for _ in range(repeats)]
    return {'delta_mrr':float(delta.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'bootstrap_repeats':repeats,'bootstrap_unit':'molecule'}

"""Certified top-k screening for s_i = b_i + weight * e_i, 0 <= e_i <= 1.

A certificate concerns this fixed scoring function, not chemical correctness.
Uncomputed entries in lower_scores remain lower bounds, not final scores.
"""
from __future__ import annotations
import math
from typing import Callable
import numpy as np


def _contract(base, weight, k):
    b=np.asarray(base,dtype=np.float64)
    if b.ndim!=1 or not np.isfinite(b).all():raise ValueError('Base scores must be a finite vector')
    if isinstance(weight,bool) or not isinstance(weight,(int,float,np.number)) or not math.isfinite(float(weight)) or weight<0:
        raise ValueError('Weight must be finite and nonnegative')
    if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or k<1:raise ValueError('k must be positive integer')
    w=float(weight)
    with np.errstate(over='ignore'):
        upper=np.nextafter(b+w,np.inf)
    if not np.isfinite(upper).all():raise ValueError('Score bound overflow')
    return b.copy(),upper,w,min(int(k),len(b))


def candidate_envelope(base, weight, *, k=25):
    """Every possible top-k lies in this envelope; labels are not an input."""
    b,upper,w,k=_contract(base,weight,k)
    if not k:return np.empty(0,dtype=np.int64)
    order=np.argsort(-b,kind='stable')
    if w==0:return order[:k]
    threshold=float(b[order[k-1]])
    return order[upper[order]>=threshold]


def certified_rerank(base, evidence: Callable[[int],float|None], *, weight, k=25, max_evaluations=None):
    """Evaluate candidates by decreasing upper bound until the top-k is proven.

    A known absence of forward evidence is None and receives zero bonus, as in
    the sealed R08 policy. Unexpected exceptions and invalid values propagate.
    A budget-limited answer has certified=False unless the bounds still prove it.
    """
    lower,upper,w,k=_contract(base,weight,k)
    if max_evaluations is not None and (isinstance(max_evaluations,bool) or
       not isinstance(max_evaluations,(int,np.integer)) or max_evaluations<0):
        raise ValueError('Evaluation budget must be a nonnegative integer or None')
    if not callable(evidence):raise ValueError('Evidence must be callable')
    n=len(lower);order=np.argsort(-lower,kind='stable');done=np.zeros(n,dtype=bool)
    count=missing=0;certified=(w==0 or k==0)
    if not certified:
        for i0 in order:
            i=int(i0)
            threshold=float(np.partition(lower,n-k)[n-k])
            # Strict exclusion keeps stable input-order ties correct.
            if upper[i]<threshold:
                certified=True;break
            if max_evaluations is not None and count>=max_evaluations:break
            value=evidence(i)
            if value is None:missing+=1;value=0.
            if isinstance(value,bool) or not isinstance(value,(int,float,np.number)):
                raise ValueError('Evidence must be numeric or explicitly missing')
            value=float(value)
            if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Evidence exceeds proven [0,1] bound')
            lower[i]+=w*value;done[i]=True;count+=1
        else:certified=True
    ranked=np.argsort(-lower,kind='stable')[:k].tolist()
    remaining=upper[~done]
    if not certified and k and ranked and remaining.size:
        certified=float(np.max(remaining))<float(lower[ranked[-1]])
    return {'top_indices':ranked,'lower_scores':lower.tolist(),'evaluated_mask':done.tolist(),
            'evaluations':count,'missing_evidence':missing,'pruned':n-count if certified else None,
            'certified':bool(certified),'score_definition':'base + weight * bounded_forward_evidence',
            'uncomputed_score_is_lower_bound':True,
            'largest_uncomputed_upper_bound':float(np.max(remaining)) if remaining.size and w else None}

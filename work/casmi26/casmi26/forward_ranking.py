"""Auditable forward-spectrum evidence; independent of target structure labels.

Entropy here means unweighted Jensen-Shannon similarity on one-to-one aligned
peaks, not a claimed reproduction of weighted MS-Entropy preprocessing.
"""
from __future__ import annotations
import numpy as np

SUPPORTED_ADDUCTS=frozenset(('[M+H]+','[M-H]-'))
ENERGIES=(10.,20.,40.,60.)


def clean_peaks(peaks,precursor,*,margin=.5):
    if not np.isfinite(precursor) or precursor<=0 or not np.isfinite(margin) or margin<0:
        raise ValueError('Invalid precursor or exclusion window')
    p=np.asarray(peaks,dtype=np.float64)
    if p.size==0:return np.empty((0,2),np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all() or np.any(p[:,0]<=0) or np.any(p[:,1]<0):
        raise ValueError('Malformed peak array')
    p=p[(p[:,0]<precursor-margin)&(p[:,1]>0)]
    if not len(p):return np.empty((0,2),np.float64)
    masses,idx=np.unique(p[:,0],return_inverse=True)
    intensity=np.bincount(idx,weights=p[:,1]);intensity/=intensity.sum()
    return np.column_stack((masses,intensity))


def compare_spectra(observed,predicted,precursor,*,ppm=20.,da=.01):
    if not np.isfinite([ppm,da]).all() or min(ppm,da)<=0:raise ValueError('Invalid match tolerances')
    a=clean_peaks(observed,precursor);b=clean_peaks(predicted,precursor)
    output={'cosine':0.,'entropy':0.,'coverage':0.,'matched_peaks':0}
    if not len(a) or not len(b):return output
    edges=[]
    for i,(mz,intensity) in enumerate(a):
        tolerance=max(da,mz*ppm*1e-6)
        left=np.searchsorted(b[:,0],mz-tolerance,side='left')
        right=np.searchsorted(b[:,0],mz+tolerance,side='right')
        edges.extend((-float(np.sqrt(intensity*b[j,1])),abs(float(mz-b[j,0])),i,j) for j in range(left,right))
    used_a=set();used_b=set()
    for _,_,i,j in sorted(edges):
        if i in used_a or j in used_b:continue
        used_a.add(i);used_b.add(j);p=float(a[i,1]);q=float(b[j,1]);total=p+q
        output['cosine']+=float(np.sqrt(p*q))
        output['entropy']+=(total*np.log(total)-p*np.log(p)-q*np.log(q))/(2*np.log(2))
        output['coverage']+=p;output['matched_peaks']+=1
    for name in ('cosine','entropy','coverage'):output[name]=float(np.clip(output[name],0.,1.))
    return output


def add_forward_evidence(base,evidence,weight):
    b=np.asarray(base,dtype=np.float64);e=np.asarray(evidence,dtype=np.float64)
    if b.ndim!=1 or e.shape!=b.shape or not np.isfinite(b).all() or not np.isfinite(weight) or weight<0:
        raise ValueError('Invalid candidate score contract')
    known=np.isfinite(e)
    if np.any(np.isinf(e)) or np.any((e[known]<0)|(e[known]>1)):
        raise ValueError('Forward evidence must be bounded or missing')
    return b+weight*np.where(known,e,0.)


def candidate_frontier(cases,limit=32):
    if isinstance(limit,bool) or not isinstance(limit,int) or limit<1:raise ValueError('Invalid frontier budget')
    selected=set()
    for data in cases.values():
        scores=np.asarray(data['scores'],dtype=np.float64);keys=data['keys']
        if scores.shape!=(len(keys),) or not np.isfinite(scores).all() or len(set(keys))!=len(keys):
            raise ValueError('Malformed candidate frontier')
        selected.update(keys[int(i)] for i in np.argsort(-scores,kind='stable')[:limit])
    return sorted(selected)


def pool_forward_scores(queries,predictions):
    """Use every supported acquisition; other adducts are not silently remapped.

    Mixed-energy acquisition: equal-intensity mixture of nearest grid spectra.
    Missing energy: uniform grid mixture. Max-grid similarity is a separately
    calibrated feature, not a claim of predicting the real collision energy.
    """
    values=[];counts=0
    for q in queries:
        mode=q['adduct']
        if mode not in SUPPORTED_ADDUCTS:continue
        candidates=sorted((float(e),p) for (m,e),p in predictions.items() if m==mode and p is not None)
        if not candidates:continue
        energies=np.array([e for e,_ in candidates]);counts+=1
        comps=[compare_spectra(q['peaks'],p,q['precursor']) for _,p in candidates]
        ce=q.get('ce')
        if not isinstance(ce,(list,tuple,np.ndarray)):ce=[] if ce is None else [ce]
        ce=[float(e) for e in ce if e is not None and np.isfinite(e) and float(e)>=0]
        indices=[int(np.argmin(abs(energies-e))) for e in ce] if ce else list(range(len(candidates)))
        parts=[]
        for i in indices:
            peaks=np.asarray(candidates[i][1],dtype=np.float64).reshape(-1,2)
            if len(peaks) and peaks[:,1].sum()>0:
                peaks=peaks.copy();peaks[:,1]/=peaks[:,1].sum();parts.append(peaks)
        mixture=np.concatenate(parts) if parts else np.empty((0,2))
        nearest=compare_spectra(q['peaks'],mixture,q['precursor'])
        values.append({**{k+'_nearest':nearest[k] for k in ('cosine','entropy','coverage')},
                       **{k+'_max':max(c[k] for c in comps) for k in ('cosine','entropy','coverage')}})
    return {'supported_spectra':counts,'total_spectra':len(queries),
            'features':{k:float(np.mean([v[k] for v in values])) for k in values[0]} if values else {}}


def rerank_case(case,features,feature,weight,limit=32):
    scores=np.asarray(case['scores'],dtype='f8')
    allowed=set(candidate_frontier({'case':case},limit))
    evidence=np.array([features.get(key,{}).get(feature,np.nan) if key in allowed else np.nan for key in case['keys']])
    return add_forward_evidence(scores,evidence,weight)


def evaluate_configuration(records,features,feature,weight,limit=32):
    from .target_domain import rank_key,metrics
    rows=[]
    for r in records:
        ranks={case:rank_key(rerank_case(data,features.get(r['key'],{}),feature,weight,limit),data['keys'],r['key'])
               for case,data in r['cases'].items()}
        rows.append({'key':r['key'],'ranks':ranks})
    return {'metrics':{case:metrics([r['ranks'][case] for r in rows]) for case in records[0]['cases']},'rows':rows}


def choose_configuration(calibration,features,names,weights,limit=32):
    if not calibration or not names or 0. not in weights:raise ValueError('Calibration and a no-change option are required')
    baseline=evaluate_configuration(calibration,features,names[0],0.,limit)['metrics']
    grid=[]
    for feature in names:
        for weight in weights:
            if weight==0 and grid:continue
            result=evaluate_configuration(calibration,features,feature,weight,limit)['metrics']
            gains={case:result[case]['mrr_at_25']-baseline[case]['mrr_at_25'] for case in baseline}
            grid.append({'feature':feature,'weight':weight,'metrics':result,'gains':gains})
    selected=max(grid,key=lambda v:(min(v['gains'].values()),np.mean(list(v['gains'].values())),-v['weight']))
    return {'selected':selected,'baseline':baseline,'grid':grid,'selected_only_on_calibration':True}

"""Bounded, graph-connected fragment evidence; not a calibrated MS simulator.

Enumerate components of the ORIGINAL heavy-atom graph after at most two edge
cuts. Attached hydrogens are counted before cutting, never implicitly capped
by fragment sanitization. A small explicit H transfer is only a hypothesis.
No spectral labels, pretrained weights, or answer formulas are used.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations
import math
import numpy as np
from .formula_evidence import PROTON, ELEMENTS, element_mass


class FragmentBudgetExceeded(ValueError):
    """Reject an oversized enumeration before traversing an arbitrary prefix."""


def _settings(max_cuts, max_scenarios):
    if type(max_cuts) is not int or max_cuts not in (1,2):raise ValueError('max_cuts must be 1 or 2')
    if type(max_scenarios) is not int or max_scenarios<1:raise ValueError('Invalid scenario budget')


def _components(adjacency, removed):
    neighbors=list(adjacency)
    for a,b in removed:neighbors[a]&=~(1<<b);neighbors[b]&=~(1<<a)
    remaining=(1<<len(neighbors))-1
    while remaining:
        component=0;front=remaining&-remaining
        while front:
            component|=front;more=0
            while front:
                bit=front&-front;front-=bit;more|=neighbors[bit.bit_length()-1]
            front=more&~component
        remaining&=~component
        yield component


@lru_cache(maxsize=1024)
def _enumerate(smiles, max_cuts, max_scenarios):
    from rdkit import Chem,rdBase
    from rdkit.Chem.MolStandardize import rdMolStandardize
    with rdBase.BlockLogs():
        mol=Chem.MolFromSmiles(smiles) if isinstance(smiles,str) and smiles else None
        if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol))!=1:
            raise ValueError('One valid connected candidate required')
        if (Chem.GetFormalCharge(mol)!=0 or any(a.GetIsotope() or a.GetNumRadicalElectrons() or a.GetSymbol() not in ELEMENTS for a in mol.GetAtoms())):
            raise ValueError('Unsupported charge, isotope, radical or element')
        mol=Chem.RemoveHs(rdMolStandardize.TautomerEnumerator().Canonicalize(mol))
    n=mol.GetNumAtoms();edges=tuple(sorted(tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in mol.GetBonds()))
    count=len(edges)+(math.comb(len(edges),2) if max_cuts==2 else 0)
    if n>96 or count>max_scenarios:raise FragmentBudgetExceeded('Graph enumeration exceeds fixed budget')
    neighbors=[0]*n; atoms=[]
    for a in mol.GetAtoms():
        atoms.append((a.GetSymbol(),int(a.GetTotalNumHs())))
    for a,b in edges:neighbors[a]|=1<<b;neighbors[b]|=1<<a
    full=(1<<n)-1;subsets={}
    for cuts in range(1,max_cuts+1):
        for removed in combinations(edges,cuts):
            for mask in _components(neighbors,removed):
                if mask!=full and mask not in subsets:subsets[mask]=cuts
    formulas={}
    for mask,cuts in subsets.items():
        counts={}
        while mask:
            bit=mask&-mask;mask-=bit;e,h=atoms[bit.bit_length()-1]
            counts[e]=counts.get(e,0)+1;counts['H']=counts.get('H',0)+h
        formula=tuple(sorted((e,c) for e,c in counts.items() if c))
        formulas[formula]=min(cuts,formulas.get(formula,cuts))
    # Immutable cache payload; callers cannot mutate shared cached results.
    return tuple((f,c,sum(element_mass(e)*v for e,v in f)) for f,c in sorted(formulas.items()))


def connected_fragments(smiles, *, max_cuts=2, max_scenarios=8192):
    _settings(max_cuts,max_scenarios)
    if not isinstance(smiles,str):raise ValueError('SMILES must be text')
    return [{'composition':f,'cuts':c,'mass':m} for f,c,m in _enumerate(smiles,max_cuts,max_scenarios)]


def _prepare(queries, ppm, da, relative_floor):
    prepared=[]
    for q in queries:
        mode=q.get('adduct')
        if mode not in ('[M+H]+','[M-H]-'):continue
        precursor=float(q['precursor']);peaks=np.asarray(q['peaks'],dtype='f8')
        if not math.isfinite(precursor) or precursor<=0:raise ValueError('Invalid precursor')
        if peaks.size==0:continue
        if peaks.ndim!=2 or peaks.shape[1]!=2 or not np.isfinite(peaks).all() or np.any(peaks[:,0]<=0) or np.any(peaks[:,1]<0):
            raise ValueError('Invalid query peaks')
        peaks=peaks[(peaks[:,0]<precursor-.5)&(peaks[:,1]>0)]
        if not len(peaks):continue
        mz,idx=np.unique(peaks[:,0],return_inverse=True);intensity=np.bincount(idx,weights=peaks[:,1])
        keep=intensity>=relative_floor*intensity.max();mz=mz[keep];intensity=intensity[keep]
        neutral=mz-(PROTON if mode=='[M+H]+' else -PROTON);keep=neutral>0
        neutral=neutral[keep];mz=mz[keep];intensity=intensity[keep]
        if not len(neutral):continue
        weight=np.sqrt(intensity);weight/=weight.sum()
        prepared.append((neutral,np.maximum(da,mz*ppm*1e-6),weight))
    return prepared


def _evidence(prepared, hypotheses):
    if not hypotheses:return 0.
    masses=np.array(sorted(hypotheses));values=np.array([hypotheses[m] for m in masses]);all_scores=[]
    for neutral,tol,weight in prepared:
        lo=np.searchsorted(masses,neutral-tol,side='left');hi=np.searchsorted(masses,neutral+tol,side='right')
        evidence=np.array([values[a:b].max() if b>a else 0. for a,b in zip(lo,hi)])
        all_scores.append(float(weight@evidence))
    return float(np.clip(np.mean(all_scores),0.,1.))


def score_structure(queries, smiles, *, max_cuts=2, max_scenarios=8192, hydrogen_shift=1,
                    ppm=5., da=.001, relative_floor=.01):
    _settings(max_cuts,max_scenarios)
    if type(hydrogen_shift) is not int or not 0<=hydrogen_shift<=2:raise ValueError('Invalid H-transfer limit')
    if not np.isfinite([ppm,da,relative_floor]).all() or min(ppm,da)<=0 or not 0<=relative_floor<=1:
        raise ValueError('Invalid tolerance or intensity floor')
    queries=list(queries);prepared=_prepare(queries,ppm,da,relative_floor)
    result={'status':'no_supported_spectra','total_spectra':len(queries),'supported_spectra':len(prepared),
            'one_cut':None,'two_cut':None,'connected_compositions':0,'hypothesis_masses':0}
    if not prepared:return result
    try:fragments=connected_fragments(smiles,max_cuts=max_cuts,max_scenarios=max_scenarios)
    except FragmentBudgetExceeded:result['status']='fragment_budget_exceeded';return result
    except ValueError:result['status']='unsupported_structure';return result
    hypotheses=[{},{}];hmass=element_mass('H')
    # Transferred H cannot make the fragment H count negative or exceed parent's
    # H inventory. It is a permissive hypothesis, not a feasible ion assignment.
    from .formula_evidence import composition
    parent_h=dict(composition(smiles)).get('H',0)
    for r in fragments:
        fragment_h=dict(r['composition']).get('H',0);limit=min(hydrogen_shift,r['cuts'])
        for dh in range(-limit,limit+1):
            if not 0<=fragment_h+dh<=parent_h:continue
            mass=r['mass']+dh*hmass
            if mass<=0:continue
            value=1./(r['cuts']+abs(dh))
            hypotheses[1][mass]=max(value,hypotheses[1].get(mass,0.))
            if r['cuts']==1:hypotheses[0][mass]=max(value,hypotheses[0].get(mass,0.))
    result.update(status='scored',one_cut=_evidence(prepared,hypotheses[0]),two_cut=_evidence(prepared,hypotheses[1]),
                  connected_compositions=len(fragments),hypothesis_masses=len(hypotheses[1]))
    return result

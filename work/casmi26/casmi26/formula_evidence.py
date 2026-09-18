"""Experimental atom-budget evidence from measured fragment masses.

This enumerates elemental subcompositions, not molecular graphs, fragment
connectivity, isotope patterns, or SIRIUS fragmentation trees. All masses come
from the candidate structure. No answer formula or answer SMILES is read from
query metadata. Only protonated/deprotonated, neutral parent structures are
supported. Missing evidence never becomes an exclusion of the candidate.
"""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
from math import prod
import numpy as np

PROTON = 1.007276466621
ELEMENTS = frozenset(('C','H','N','O','P','S','F','Cl','Br','I'))
SHIFTS = (-0.2345678, 0.1234567, 0.3456789)


class FormulaBudgetExceeded(ValueError):
    """The declared bound prevents uncontrolled combinatorial allocation."""


@lru_cache(maxsize=16)
def element_mass(element: str) -> float:
    if element not in ELEMENTS:
        raise ValueError('Unsupported element: '+str(element))
    from rdkit import Chem
    return float(Chem.GetPeriodicTable().GetMostCommonIsotopeMass(element))


def composition(smiles: str) -> tuple[tuple[str,int], ...]:
    from rdkit import Chem
    mol=Chem.MolFromSmiles(smiles) if isinstance(smiles,str) else None
    if mol is None or mol.GetNumAtoms()==0 or len(Chem.GetMolFrags(mol))!=1:
        raise ValueError('Candidate must be one valid molecule')
    if Chem.GetFormalCharge(mol)!=0 or any(a.GetIsotope()!=0 for a in mol.GetAtoms()):
        raise ValueError('Charged or isotope-labelled candidate is unsupported')
    counts=Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms())
    if set(counts)-ELEMENTS:
        raise ValueError('Candidate contains unsupported elements')
    return tuple(sorted(counts.items()))


def subformula_masses(formula, *, max_states: int=2_000_000) -> np.ndarray:
    if type(max_states) is not int or max_states<1:
        raise ValueError('Invalid enumeration budget')
    formula=tuple(formula)
    if (not formula or len({e for e,n in formula})!=len(formula) or
        any(e not in ELEMENTS or type(n) is not int or n<0 for e,n in formula)):
        raise ValueError('Invalid elemental composition')
    states=prod(n+1 for e,n in formula)
    if states>max_states:
        raise FormulaBudgetExceeded('Subformula state count '+str(states)+' exceeds '+str(max_states))
    masses=np.array([0.],dtype=np.float64)
    for element,count in sorted(formula):
        masses=(masses[:,None]+np.arange(count+1,dtype=np.float64)*element_mass(element)).reshape(-1)
    return np.unique(masses)


def nearest_error(sorted_masses: np.ndarray, values: np.ndarray) -> np.ndarray:
    a=np.asarray(sorted_masses,dtype=np.float64);v=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or np.any(np.diff(a)<0) or not np.isfinite(v).all():
        raise ValueError('Invalid sorted masses or query values')
    idx=np.searchsorted(a,v)
    return np.minimum(abs(v-a[np.clip(idx,0,len(a)-1)]),abs(v-a[np.clip(idx-1,0,len(a)-1)]))


def score_formula(queries, formula, *, ppm=5., da=.001, relative_floor=.01, max_states=2_000_000):
    if (not np.isfinite([ppm,da,relative_floor]).all() or min(ppm,da)<=0 or
        not 0<=relative_floor<=1):
        raise ValueError('Invalid fragment tolerance or relative floor')
    queries=list(queries)
    result={'status':'no_supported_spectra','supported_spectra':0,'total_spectra':len(queries),
            'explained':None,'chance':None,'excess':None,'subformula_states':0}
    prepared=[]
    for q in queries:
        mode=q.get('adduct')
        if mode not in ('[M+H]+','[M-H]-'):continue
        precursor=float(q['precursor']);p=np.asarray(q['peaks'],dtype=np.float64)
        if not np.isfinite(precursor) or precursor<=0:
            raise ValueError('Invalid query precursor')
        if not p.size:continue
        if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all() or np.any(p[:,0]<=0) or np.any(p[:,1]<0):
            raise ValueError('Invalid query peaks')
        p=p[(p[:,0]<precursor-.5)&(p[:,1]>0)]
        if not len(p):continue
        mz,idx=np.unique(p[:,0],return_inverse=True);intensity=np.bincount(idx,weights=p[:,1])
        keep=intensity>=relative_floor*intensity.max();mz=mz[keep];intensity=intensity[keep]
        shift=PROTON if mode=='[M+H]+' else -PROTON
        neutral=mz-shift;keep=neutral>0
        neutral=neutral[keep];mz=mz[keep];intensity=intensity[keep]
        if not len(neutral):continue
        weights=np.sqrt(intensity);weights/=weights.sum()
        prepared.append((neutral,np.maximum(da,mz*ppm*1e-6),weights))
    if not prepared:return result
    try:masses=subformula_masses(formula,max_states=max_states)
    except FormulaBudgetExceeded:
        result['status']='formula_budget_exceeded';return result
    # One search per spectrum, jointly including three deterministic shifted controls.
    # These shifts are arithmetic controls, not chemically valid decoy structures.
    values=[]
    for neutral,tol,weight in prepared:
        targets=neutral[:,None]+np.array((0.,)+SHIFTS)
        errors=nearest_error(masses,targets)
        matched=(errors<=tol[:,None])&(targets>0)
        evidence=weight@matched.astype(np.float64)
        values.append((float(evidence[0]),float(evidence[1:].mean())))
    a=np.asarray(values);explained=float(a[:,0].mean());chance=float(a[:,1].mean())
    result.update(status='scored',supported_spectra=len(prepared),explained=explained,chance=chance,
                  excess=float(np.clip(explained-chance,0.,1.)),subformula_states=len(masses))
    return result

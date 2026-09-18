"""Graph-fragment contracts, checked against an independent small-graph oracle."""
from itertools import combinations
import importlib
import math
import numpy as np
import pytest


def implementation():
    name='casmi26.connected_evidence'
    assert importlib.util.find_spec(name) is not None, 'Connected-fragment implementation is missing'
    return importlib.import_module(name)


def oracle(smiles, max_cuts):
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    m=rdMolStandardize.TautomerEnumerator().Canonicalize(Chem.MolFromSmiles(smiles))
    m=Chem.RemoveHs(m); n=m.GetNumAtoms()
    edges=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in m.GetBonds()]
    out={}
    for count in range(1,max_cuts+1):
        for removed in combinations(range(len(edges)),count):
            adjacency=[set() for _ in range(n)]
            for j,(a,b) in enumerate(edges):
                if j not in removed:adjacency[a].add(b);adjacency[b].add(a)
            todo=set(range(n))
            while todo:
                start=next(iter(todo)); component={start};queue=[start]
                while queue:
                    a=queue.pop()
                    for b in adjacency[a]-component:component.add(b);queue.append(b)
                todo-=component
                if len(component)==n:continue
                atoms={}
                for i in component:
                    atom=m.GetAtomWithIdx(i);e=atom.GetSymbol();atoms[e]=atoms.get(e,0)+1
                    atoms['H']=atoms.get('H',0)+atom.GetTotalNumHs()
                formula=tuple(sorted((e,c) for e,c in atoms.items() if c))
                out[formula]=min(count,out.get(formula,count))
    return out


@pytest.mark.parametrize('smiles',['CCCO','CC(C)(C)O','C1CCCCC1','c1ccccc1O','CCOC(=O)C','CCN(CC)CC','O=C1CCCCC1','CC(=O)O','ClCCBr'])
@pytest.mark.parametrize('cuts',[1,2])
def test_enumeration_equals_independent_all_edge_cut_oracle(smiles,cuts):
    records=implementation().connected_fragments(smiles,max_cuts=cuts)
    actual={tuple(map(tuple,r['composition'])):r['cuts'] for r in records}
    assert actual==oracle(smiles,cuts)
    for r in records:
        assert r['cuts']<=cuts
        assert r['mass']>0


def test_rings_need_two_cuts_without_invented_hydrogen_capping():
    m=implementation()
    assert m.connected_fragments('C1CCCCC1',max_cuts=1)==[]
    records=m.connected_fragments('C1CCCCC1',max_cuts=2)
    for r in records:
        f=dict(r['composition']);assert f['H']==2*f['C'];assert r['cuts']==2
    assert any(dict(r['composition'])=={'C':1,'H':2} for r in records)


def test_same_formula_isomers_can_have_different_connected_fragments():
    m=implementation()
    from casmi26.formula_evidence import composition
    assert composition('CCCCO')==composition('CC(C)(C)O')
    a=m.connected_fragments('CCCCO',max_cuts=1);b=m.connected_fragments('CC(C)(C)O',max_cuts=1)
    assert a!=b
    target=next(r['mass'] for r in a if dict(r['composition'])=={'C':1,'H':3,'O':1})
    q={'adduct':'[M+H]+','precursor':75.08,'peaks':[[target+m.PROTON,1.]]}
    aa=m.score_structure([q],'CCCCO');bb=m.score_structure([q],'CC(C)(C)O')
    assert aa['one_cut']>bb['one_cut']


def test_representation_and_peak_order_invariance():
    from rdkit import Chem
    m=implementation();mol=Chem.MolFromSmiles('CCC(C)CO');reversed_=Chem.RenumberAtoms(mol,list(reversed(range(mol.GetNumAtoms()))))
    assert m.connected_fragments('CCC(C)CO')==m.connected_fragments(Chem.MolToSmiles(reversed_,canonical=False))
    q={'adduct':'[M+H]+','precursor':89.1,'peaks':[[31.0183897+m.PROTON,.8],[43.02,.2]]}
    a=m.score_structure([q],'CCC(C)CO')
    q2={**q,'peaks':[[43.02,20],[31.0183897+m.PROTON,80]]}
    b=m.score_structure([q2],'CCC(C)CO')
    assert a['one_cut']==pytest.approx(b['one_cut'],abs=1e-14)
    assert a['two_cut']==pytest.approx(b['two_cut'],abs=1e-14)


@pytest.mark.parametrize('smiles',['bad','CC.O','[Na+]','[13CH3]CO','[CH3]','[SiH4]',''])
def test_invalid_or_unsupported_is_explicit(smiles):
    m=implementation()
    with pytest.raises(ValueError):m.connected_fragments(smiles)
    q={'adduct':'[M+H]+','precursor':75.,'peaks':[[31.,1.]]}
    a=m.score_structure([q],smiles)
    assert a['status']=='unsupported_structure' and a['two_cut'] is None


def test_budget_is_fail_closed_not_prefix_dependent():
    m=implementation()
    with pytest.raises(m.FragmentBudgetExceeded):m.connected_fragments('CCCCCCCC',max_scenarios=1)
    q={'adduct':'[M+H]+','precursor':100.,'peaks':[[31.,1.]]}
    a=m.score_structure([q],'CCCCCCCC',max_scenarios=1)
    assert a['status']=='fragment_budget_exceeded' and a['two_cut'] is None


@pytest.mark.parametrize('kwargs',[{'max_cuts':0},{'max_cuts':3},{'max_cuts':True},{'max_scenarios':0},{'hydrogen_shift':3},{'ppm':0},{'da':float('nan')},{'relative_floor':-1}])
def test_invalid_parameters_rejected(kwargs):
    m=implementation();q={'adduct':'[M+H]+','precursor':100.,'peaks':[[31.,1.]]}
    with pytest.raises(ValueError):m.score_structure([q],'CCCCO',**kwargs)


def test_missing_and_unsupported_queries_do_not_invent_evidence():
    m=implementation()
    assert m.score_structure([],'CCCCO')['two_cut'] is None
    q={'adduct':'[M+Na]+','precursor':97.1,'peaks':[[31.,1.]]}
    assert m.score_structure([q],'CCCCO')['status']=='no_supported_spectra'
    q={'adduct':'[M+H]+','precursor':75.,'peaks':[[75.,1.]]}
    assert m.score_structure([q],'CCCCO')['status']=='no_supported_spectra'


def test_query_answers_never_used_and_evidence_bounded():
    m=implementation();q={'adduct':'[M-H]-','precursor':73.,'peaks':[[31.,.2],[29.,1.]]}
    a=m.score_structure([q],'CCCCO')
    b=m.score_structure([{**q,'formula':'BOGUS','normalized_smiles':'not used','key':'wrong'}],'CCCCO')
    assert a==b
    assert 0<=a['one_cut']<=1 and 0<=a['two_cut']<=1


@pytest.mark.parametrize('peaks',[[[1.,-1.]],[[float('nan'),1.]],[[31.,float('inf')]],[[0.,1.]],[[1,2,3]]])
def test_malformed_observations_rejected(peaks):
    with pytest.raises(ValueError):implementation().score_structure([{'adduct':'[M+H]+','precursor':100.,'peaks':peaks}],'CCCCO')

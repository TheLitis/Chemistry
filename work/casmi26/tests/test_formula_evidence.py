"""Exact atom-budget arithmetic, not an assertion of chemical connectivity."""
from itertools import product
from pathlib import Path
import importlib
import numpy as np
import pytest


def module():
    assert (Path(__file__).parents[1]/'casmi26/formula_evidence.py').is_file(), 'Formula evidence module is missing'
    return importlib.import_module('casmi26.formula_evidence')


def query(mode='[M+H]+'):
    m=module(); f=m.composition('CCO')
    neutral=sum(m.element_mass(e)*n for e,n in f)
    fragment=m.element_mass('C')+2*m.element_mass('H')+m.element_mass('O')
    shift=m.PROTON if mode=='[M+H]+' else -m.PROTON
    return {'adduct':mode,'precursor':neutral+shift,'peaks':[[fragment+shift,1.]],'ce':[20.]}


def test_composition_isomer_order_and_explicit_hydrogen():
    m=module();assert m.composition('CCO')==m.composition('COC')==m.composition('[H]OC([H])([H])C')
    assert dict(m.composition('CCO'))=={'C':2,'H':6,'O':1}


@pytest.mark.parametrize('smiles',['not smiles','[Na+]','[NH4+]','[13CH4]','CCO.C','[O-]C=O'])
def test_unsupported_composition_is_explicit(smiles):
    with pytest.raises(ValueError):module().composition(smiles)


def test_small_formula_enumeration_matches_independent_cartesian_product():
    m=module();f=(('C',2),('H',4),('O',1));actual=m.subformula_masses(f)
    expected=np.array(sorted(sum(k*m.element_mass(e) for k,(e,n) in zip(ns,f)) for ns in product(*(range(n+1) for e,n in f))))
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-13)
    assert len(actual)==30 and actual[0]==0 and np.all(np.diff(actual)>0)


def test_budget_rejects_before_allocation():
    m=module()
    with pytest.raises(m.FormulaBudgetExceeded):m.subformula_masses((('C',100),('H',200),('N',20),('O',40)),max_states=100)


@pytest.mark.parametrize('f',[(('C',-1),),(('C',1.5),),(('C',True),),(('C',1),('C',1)),(('Xe',1),),()])
def test_bad_formula_rejected(f):
    with pytest.raises(ValueError):module().subformula_masses(f)


@pytest.mark.parametrize('mode',['[M+H]+','[M-H]-'])
def test_correct_formula_explains_exact_peak_but_oxygen_free_formula_does_not(mode):
    m=module();q=query(mode)
    good=m.score_formula([q],m.composition('CCO'))
    bad=m.score_formula([q],m.composition('CCC'))
    assert good['supported_spectra']==1 and good['explained']==pytest.approx(1.)
    assert good['excess']==pytest.approx(1.) and bad['explained']==pytest.approx(0.)


def test_scaling_permutation_label_and_duplicate_invariance():
    m=module();q=query();r={**q,'peaks':q['peaks']*2,'normalized_smiles':'LEAK','molecular_formula':'WRONG'}
    r['peaks']=[[p[0],p[1]*100] for p in r['peaks']]
    a=m.score_formula([q],m.composition('CCO'));b=m.score_formula([r],m.composition('CCO'))
    assert a==b


def test_no_model_for_unsupported_ion_or_oversized_formula():
    m=module();q=query();q['adduct']='[M+Na]+'
    r=m.score_formula([q],m.composition('CCO'))
    assert r['status']=='no_supported_spectra' and r['excess'] is None
    r=m.score_formula([query()],(('C',100),('H',200),('O',40)),max_states=5)
    assert r['status']=='formula_budget_exceeded' and r['excess'] is None


def test_noise_floor_and_precursor_are_excluded():
    m=module();q=query();q['peaks'] += [[q['precursor'],100.],[22.12345,.0001]]
    r=m.score_formula([q],m.composition('CCO'))
    assert r['explained']==pytest.approx(1.)


@pytest.mark.parametrize('kwargs',[{'ppm':0},{'da':-1},{'relative_floor':2},{'ppm':float('nan')}])
def test_invalid_tolerance_is_not_a_quality_result(kwargs):
    with pytest.raises(ValueError):module().score_formula([query()],module().composition('CCO'),**kwargs)


def test_nearest_mass_error_tie_and_boundaries():
    m=module();a=np.array([0.,10.,20.]);q=np.array([-.1,0.,6.,15.,21.])
    np.testing.assert_allclose(m.nearest_error(a,q),[.1,0.,4.,5.,1.])

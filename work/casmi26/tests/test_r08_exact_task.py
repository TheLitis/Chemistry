import importlib.util
from pathlib import Path
import pytest


def load():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_r08_exact.py'
    assert path.is_file(), 'Exact forward comparison task is missing'
    spec=importlib.util.spec_from_file_location('r08_exact_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def records():
    return [{'key':'DO_NOT_USE_AS_TRUTH','queries':[{'adduct':'[M+H]+'}],
             'cases':{'one':{'keys':['A','B','C'],'smiles':['CC','CCC','CCCC'],'scores':[1.,.99,-10.]}}}]


def test_request_envelope_ignores_query_answers_and_keeps_prior_representative():
    m=load();r=records()
    a=m.make_requests(r,{'A':{'smiles':'C(C)'}},weight=.25,k=1)
    assert [v['key'] for v in a]==['A','B']
    assert a[0]['smiles']=='C(C)'
    r[0]['key']='B';r[0]['truth']='C';r[0]['queries'][0]['smiles']='PRIVATE_ANSWER'
    assert m.make_requests(r,{'A':{'smiles':'C(C)'}},weight=.25,k=1)==a


def test_no_simulations_for_unsupported_adduct():
    m=load();r=records();r[0]['queries'][0]['adduct']='[M+Na]+'
    assert m.make_requests(r,{},weight=.25,k=1)==[]


def test_request_modes_union_and_deterministic_order():
    m=load();r=records();s=records()[0];s['queries']=[{'adduct':'[M-H]-'}];r.append(s)
    a=m.make_requests(r,{},weight=.25,k=1)
    assert a[0]['modes']==['[M+H]+','[M-H]-']
    assert a==m.make_requests(list(reversed(r)),{},weight=.25,k=1)

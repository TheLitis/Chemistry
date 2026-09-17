"""Exact branch-and-bound ranking with a bounded nonnegative evidence bonus."""
import importlib.util
import numpy as np
import pytest


def functions():
    assert importlib.util.find_spec('casmi26.bounded_forward') is not None, 'Bounded forward ranker is not implemented'
    from casmi26.bounded_forward import candidate_envelope, certified_rerank
    return candidate_envelope, certified_rerank


def test_ranker_can_recover_a_candidate_beyond_fixed_top32():
    envelope, rank = functions()
    base=np.arange(100,0,-1,dtype=float)/1000
    bonuses=np.zeros(100);bonuses[70]=1
    called=[]
    result=rank(base,lambda i:(called.append(i),bonuses[i])[1],weight=.25,k=25)
    assert result['top_indices'][0]==70
    assert result['certified'] is True
    assert len(called)==len(set(called))
    assert 70 in envelope(base,.25,k=25)


def test_pruning_saves_calls_and_agrees_with_exhaustive_scores():
    _,rank=functions();base=np.array([4.,3.,2.,-20.,-30.]);calls=[]
    result=rank(base,lambda i:(calls.append(i),.2)[1],weight=.25,k=2)
    assert result['top_indices']==[0,1] and result['certified']
    assert calls==[0,1]
    assert result['evaluations']==2 and result['pruned']==3


def test_zero_weight_never_calls_model_and_preserves_ties():
    _,rank=functions()
    def fail(_):raise AssertionError('No prediction needed for zero weight')
    r=rank([1.,1.,0.],fail,weight=0.,k=2)
    assert r['top_indices']==[0,1] and r['certified'] and r['evaluations']==0


def test_equal_upper_bound_is_evaluated_not_silently_pruned():
    _,rank=functions()
    r=rank([1.,1.],lambda _:0.,weight=.1,k=1)
    assert r['top_indices']==[0] and r['evaluations']==2 and r['certified']


def test_budget_exhaustion_is_not_an_exact_certificate():
    _,rank=functions()
    r=rank([1.,.99,.98],lambda i:float(i==2),weight=.25,k=1,max_evaluations=1)
    assert r['certified'] is False and r['evaluations']==1
    assert r['top_indices']==[0]


def test_missing_forward_evidence_is_explicit_zero_bonus():
    _,rank=functions()
    r=rank([1.,.9],lambda _:None,weight=.25,k=2)
    assert r['certified'] and r['missing_evidence']==2
    assert r['top_indices']==[0,1]


@pytest.mark.parametrize('bonus',[float('nan'),float('inf'),-.1,1.1,'bad'])
def test_invalid_bonus_fails_without_a_certificate(bonus):
    _,rank=functions()
    with pytest.raises((ValueError,TypeError)):
        rank([1.],lambda _:bonus,weight=.25,k=1)


@pytest.mark.parametrize('base,weight,k',[([float('nan')],.25,1),([1.],-.1,1),([1.],float('inf'),1),([1.],.25,0),([1.],.25,True),([[1.]],.25,1)])
def test_bad_contract_rejected(base,weight,k):
    envelope,rank=functions()
    with pytest.raises(ValueError):envelope(base,weight,k=k)
    with pytest.raises(ValueError):rank(base,lambda _:0.,weight=weight,k=k)


def test_empty_inputs_have_empty_certified_toplist():
    _,rank=functions()
    r=rank([],lambda _:0.,weight=.25,k=25)
    assert r['top_indices']==[] and r['certified']


def test_randomized_topk_and_envelope_match_brute_force():
    envelope,rank=functions();rng=np.random.default_rng(26091708)
    for n in (1,7,33,129):
        for _ in range(30):
            base=np.round(rng.normal(size=n),2);bonus=rng.uniform(size=n);weight=float(rng.uniform(.01,2));k=min(n,25)
            expected=np.argsort(-(base+weight*bonus),kind='stable')[:k].tolist()
            possible=set(map(int,envelope(base,weight,k=k)))
            assert set(expected)<=possible
            result=rank(base,lambda i:bonus[i],weight=weight,k=k)
            assert result['top_indices']==expected and result['certified']
            assert result['evaluations']<=len(possible)

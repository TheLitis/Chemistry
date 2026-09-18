"""Exact selected-feature path; do not compute unused max-grid comparisons."""
import numpy as np
import pytest
from casmi26.forward_ranking import pool_forward_scores


def fast():
    from importlib.util import find_spec
    assert find_spec('casmi26.forward_nearest') is not None, 'Missing selected-feature-only evaluator'
    from casmi26.forward_nearest import pool_cosine_nearest_only
    return pool_cosine_nearest_only


def example(ce):
    queries=[{'adduct':'[M+H]+','precursor':200.,'peaks':[[40.001,2],[30.,1],[40.001,2]],'ce':ce},
             {'adduct':'[M-H]-','precursor':198.,'peaks':[[40.,1],[70.,2]],'ce':None}]
    predictions={('[M+H]+',40.):[[40.,2],[70.,1]],('[M+H]+',10.):[[30.,10],[50.,1]],
                 ('[M-H]-',10.):[[40.005,1],[70.,1]],('[M-H]-',40.):[[40.005,1],[150.,1]]}
    return queries,predictions


@pytest.mark.parametrize('ce',[None,[],[10.],[20.,40.,40.],30.,[float('nan'),40.],[-10.,10.]])
def test_identical_to_selected_original_feature(ce):
    q,p=example(ce)
    assert fast()(q,p)==pool_forward_scores(q,p)['features']['cosine_nearest']


def test_unsupported_or_missing_evidence_is_none_not_zero():
    q,p=example([20.])
    q[0]['adduct']='[M+Na]+';q[1]['adduct']='[M+Cl]-'
    assert fast()(q,p) is None
    q,p=example([20.])
    assert fast()(q,{}) is None
    assert fast()(q,{('[M+H]+',20.):None}) is None


def test_empty_predicted_peaks_remain_zero_evidence():
    q,_=example([20.])
    p={('[M+H]+',20.):[]}
    assert fast()(q,p)==0.0


def test_unused_malformed_grid_peak_still_rejected():
    q,p=example([40.])
    p[('[M+H]+',10.)]=[[30.,-1.]]
    with pytest.raises(ValueError): pool_forward_scores(q,p)
    with pytest.raises(ValueError): fast()(q,p)


def test_target_fields_do_not_change_score():
    q,p=example([10.]);expected=fast()(q,p)
    for row in q:row.update(smiles='SECRET-ANSWER',formula='DO-NOT-USE',target='IGNORE')
    assert fast()(q,p)==expected


def test_repeated_random_peaks_and_scaling_match_exactly():
    rng=np.random.default_rng(26091808)
    f=fast()
    for _ in range(25):
        q,p=example([10.,40.])
        p={k:np.column_stack((rng.integers(10,180,40),rng.uniform(.01,100,40))) for k in p}
        q[0]['peaks']=np.column_stack((rng.integers(10,180,30),rng.uniform(.01,100,30)))
        assert f(q,p)==pool_forward_scores(q,p)['features']['cosine_nearest']

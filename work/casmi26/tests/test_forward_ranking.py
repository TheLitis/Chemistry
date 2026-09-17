import numpy as np
import pytest


def test_clean_peaks_is_order_scale_invariant_and_excludes_precursor():
    from casmi26.forward_ranking import clean_peaks
    a=clean_peaks([[20,2],[10,1],[20,3],[100,100]],100)
    b=clean_peaks([[100,1000],[20,50],[10,10]],100)
    np.testing.assert_array_equal(a,b)
    assert a[:,0].tolist()==[10,20]
    assert a[:,1].sum()==pytest.approx(1)
    assert len(clean_peaks([[100,1]],100))==0


@pytest.mark.parametrize('bad', [[[float('nan'),1]], [[10,-1]], [[-5,1]], [[10,1,2]]])
def test_invalid_peaks_not_silently_used(bad):
    from casmi26.forward_ranking import clean_peaks
    with pytest.raises(ValueError):clean_peaks(bad,100)


def test_similarity_identity_disjoint_and_precursor_leakage():
    from casmi26.forward_ranking import compare_spectra
    x=[[20,1],[30,3]]
    a=compare_spectra(x,x,100)
    assert a['cosine']==pytest.approx(1)
    assert a['entropy']==pytest.approx(1)
    b=compare_spectra(x,[[40,2]],100)
    assert b['cosine']==0 and b['entropy']==0
    c=compare_spectra([[100,9]],[[100,9]],100)
    assert c['cosine']==0 and c['matched_peaks']==0


def test_matching_is_one_to_one_and_bounded():
    from casmi26.forward_ranking import compare_spectra
    a=compare_spectra([[20,1],[20.001,1]],[[20.0005,1]],100)
    assert a['matched_peaks']==1
    assert 0<=a['cosine']<1 and 0<=a['entropy']<1


def test_comparison_invariant_to_peak_permutation():
    from casmi26.forward_ranking import compare_spectra
    a=[[20,1],[20.005,5],[30,2]];b=[[20.004,2],[20.001,3],[40,1]]
    assert compare_spectra(a,b,100)==compare_spectra(a[::-1],b[::-1],100)


def test_partial_rerank_never_changes_unscored_or_unsupplied_indices():
    from casmi26.forward_ranking import add_forward_evidence
    base=np.array([.7,.6,.5]);f=np.array([0.,.8,np.nan])
    out=add_forward_evidence(base,f,.5)
    np.testing.assert_allclose(out,[.7,1.,.5]);np.testing.assert_allclose(base,[.7,.6,.5])
    np.testing.assert_array_equal(add_forward_evidence(base,f,0),base)
    with pytest.raises(ValueError):add_forward_evidence(base,f,-1)


def test_candidate_frontier_has_no_target_argument_and_preserves_budget():
    from casmi26.forward_ranking import candidate_frontier
    groups={'available':{'keys':['a','b','c'],'scores':[3,2,1]},
            'absent':{'keys':['c','b','a'],'scores':[3,2,1]}}
    assert candidate_frontier(groups,1)==['a','c']
    assert set(candidate_frontier(groups,2))=={'a','b','c'}
    with pytest.raises(ValueError):candidate_frontier(groups,0)


def test_partial_pooling_has_no_fake_evidence_for_unsupported_adduct():
    from casmi26.forward_ranking import pool_forward_scores
    q=[{'adduct':'[M+Na]+','precursor':100,'peaks':[[20,1]],'ce':[20]}]
    a=pool_forward_scores(q,{})
    assert a['supported_spectra']==0 and a['features']=={}


def test_pooling_uses_all_supported_spectra_and_distinguishes_energy_rule():
    from casmi26.forward_ranking import pool_forward_scores
    p={('[M+H]+',10.):[[20,1]],('[M+H]+',40.):[[30,1]]}
    q=[{'adduct':'[M+H]+','precursor':100,'peaks':[[20,1]],'ce':[10]},
       {'adduct':'[M+H]+','precursor':100,'peaks':[[30,1]],'ce':[10]}]
    a=pool_forward_scores(q,p)
    assert a['supported_spectra']==2
    assert a['features']['cosine_nearest']==pytest.approx(.5)
    assert a['features']['cosine_max']==pytest.approx(1.)


def test_energy_mixture_is_a_spectrum_not_a_best_answer_oracle():
    from casmi26.forward_ranking import pool_forward_scores
    p={('[M+H]+',10.):[[20,1]],('[M+H]+',40.):[[30,1]]}
    q=[{'adduct':'[M+H]+','precursor':100,'peaks':[[20,1]],'ce':[10,40]}]
    a=pool_forward_scores(q,p)
    assert a['features']['cosine_nearest']==pytest.approx(2**-.5)

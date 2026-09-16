import numpy as np
import pytest
from casmi26.ranking import neural_scores, ranked_indices, paired_effect, spectrum_signature


def test_neural_scores_use_present_and_absent_bits():
    p=np.array([.9,.1]); fp=np.array([[1,0],[0,1],[1,1]])
    s=neural_scores(p,fp,np.array([100.,100.,100.]),100.)
    assert s[0]>s[2]>s[1]


def test_mass_prior_only_breaks_chemical_ties():
    s=neural_scores(np.array([.7]),np.ones((2,1)),np.array([100.,100.003]),100.)
    assert s[0]>s[1]


def test_extreme_posteriors_do_not_make_infinite_scores():
    assert np.isfinite(neural_scores([0.,1.],[[1,0],[0,1]],[100.,100.],100.)).all()


def test_zero_reference_evidence_cannot_override_neural():
    idx, gate=ranked_indices([1.,3.,2.],[0.,0.,0.],['a','b','c'],threshold=.8,margin=.1)
    assert idx.tolist()==[1,2,0] and not gate['activated']


def test_weak_reference_evidence_cannot_override_neural():
    idx,gate=ranked_indices([3.,1.],[.1,.7],['a','b'],threshold=.9,margin=.05)
    assert idx.tolist()==[0,1] and not gate['activated']


def test_strong_but_ambiguous_match_cannot_override():
    idx,gate=ranked_indices([3.,1.],[.94,.95],['a','b'],threshold=.9,margin=.1)
    assert idx.tolist()==[0,1] and not gate['activated']


def test_confident_match_promotes_one_preserving_remaining_order():
    idx,gate=ranked_indices([4.,3.,2.,1.],[.2,.1,.95,.3],['a','b','c','d'],threshold=.9,margin=.1)
    assert idx.tolist()==[2,0,1,3] and gate['activated']


def test_tautomer_equivalents_do_not_hide_distinct_runner_up():
    idx,gate=ranked_indices([4.,3.,2.],[.1,.95,.94],['a','b','b'],threshold=.9,margin=.1)
    assert gate['activated'] and gate['margin']==pytest.approx(.85)
    assert idx[0]==1


def test_exact_threshold_does_not_treat_zero_as_confident():
    with pytest.raises(ValueError):ranked_indices([1.],[0.],['a'],threshold=0,margin=0)


@pytest.mark.parametrize('bad',[[1.,float('nan')],[1.,float('inf')]])
def test_nonfinite_scores_rejected(bad):
    with pytest.raises(ValueError):ranked_indices(bad,[.1,.2],['a','b'])


def test_signature_ignores_peak_order_and_intensity_scale():
    a={'adduct':'[M+H]+','precursor':100.,'peaks':np.array([[20.,1.],[30.,2.]])}
    b={**a,'peaks':np.array([[30.,20.],[20.,10.]])}
    assert spectrum_signature(a)==spectrum_signature(b)


def test_paired_effect_retains_molecule_as_bootstrap_unit():
    a=np.array([.2,.5,.7]);b=a+.1
    result=paired_effect(a,b,repeats=200)
    assert result['delta_mrr']==pytest.approx(.1)
    assert result['ci95']==pytest.approx([.1,.1])

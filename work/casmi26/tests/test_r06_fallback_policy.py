import numpy as np


def test_r06_candidate_policy_preserves_strict_window_when_available():
    from casmi26.r06_candidate import select_mass_candidates
    masses=np.array([99.0,100.0005,100.0010,101.0])
    idx,mode=select_mass_candidates(masses,100.0)
    assert mode=='mass_compatible'
    assert idx.tolist()==[1,2]


def test_r06_candidate_policy_expands_then_falls_back_instead_of_empty_cell():
    from casmi26.r06_candidate import select_mass_candidates
    masses=np.array([99.0,100.0,101.0,102.0])
    idx,mode=select_mass_candidates(masses,100.015)
    assert mode=='expanded_mass_window'
    assert len(idx)>0
    idx,mode=select_mass_candidates(masses,500.0)
    assert mode=='nearest_mass_no_compatible_structure'
    assert 1<=len(idx)<=64
    assert np.all((idx>=0)&(idx<len(masses)))

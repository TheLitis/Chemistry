import importlib.util
from pathlib import Path
import numpy as np


def load():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_ensemble_research.py'
    assert path.exists(), 'R02 implementation absent'
    spec=importlib.util.spec_from_file_location('r02test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_fresh_split():
    m=load();a,b=m.fresh_split([str(i) for i in range(40)],{'1','2'},5,8)
    assert len(a)==5 and len(b)==8 and not (set(a)&set(b))
    assert not ({'1','2'}&(set(a)|set(b)))


def test_zero_spectral_information_does_not_change_r01():
    m=load();s=np.zeros(3);t=np.array([.1,.2,.5]);b=np.array([2.,3.,4.]);p=np.zeros(3)
    result=m.score_variants(s,t,b,p)
    np.testing.assert_allclose(result['fusion:4'],result['r01'])


def test_original_blend_preserved():
    m=load();s=np.array([.2,.9]);t=np.array([.4,.1])
    scores=m.score_variants(s,t,np.ones(2),np.zeros(2))
    np.testing.assert_allclose(scores['old_blend'],.75*s+.25*t)


def test_holdout_keys_never_enter_references():
    m=load();catalog=[['a','a','TRAIN',1],['b','b','VAL',1],['c','c','VAL',1]]
    assert m.reference_ids(catalog,[0,1,2],{'VAL'})==[0]

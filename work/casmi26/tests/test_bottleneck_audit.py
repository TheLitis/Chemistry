import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    p=Path(__file__).resolve().parents[3]/'tasks/casmi_bottleneck_audit.py'
    assert p.is_file(), 'Bottleneck audit not implemented'
    spec=importlib.util.spec_from_file_location('audit',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_correct_adducts_and_water_loss_mislabelling():
    m=module()
    from casmi26.production import ion
    exact=np.array([200.,200.,200.]);ads=['[M+H]+','[M-H2O+H]+','[M+H]+']
    prec=np.array([200+ion(a)[2] for a in ads]);prec[2]-=18.010564684
    got=m.classify_precursors(prec,exact,ads)
    assert got['accepted'].tolist()==[True,True,False]
    assert got['water_loss_compatible'].tolist()==[False,False,True]


def test_unknown_or_nonfinite_is_not_validated():
    m=module();got=m.classify_precursors([201.,np.nan,201.],[200.,200.,np.nan],['invalid','[M+H]+','[M+H]+'])
    assert not got['accepted'].any()
    assert not got['water_loss_compatible'].any()


def test_external_pull_is_only_source_not_execution(tmp_path):
    m=module()
    args=m.public_pull('prvsiyan/analog-propagation-casmi-2026-baseline',tmp_path)
    assert args[:2]==['kernels','pull'] and '--metadata' in args
    assert not any(x in args for x in ['push','submit','output','run'])
    with pytest.raises(ValueError):m.public_pull('unexpected/user',tmp_path)


def test_summary_keeps_zero_support_and_cannot_infer_hidden_accuracy():
    m=module();out=m.numeric_summary([])
    assert out['count']==0 and out['median'] is None
    assert m.numeric_summary([1,2,3])['median']==2.

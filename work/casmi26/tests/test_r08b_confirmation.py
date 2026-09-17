"""Fresh confirmation cannot silently change cohorts or use a new weight search."""
import importlib.util
from pathlib import Path
import pytest


def module():
    p=Path(__file__).resolve().parents[3]/'tasks/casmi_r08b_confirmation.py'
    assert p.exists(), 'R08B prospective confirmation is missing'
    spec=importlib.util.spec_from_file_location('confirmation',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_cohorts_are_fresh_disjoint_and_deterministic():
    m=module();t=[f'k{i:03}' for i in range(400)];e=t+[f'e{i}' for i in range(250)];used={'k000','k001','e1'}
    a,b=m.choose_keys(t,e,used,128,96)
    assert len(a)==128 and len(b)==96 and not set(a)&set(b)
    assert not (set(a)|set(b))&used
    assert (a,b)==m.choose_keys(list(reversed(t)),list(reversed(e)),used,128,96)


def test_confirmation_fails_without_requested_cohorts():
    with pytest.raises(ValueError,match='Insufficient'):
        module().choose_keys(['a'],['a'],set(),2,2)


def test_fixed_score_is_not_selected_on_new_audit():
    m=module();assert m.FORWARD_WEIGHT==.25 and m.FEATURE=='cosine_nearest'
    assert m.TARGET_SIZE==128 and m.EXTERNAL_SIZE==96


def report():
    return {'fresh_timsTOF':{'absent':{'paired':{'ci95':[.02,.1]}}},
      'external_covered_mixed':{
        'available':{'paired':{'ci95':[-.01,.02]}},
        'external_recovery':{'paired':{'ci95':[.001,.03]},'coverage':.9}}}


def test_new_gate_does_not_promote_missing_or_inconclusive_data():
    m=module();assert m.decide(report())['eligible']
    r=report();r['external_covered_mixed']['external_recovery']['paired']['ci95'][0]=0.
    assert not m.decide(r)['eligible']
    assert not m.decide({})['eligible']
    r=report();r['external_covered_mixed']['available']['paired']['ci95'][0]=-.04
    assert not m.decide(r)['eligible']
    r=report();r['fresh_timsTOF']['absent']['paired']['ci95'][0]=float('nan')
    assert not m.decide(r)['eligible']


def test_gate_keeps_original_failed_gate_distinct():
    r=module().decide(report())
    assert r['replaces_original_r08_gate'] is False
    assert r['requires_new_independent_cohorts'] is True

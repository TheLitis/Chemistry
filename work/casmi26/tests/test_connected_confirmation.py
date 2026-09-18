import importlib.util
from pathlib import Path
import pytest


def module():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_connected_confirmation.py'
    assert path.is_file(),'Prospective connected confirmation is missing'
    spec=importlib.util.spec_from_file_location('connected_confirmation_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_new_keys_deterministic_disjoint_and_no_resampling_fallback():
    m=module();pool=[f'k{i}' for i in range(60)];exclude=set(pool[:8])
    a,b=m.choose_keys(pool,pool,exclude,8,12)
    assert len(a)==8 and len(b)==12 and not set(a)&set(b) and not (set(a)|set(b))&exclude
    assert (a,b)==m.choose_keys(list(reversed(pool)),list(reversed(pool)),exclude,8,12)
    with pytest.raises(ValueError):m.choose_keys(pool,pool,set(pool),8,12)


def metrics():
    return {'target':{'absent':{'paired':{'ci95':[.001,.03]}}},
            'external':{'external_recovery':{'paired':{'ci95':[.001,.04]},'full_candidate_coverage':.95},
                        'available':{'paired':{'ci95':[-.01,.02]}}}}


def test_promotion_requires_all_frozen_conditions_not_average_gain():
    m=module();assert m.decide(metrics(),recall_preserved=True)['eligible']
    a=metrics();a['target']['absent']['paired']['ci95']=[-.001,.1]
    assert not m.decide(a,recall_preserved=True)['eligible']
    a=metrics();a['external']['available']['paired']['ci95']=[-.021,.1]
    assert not m.decide(a,recall_preserved=True)['eligible']
    a=metrics();a['external']['external_recovery']['full_candidate_coverage']=.79
    assert not m.decide(a,recall_preserved=True)['eligible']
    assert not m.decide(metrics(),recall_preserved=False)['eligible']
    assert not m.decide({},recall_preserved=True)['eligible']
    a=metrics();a['external']['external_recovery']['paired']['ci95']=[float('nan'),.04]
    assert not m.decide(a,recall_preserved=True)['eligible']


def test_locked_feature_does_not_search_multiple_variants():
    m=module()
    assert m.SELECTION['feature']=='two_cut' and m.SELECTION['weight']==.25
    assert m.SELECTION['scope']=='complete_R08B_top25'
    assert m.ROOT_RELATIVE=='artifacts/casmi26/research-r10-connected/confirmation-v1'
    assert m.SEED!='R08B-fixed-confirmation-20260917'


def test_reused_scores_can_only_reorder_exact_top25():
    m=module()
    case={'keys':['a','b','c'],'smiles':['CCCCO','CC(C)(C)O','CCO'],'scores':[.3,.2,.1]}
    features={'CCCCO':{'two_cut':0.},'CC(C)(C)O':{'two_cut':1.},'CCO':{'two_cut':None}}
    order=m.refine_top(case,features)
    assert order==[1,0,2]
    assert set(order)=={0,1,2}
    with pytest.raises(ValueError):m.refine_top(case,{'CCCCO':{'two_cut':float('nan')},**{s:f for s,f in features.items() if s!='CCCCO'}})


def test_prepared_record_identity_cannot_reuse_prior_or_missing_keys():
    m=module()
    assert hasattr(m,'validate_partitions'), 'Confirmation lineage guard is missing'
    plan={'target_keys':['a','b'],'external_keys':['c'],'training_query_overlap':0}
    records=[{'key':k,'cohort':'fresh_timsTOF' if k in ('a','b') else 'external_covered_mixed'} for k in 'abc']
    assert m.validate_partitions(plan,records,{'old'},expected_sizes=(2,1))
    with pytest.raises(ValueError):m.validate_partitions(plan,records,{'a'},expected_sizes=(2,1))
    with pytest.raises(ValueError):m.validate_partitions(plan,records[:-1],set(),expected_sizes=(2,1))
    with pytest.raises(ValueError):m.validate_partitions(plan,records+[records[0]],set(),expected_sizes=(2,1))
    with pytest.raises(ValueError):m.validate_partitions(plan,[{**r,'cohort':'external_covered_mixed'} for r in records],set(),expected_sizes=(2,1))

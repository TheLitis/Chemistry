import importlib.util
from pathlib import Path
import pytest


def module():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_r08_verify.py'
    assert path.exists(), 'Independent R08 verifier is not implemented'
    spec=importlib.util.spec_from_file_location('r08_check',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def example():
    records=[];features={}
    for key in ('cal','audit','fresh'):
        case={'keys':['rival',key],'scores':[.2,0.]}
        records.append({'key':key,'cases':{c:dict(case) for c in ('available','absent','external_recovery')}})
        features[key]={key:{'cosine_nearest':.9},'rival':{'cosine_nearest':.1}}
    plan={'calibration_keys':['cal'],'reused_np_audit_keys':['audit'],'fresh_transfer_keys':['fresh'],
          'shortlist_per_regime':2,'features':['cosine_nearest'],'forward_weights':[0.,.5],
          'fitting_overlap':0}
    return plan,records,features


def test_known_scores_rank_at_25_and_no_label_addition():
    m=module();case={'keys':['other','truth'],'scores':[.2,0.]}
    assert m.rank_case(case,{},'cosine_nearest',0.,2,'truth')==2
    assert m.rank_case(case,{'truth':{'cosine_nearest':.9}},'cosine_nearest',.5,2,'truth')==1
    assert m.rank_case(case,{},'cosine_nearest',100.,2,'absent')==0
    assert m.rank_case({'keys':[str(i) for i in range(30)],'scores':[0.]*30},{},'cosine_nearest',0.,32,'25')==0


def test_forward_only_modifies_predeclared_frontier():
    m=module();case={'keys':['first','second'],'scores':[.2,0.]}
    assert m.rank_case(case,{'second':{'cosine_nearest':1.}},'cosine_nearest',10.,1,'second')==2


def test_calibration_selection_does_not_consume_audit_labels():
    m=module();plan,records,features=example()
    selected,ranks=m.recompute(plan,records,features)
    assert selected['weight']==.5
    assert ranks['fresh']['selected']['absent']==1
    features['fresh']={'rival':{'cosine_nearest':1.},'fresh':{'cosine_nearest':0.}}
    again,ranks=m.recompute(plan,records,features)
    assert selected==again
    assert ranks['fresh']['selected']['absent']==2


def test_no_change_candidate_wins_a_tie():
    m=module();plan,records,_=example()
    selected,_=m.recompute(plan,records,{})
    assert selected['weight']==0.


@pytest.mark.parametrize('which',['key_overlap','missing_query','duplicate_query','nonfinite_score','duplicate_candidate','bad_evidence'])
def test_malformed_inputs_stop_verification(which):
    m=module();plan,records,features=example()
    if which=='key_overlap':plan['fresh_transfer_keys']=['cal']
    elif which=='missing_query':records.pop()
    elif which=='duplicate_query':records.append(records[0])
    elif which=='nonfinite_score':records[0]['cases']['available']['scores'][0]=float('nan')
    elif which=='duplicate_candidate':records[0]['cases']['available']['keys']=['rival','rival']
    else:features['cal']['cal']['cosine_nearest']=2.
    with pytest.raises(ValueError):m.recompute(plan,records,features)


def test_statistical_metrics_are_independently_computed():
    m=module();result=m.metrics([0,1,2,25])
    assert result=={'molecules':4,'mrr_at_25':pytest.approx(.385),'top1':.25,'recall_at_25':.75}
    effect=m.paired([2,2,2],[1,1,1])
    assert effect['delta_mrr']==.5
    assert effect['ci95']==[.5,.5]


def test_comparison_rejects_modified_numbers():
    m=module()
    with pytest.raises(ValueError,match='test.metric'):m.close(.2,.3,'test.metric')
    m.close(.2,.2+1e-14,'test.metric')

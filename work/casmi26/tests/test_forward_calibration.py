import numpy as np


def test_case_specific_frontier_does_not_borrow_another_regimes_shortlist():
    from casmi26.forward_ranking import rerank_case
    case={'keys':['a','b','c'],'scores':[.9,.8,.7]}
    scores=rerank_case(case,{'c':{'cosine_max':1.}},'cosine_max',2.,1)
    np.testing.assert_array_equal(scores,case['scores'])
    scores=rerank_case(case,{'a':{'cosine_max':.5}},'cosine_max',2.,1)
    np.testing.assert_array_equal(scores,[1.9,.8,.7])


def test_calibration_prefers_no_change_when_every_feature_harms():
    from casmi26.forward_ranking import choose_configuration
    case={'keys':['truth','wrong'],'scores':[.9,.8]}
    records=[{'key':'truth','cases':{r:case for r in ('available','absent','external_recovery')}}]
    features={'truth':{'wrong':{'cosine_max':1.}}}
    result=choose_configuration(records,features,['cosine_max'],[0.,1.],limit=32)
    assert result['selected']['weight']==0.


def test_calibration_uses_all_regimes_and_can_improve_without_answer_insertion():
    from casmi26.forward_ranking import choose_configuration
    case={'keys':['wrong','truth'],'scores':[.9,.8]}
    records=[{'key':'truth','cases':{r:case for r in ('available','absent','external_recovery')}}]
    features={'truth':{'truth':{'cosine_max':1.}}}
    result=choose_configuration(records,features,['cosine_max'],[0.,.25],limit=32)
    assert result['selected']['weight']==.25
    assert all(v==.5 for v in result['selected']['gains'].values())

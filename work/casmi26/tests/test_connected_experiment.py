import gzip
import importlib
import json
from pathlib import Path
import pytest


def module():
    assert importlib.util.find_spec('casmi26.connected_experiment') is not None, 'Graph comparison experiment is missing'
    return importlib.import_module('casmi26.connected_experiment')


def fixture(tmp_path):
    base=[.12,.10,.0];keys=['wrong','truth','other']
    case={'keys':keys,'smiles':['CC(C)(C)O','CCCCO','CCO'],'scores':base}
    records=[{'key':'truth','cohort':'fixture','queries':[{'adduct':'[M+H]+','precursor':75.08,
        'peaks':[[32.02566618,1.]]}],'cases':{'absent':case,'available':case}}]
    certs=[{'query_id':'truth','case':c,'base_scores':base,'lower_scores':base,'evaluated_mask':[True]*3,
            'weight':.25,'top_indices':[0,1,2],'evaluations':3,'certified':True} for c in records[0]['cases']]
    source=tmp_path/'evidence.json.gz';certificate=tmp_path/'certificates.json.gz'
    source.write_bytes(gzip.compress(json.dumps(records).encode(),mtime=0))
    certificate.write_bytes(gzip.compress(json.dumps(certs).encode(),mtime=0))
    return source,certificate,records,certs


def test_fixed_top25_requires_completed_numeric_certificate(tmp_path):
    a,b,records,certs=fixture(tmp_path);m=module()
    result=m.fixed_top(records[0]['cases']['absent'],certs[0])
    assert result['keys']==['wrong','truth','other'] and result['scores']==[.12,.1,0.]
    for field,value in [('certified',False),('base_scores',[1.,1.,1.]),('lower_scores',[.1,.12,0.]),('evaluated_mask',[False,True,True])]:
        with pytest.raises(ValueError):m.fixed_top(records[0]['cases']['absent'],{**certs[0],field:value})


def test_topk_cannot_hide_unknown_forward_scores(tmp_path):
    _,_,records,certs=fixture(tmp_path);m=module();case=records[0]['cases']['absent']
    keys=[str(i) for i in range(26)];data={'keys':keys,'smiles':['C']*26,'scores':[1.]*25+[.99]}
    cert={'base_scores':data['scores'],'lower_scores':data['scores'],'evaluated_mask':[True]*25+[False],
          'weight':.25,'top_indices':list(range(25)),'evaluations':25,'certified':True}
    with pytest.raises(ValueError,match='uncomputed'):m.fixed_top(data,cert)


def test_isomer_reranker_keeps_formula_slots_and_exact_candidate_set():
    m=module();scores=[.5,.49,.48,.1];formula=['A','B','A','B'];feature=[0.,0.,1.,1.]
    order=m.reorder(scores,feature,formula,isomer_only=True)
    assert [formula[i] for i in order]==formula
    assert order==[2,3,0,1] or order==[2,1,0,3]
    assert set(order)==set(range(4))
    assert m.reorder(scores,[None]*4,formula)==list(range(4))


@pytest.mark.parametrize('feature',[[0,float('nan')],[0,-.1],[1,1.1],[.1]])
def test_bad_feature_never_enters_ranking(feature):
    with pytest.raises(ValueError):module().reorder([1.,0.],feature,['A','A'])


def test_end_to_end_uses_full_r08b_baseline_preserves_recall_and_reuses_results(tmp_path):
    source,certificate,_,_=fixture(tmp_path);m=module();out=tmp_path/'result'
    report=m.run_study(source,certificate,out,workers=1)
    assert report['baseline']=='complete_R08B_certified_top25'
    assert report['query_count']==1 and report['recall_preserved_for_every_query']
    assert report['groups']['fixture']['absent']['r08b']['mrr_at_25']==.5
    assert report['groups']['fixture']['absent']['one_cut_global']['mrr_at_25']==1.
    assert report['production_scoring_changed'] is False and report['new_submissions']==0
    assert report==m.run_study(source,certificate,out,workers=1)
    (out/'results.json.gz').write_bytes(b'corruption')
    with pytest.raises(ValueError,match='changed'):m.run_study(source,certificate,out,workers=1)


def test_no_silent_missing_or_repeated_cases(tmp_path):
    source,certificate,records,certs=fixture(tmp_path);m=module()
    certificate.write_bytes(gzip.compress(json.dumps(certs[:1]).encode(),mtime=0))
    with pytest.raises(ValueError,match='certificate'):m.run_study(source,certificate,tmp_path/'r',workers=1)


def test_source_identity_locked(tmp_path):
    source,certificate,records,certs=fixture(tmp_path);m=module()
    m.run_study(source,certificate,tmp_path/'r',workers=1)
    records[0]['queries'][0]['peaks'][0][0]+=1
    source.write_bytes(gzip.compress(json.dumps(records).encode(),mtime=0))
    with pytest.raises(ValueError,match='identity'):m.run_study(source,certificate,tmp_path/'r',workers=1)

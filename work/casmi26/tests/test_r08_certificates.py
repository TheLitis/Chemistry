"""Independent verification must reject claims stronger than stored bounds."""
import copy
import importlib.util
from pathlib import Path
import pytest


def checker():
    p=Path(__file__).resolve().parents[3]/'tasks/casmi_r08_exact_verify.py'
    assert p.is_file(), 'Independent exact-ranking certificate verifier is missing'
    s=importlib.util.spec_from_file_location('exact_checker',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m.check_certificate


def certificate():
    return {'base_scores':[3.,2.,0.],'lower_scores':[3.1,2.2,0.],'weight':.25,
            'evaluated_mask':[True,True,False],'top_indices':[0,1],
            'evaluations':2,'certified':True}


def test_independent_certificate_accepts_strictly_excluded_tail():
    assert checker()(certificate(),k=2)


@pytest.mark.parametrize('mutation', ['count','uncomputed','order','unknown_can_enter','uncertified','range'])
def test_independent_certificate_rejects_false_claim(mutation):
    c=certificate()
    if mutation=='count':c['evaluations']=1
    elif mutation=='uncomputed':c['lower_scores'][2]=.1
    elif mutation=='order':c['top_indices']=[1,0]
    elif mutation=='unknown_can_enter':c['base_scores'][2]=c['lower_scores'][2]=2.19
    elif mutation=='uncertified':c['certified']=False
    elif mutation=='range':c['lower_scores'][0]=3.4
    with pytest.raises(ValueError):checker()(c,k=2)


def test_independent_certificate_conservative_about_unknown_ties():
    c=certificate();c['base_scores'][2]=c['lower_scores'][2]=c['lower_scores'][1]-.25
    with pytest.raises(ValueError):checker()(c,k=2)


def verifier():
    p=Path(__file__).resolve().parents[3]/'tasks/casmi_r08_exact_verify.py'
    s=importlib.util.spec_from_file_location('complete_checker',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def complete_fixture(tmp_path):
    import gzip,hashlib,json
    root=tmp_path/'result';root.mkdir();source=tmp_path/'original';source.mkdir()
    dump=lambda p,v:p.write_text(json.dumps(v))
    z=lambda p,v:p.write_bytes(gzip.compress(json.dumps(v).encode(),mtime=0))
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    records=[];certificates=[];features={};old={};raw=[]
    for key in ('query-a','query-b'):
        cases={c:{'keys':[key,'other'],'scores':[.1,.2]} for c in ('available','absent','external_recovery')}
        records.append({'key':key,'queries':[{'adduct':'[M+H]+'}],'cases':cases})
        features[key]={key:.9,'other':.1};old[key]={k:{'cosine_nearest':v} for k,v in features[key].items()}
        raw.append({'key':key,'cohort':'audit','ranks':{c:{'r07':2,'r08_capped32':1,'r08_exact':1} for c in cases}})
        for case in cases:
            certificates.append({'query_id':key,'case':case,'base_scores':[.1,.2],
                'lower_scores':[.1+.25*.9,.2+.25*.1],'weight':.25,'top_indices':[0,1],
                'evaluated_mask':[True,True],'evaluations':2,'certified':True})
    z(source/'evidence.json.gz',records);z(source/'forward-features.json.gz',old)
    z(root/'certificates.json.gz',certificates);z(root/'features.json.gz',features)
    dump(root/'audit-ranks.json',raw)
    dump(root/'protocol.json',{'selected':{'feature':'cosine_nearest','weight':.25},'candidate_data_sha256':h(source/'evidence.json.gz')})
    comparison={}
    for case in records[0]['cases']:
        comparison[case]={'r07':{'molecules':2,'mrr_at_25':.5,'top1':0.,'recall_at_25':1.},
            'r08_capped32':{'molecules':2,'mrr_at_25':1.,'top1':1.,'recall_at_25':1.},
            'r08_exact':{'molecules':2,'mrr_at_25':1.,'top1':1.,'recall_at_25':1.},
            'exact_minus_r07':{'delta_mrr':.5,'ci95':[.5,.5]},
            'exact_minus_capped32':{'delta_mrr':0.,'ci95':[0.,0.]}}
    report={'candidate_pairs_across_regimes':12,'evaluated_pairs_across_regimes':12,
        'comparison':{'audit':comparison},'files':{p.name:h(p) for p in root.iterdir()}}
    dump(root/'report.json',report)
    return root,source


def test_independent_verifier_recomputes_entire_result_and_intervals(tmp_path):
    root,source=complete_fixture(tmp_path);r=verifier().verify(root,source)
    assert r['certificates']==6 and r['all_three_rankings_recomputed']
    assert r['all_paired_intervals_recomputed'] and not r['production_ranker_imported']


def test_independent_verifier_does_not_trust_hash_consistent_false_baseline(tmp_path):
    import hashlib,json
    root,source=complete_fixture(tmp_path)
    p=root/'audit-ranks.json';a=json.loads(p.read_text());a[0]['ranks']['available']['r07']=1;p.write_text(json.dumps(a))
    p=root/'report.json';r=json.loads(p.read_text());r['files']['audit-ranks.json']=hashlib.sha256((root/'audit-ranks.json').read_bytes()).hexdigest();p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Baseline'):verifier().verify(root,source)

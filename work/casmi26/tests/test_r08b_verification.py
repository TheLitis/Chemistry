"""Independent R08B checks must reject incorrect ranks even with valid hashes."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest


def verifier():
    root=Path(__file__).resolve().parents[3]
    path=root/'tasks/casmi_r08b_verify.py'
    assert path.is_file(), 'Independent R08B verifier is missing'
    sys.path.insert(0,str(root/'tasks'))
    spec=importlib.util.spec_from_file_location('confirmation_verify',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def fixture(tmp_path):
    root=tmp_path/'result';root.mkdir()
    keys=[];i=0
    while len(keys)<4:
        k='test-key-'+str(i);i+=1
        if int.from_bytes(hashlib.sha256(('official-split-v1:'+k).encode()).digest()[:8],'big')%10==0:
            keys.append(k)
    records=[];cert=[];raw=[];features={}
    for j,key in enumerate(keys):
        cohort='fresh_timsTOF' if j<2 else 'external_covered_mixed'
        cases={c:{'keys':['wrong',key],'smiles':['C','CC'],'scores':[.2,.1]} for c in ('available','absent','external_recovery')}
        records.append({'key':key,'cohort':cohort,'queries':[{'adduct':'[M+H]+'}],
                        'cases':cases,'instruments':{'timsTOF':1}})
        features[key]={'wrong':.1,key:.9}
        raw.append({'key':key,'cohort':cohort,'supported_modes':['[M+H]+'],
           'ranks':{c:{'r07':2,'r08':1,'coverage':True,'candidate_count':2,'evaluated':2} for c in cases}})
        for c,data in cases.items():
            cert.append({'query_id':key,'case':c,'base_scores':data['scores'],'lower_scores':[.225,.325],
                'weight':.25,'evaluated_mask':[True,True],'evaluations':2,'top_indices':[1,0],'certified':True})
    plan={'target_keys':keys[:2],'external_keys':keys[2:],'frozen_selection':{'feature':'cosine_nearest','weight':.25},
      'training_query_overlap':0,'new_weight_search':False,
      'gates':{'target_absent_delta_lower95_gt':0.,'external_recovery_delta_lower95_gt':0.,
               'external_available_delta_lower95_gt':-.03,'external_recovery_coverage_gte':.8}}
    def dump(name,data):
        p=root/name
        if name.endswith('.gz'):p.write_bytes(gzip.compress(json.dumps(data).encode(),mtime=0))
        else:p.write_text(json.dumps(data))
    dump('protocol.json',plan);dump('evidence.json.gz',records);dump('features.json.gz',features)
    dump('certificates.json.gz',cert);dump('audit-ranks.json',raw)
    baseline={'molecules':2,'mrr_at_25':.5,'top1':0.,'recall_at_25':1.}
    improved={'molecules':2,'mrr_at_25':1.,'top1':1.,'recall_at_25':1.}
    cohorts={g:{c:{'r07':baseline,'r08':improved,'paired':{'delta_mrr':.5,'ci95':[.5,.5]},'coverage':1.}
         for c in cases} for g in ('fresh_timsTOF','external_covered_mixed')}
    dump('report.json',{'cohorts':cohorts,'decision':{'eligible':True},'certificates':12,
        'previous_query_keys_reused':0,'fitting_overlap':0,
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}})
    return root,keys


def rehash(root,name):
    p=root/'report.json';data=json.loads(p.read_text());data['files'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest();p.write_text(json.dumps(data))


def test_confirmation_result_and_gate_recomputed(tmp_path):
    root,keys=fixture(tmp_path);r=verifier().verify(root,expected_sizes=(2,2))
    assert r['certificates']==12 and r['queries']==4 and r['decision']['eligible']
    assert not r['production_ranker_imported'] and r['paired_intervals_recomputed']


def test_verifier_rejects_hash_consistent_wrong_baseline(tmp_path):
    root,keys=fixture(tmp_path);p=root/'audit-ranks.json';rows=json.loads(p.read_text())
    rows[0]['ranks']['available']['r07']=1;p.write_text(json.dumps(rows));rehash(root,p.name)
    with pytest.raises(ValueError,match='baseline'):verifier().verify(root,expected_sizes=(2,2))


def test_verifier_rejects_false_freshness(tmp_path):
    root,keys=fixture(tmp_path)
    with pytest.raises(ValueError,match='previous'):verifier().verify(root,previous_keys={keys[0]},expected_sizes=(2,2))


def test_verifier_rejects_changed_fixed_score(tmp_path):
    root,keys=fixture(tmp_path);p=root/'protocol.json';data=json.loads(p.read_text())
    data['frozen_selection']['weight']=.5;p.write_text(json.dumps(data));rehash(root,p.name)
    with pytest.raises(ValueError,match='score'):verifier().verify(root,expected_sizes=(2,2))


def test_verifier_rejects_fabricated_success_decision(tmp_path):
    root,keys=fixture(tmp_path);p=root/'report.json';data=json.loads(p.read_text());data['decision']['eligible']=False;p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='decision'):verifier().verify(root,expected_sizes=(2,2))


def test_verifier_rejects_nonfinite_computed_forward_evidence(tmp_path):
    root,keys=fixture(tmp_path);p=root/'features.json.gz';data=json.loads(gzip.decompress(p.read_bytes()))
    data[keys[0]]['wrong']=float('nan');p.write_bytes(gzip.compress(json.dumps(data).encode()));rehash(root,p.name)
    with pytest.raises(ValueError,match='evidence'):verifier().verify(root,expected_sizes=(2,2))

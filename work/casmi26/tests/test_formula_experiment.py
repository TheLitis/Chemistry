import gzip
import importlib
import json
from pathlib import Path
import pytest
from casmi26.formula_evidence import PROTON,element_mass


def module():
    assert (Path(__file__).parents[1]/'casmi26/formula_experiment.py').is_file(), 'Formula experiment is missing'
    return importlib.import_module('casmi26.formula_experiment')


def fixture_records():
    fragment=element_mass('C')+2*element_mass('H')+element_mass('O')+PROTON
    q={'adduct':'[M+H]+','precursor':47.04914,'peaks':[[fragment,1.]],'ce':[20.]}
    case={'keys':['wrong','truth'],'smiles':['CCC','CCO'],'scores':[.2,.1]}
    return [{'key':'truth','cohort':'fixture','queries':[q],
             'cases':{'available':case,'absent':case,'external_recovery':{'keys':['wrong'],'smiles':['CCC'],'scores':[.2]}}}]


def test_record_research_does_not_inject_missing_truth():
    r=module().evaluate_record(fixture_records()[0])
    assert r['ranks']['absent']['baseline']==2
    assert r['ranks']['absent']['excess']==1
    assert r['ranks']['external_recovery']['excess']==0
    assert r['candidate_counts']=={'available':2,'absent':2,'external_recovery':1}


def test_experiment_executes_reuses_frozen_report_and_detects_input_changes(tmp_path):
    m=module();source=tmp_path/'evidence.json.gz';source.write_bytes(gzip.compress(json.dumps(fixture_records()).encode()))
    result=m.run_study(source,tmp_path/'study',workers=1)
    assert result['quality_claim']=='exploratory_reused_cohorts'
    assert result['parameter_selection_performed'] is False
    assert result['new_submissions']==0
    assert result['groups']['fixture']['absent']['excess']['mrr_at_25']==1.
    assert m.run_study(source,tmp_path/'study',workers=1)==result
    source.write_bytes(gzip.compress(json.dumps(fixture_records()*2).encode()))
    with pytest.raises(ValueError,match='identity'):m.run_study(source,tmp_path/'study',workers=1)


def test_wrong_arrays_and_duplicate_query_keys_are_not_silently_scored(tmp_path):
    m=module();r=fixture_records()[0];r['cases']['available']['scores']=[.2]
    with pytest.raises(ValueError):m.evaluate_record(r)
    source=tmp_path/'dup.gz';source.write_bytes(gzip.compress(json.dumps(fixture_records()*2).encode()))
    with pytest.raises(ValueError,match='Duplicate'):m.run_study(source,tmp_path/'study')

from __future__ import annotations
import csv
import importlib.util
from pathlib import Path


def load():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_r06_scoring_diagnose.py'
    assert path.exists(), 'R06 scoring diagnostic is not implemented'
    spec=importlib.util.spec_from_file_location('r06_scoring_diag',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def write_submission(path, rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['molecule_id','smiles']);w.writeheader();w.writerows(rows)


def test_audit_submission_accepts_valid_and_checks_structure_keys(tmp_path):
    m=load();path=tmp_path/'submission.csv'
    write_submission(path,[{'molecule_id':'a','smiles':'CCO;CCN'},{'molecule_id':'b','smiles':'CC(=O)C'}])
    result=m.audit_submission(path,expected_ids=['a','b'])
    assert result['rows']==2 and result['guesses']==3
    assert result['empty_cells']==[] and result['invalid_guesses']==[]
    assert result['id_set_matches'] is True and result['structure_key_failures']==[]


def test_audit_submission_reports_format_and_chemical_failures(tmp_path):
    m=load();path=tmp_path/'submission.csv'
    write_submission(path,[{'molecule_id':'a','smiles':'CCO;;bad['},{'molecule_id':'x','smiles':''}])
    result=m.audit_submission(path,expected_ids=['a','b'])
    assert result['id_set_matches'] is False
    assert result['empty_cells']==['x']
    assert result['empty_guess_tokens']
    assert result['invalid_guesses']
    assert result['structure_key_failures']


def test_history_complete_without_score_is_not_success():
    m=load()
    result=m.classify_submission_row({'status':'SubmissionStatus.COMPLETE','publicScore':''})
    assert result['terminal'] is True
    assert result['scored'] is False
    assert result['ambiguous_complete_without_score'] is True

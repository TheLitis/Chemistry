import importlib.util
from pathlib import Path
import json
import hashlib
import pytest


def load():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_kaggle_publish.py'
    spec=importlib.util.spec_from_file_location('publish_output_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_downloaded_output_verified_by_report_hash(tmp_path):
    m=load();f=tmp_path/'submission.csv';f.write_text('molecule_id,smiles\n01,CCO\n')
    (tmp_path/'submission.csv.report.json').write_text(json.dumps({'submission_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'prediction_count':1}))
    assert m.validate_downloaded_output(tmp_path)['rows']==1


def test_report_cannot_authorize_missing_csv(tmp_path):
    m=load();(tmp_path/'submission.csv.report.json').write_text('{}')
    with pytest.raises((ValueError,FileNotFoundError)):m.validate_downloaded_output(tmp_path)


def test_mismatched_hash_rejected(tmp_path):
    m=load();(tmp_path/'submission.csv').write_text('molecule_id,smiles\n01,C\n')
    (tmp_path/'submission.csv.report.json').write_text(json.dumps({'submission_sha256':'wrong','prediction_count':1}))
    with pytest.raises(ValueError):m.validate_downloaded_output(tmp_path)

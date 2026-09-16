import importlib.util
import json
from pathlib import Path
import pytest


def verifier():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_v3_verify.py'
    spec=importlib.util.spec_from_file_location('v3verify',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture(tmp_path,text):
    from casmi26.production import sha256
    path=tmp_path/'submission.csv';path.write_text(text)
    report=tmp_path/'report.json';report.write_text(json.dumps({'submission_sha256':sha256(path),'prediction_count':1}))
    return path,report


def test_artifact_verification_reads_real_csv_and_hash(tmp_path):
    mod=verifier();path,report=fixture(tmp_path,'molecule_id,smiles\n001,CCO;COC\n')
    verified=mod.verify_prediction(path,report)
    assert verified['rows']==1 and verified['guesses']==2
    path.write_text('molecule_id,smiles\n001,C\n')
    with pytest.raises(ValueError,match='hash'):mod.verify_prediction(path,report)


def test_artifact_verification_rejects_invalid_or_duplicate_structures(tmp_path):
    mod=verifier()
    for smiles in ('CCO;OCC','broken-molecule','CCO;;COC'):
        path,report=fixture(tmp_path,'molecule_id,smiles\n001,'+smiles+'\n')
        with pytest.raises(ValueError):mod.verify_prediction(path,report)

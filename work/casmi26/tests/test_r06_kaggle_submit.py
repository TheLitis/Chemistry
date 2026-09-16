from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import pytest


def load():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_r06_kaggle_submit.py'
    assert path.exists(), 'R06 Kaggle publish task is not implemented'
    spec=importlib.util.spec_from_file_location('r06submit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_metadata_is_private_offline_and_uses_two_owner_datasets():
    m=load();meta=m.kernel_metadata('owner')
    assert meta['id']=='owner/'+m.KERNEL
    assert meta['is_private']=='true' and meta['enable_internet']=='false'
    assert meta['competition_sources']==[m.SLUG]
    assert meta['dataset_sources']==['owner/'+m.V1_DATASET,'owner/'+m.R06_DATASET]


def test_notebook_reads_auto_extracted_r06_mount_not_missing_zip(tmp_path):
    m=load();path=m.build_notebook(tmp_path/'r06.ipynb');book=json.loads(path.read_text())
    source='\n'.join(''.join(c.get('source',[])) for c in book['cells'] if c['cell_type']=='code')
    assert "rglob('r06-bundle.json')" in source
    assert "PACKAGE/'predict_r06.py'" in source
    assert "rglob('r06-research-candidate.zip')" not in source
    assert 'extractall' not in source


def test_user_authorized_single_submission_still_respects_official_limit():
    m=load()
    assert m.may_submit({'numToday':2,'numAllowedNow':3},pending=0,journal={})
    assert not m.may_submit({'numToday':5,'numAllowedNow':0},pending=0,journal={})
    assert not m.may_submit({'numToday':2,'numAllowedNow':3},pending=1,journal={})
    assert not m.may_submit({'numToday':2,'numAllowedNow':3},pending=0,journal={'submission_attempted':True})


def test_validate_output_requires_exact_r06_contract(tmp_path):
    m=load();csv_path=tmp_path/'submission.csv';report_path=tmp_path/'submission.csv.report.json'
    csv_path.write_text('molecule_id,smiles\n001,CCO;COC\n',encoding='utf-8')
    digest=hashlib.sha256(csv_path.read_bytes()).hexdigest()
    report={'format':6,'budget':'all','prediction_count':1,'test_spectra':3,'test_labels_used':False,
            'empty_candidate_rows':[],'submission_sha256':digest,'bundle_manifest_sha256':'a'*64,
            'selection':{'models':['a.npz','b.npz'],'pooling':'early','weights':[.5,.25,.25],'sigma_ppm':2.0}}
    report_path.write_text(json.dumps(report),encoding='utf-8')
    result=m.validate_r06_output(tmp_path,expected_bundle='a'*64)
    assert result['rows']==1 and result['guesses']==2 and result['budget']=='all'
    report['budget']=3;report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='budget'):m.validate_r06_output(tmp_path,expected_bundle='a'*64)


def test_validate_output_rejects_hash_labels_empty_candidates_and_bundle(tmp_path):
    m=load();csv_path=tmp_path/'submission.csv';report_path=tmp_path/'submission.csv.report.json'
    csv_path.write_text('molecule_id,smiles\n001,CCO\n',encoding='utf-8')
    digest=hashlib.sha256(csv_path.read_bytes()).hexdigest()
    base={'format':6,'budget':'all','prediction_count':1,'test_spectra':1,'test_labels_used':False,
          'empty_candidate_rows':[],'submission_sha256':digest,'bundle_manifest_sha256':'b'*64,
          'selection':{'models':['a.npz'],'pooling':'early','weights':[1,0,0],'sigma_ppm':2.0}}
    for field,value,match in [('submission_sha256','0'*64,'hash'),('test_labels_used',True,'labels'),
                              ('empty_candidate_rows',['001'],'candidate'),('bundle_manifest_sha256','c'*64,'bundle')]:
        report=dict(base);report[field]=value;report_path.write_text(json.dumps(report))
        with pytest.raises(ValueError,match=match):m.validate_r06_output(tmp_path,expected_bundle='b'*64)

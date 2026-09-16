from __future__ import annotations
import base64
import io
import importlib.util
import json
from pathlib import Path
import zipfile


def load():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_r06_kaggle_v3.py'
    assert path.exists(), 'R06 Kaggle v3 task is not implemented'
    spec=importlib.util.spec_from_file_location('r06_v3',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_v3_notebook_embeds_current_inference_source_not_old_dataset_code(tmp_path):
    m=load();path=m.build_notebook(tmp_path/'r06-v3.ipynb');book=json.loads(path.read_text())
    cells=[''.join(c.get('source',[])) for c in book['cells'] if c['cell_type']=='code']
    text='\n'.join(cells)
    assert 'SOURCE_ZIP_BASE64' in text
    assert "rglob('r06-bundle.json')" in text
    assert "from casmi26.r06_candidate import main" in text
    assert "PACKAGE/'predict_r06.py'" not in text
    for cell in cells:compile(cell,'<v3-cell>','exec')
    ns={};exec(next(c for c in cells if c.startswith('SOURCE_ZIP_BASE64')),ns)
    payload=base64.b64decode(ns['SOURCE_ZIP_BASE64'])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names=set(archive.namelist())
        assert 'casmi26/r06_candidate.py' in names
        source=archive.read('casmi26/r06_candidate.py').decode()
    assert 'def select_mass_candidates(' in source
    assert "'nearest_mass_no_compatible_structure'" in source
    assert m.VISIBLE_HASH=='750e3410dbb73e1cdf7dcb5b0b6a7e36c142ef05cd6eef8618b2fd016508cffe'


def test_v3_kernel_metadata_is_private_offline_and_reuses_existing_datasets():
    m=load();meta=m.kernel_metadata('owner')
    assert meta['id']=='owner/'+m.KERNEL
    assert meta['is_private']=='true' and meta['enable_internet']=='false'
    assert meta['competition_sources']==[m.SLUG]
    assert meta['dataset_sources']==['owner/'+m.V1_DATASET,'owner/'+m.R06_DATASET]


def test_v3_submission_gate_requires_known_scoring_error_and_verified_kernel_output():
    m=load()
    assert not m.may_submit_correction({'numToday':3,'numAllowedNow':2},0,{})
    journal={'submission_attempted':True,'submission_ref':56278642,
             'submission_v3_attempted':False,'kernel_v3_output_verified':True,
             'scoring_error_confirmed':True}
    assert m.may_submit_correction({'numToday':3,'numAllowedNow':2},0,journal)
    assert not m.may_submit_correction({'numToday':5,'numAllowedNow':0},0,journal)
    assert not m.may_submit_correction({'numToday':3,'numAllowedNow':2},1,journal)
    journal['submission_v3_attempted']=True
    assert not m.may_submit_correction({'numToday':3,'numAllowedNow':2},0,journal)


def test_server_error_confirmation_requires_the_exact_old_ref_and_format_failure():
    m=load()
    text='Your notebook generated a submission file with incorrect format. Some examples causing this are: wrong number of rows or columns, empty values, an incorrect data type for a value, or invalid submission values from what is expected.'
    assert m.confirm_scoring_error({'ref':56278642,'error_description':text})==text
    assert m.confirm_scoring_error({'ref':56278642,'error_description':''}) is None
    assert m.confirm_scoring_error({'ref':123,'error_description':text}) is None
    assert m.confirm_scoring_error({'ref':56278642,'error_description':'unrelated failure'}) is None

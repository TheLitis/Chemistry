import copy
import importlib.util
import json
from pathlib import Path
import pytest


def mod():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_public_repro.py'
    assert path.is_file(), 'Public baseline reproduction controller not implemented'
    spec=importlib.util.spec_from_file_location('public_repro_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def sample_book():
    return {'nbformat':4,'nbformat_minor':5,'metadata':{},'cells':[
        {'cell_type':'markdown','metadata':{},'source':['Author explanation']},
        {'cell_type':'code','metadata':{},'source':['x = 1\n'],'execution_count':2,'outputs':[{'output_type':'stream','text':'old'}]}]}


def test_author_code_is_preserved_exactly_and_outputs_cleared():
    m=mod();b=sample_book();original=copy.deepcopy(b);out=m.make_notebook(b)
    assert b==original
    assert m.code_digest(b)==m.code_digest({'cells':out['cells'][2:-1]})
    assert out['cells'][3]['outputs']==[] and out['cells'][3]['execution_count'] is None
    assert 'haideptry' in ''.join(out['cells'][0]['source'])
    assert 'Apache' in ''.join(out['cells'][0]['source'])
    compile(''.join(out['cells'][-1]['source']),'<audit>','exec')


def test_private_offline_identity_uses_only_frozen_public_sources():
    m=mod();source={'id':m.AUTHOR,'dataset_sources':list(m.DATASETS),'competition_sources':[m.SLUG],
                    'kernel_sources':[],'model_sources':[],'enable_internet':False,'docker_image':'same-image'}
    result=m.kernel_metadata(source)
    assert result['id']==m.KERNEL and result['is_private'] is True and result['enable_internet'] is False
    assert result['dataset_sources']==list(m.ACTIVE_DATASETS) and result['docker_image']=='same-image'
    assert result['enable_gpu'] is True
    with pytest.raises(ValueError):m.kernel_metadata({**source,'dataset_sources':['unreviewed/private']})
    with pytest.raises(ValueError):m.kernel_metadata({**source,'id':'different/notebook'})


@pytest.mark.parametrize('license_name',['CC0-1.0','CC-BY-4.0','Apache-2.0','MIT'])
def test_accepts_explicit_permissive_dataset_metadata(license_name):
    assert mod().permitted_license({'licenses':[{'name':license_name}]})==license_name.lower()


@pytest.mark.parametrize('metadata',[{}, {'licenses':[]},{'licenses':[{'name':'other'}]}, {'licenses':[{'name':'CC-BY-NC-4.0'}]}])
def test_unclear_or_restricted_metadata_stops_automatic_publication(metadata):
    with pytest.raises(ValueError):mod().permitted_license(metadata)


def test_output_identity_cannot_use_stale_template_ids():
    m=mod()
    assert m.validate_rows(['new-a','new-b'],[{'molecule_id':'new-b','smiles':'CCO;CCO'},{'molecule_id':'new-a','smiles':'C'}])['duplicate_guess_rows']==1
    with pytest.raises(ValueError):m.validate_rows(['new-a','new-b'],[{'molecule_id':'old-a','smiles':'C'},{'molecule_id':'old-b','smiles':'C'}])


@pytest.mark.parametrize('rows',[
    [{'molecule_id':'a','smiles':''}],
    [{'molecule_id':'a','smiles':'C;'}],
    [{'molecule_id':'a','smiles':';'.join(['C']*26)}],
    [{'molecule_id':'a','smiles':'C'},{'molecule_id':'a','smiles':'C'}],
])
def test_invalid_prediction_rows_rejected(rows):
    with pytest.raises(ValueError):mod().validate_rows(['a'],rows)


def test_publication_and_submission_never_repeat_ambiguous_write():
    m=mod()
    assert m.write_allowed({},'publish') is True
    assert m.write_allowed({'publish_attempted':True},'publish') is False
    assert m.write_allowed({'submit_attempted':True},'submit') is False
    with pytest.raises(ValueError):m.write_allowed({},'delete')


def test_frozen_source_and_notebook_hashes_required(tmp_path):
    m=mod();p=tmp_path/'x.ipynb';p.write_text(json.dumps(sample_book()))
    assert m.sha256(p)==m.sha256(p)
    with pytest.raises(ValueError):m.verify_file(p,'0'*64)


def test_global_submission_guard_does_not_replace_existing_lock(tmp_path):
    m=mod();p=tmp_path/'competition-submit-exclusive.lock'
    assert hasattr(m,'create_exclusive_lock'), 'Global submission lock missing'
    m.create_exclusive_lock(p)
    with pytest.raises(FileExistsError):m.create_exclusive_lock(p)
    assert p.is_file()


def test_nested_cli_metadata_and_verbose_ccby_are_supported():
    m=mod()
    assert m.permitted_license({'info':{'licenses':[{'name':'CC0-1.0'}]}})=='cc0-1.0'
    assert m.permitted_license({'info':{'licenses':[{'name':'Attribution 4.0 International (CC BY 4.0)'}]}})=='cc-by-4.0'


def test_unused_restricted_bio_and_raw_coconut_are_not_mounted():
    m=mod();source={'id':m.AUTHOR,'dataset_sources':list(m.DATASETS),'competition_sources':[m.SLUG]}
    actual=m.kernel_metadata(source)['dataset_sources']
    assert 'prvsiyan/chebi-lipidmaps-casmi26' not in actual
    assert 'thedevastator/open-source-natural-product-annotations' not in actual
    assert 'prvsiyan/casmi26-fp-models-v2' in actual
    assert 'prvsiyan/casmi26-ranker-features' in actual
    assert 'prvsiyan/coconut-casmi26-candidates' in actual


def test_rdkit_artifact_must_match_official_pypi_bytes_before_execution():
    m=mod();fn=getattr(m,'rdkit_wheel_contract',None)
    assert callable(fn),'Publisher-wheel verification missing'
    payload={'info':{'name':'rdkit','version':'2026.3.3','license':'BSD-3-Clause'},'urls':[
      {'filename':'rdkit-2026.3.3-cp312-cp312-manylinux_2_28_x86_64.whl','digests':{'sha256':'a'*64},'size':4000}]}
    contract=fn(payload)
    assert contract['files'][payload['urls'][0]['filename']]=='a'*64
    assert contract['license']=='bsd-3-clause'
    for bad in ({**payload,'info':{**payload['info'],'version':'2025.9.4'}},
                {**payload,'urls':[]}):
        with pytest.raises(ValueError):fn(bad)


def test_wheel_hash_assertion_precedes_original_scientific_cells():
    m=mod();b=m.make_notebook(sample_book(),wheels={'files':{'rdkit-x.whl':'a'*64}})
    prefix=''.join(b['cells'][1]['source'])
    assert 'file_digest' in prefix and 'PyPI publisher bytes' in prefix
    compile(prefix,'<preinstall>','exec')
    assert m.code_digest(sample_book())==m.code_digest({'cells':b['cells'][2:-1]})

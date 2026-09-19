import importlib.util
import io
from pathlib import Path
import pytest


def mod():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_foundation_probe.py'
    if not path.exists():
        pytest.fail('Foundation-model probe not implemented')
    spec=importlib.util.spec_from_file_location('foundation_probe',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def metadata(license_id='cc-by-4.0'):
    return {'metadata':{'license':{'id':license_id}},'files':[{
        'key':'ssl_model.ckpt','size':3,'checksum':'md5:900150983cd24fb0d6963f7d28e17f72',
        'links':{'self':'https://zenodo.org/api/records/10997887/files/ssl_model.ckpt/content'}},
        {'key':'other.ckpt','size':3,'checksum':'md5:'+32*'0','links':{'self':'https://zenodo.org/a'}}]}


def test_select_only_explicitly_licensed_ssl_backbone():
    assert mod().ssl_entry(metadata())['key']=='ssl_model.ckpt'


@pytest.mark.parametrize('license_id',['cc-by-nc-4.0','other',None,''])
def test_noncommercial_or_unspecified_weight_license_is_not_accepted(license_id):
    with pytest.raises(ValueError):mod().ssl_entry(metadata(license_id))


@pytest.mark.parametrize('url',['http://zenodo.org/a','https://zenodo.org.evil.com/a',
    'https://u:p@zenodo.org/a','https://127.0.0.1/a','file:///tmp/a','https://zenodo.org:8080/a'])
def test_download_host_contract(url):
    with pytest.raises(ValueError):mod().check_url(url)


def test_digest_and_size_bounded_copy(tmp_path):
    d=tmp_path/'ok';info=mod().copy_public(io.BytesIO(b'abc'),d,3,expected='md5:900150983cd24fb0d6963f7d28e17f72')
    assert info['bytes']==3 and d.read_bytes()==b'abc'
    with pytest.raises(ValueError):mod().copy_public(io.BytesIO(b'abcd'),tmp_path/'bad',3)
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError):mod().copy_public(io.BytesIO(b'xyz'),tmp_path/'wrong',3,expected='md5:900150983cd24fb0d6963f7d28e17f72')
    assert not (tmp_path/'wrong').exists()


def test_checkpoint_size_unknown_or_duplicate_denied():
    x=metadata();x['files'][0]['size']=None
    with pytest.raises(ValueError):mod().ssl_entry(x)
    x=metadata();x['files']*=2
    with pytest.raises(ValueError):mod().ssl_entry(x)

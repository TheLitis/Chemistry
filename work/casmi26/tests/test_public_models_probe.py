import importlib.util
from pathlib import Path
import io
import zipfile
import pytest


def module():
    path = Path(__file__).resolve().parents[3] / 'tasks/casmi_public_models_probe.py'
    assert path.is_file(), 'Public model probe is not implemented'
    spec = importlib.util.spec_from_file_location('public_models_probe', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_download_allowlist_rejects_other_hosts_and_schemes():
    m = module()
    assert m.check_url('https://codeload.github.com/BAMeScience/fiora/zip/' + m.FIORA_COMMIT)
    for url in ('http://zenodo.org/records/1', 'https://evil.invalid/a', 'file:///etc/passwd',
                'https://zenodo.org.evil.invalid/a', 'https://name:secret@zenodo.org/a'):
        with pytest.raises(ValueError): m.check_url(url)


def test_pinned_archive_inspection_does_not_execute_weights(tmp_path):
    m = module(); path = tmp_path/'source.zip'; prefix = 'fiora-'+m.FIORA_COMMIT+'/'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(prefix+'LICENSE', 'MIT')
        z.writestr(prefix+'fiora/resources/models/test_state.pt', b'not-executed')
        z.writestr(prefix+'fiora/resources/models/test_params.json', '{"a":1}')
    result = m.inspect_archive(path)
    assert result['model_assets'][0]['bytes'] > 0
    assert result['license_text'] == 'MIT'
    assert result['parameters'][prefix+'fiora/resources/models/test_params.json'] == {'a':1}


@pytest.mark.parametrize('name', ['../x', '/x', 'C:/x', 'x/../../y', 'x\\..\\y'])
def test_zip_traversal_rejected(tmp_path, name):
    m = module(); path = tmp_path/'bad.zip'
    with zipfile.ZipFile(path, 'w') as z: z.writestr(name, b'payload')
    with pytest.raises(ValueError): m.inspect_archive(path)


def test_bounded_transfer_preserves_old_input_and_rejects_large_body(tmp_path):
    m = module(); target = tmp_path/'body'
    r = m.copy_bounded(io.BytesIO(b'12345'), target, 10)
    assert r['bytes'] == 5 and len(r['sha256']) == 64
    with pytest.raises(ValueError): m.copy_bounded(io.BytesIO(b'x'*11), target, 10)
    assert target.read_bytes() == b'12345'
    assert not list(tmp_path.glob('*.part'))


def test_metadata_summary_does_not_require_model_execution():
    m = module()
    metadata={'metadata':{'title':'FRIGID','license':{'id':'cc-by-4.0'}},
              'files':[{'key':'checkpoint.ckpt','size':20,'checksum':'md5:x',
                        'links':{'self':'https://zenodo.org/api/records/19685145/files/checkpoint.ckpt/content'}}]}
    s=m.summarize_zenodo(metadata)
    assert s['files'][0]['key']=='checkpoint.ckpt'
    assert s['license']=={'id':'cc-by-4.0'}

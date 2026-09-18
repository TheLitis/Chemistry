import importlib.util
from pathlib import Path
import pytest


def task():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_r08b_preview.py'
    assert path.is_file(),'Forward private-preview task absent'
    s=importlib.util.spec_from_file_location('preview',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_preview_metadata_uses_private_new_id_and_old_assets():
    m=task();meta=m.kernel_metadata()
    assert meta['id']=='thelindortis/casmi26-r08b-forward-hybrid'
    assert meta['enable_internet']=='false' and meta['is_private']=='true'
    assert meta['enable_gpu']=='false'
    assert meta['competition_sources']==['enveda-CASMI26-molecule-id-mass-spectra']
    assert meta['dataset_sources']==['thelindortis/casmi26-assets-v1','thelindortis/casmi26-r07-assets-v1','thelindortis/casmi26-r08b-forward-assets-v1']


@pytest.mark.parametrize('args',[
    ['competitions','submit','x'],['datasets','create','--public'],['datasets','create','-u'],
    ['datasets','delete','x'],['kernels','push','--public'],['datasets','version','x']])
def test_preview_whitelist_cannot_spend_competition_slot_or_publish_public(args):
    with pytest.raises(ValueError):task().allow_command(args)


def test_cross_python_wheels_stay_linux_and_do_not_install_pc_packages(tmp_path):
    m=task();c=m.download_command('python','313',tmp_path)
    assert c[:4]==['python','-m','pip','download']
    assert '--only-binary=:all:' in c and c[c.index('--python-version')+1]=='313'
    assert c[c.index('--abi')+1]=='cp313'
    assert 'https://pypi.org/simple' in c and 'numpy==2.3.5' in c
    assert not any(s.startswith(('torch==','rdkit==','pyarrow==')) for s in c)
    assert '--no-deps' not in c
    with pytest.raises(ValueError):m.download_command('py','314',tmp_path)

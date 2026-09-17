import importlib.util
from pathlib import Path
import pytest


def load():
    p=Path(__file__).resolve().parents[3]/'tasks/casmi_r07_shadow.py'
    assert p.exists(), 'R07 shadow publication task is missing'
    spec=importlib.util.spec_from_file_location('r07_shadow',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_shadow_kernel_is_private_offline_cpu_and_uses_full_snapshot_assets():
    m=load();d=m.kernel_metadata('owner')
    assert d['is_private']=='true' and d['enable_internet']=='false' and d['enable_gpu']=='false'
    assert d['dataset_sources']==['owner/casmi26-assets-v1','owner/casmi26-r07-assets-v1']
    assert d['competition_sources']==[m.SLUG]
    with pytest.raises(ValueError):m.kernel_metadata('../other')


def test_shadow_transport_has_no_outputs_or_credentials():
    m=load();a=m.transport_map()
    assert a['coconut.snapshot']=='coconut.zip'
    assert set(a)=={'r07-bundle.json','catalog.json','fingerprints.npy','model.npz','coconut.snapshot'}
    assert not any('submission' in s or 'token' in s for s in a)


@pytest.mark.parametrize('args',[['competitions','submit','x'],['datasets','create','--public'],
    ['datasets','delete','x'],['kernels','delete','x'],['competitions','settings','update'],['models','create']])
def test_shadow_cannot_spend_submission_or_publish_publicly(args):
    with pytest.raises(ValueError):load().allow_command(args)


def test_shadow_allows_only_needed_write_and_read_actions():
    m=load()
    for args in [['datasets','create','-p','assets','-t','-r','skip'],['kernels','push','-p','nb'],
                  ['kernels','output','owner/k'],['competitions','submissions','x','--format','json']]:
        assert m.allow_command(args) is True

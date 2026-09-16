"""Archive contract must reject extra paths before extraction."""
import importlib.util
from pathlib import Path
import zipfile
import pytest


def module():
    p=Path(__file__).resolve().parents[3]/'tasks/casmi_r06_archive_acceptance.py'
    s=importlib.util.spec_from_file_location('r06archive',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_archive_validation_rejects_traversal_or_extra_files(tmp_path):
    p=tmp_path/'x.zip'
    for bad in ['../bad.py','/absolute.py','bundle/../bad.py','work/casmi26/casmi26/nested/a.py','secrets.txt']:
        with zipfile.ZipFile(p,'w') as z:z.writestr(bad,'x')
        with zipfile.ZipFile(p) as z:
            with pytest.raises(ValueError):module().validate_members(z,{'files':{'model.npz':'abc'}})


def test_archive_validation_requires_model_and_entrypoint(tmp_path):
    p=tmp_path/'x.zip'
    manifest={'files':{'model.npz':'abc'}}
    names=['bundle/model.npz','bundle/r06-bundle.json','predict_r06.py','README.txt','runtime.json','work/casmi26/casmi26/__init__.py']
    with zipfile.ZipFile(p,'w') as z:
        for name in names:z.writestr(name,'x')
    with zipfile.ZipFile(p) as z:assert module().validate_members(z,manifest)==set(names)

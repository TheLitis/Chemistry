"""Source export under a changed runner identity must not change global trust."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest


def module():
    path=Path(__file__).resolve().parents[3]/'tasks/source_archive.py'
    spec=importlib.util.spec_from_file_location('source_archive',path)
    out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out)
    return out


def test_archive_scopes_ownership_exception_to_exact_repository(tmp_path,monkeypatch):
    if not shutil.which('git'):pytest.skip('Git missing')
    repo=tmp_path/'owned-before-service-change';repo.mkdir()
    global_config=tmp_path/'global-config';global_config.write_text('[user]\n\tname = Example\n')
    env=dict(os.environ,GIT_CONFIG_GLOBAL=str(global_config),GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_COUNT='0')
    for key in ('GIT_CONFIG_PARAMETERS','GIT_TEST_ASSUME_DIFFERENT_OWNER'):
        env.pop(key,None);monkeypatch.delenv(key,raising=False)
    subprocess.run(['git','init',str(repo)],env=env,check=True,capture_output=True)
    (repo/'science.py').write_text('print(42)\n')
    subprocess.run(['git','-C',str(repo),'add','science.py'],env=env,check=True,capture_output=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-m','fixture'],env=env,check=True,capture_output=True)
    for key,value in env.items():
        if key.startswith('GIT_CONFIG_'):monkeypatch.setenv(key,value)
    monkeypatch.setenv('GIT_TEST_ASSUME_DIFFERENT_OWNER','1')
    bare=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True)
    if bare.returncode==0:pytest.skip('Git does not implement different-owner test override')
    assert 'dubious ownership' in bare.stderr
    before=global_config.read_bytes()
    target=tmp_path/'source.zip';result=module().archive_source(repo,target)
    assert result['configuration_scope']=='command' and result['archive_members']==1
    assert zipfile.ZipFile(target).read('science.py')==b'print(42)\n'
    assert before==global_config.read_bytes()
    again=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True)
    assert again.returncode!=0, 'Trust must not persist outside this one command'


def test_archive_refuses_checkout_output(tmp_path):
    repo=tmp_path/'repo';repo.mkdir();(repo/'.git').mkdir()
    with pytest.raises(ValueError,match='outside'):
        module().archive_source(repo,repo/'output.zip')

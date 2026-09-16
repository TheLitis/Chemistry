import importlib.util
from pathlib import Path


def module():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_access_check.py'
    assert path.exists(), 'Scoped persisted-environment access check is absent'
    spec=importlib.util.spec_from_file_location('casmi_access_test',path)
    out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out)
    return out


def prepare_module():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_prepare.py'
    assert path.exists(), 'Kaggle environment resolver is absent'
    spec=importlib.util.spec_from_file_location('casmi_prepare_test',path)
    out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out)
    return out


def test_existing_process_token_takes_precedence():
    m=module()
    env,source=m.select_environment({'KAGGLE_API_TOKEN':'synthetic-process','OTHER':'keep'},
                                    [('user',{'KAGGLE_API_TOKEN':'synthetic-user'})])
    assert env['KAGGLE_API_TOKEN']=='synthetic-process'
    assert env['OTHER']=='keep' and source=='process'


def test_legacy_pairs_are_not_mixed_across_accounts():
    m=module()
    env,source=m.select_environment({'KAGGLE_USERNAME':'different'},
                                    [('user',{'KAGGLE_KEY':'synthetic-key-only'})])
    assert source=='not_provisioned'
    assert 'KAGGLE_KEY' not in env


def test_complete_user_pair_replaces_partial_process_pair():
    m=module()
    env,source=m.select_environment({'KAGGLE_USERNAME':'partial','OTHER':'keep'},
                                    [('user',{'KAGGLE_USERNAME':'synthetic-user','KAGGLE_KEY':'synthetic-key'})])
    assert env['KAGGLE_USERNAME']=='synthetic-user'
    assert env['KAGGLE_KEY']=='synthetic-key' and env['OTHER']=='keep'
    assert source=='user'


def test_scoped_files_do_not_enumerate_unrelated_downloads(tmp_path):
    m=module()
    (tmp_path/'train.parquet').write_bytes(b'x')
    (tmp_path/'personal-bank.pdf').write_bytes(b'private')
    records=m.official_files([tmp_path])
    assert [Path(r['path']).name for r in records]==['train.parquet']


def test_staged_access_token_is_promoted_to_documented_environment_variable(tmp_path):
    m=prepare_module()
    state=tmp_path/'state';root=state/'kaggle';root.mkdir(parents=True)
    token='synthetic-access-token-1234567890'
    (root/'access_token').write_text(token,encoding='utf-8')
    env=m.kaggle_environment(state,{'OTHER':'keep'})
    assert env['KAGGLE_API_TOKEN']==token
    assert env['OTHER']=='keep'
    assert 'KAGGLE_USERNAME' not in env and 'KAGGLE_KEY' not in env

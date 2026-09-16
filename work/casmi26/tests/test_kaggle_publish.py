from pathlib import Path
import importlib.util
import datetime as dt
import pytest


def load():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_kaggle_publish.py'
    assert path.exists(), 'Submission guard is not implemented'
    spec=importlib.util.spec_from_file_location('pub',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_empty_history_permits_one():
    m=load();now=dt.datetime(2026,9,16,8,tzinfo=dt.timezone.utc)
    assert m.budget_status([],now,5)['may_submit']


def test_rolling_budget_blocks_two_attempts():
    m=load();now=dt.datetime(2026,9,16,8,tzinfo=dt.timezone.utc)
    rows=[{'date':'2026-09-15T23:00:00Z','status':'ERROR'}, {'date':'2026-09-16T04:00:00Z','status':'COMPLETE'}]
    assert not m.budget_status(rows,now,5)['may_submit']


def test_unreadable_dates_fail_closed():
    m=load()
    with pytest.raises(ValueError):m.budget_status([{'date':'unknown'}],dt.datetime.now(dt.timezone.utc),5)


def test_zero_daily_limit_blocks():
    m=load()
    assert not m.budget_status([],dt.datetime.now(dt.timezone.utc),0)['may_submit']


def test_metadata_private_and_offline():
    m=load();book=m.kernel_metadata('myuser','myuser/casmi26-assets-v1')
    assert book['is_private']=='true' and book['enable_internet']=='false'
    assert book['competition_sources']==[m.SLUG] and book['dataset_sources']==['myuser/casmi26-assets-v1']


def test_token_not_allowed_in_assets(tmp_path):
    m=load();(tmp_path/'access_token').write_text('synthetic')
    with pytest.raises(ValueError):m.asset_files(tmp_path)


def test_no_test_answers_or_csv_uploaded(tmp_path):
    m=load();(tmp_path/'submission.csv').write_text('private')
    with pytest.raises(ValueError):m.asset_files(tmp_path)


def test_idempotency_blocks_unknown_outcome():
    m=load()
    assert m.already_attempted({'submission_attempted':True})
    assert not m.already_attempted({})


def test_cross_account_assets_rejected():
    m=load()
    with pytest.raises(ValueError):m.kernel_metadata('myuser','another/casmi26-assets-v1')

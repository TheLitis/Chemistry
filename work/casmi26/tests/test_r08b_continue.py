"""Monotonic continuation of already-approved stages, not an unbounded agent."""
import importlib.util
from pathlib import Path
import pytest


def module():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_r08b_continue.py'
    assert path.exists(), 'Missing R08B continuation controller'
    spec=importlib.util.spec_from_file_location('r08b_controller_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def good_build():
    return {'status':'candidate_built_and_locally_executed','gate':{'eligible':True}}


def test_never_builds_or_retrains_missing_candidate():
    assert module().next_stage(None,{}, {}) is None


def test_stage_then_publish_then_verify_then_check():
    f=module().next_stage;b=good_build()
    assert f(b,{}, {})==('preview','stage')
    assert f(b,{'assets_staged':True},{})==('preview','publish')
    p={'assets_staged':True,'kernel_pushed':True,'kernel_version':1}
    assert f(b,p,{})==('preview','verify')
    p['preview_verified']=True
    assert f(b,p,{})==('submission','check')


def test_any_recorded_attempt_only_reads_even_without_assets():
    assert module().next_stage(None,{}, {'attempted':True})==('submission','status')


@pytest.mark.parametrize('flag',['dataset_attempted','kernel_attempted'])
def test_ambiguous_external_write_is_not_retried(flag):
    with pytest.raises(ValueError,match='ambiguous'):
        module().next_stage(good_build(),{'assets_staged':True,flag:True},{})


def test_completed_dataset_can_continue_to_first_kernel_push():
    p={'assets_staged':True,'dataset_attempted':True,'dataset_created':True}
    assert module().next_stage(good_build(),p,{})==('preview','publish')


def test_unaccepted_build_and_wrong_notebook_fail_closed():
    with pytest.raises(ValueError):module().next_stage({'status':'failed'}, {},{})
    with pytest.raises(ValueError):module().next_stage({'status':'candidate_built_and_locally_executed','gate':{'eligible':False}}, {},{})
    with pytest.raises(ValueError):module().next_stage(good_build(),{'kernel_pushed':True,'kernel_version':2}, {})


def test_git_blob_identity_is_line_ending_stable(tmp_path):
    m=module();a=tmp_path/'a.py';b=tmp_path/'b.py'
    a.write_bytes(b'print(1)\n');b.write_bytes(b'print(1)\r\n')
    assert m.normalized_blob(a)==m.normalized_blob(b)
    a.write_bytes(b'print(2)\n')
    assert m.normalized_blob(a)!=m.normalized_blob(b)


def test_only_check_success_can_trigger_first_submit():
    f=module().may_follow_with_submit
    assert f({'status':'ready_for_one_submission','budget':{'may_submit':True},'journal':{}})
    assert not f({'status':'ready_for_one_submission','budget':{'may_submit':False}})
    assert not f({'status':'pending','budget':{'may_submit':True}})
    assert not f({'status':'ready_for_one_submission','budget':{'may_submit':True},'journal':{'attempted':True}})

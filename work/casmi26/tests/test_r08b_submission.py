"""Fail-closed promotion and at-most-once submission guards, without API calls."""
import copy
import datetime as dt
import importlib.util
from pathlib import Path
import pytest


def module():
    file = Path(__file__).resolve().parents[3] / 'tasks/casmi_r08b_submission.py'
    assert file.is_file(), 'Missing bounded R08B submission task'
    spec = importlib.util.spec_from_file_location('r08b_submit_tests', file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def documents(m):
    forward = dict(weight=.25, feature='cosine_nearest', all_top25_numerically_certified=True,
                   model_sha256=m.FIORA, params_sha256=m.PARAMS)
    output = dict(format=8, model_sha256=m.MODEL, bundle_manifest_sha256=m.BUNDLE,
                  prediction_count=2, test_spectra=3, test_sha256='4'*64, train_sha256='5'*64,
                  submission_sha256='6'*64, test_labels_used=False, empty_candidate_rows=[],
                  forward=forward, seconds_total=600.)
    build = dict(status='candidate_built_and_locally_executed', gate={'eligible':True},
                 package_sha256='1'*64, prediction=copy.deepcopy(output),
                 all_guesses_valid_and_distinct=True)
    asset = dict(format=8, package_sha256='1'*64, base_bundle_manifest_sha256=m.BUNDLE,
                 contains_test_ids_or_predictions=False)
    preview = dict(preview_verified=True, kernel=m.KERNEL, kernel_version=1,
                   package_sha256='1'*64, asset_manifest_sha256='2'*64, notebook_sha256='3'*64,
                   visible_submission_sha256='6'*64, verified_submission_sha256='6'*64)
    return build, preview, asset, output


def test_identity_requires_actual_successful_preview():
    m=module();args=documents(m)
    identity=m.candidate_identity(*args)
    assert identity['kernel']==m.KERNEL and identity['kernel_version']==1
    args[1]['preview_verified']=False
    with pytest.raises(ValueError):m.candidate_identity(*args)


@pytest.mark.parametrize('document,field,value',[
    (0,'status','still_building'),(1,'kernel','other/notebook'),(1,'kernel_version',2),
    (1,'package_sha256','0'*64),(1,'verified_submission_sha256','0'*64),
    (1,'notebook_sha256','not-a-hash'),(2,'contains_test_ids_or_predictions',True),
    (3,'test_labels_used',True),(3,'empty_candidate_rows',['x']),
    (3,'submission_sha256','0'*64),(3,'model_sha256','0'*64),
    (3,'prediction_count',0),(3,'seconds_total',float('nan')),(3,'seconds_total',30000.)
])
def test_identity_rejects_changed_or_incomplete_candidate(document,field,value):
    m=module();args=documents(m);args[document][field]=value
    with pytest.raises(ValueError):m.candidate_identity(*args)


@pytest.mark.parametrize('field,value',[
    ('weight',.5),('feature','entropy'),('all_top25_numerically_certified',False),
    ('model_sha256','0'*64),('params_sha256','0'*64)])
def test_forward_contract_is_immutable(field,value):
    m=module();args=documents(m);args[3]['forward'][field]=value
    with pytest.raises(ValueError):m.candidate_identity(*args)


def test_gate_failure_is_not_promoted():
    m=module();args=documents(m);args[0]['gate']['eligible']=False
    with pytest.raises(ValueError):m.candidate_identity(*args)


def test_command_cannot_push_or_submit_another_version():
    m=module();identity=m.candidate_identity(*documents(m));cmd=m.submit_arguments(identity)
    assert cmd[:3]==['competitions','submit',m.SLUG]
    assert cmd[cmd.index('-k')+1]==m.KERNEL and cmd[cmd.index('-v')+1]=='1'
    identity['kernel_version']=2
    with pytest.raises(ValueError):m.submit_arguments(identity)


def test_timeout_or_missing_ref_never_permits_retry():
    m=module();now=dt.datetime(2026,9,18,12,tzinfo=dt.timezone.utc)
    journal=dict(attempted=True,attempted_utc=now.isoformat(),description=m.DESCRIPTION,command_timeout=True)
    assert m.reconcile([],journal) is None
    assert m.next_action(journal,{'may_submit':True},'submit')=='read_only'


def test_reconcile_requires_unique_bound_identity():
    m=module();stamp='2026-09-18T12:00:00+00:00'
    journal={'attempted':True,'attempted_utc':stamp,'description':m.DESCRIPTION}
    row={'ref':4,'date':stamp,'description':m.DESCRIPTION,'status':'COMPLETE','public_score':'0.0'}
    assert m.reconcile([row],journal)['ref']==4
    with pytest.raises(ValueError):m.reconcile([row,{**row,'ref':5}],journal)
    journal['submission_ref']=4
    with pytest.raises(ValueError):m.reconcile([{**row,'description':'another candidate'}],journal)


def test_checks_never_write_and_pending_budget_stops_submission():
    m=module()
    assert m.next_action({}, {'may_submit':True},'check')=='read_only'
    assert m.next_action({}, {'may_submit':False},'submit')=='wait'
    assert m.next_action({}, {'may_submit':True},'submit')=='one_attempt'


def test_history_must_be_complete_before_any_write():
    m=module()
    assert m.validate_history({'history':[], 'limits':{'numTotal':0}})
    with pytest.raises(ValueError):m.validate_history({'history':[], 'limits':{'numTotal':5}})
    with pytest.raises(ValueError):m.validate_history({'history':[], 'limits':{}})

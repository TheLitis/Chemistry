from datetime import datetime, timezone, timedelta
import importlib.util
from pathlib import Path
import pytest


def module():
    path = Path(__file__).resolve().parents[3]/'tasks/casmi_r07_submission.py'
    assert path.is_file(), 'R07 submission gate is not implemented'
    s=importlib.util.spec_from_file_location('r07_submission_gate',path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def row(ref, date, score='0.147', error='', state='COMPLETE', description='old'):
    return {'ref':ref,'date':date,'public_score':score,'error_description':error,
            'status':state,'description':description}


def test_rolling_budget_counts_failed_submissions_and_handles_omitted_zero():
    m=module();now=datetime(2026,9,17,2,tzinfo=timezone.utc)
    h=[row(1,'2026-09-16T03:49:00'),row(2,'2026-09-16T04:34:00'),
       row(3,'2026-09-16T12:35:00','',error='incorrect format'),row(4,'2026-09-16T13:59:00')]
    r=m.budget_decision(h,{'numTotal':4,'numAllowedNow':5},now)
    assert r['may_submit'] is False and r['attempts_last_24h']==4
    assert r['next_internal_window_utc']=='2026-09-17T12:35:00+00:00'
    assert m.budget_decision(h,{'numAllowedNow':5},datetime(2026,9,17,12,35,tzinfo=timezone.utc))['may_submit'] is True


def test_pending_or_ambiguous_complete_blocks_new_submission():
    m=module();now=datetime(2026,9,17,2,tzinfo=timezone.utc)
    for status in ('PENDING','RUNNING','COMPLETE'):
        d=m.budget_decision([row(1,'2026-09-17T01:00:00','',state=status)],{'numAllowedNow':5},now)
        assert d['may_submit'] is False
    assert m.budget_decision([row(1,'2026-09-17T01:00:00','0')],{'numAllowedNow':5},now)['may_submit']


def test_score_classification_uses_error_field_before_complete():
    m=module()
    assert m.classify(row(1,'2026-09-16','',error='incorrect format'))['state']=='scoring_error'
    assert m.classify(row(1,'2026-09-16',''))['state']=='complete_without_score'
    assert m.classify(row(1,'2026-09-16','0'))['state']=='scored'
    assert m.classify(row(1,'2026-09-16','0'))['public_score']==0.


@pytest.mark.parametrize('score',['NaN','Infinity','-0.1','1.1','not-a-score'])
def test_invalid_score_is_not_success(score):
    m=module()
    assert m.classify(row(1,'2026-09-16',score))['state']=='invalid_score'


def test_reconciliation_matches_exact_request_and_never_retries():
    m=module();now='2026-09-17T12:36:00Z'
    j={'attempted':True,'attempted_utc':now,'description':m.DESCRIPTION}
    h=[row(20,now,'',state='PENDING',description=m.DESCRIPTION),row(19,now,description='other')]
    assert m.reconcile(h,j)['ref']==20
    h.append(row(21,now,'',state='PENDING',description=m.DESCRIPTION))
    with pytest.raises(ValueError,match='Ambiguous'): m.reconcile(h,j)
    j['submission_ref']=20
    assert m.reconcile(h,j)['ref']==20


def test_missing_or_future_dates_fail_closed():
    m=module();now=datetime(2026,9,17,2,tzinfo=timezone.utc)
    for stamp in (None,'bad','2026-09-18T01:00:00'):
        with pytest.raises(ValueError):m.budget_decision([row(1,stamp)],{'numAllowedNow':5},now)
    assert not m.budget_decision([],{'numAllowedNow':0},now)['may_submit']


def test_retained_submission_id_must_not_point_to_another_description():
    m=module();j={'attempted':True,'attempted_utc':'2026-09-17T12:36:00Z',
        'description':m.DESCRIPTION,'submission_ref':20}
    with pytest.raises(ValueError):m.reconcile([row(20,'2026-09-17T12:36:00Z',description='other')],j)


def test_submit_command_is_one_immutable_private_notebook_version():
    m=module();command=m.submit_arguments('thelindortis/'+m.KERNEL,1)
    assert command[:2]==['competitions','submit']
    assert command[command.index('-v')+1]=='1'
    assert command[command.index('-f')+1]=='submission.csv'
    assert command[command.index('-m')+1]==m.DESCRIPTION
    for bad in ('someone/another-notebook','evil;rm -rf /',''):
        with pytest.raises(ValueError):m.submit_arguments(bad,1)
    with pytest.raises(ValueError):m.submit_arguments('thelindortis/'+m.KERNEL,2)

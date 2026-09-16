from pathlib import Path
import importlib.util
import pytest


def load():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_score_status.py'
    assert path.exists(), 'Read-only scoring reconciliation is absent'
    spec=importlib.util.spec_from_file_location('score_status_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def record(status='SubmissionStatus.COMPLETE',score='0.123'):
    return {'ref':'56268669','fileName':'submission.csv','date':'2026-09-16 03:49:28.787000',
            'description':'CASMI26 baseline v1 - trained catalog + spectral matching; first official evaluation',
            'status':status,'publicScore':score,'privateScore':''}


def journal():
    return {'submission_accepted':True,'submission_attempted':True,'attempted_utc':'2026-09-16T03:49:26.127338+00:00'}


def test_known_submission_matches_journal():
    m=load();row=m.verified_submission([record()],journal(),56268669)
    assert row['ref']=='56268669'


def test_unrelated_submission_rejected():
    m=load();r=record();r['description']='Somebody else'
    with pytest.raises(ValueError):m.verified_submission([r],journal(),56268669)


def test_pending_does_not_mean_zero():
    m=load();result=m.score_summary(record('SubmissionStatus.PENDING',''))
    assert result['public_score'] is None and not result['terminal']


def test_true_zero_score_is_preserved():
    m=load();result=m.score_summary(record(score='0'))
    assert result['public_score']==0.0 and result['private_score'] is None and result['terminal']


def test_invalid_score_is_not_reported():
    m=load()
    with pytest.raises(ValueError):m.score_summary(record(score='nan'))


def test_terminal_error_has_no_fabricated_score():
    m=load();result=m.score_summary(record('SubmissionStatus.ERROR',''))
    assert result['terminal'] and result['public_score'] is None

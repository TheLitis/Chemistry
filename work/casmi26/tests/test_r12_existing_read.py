import importlib.util
from pathlib import Path
import pytest


def m():
    p=Path(__file__).resolve().parents[3]/'tasks/casmi_r12_collect_existing.py'
    assert p.is_file(), 'Read-only existing preview collector missing'
    s=importlib.util.spec_from_file_location('existing_preview',p);x=importlib.util.module_from_spec(s);s.loader.exec_module(x);return x


def test_read_only_actions_are_fixed():
    x=m()
    for a in ('status','pull','output'):
        cmd=x.command(a,Path('/tmp/output'))
        assert cmd[:3]==['kernels',a,'thelindortis/casmi26-r12-ion-view-diagnostic']
    for a in ('submit','push','delete','create'):
        with pytest.raises(ValueError):x.command(a,Path('/tmp/output'))


def test_no_publication_or_attempt_journal_forgery():
    text=Path(m().__file__).read_text()
    assert 'publication_performed_by_this_task' in text
    assert 'journal.update' not in text and "journal['" not in text
    assert 'verify_remote' in text and 'compare_predictions' in text

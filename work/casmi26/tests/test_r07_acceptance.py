import importlib.util
from pathlib import Path
import pytest


def load():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_r07_acceptance.py'
    assert path.exists(), 'R07 standalone acceptance task is missing'
    spec=importlib.util.spec_from_file_location('r07_acceptance',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def reports():
    return {'format':7,'prediction_count':400,'test_spectra':1213,'submission_sha256':'csv',
        'bundle_manifest_sha256':'manifest','model_sha256':'weights','test_sha256':'test','train_sha256':'train',
        'selection':{'kind':'v1_tanimoto'},'test_labels_used':False,'empty_candidate_rows':[]}


def test_exact_standalone_output_contract():
    a=reports();assert load().same_output(a,dict(a)) is True


@pytest.mark.parametrize('key',list(reports()))
def test_acceptance_rejects_changed_output_or_model(key):
    a=reports();b=dict(a);b[key]='different'
    with pytest.raises(ValueError,match=key):load().same_output(a,b)

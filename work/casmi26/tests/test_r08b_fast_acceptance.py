import importlib.util
from pathlib import Path
import pytest


def module():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_r08b_fast_acceptance.py'
    assert path.exists(), 'Missing full-data fast-backend acceptance task'
    spec=importlib.util.spec_from_file_location('fast_acceptance',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def reports():
    base={k:k for k in ('submission_sha256','train_sha256','test_sha256','model_sha256','bundle_manifest_sha256')}
    base.update(prediction_count=400,test_spectra=1213,format=8,empty_candidate_rows=[],test_labels_used=False,
        forward=dict(weight=.25,feature='cosine_nearest',all_top25_numerically_certified=True))
    return base,{**base,'forward':{**base['forward'],'scoring_backend':'nearest-only'}}


def test_requires_identical_whole_output():
    a,b=reports();assert module().verify_equivalence(a,b) is True


@pytest.mark.parametrize('key',['submission_sha256','train_sha256','prediction_count','test_spectra','model_sha256','bundle_manifest_sha256'])
def test_rejects_changed_predictions_or_inputs(key):
    a,b=reports();b[key]='changed'
    with pytest.raises(ValueError,match=key):module().verify_equivalence(a,b)


@pytest.mark.parametrize('key,value',[('weight',.5),('feature','cosine_max'),('all_top25_numerically_certified',False),('scoring_backend','all-features')])
def test_rejects_changed_scoring(key,value):
    a,b=reports();b['forward'][key]=value
    with pytest.raises(ValueError,match='forward'):module().verify_equivalence(a,b)

import importlib.util
from pathlib import Path
import pytest


def acceptance():
    path = Path(__file__).resolve().parents[3]/'tasks/casmi_v3_acceptance.py'
    spec = importlib.util.spec_from_file_location('v3_acceptance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report():
    return dict(submission_sha256='csv-hash', model_sha256='model-hash',
                train_sha256='train-hash', test_sha256='test-hash',
                prediction_count=400, test_spectra=1213, model_format=3)


def test_entrypoint_acceptance_requires_same_predictions():
    expected = report()
    assert acceptance().verify_equivalent_prediction(expected, dict(expected)) is True


@pytest.mark.parametrize('field', list(report()))
def test_entrypoint_acceptance_rejects_changed_contract(field):
    expected = report(); actual = dict(expected); actual[field] = 'changed'
    with pytest.raises(ValueError, match=field):
        acceptance().verify_equivalent_prediction(expected, actual)

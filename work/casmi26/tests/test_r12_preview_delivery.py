import importlib.util
import json
from pathlib import Path
import pytest


def controller():
    path = Path(__file__).resolve().parents[3] / 'tasks/casmi_r12_preview.py'
    assert path.is_file(), 'R12 preview controller missing'
    spec = importlib.util.spec_from_file_location('r12_controller', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('status', ['RUNNING', 'QUEUED', 'COMPLETE', 'ERROR', 'CANCELLED'])
def test_only_explicit_worker_status_is_accepted(status):
    m = controller()
    assert m.worker_state('Kernel has status "KernelWorkerStatus.' + status + '"') == status.lower()
    with pytest.raises(ValueError):
        m.worker_state('Network request failed to return a status')


def test_preview_publication_is_at_most_once():
    m = controller()
    assert m.may_publish({})
    assert not m.may_publish({'publish_attempted': True})
    assert not m.may_publish({'publish_attempted': True, 'kernel_version': None})


def test_current_ids_and_nonempty_rows_are_required(tmp_path):
    m = controller()
    p = tmp_path / 'a.csv'
    p.write_text('molecule_id,smiles\n001,CCO;CCO\n002,C\n')
    rows = m.read_predictions(p)
    assert list(rows) == ['001', '002'] and len(rows['001']) == 2
    p.write_text('molecule_id,smiles\n001,CCO\n001,C\n')
    with pytest.raises(ValueError): m.read_predictions(p)
    p.write_text('molecule_id,smiles\n001,\n')
    with pytest.raises(ValueError): m.read_predictions(p)


def test_changes_are_not_reported_as_accuracy():
    m = controller()
    a = {'001': ['C', 'CCO'], '002': ['O']}
    b = {'002': ['O'], '001': ['CCO', 'C']}
    report = m.compare_predictions(a, b)
    assert report == {'rows': 2, 'changed_top1': 1, 'changed_rows': 1}
    assert 'score' not in report
    with pytest.raises(ValueError): m.compare_predictions(a, {'old': ['C']})


def test_remote_identity_matches_the_exact_all_cells():
    m = controller()
    a = {'cells': [{'cell_type': 'code', 'source': ['x=1\n']}]}
    meta = {'id': m.KERNEL, 'id_no': 17, 'is_private': True, 'enable_internet': False,
            'dataset_sources': list(m.DATASETS), 'competition_sources': [m.COMPETITION],
            'kernel_sources': [], 'model_sources': []}
    assert m.verify_remote(a, a, meta)['id_no'] == 17
    with pytest.raises(ValueError): m.verify_remote(a, a, {**meta, 'enable_internet': True})
    with pytest.raises(ValueError): m.verify_remote(a, a, {**meta, 'is_private': False})
    with pytest.raises(ValueError): m.verify_remote(a, a, {**meta, 'model_sources': ['foreign']})
    b = {'cells': [{'cell_type': 'code', 'source': ['x=2\n']}]}
    with pytest.raises(ValueError): m.verify_remote(a, b, meta)


def test_no_competition_write_or_existing_controller_mutation():
    m = controller()
    text = Path(m.__file__).read_text()
    assert "choices=('publish', 'collect')" in text
    assert "'competitions', 'submit'" not in text
    assert "'datasets', 'create'" not in text
    assert m.ROOT == 'r12-ion-view-preview-v1'

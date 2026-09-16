"""Windows embeddable Python omits the task directory from sys.path."""
import importlib.util
from pathlib import Path
import sys


def test_dataset_locator_works_without_tasks_on_import_path(tmp_path,monkeypatch):
    repo=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('v3task',repo/'tasks/casmi_v3.py')
    task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
    root=tmp_path/'external';data=root/'competition';data.mkdir(parents=True)
    for name in ('train.parquet','test.parquet','sample_submission.csv'):(data/name).write_bytes(b'fixture')
    monkeypatch.setattr(sys,'path',[p for p in sys.path if Path(p or '.').resolve()!=repo/'tasks'])
    assert task.dataset_path(repo,root)==data

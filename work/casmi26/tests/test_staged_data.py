import importlib.util
from pathlib import Path
import pytest


def module():
    path = Path(__file__).resolve().parents[3] / 'tasks/casmi_staged.py'
    assert path.exists(), 'Staged data inspector missing'
    spec = importlib.util.spec_from_file_location('staged', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_complete_same_folder(tmp_path):
    m = module()
    d = tmp_path / 'named-data'; d.mkdir()
    for name in m.NAMES:
        (d / name).write_bytes(b'fixture')
    assert m.find_dataset(tmp_path) == d


def test_no_mix_across_folders(tmp_path):
    m = module()
    for n, name in enumerate(m.NAMES):
        d = tmp_path / str(n); d.mkdir(); (d/name).write_bytes(b'fixture')
    with pytest.raises(FileNotFoundError):
        m.find_dataset(tmp_path)


def test_ambiguous_dataset_rejected(tmp_path):
    m = module()
    for folder in ('a', 'b'):
        d = tmp_path / folder; d.mkdir()
        for name in m.NAMES: (d/name).write_bytes(b'fixture')
    with pytest.raises(ValueError, match='Multiple'):
        m.find_dataset(tmp_path)


def test_zero_length_file_rejected(tmp_path):
    m = module()
    for name in m.NAMES: (tmp_path / name).touch()
    with pytest.raises(FileNotFoundError): m.find_dataset(tmp_path)

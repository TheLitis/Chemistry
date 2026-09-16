import importlib.util
from pathlib import Path


def test_linux_wheels_are_requested_for_kaggle_not_windows(tmp_path):
    source=Path(__file__).resolve().parents[3]/'tasks/casmi_package.py'
    assert source.exists()
    spec=importlib.util.spec_from_file_location('package_task',source)
    task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
    args=task.download_arguments(tmp_path)
    assert args[:3] == ['-m','pip','download']
    assert '312' in args and 'cp312' in args
    assert 'manylinux_2_28_x86_64' in args
    assert '--only-binary=:all:' in args
    assert 'rdkit==2026.3.3' in args

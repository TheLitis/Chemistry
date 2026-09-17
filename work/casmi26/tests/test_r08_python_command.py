import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def module():
    path=Path(__file__).resolve().parents[3]/'tasks/r08_python_command.py'
    assert path.is_file(), 'Explicit isolated child Python launcher is missing'
    spec=importlib.util.spec_from_file_location('r08_python_command_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def test_isolated_interpreter_reproduces_ignored_pythonpath(tmp_path):
    (tmp_path/'chemistry_marker.py').write_text('ANSWER=42\n')
    env={**os.environ,'PYTHONPATH':str(tmp_path)}
    result=subprocess.run([sys.executable,'-I','-c','import chemistry_marker'],env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'ModuleNotFoundError' in result.stderr


def test_explicit_path_child_works_in_isolated_python(tmp_path):
    (tmp_path/'chemistry_marker.py').write_text('ANSWER=42\n')
    command=module().python_command(sys.executable,[tmp_path],'import chemistry_marker; print(chemistry_marker.ANSWER)')
    result=subprocess.run(command,env={**os.environ,'PYTHONPATH':'irrelevant'},capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    assert result.stdout.strip()=='42'
    assert '-I' in command


def test_explicit_paths_and_arguments_preserve_quotes_spaces(tmp_path):
    folder=tmp_path/"quoted ' source";folder.mkdir()
    (folder/'chemistry_marker.py').write_text('ANSWER=73\n')
    code='import chemistry_marker,sys,json; print(json.dumps([chemistry_marker.ANSWER,sys.argv[1:]]))'
    result=subprocess.run(module().python_command(sys.executable,[folder],code,['a b',"x'y"]),capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    assert json.loads(result.stdout)==[73,['a b',"x'y"]]


def test_missing_source_path_rejected_before_launch(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        module().python_command(sys.executable,[tmp_path/'missing'],'pass')

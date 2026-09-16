import importlib.util
import json
from pathlib import Path
import pytest


def module():
    path = Path(__file__).resolve().parents[3]/'tasks/casmi_deliver.py'
    assert path.exists(), 'Delivery task missing'
    spec = importlib.util.spec_from_file_location('casmi_delivery',path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_notebook_execution_records_stdout_and_order(tmp_path):
    m=module();book={'cells':[
        {'cell_type':'markdown','source':['# fixture']},
        {'cell_type':'code','source':['x=40\nprint("started")'],'metadata':{},'outputs':[],'execution_count':None},
        {'cell_type':'code','source':['assert x==40\nprint(x+2)'],'metadata':{},'outputs':[],'execution_count':None}]}
    source=tmp_path/'a.ipynb';target=tmp_path/'executed.ipynb';source.write_text(json.dumps(book))
    result=m.execute_notebook(source,target)
    assert result['code_cells_executed']==2
    saved=json.loads(target.read_text())
    assert saved['cells'][2]['execution_count']==2
    assert '42' in saved['cells'][2]['outputs'][0]['text']


def test_notebook_failure_is_not_success(tmp_path):
    m=module();source=tmp_path/'a.ipynb';target=tmp_path/'b.ipynb'
    source.write_text(json.dumps({'cells':[{'cell_type':'code','source':['raise ValueError("bad")']}]}))
    with pytest.raises(ValueError,match='bad'):m.execute_notebook(source,target)
    saved=json.loads(target.read_text());assert saved['cells'][0]['outputs'][-1]['output_type']=='error'


def test_wheels_are_offline_linux_binary_only():
    m=module();args=m.wheel_arguments(Path('wheels'),'312')
    assert '--only-binary=:all:' in args and 'cp312' in args
    assert 'manylinux_2_28_x86_64' in args
    assert 'rdkit==2026.3.3' in args


def test_wheel_python_version_is_not_arbitrary():
    m=module()
    with pytest.raises(ValueError):m.wheel_arguments(Path('x'),'999')

from pathlib import Path
import importlib
import sys
import json
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'casmi26/submission_notebook.py'
    assert path.exists(), 'Notebook builder absent'
    sys.path.insert(0,str(path.parents[1]))
    return importlib.import_module('casmi26.submission_notebook')


def test_notebook_format_and_code(tmp_path):
    m=module(); path=tmp_path/'submission.ipynb'
    m.build_notebook(path)
    book=json.loads(path.read_text())
    assert book['nbformat']==4
    for cell in book['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),str(path),'exec')
            assert cell['outputs']==[]
    nbformat=pytest.importorskip('nbformat')
    nbformat.validate(nbformat.read(path,as_version=4))


def test_notebook_uses_current_mount_not_visible_predictions(tmp_path):
    m=module();path=tmp_path/'submission.ipynb';m.build_notebook(path)
    source='\n'.join(''.join(c['source']) for c in json.loads(path.read_text())['cells'])
    assert '/kaggle/input' in source and 'test.parquet' in source
    assert 'current_test' in source
    assert 'predictions_generated' not in source
    assert '--no-index' in source and '--find-links' in source
    assert 'KAGGLE_API_TOKEN' not in source

import json
from pathlib import Path
import pytest


def test_forward_notebook_is_dynamic_offline_and_uses_frozen_extension(tmp_path):
    from casmi26.notebook_forward import build_notebook
    path=build_notebook(tmp_path/'candidate.ipynb',manifest_sha256='a'*64)
    book=json.loads(path.read_text());cells=[ ''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
    assert book['nbformat']==4
    for text in cells:compile(text,'<notebook>','exec')
    source='\n'.join(cells)
    assert "rglob('forward-assets.json')" in source
    assert "rglob('r07-bundle.json')" in source
    assert "rglob('test.parquet')" in source
    assert "'--no-index'" in source and "'--no-deps'" in source
    assert 'casmi26.forward_release' in source
    assert 'FIORA_TEST_MODEL' not in source
    assert 'KAGGLE_API_TOKEN' not in source
    assert '400' not in source and '1213' not in source
    assert 'a'*64 in source
    assert "'--train'" in source and "'--cache'" in source
    assert 'scored' not in source


@pytest.mark.parametrize('value',['',None,'not-a-sha','a'*63,'../file'])
def test_forward_notebook_rejects_bad_asset_identity(tmp_path,value):
    from casmi26.notebook_forward import build_notebook
    with pytest.raises(ValueError):build_notebook(tmp_path/'bad.ipynb',manifest_sha256=value)
    assert not (tmp_path/'bad.ipynb').exists()

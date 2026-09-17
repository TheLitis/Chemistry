import base64,io,json,zipfile
from pathlib import Path


def test_notebook_is_self_contained_private_data_compatible_and_offline(tmp_path):
    from casmi26.notebook_r07 import build_notebook
    p=build_notebook(tmp_path/'r07.ipynb');n=json.loads(p.read_text())
    cells=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    for c in cells:compile(c,'<r07-cell>','exec')
    text='\n'.join(cells)
    assert '--no-index' in text and "rglob('r07-bundle.json')" in text
    assert 'KAGGLE_API_TOKEN' not in text
    assert '400' not in '\n'.join(cells[1:]) and '1213' not in '\n'.join(cells[1:])
    ns={};exec(cells[0],ns)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ns['SOURCE_B64']))) as z:
        assert 'casmi26/r07_release.py' in z.namelist()
        assert all(Path(x).suffix=='.py' for x in z.namelist())


def test_notebook_source_archive_is_reproducible(tmp_path):
    from casmi26.notebook_r07 import build_notebook
    a=build_notebook(tmp_path/'a.ipynb').read_bytes()
    b=build_notebook(tmp_path/'b.ipynb').read_bytes()
    assert a==b


def test_notebook_restores_opaque_archive_transport_without_repacking(tmp_path):
    from casmi26.notebook_r07 import build_notebook
    n=json.loads(build_notebook(tmp_path/'snapshot.ipynb').read_text())
    text='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
    assert "BUNDLE/'coconut.snapshot'" in text
    assert "'coconut.zip'" in text and "'model.npz'" in text
    assert 'copyfile' in text

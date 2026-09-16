import importlib.util
from pathlib import Path
import zipfile
import pytest


def module():
    path=Path(__file__).resolve().parents[3]/'tasks/casmi_import.py'
    assert path.exists(), 'CASMI archive import task is absent'
    spec=importlib.util.spec_from_file_location('casmi_import_test',path)
    out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out)
    return out


def test_archive_copies_only_verified_data(tmp_path):
    m=module(); source=tmp_path/'a.zip'
    with zipfile.ZipFile(source,'w') as z:
        z.writestr('dataset/train.parquet',b'fixture')
        z.writestr('dataset/sample_submission.csv','molecule_id,smiles\n01,C\n')
    files=m.import_archive(source,tmp_path/'dest')
    assert (tmp_path/'dest/dataset/train.parquet').read_bytes()==b'fixture'
    assert len(files)==2
    assert m.import_archive(source,tmp_path/'dest')==files


@pytest.mark.parametrize('bad',['../outside','/absolute','C:/absolute','x/../../oops','x\\..\\oops'])
def test_archive_traversal_rejected_before_any_writes(tmp_path,bad):
    m=module(); source=tmp_path/'a.zip'
    with zipfile.ZipFile(source,'w') as z:
        z.writestr('ok.csv',b'x'); z.writestr(bad,b'y')
    with pytest.raises(ValueError):m.import_archive(source,tmp_path/'dest')
    assert not (tmp_path/'dest/ok.csv').exists()


def test_archive_does_not_overwrite_existing_data(tmp_path):
    m=module(); source=tmp_path/'a.zip'; dest=tmp_path/'dest';dest.mkdir()
    (dest/'train.parquet').write_bytes(b'other')
    with zipfile.ZipFile(source,'w') as z:z.writestr('train.parquet',b'new')
    with pytest.raises(ValueError,match='different'):m.import_archive(source,dest)
    assert (dest/'train.parquet').read_bytes()==b'other'


def test_duplicate_member_rejected(tmp_path):
    m=module(); source=tmp_path/'a.zip'
    with zipfile.ZipFile(source,'w') as z:
        z.writestr('TRAIN.parquet',b'1');z.writestr('train.parquet',b'2')
    with pytest.raises(ValueError):m.import_archive(source,tmp_path/'dest')

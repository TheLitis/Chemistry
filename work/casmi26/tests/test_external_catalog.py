from collections import Counter
import csv
import io
import zipfile
import pytest
from casmi26.catalog_candidates import MassWindows, candidate_rows, distinct_candidates


def test_mass_windows_cover_boundaries_without_huge_grid():
    w=MassWindows([100.,100.002,200.],ppm=20,da=.005,padding=.001)
    assert w.contains(99.994) and w.contains(100.008)
    assert w.contains(200.006) and not w.contains(150.)
    assert len(w.lower)==2


def test_mass_windows_reject_bad_inputs():
    with pytest.raises(ValueError):MassWindows([float('nan')])
    with pytest.raises(ValueError):MassWindows([-1.])
    with pytest.raises(ValueError):MassWindows([10.],ppm=-1)


def archive(tmp_path,rows):
    p=tmp_path/'coconut.zip';s=io.StringIO(newline='')
    writer=csv.DictWriter(s,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with zipfile.ZipFile(p,'w') as z:z.writestr('snapshot.csv',s.getvalue())
    return p


def test_reader_uses_exact_mass_not_average_mass(tmp_path):
    p=archive(tmp_path,[{'identifier':'a','canonical_smiles':'CCO','exact_molecular_weight':'46.04186','molecular_weight':'46.069'},
                        {'identifier':'b','canonical_smiles':'C','exact_molecular_weight':'16.0313','molecular_weight':'46.04186'}])
    stats=Counter();rows=list(candidate_rows(p,MassWindows([46.04186]),stats))
    assert len(rows)==1 and rows[0][0]=='a'
    assert stats['csv_rows']==2 and stats['mass_selected']==1


def test_missing_schema_does_not_silently_guess(tmp_path):
    p=archive(tmp_path,[{'smiles':'CCO','weight':'46'}])
    with pytest.raises(ValueError,match='schema'):list(candidate_rows(p,MassWindows([46]),Counter()))


def test_invalid_mass_is_counted(tmp_path):
    p=archive(tmp_path,[{'identifier':'a','canonical_smiles':'C','exact_molecular_weight':'nan'}])
    stats=Counter();assert list(candidate_rows(p,MassWindows([16]),stats))==[]
    assert stats['invalid_mass']==1


def test_equivalent_keys_choose_closest_mass_then_stable_source():
    records=[['a','CCO','same',46.0,'00'],['b','OCC','same',46.001,'00'],['c','COC','other',46.0,'00']]
    chosen=distinct_candidates(records,46.001)
    assert [r[0] for r in chosen]==['b','c']

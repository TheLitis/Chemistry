import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'casmi26/production.py'
    assert path.exists(), 'Production module missing'
    import sys, importlib
    sys.path.insert(0,str(path.parents[1]))
    return importlib.import_module('casmi26.production')


def test_vectorized_features_scale_and_row_isolation():
    m=module()
    mz=np.array([20.,31.,20.,31.]); intensity=np.array([4.,9.,40.,90.]);offsets=np.array([0,2,4])
    x,ok,masses=m.batch_features(mz,intensity,offsets,[47.0491412786]*2,['[M+H]+']*2,[[20.,40.]]*2)
    assert x.shape==(2,4104) and ok.all()
    np.testing.assert_allclose(x[0],x[1],atol=1e-6)
    assert masses[0]==pytest.approx(46.041864812,abs=1e-7)
    assert x[0,20]==pytest.approx(2/np.sqrt(13))


def test_adduct_mass_and_unknown_rejection():
    m=module()
    n,z,s=m.ion('[2M+Na]+'); assert n==2 and z==1
    assert (2*100+s-s)/n==100
    with pytest.raises(ValueError):m.ion('unknown')


def test_invalid_peaks_mark_only_their_spectrum():
    m=module()
    x,ok,_=m.batch_features(np.array([20.,np.nan]),np.array([1.,1.]),np.array([0,1,2]),
                           [100.,100.],['[M+H]+']*2,[[],[]])
    assert ok.tolist()==[True,False]
    assert np.isfinite(x).all()


def test_no_false_feature_from_precursor():
    m=module()
    x,ok,_=m.batch_features(np.array([100.,101.]),np.array([1.,1.]),np.array([0,2]),
                           [100.],['[M+H]+'],[[]])
    assert ok.tolist()==[True] and not x[0,:4096].any()


def test_cosine_one_to_one_and_scale():
    m=module()
    a=np.array([[20.,.3],[31.,.7]])
    assert m.fast_cosine(a,a)==pytest.approx(1)
    assert m.fast_cosine(a,a*[1,10])==pytest.approx(1)
    assert m.fast_cosine(a,np.array([[40.,1.]]))==0
    assert m.fast_cosine(np.array([[10.,.5],[10.001,.5]]),np.array([[10.,1.]]))<1


def test_candidate_selection_never_forces_target():
    m=module()
    assert m.mass_candidates(np.array([10.,20.,30.]),20.).tolist()==[1]
    assert not len(m.mass_candidates(np.array([10.,20.,30.]),25.))


def test_molecule_record_tautomer_and_stereo():
    m=module()
    a=m.molecule_record('C[C@H](O)F'); b=m.molecule_record('C[C@@H](O)F')
    assert a[2]==b[2] and a[4]==b[4]
    assert m.molecule_record('bogus')[2] is None


def test_split_by_structure_key():
    m=module()
    assert m.is_validation('TESTKEY')==m.is_validation('TESTKEY')
    assert len({m.is_validation(str(i)) for i in range(100)})==2


def test_arrow_feature_slice_matches_full_batch():
    pa=pytest.importorskip('pyarrow'); m=module()
    batch=pa.RecordBatch.from_pylist([
        {'ms2_mzs':[20.,31.], 'ms2_normalized_intensities':[4.,9.], 'precursor_mz':47.0491412786,
         'adduct':'[M+H]+', 'collision_energy_ev':[20.]},
        {'ms2_mzs':[25.], 'ms2_normalized_intensities':[1.], 'precursor_mz':100.,
         'adduct':'[M-H]-', 'collision_energy_ev':None}])
    full=m.arrow_features(batch);part=m.arrow_features(batch.slice(1,1))
    np.testing.assert_allclose(full[0][1:],part[0])
    assert full[1].all() and part[1].all()


def test_real_parquet_preparation_and_cache(tmp_path):
    pa=pytest.importorskip('pyarrow'); import pyarrow.parquet as pq
    m=module()
    train=tmp_path/'train.parquet';test=tmp_path/'test.parquet'; cache=tmp_path/'cache'
    row={'normalized_smiles':'CCO','ms2_mzs':[20.,31.], 'ms2_normalized_intensities':[4.,9.],
         'precursor_mz':47.0491412786,'adduct':'[M+H]+','collision_energy_ev':[20.]}
    pq.write_table(pa.Table.from_pylist([row]),train)
    query={**row,'molecule_id':'001','normalized_smiles':'deliberate wrong answer ignored'}
    pq.write_table(pa.Table.from_pylist([query]),test)
    report=m.prepare(train,test,cache,workers=1)
    assert report['accepted_spectra']==1 and report['test_spectra']==1
    assert report['broad_mass_candidate_counts']=={'001':1}
    x=np.load(cache/'features.npy'); assert x.shape==(1,4104)
    assert m.prepare(train,test,cache,workers=1)==report

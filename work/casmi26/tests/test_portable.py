from pathlib import Path
import importlib
import sys
import json
import numpy as np
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'casmi26/portable.py'
    assert path.exists(), 'Portable hidden-test inference is absent'
    sys.path.insert(0,str(path.parents[1]))
    return importlib.import_module('casmi26.portable')


def test_candidate_mass_fallback_is_explicit():
    m=module(); masses=np.array([100.,200.,300.])
    idx,mode=m.select_candidates(masses,200.)
    assert idx.tolist()==[1] and mode=='mass_compatible'
    idx,mode=m.select_candidates(masses,260.)
    assert len(idx)>0 and mode=='nearest_mass_no_compatible_structure'


def test_equivalent_graphs_do_not_consume_submission_ranks():
    m=module()
    catalog=[['a','CCO','same',46.],['b','OCC','same',46.],['c','COC','other',46.]]
    scores=[(1.,0,1.,.5),(.9,1,.9,.5),(.8,2,.8,.4)]
    assert m.select_distinct(scores,catalog)==['CCO','COC']


def test_template_for_old_visible_ids_cannot_block_hidden_ids(tmp_path):
    m=module();template=tmp_path/'sample_submission.csv'
    template.write_text('molecule_id,smiles\nvisible,CCO\n')
    order,source=m.output_order(template,{'hidden_b':[], 'hidden_a':[]})
    assert order==['hidden_a','hidden_b'] and source=='current_test_ids'


def test_matching_template_preserves_order(tmp_path):
    m=module();template=tmp_path/'sample_submission.csv'
    template.write_text('molecule_id,smiles\n002,IGNORE\n001,IGNORE\n')
    order,source=m.output_order(template,{'001':[], '002':[]})
    assert order==['002','001'] and source=='matching_template_ids'


def test_bundle_validation_detects_modification(tmp_path):
    m=module()
    files={}
    for name in m.ASSET_NAMES:
        (tmp_path/name).write_bytes(b'fixture')
        files[name]=m.sha256(tmp_path/name)
    (tmp_path/'bundle.json').write_text(json.dumps({'format':1,'files':files,'source':{'train_sha256':'abc'}}))
    assert m.verify_bundle(tmp_path)['format']==1
    (tmp_path/'model.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):m.verify_bundle(tmp_path)


def test_bundle_never_contains_query_specific_cache(tmp_path):
    m=module()
    assert 'query-library.parquet' not in m.ASSET_NAMES
    assert 'features.npy' not in m.ASSET_NAMES
    assert set(m.ASSET_NAMES)=={'catalog.json','fingerprints.npy','model.npz'}


def test_hidden_input_integration(tmp_path):
    pa=pytest.importorskip('pyarrow');import pyarrow.parquet as pq
    m=module();p=m.production
    raw=[p.molecule_record(s) for s in ('CCO','COC')];raw.sort(key=lambda r:(r[3],r[2],r[0]))
    bundle=tmp_path/'assets';bundle.mkdir()
    (bundle/'catalog.json').write_text(json.dumps([list(r[:4]) for r in raw]))
    np.save(bundle/'fingerprints.npy',np.array([np.frombuffer(bytes.fromhex(r[4]),dtype=np.uint8) for r in raw]))
    np.savez(bundle/'model.npz',version=np.array(1),w1=np.zeros((2,4104),np.float32),b1=np.zeros(2,np.float32),
             w2=np.zeros((2048,2),np.float32),b2=np.zeros(2048,np.float32))
    rows=[dict(normalized_smiles='CCO',ms2_mzs=[31.,29.],ms2_normalized_intensities=[10.,1.],
               precursor_mz=47.0491412786,adduct='[M+H]+',collision_energy_ev=[20.]),
          dict(normalized_smiles='COC',ms2_mzs=[45.,29.],ms2_normalized_intensities=[10.,1.],
               precursor_mz=47.0491412786,adduct='[M+H]+',collision_energy_ev=[20.])]
    train=tmp_path/'train.parquet';pq.write_table(pa.Table.from_pylist(rows),train)
    manifest={'format':1,'version':p.VERSION,'source':{'train_sha256':m.sha256(train)},
              'files':{name:m.sha256(bundle/name) for name in m.ASSET_NAMES}}
    (bundle/'bundle.json').write_text(json.dumps(manifest))
    query={**rows[1],'molecule_id':'new_hidden_id','normalized_smiles':'false answer ignored'}
    test=tmp_path/'test.parquet';pq.write_table(pa.Table.from_pylist([query]),test)
    out=tmp_path/'submission.csv'
    report=m.infer(test,train,bundle,out)
    assert report['prediction_count']==1 and report['test_spectra']==1
    import csv
    result=list(csv.DictReader(out.open()))
    assert result[0]['molecule_id']=='new_hidden_id'
    assert result[0]['smiles'].split(';')[0]=='COC'
    assert report['official_score'] is None

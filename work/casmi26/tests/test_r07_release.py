import json
from pathlib import Path
import numpy as np
import pytest


def test_selection_rejects_unvalidated_configurations():
    from casmi26.r07_release import validate_selection
    good={'kind':'v1_tanimoto','spectral_weight':1.0,'mass_weight':.5,'external_penalty':.2}
    assert validate_selection(good,1.4)['kind']=='v1_tanimoto'
    for k,v in [('kind','v3_ll'),('spectral_weight',-1),('mass_weight',float('nan')),('external_penalty',True)]:
        with pytest.raises(ValueError):validate_selection({**good,k:v},1.4)
    with pytest.raises(ValueError):validate_selection(good,-1e6)


def test_source_identity_survives_line_endings_but_not_edits(tmp_path):
    from casmi26.r07_release import source_identity
    p=tmp_path/'a.py';p.write_bytes(b'x = 1\r\n')
    a=source_identity(tmp_path,['a.py'])
    p.write_bytes(b'x = 1\n');assert source_identity(tmp_path,['a.py'])==a
    p.write_bytes(b'x = 2\n');assert source_identity(tmp_path,['a.py'])!=a


def test_selected_candidate_scores_match_sealed_protocol():
    from casmi26.r07_release import selected_scores
    from casmi26.target_domain import mass_prior,hybrid_scores
    bits=np.array([[1,0,1],[0,1,0]],dtype='u1');logits=np.array([.3,-.2,1.])
    sp=np.array([.8,0.]);m=np.array([100.001,100.]);ext=np.array([0.,1.])
    config={'kind':'v1_tanimoto','spectral_weight':1.,'mass_weight':.5,'external_penalty':.2}
    probabilities=1/(1+np.exp(-np.clip(logits,-40,40)))
    dots=bits@probabilities;neural=dots/np.maximum(probabilities.sum()+bits.sum(1)-dots,1e-8)
    expected=hybrid_scores(neural,sp,mass_prior(m,100.,offset_ppm=1.4,scale_ppm=5.),spectral_weight=1.,mass_weight=.5)-.2*ext
    np.testing.assert_allclose(selected_scores(logits,bits,sp,m,100.,ext,config,1.4),expected,rtol=0,atol=1e-12)


def fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import io,csv,zipfile
    from casmi26.production import molecule_record,sha256
    root=tmp_path/'bundle';root.mkdir()
    recs=sorted([molecule_record(s) for s in ('CCO','CC(=O)C')],key=lambda x:x[3])
    (root/'catalog.json').write_text(json.dumps([list(r[:4]) for r in recs]))
    np.save(root/'fingerprints.npy',np.stack([np.frombuffer(bytes.fromhex(r[4]),dtype='u1') for r in recs]))
    np.savez(root/'model.npz',format=np.array(3),feature_dim=np.array(4104),head_sizes=np.array([2048]),
        w1=np.zeros((4,4104),'f4'),b1=np.zeros(4,'f4'),w2=np.zeros((2048,4),'f4'),b2=np.zeros(2048,'f4'))
    stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['identifier','canonical_smiles','exact_molecular_weight'])
    ext=molecule_record('COC');writer.writerow(['EXTERNAL-1',ext[1],ext[3]])
    with zipfile.ZipFile(root/'coconut.zip','w') as z:z.writestr('coconut.csv',stream.getvalue())
    rows=[]
    for i,r in enumerate(recs):
        rows.append(dict(molecule_id=f'{i:03d}',normalized_smiles=r[0],ms2_mzs=[20.,31.15],
          ms2_normalized_intensities=[.2,.8],precursor_mz=r[3]+1.007276466621,adduct='[M+H]+',collision_energy_ev=[20.]))
    train=tmp_path/'train.parquet';test=tmp_path/'test.parquet'
    pq.write_table(pa.Table.from_pylist(rows),train)
    pq.write_table(pa.Table.from_pylist([{**r,'normalized_smiles':'FORBIDDEN_TEST_LABEL'} for r in rows+[rows[0]]]),test)
    manifest=dict(format=7,algorithm='r07-selected-v1-hybrid',feature_version='official-corpus-v1',
        selection={'kind':'v1_tanimoto','spectral_weight':1.,'mass_weight':.5,'external_penalty':.2},
        mass_offset_ppm=1.4,mass_scale_ppm=5.,train_sha256=sha256(train),
        files={n:sha256(root/n) for n in ('model.npz','catalog.json','fingerprints.npy','coconut.zip')},
        contains_test_ids_or_predictions=False,weights_kind='synthetic-test')
    (root/'r07-bundle.json').write_text(json.dumps(manifest))
    return root,train,test,recs


def test_candidate_end_to_end_and_offline_external_recovery(tmp_path):
    import csv
    from casmi26.r07_release import infer
    bundle,train,test,recs=fixture(tmp_path)
    output=tmp_path/'sub.csv'
    template=tmp_path/'sample.csv';template.write_text('molecule_id,smiles\n001,ignored\n000,ignored\n')
    report=infer(test,train,bundle,output,template=template,workers=1)
    rows=list(csv.DictReader(output.open()));assert [r['molecule_id'] for r in rows]==['001','000']
    assert report['test_spectra']==3 and report['prediction_count']==2
    assert report['external']['accepted']==1 and report['test_labels_used'] is False
    assert report['details']['000']['spectra_used']==2
    assert 'COC' in rows[1]['smiles'].split(';')
    assert report['official_score'] is None
    report2=infer(test,train,bundle,tmp_path/'again.csv',workers=1)
    assert report2['prediction_count']==2


def test_bundle_and_input_tamper_rejected(tmp_path):
    from casmi26.r07_release import infer,verify_bundle
    bundle,train,test,_=fixture(tmp_path)
    verify_bundle(bundle)
    with pytest.raises(ValueError,match='overwrite'):infer(test,train,bundle,test,workers=1)
    with pytest.raises(ValueError,match='overlap'):infer(train,train,bundle,tmp_path/'no.csv',workers=1)
    (bundle/'catalog.json').write_text('[]')
    with pytest.raises(ValueError,match='hash'):verify_bundle(bundle)


def test_empty_mass_window_uses_explicit_nonempty_fallback(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from casmi26.r07_release import infer
    bundle,train,test,_=fixture(tmp_path)
    rows=pq.read_table(test).to_pylist()
    for r in rows:r['precursor_mz']=999.0
    pq.write_table(pa.Table.from_pylist(rows),test)
    report=infer(test,train,bundle,tmp_path/'sub.csv',workers=1)
    assert len(report['mass_incompatible_fallbacks'])==2
    assert report['empty_candidate_rows']==[]


def test_unsupported_query_never_silently_drops_compound(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from casmi26.r07_release import infer
    bundle,train,test,_=fixture(tmp_path)
    rows=pq.read_table(test).to_pylist();rows[0]['molecule_id']=None
    pq.write_table(pa.Table.from_pylist(rows),test)
    with pytest.raises(ValueError,match='molecule'):infer(test,train,bundle,tmp_path/'sub.csv',workers=1)
    assert not (tmp_path/'sub.csv').exists()

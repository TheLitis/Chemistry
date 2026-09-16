import json
from pathlib import Path
import numpy as np
import pytest


def test_fractional_features_preserve_base_and_resolve_collision():
    from casmi26.features_v3 import highres_features
    from casmi26.production import batch_features
    a=(np.array([50.1,100.1,50.2,100.2]),np.array([.6,.4,.6,.4]),np.array([0,2,4]),
       np.array([200.5,200.5]),['[M+H]+']*2,[[30.],[30.]])
    base,v,m=batch_features(*a);x,w,n=highres_features(*a)
    assert x.shape==(2,8200)
    np.testing.assert_array_equal(x[:,:4104],base)
    np.testing.assert_array_equal(v,w);np.testing.assert_array_equal(m,n)
    np.testing.assert_array_equal(base[0],base[1]);assert np.max(abs(x[0]-x[1]))>.01
    a2=list(a);a2[1]=a[1]*100
    np.testing.assert_allclose(highres_features(*a2)[0],x,atol=2e-7)


def test_highres_bad_spectrum_is_zero_and_order_invariant():
    from casmi26.features_v3 import highres_features
    x,v,_=highres_features([float('nan'),50],[1,1],[0,1,2],[100,100],['[M+H]+']*2,[None,None])
    assert v.tolist()==[False,True];assert not x[0].any();assert np.isfinite(x).all()
    x,_,_=highres_features([31.15,20.34,20.34,31.15],[3,1,1,3],[0,2,4],[100,100],['[M+H]+']*2,[None,None])
    np.testing.assert_array_equal(x[0],x[1])


def test_multitarget_definition_and_tautomer_normalization():
    from casmi26.features_v3 import fingerprint_targets,HEAD_SIZES
    from casmi26.production import molecule_record
    a=fingerprint_targets('CCO');b=fingerprint_targets('OCC')
    assert HEAD_SIZES==(2048,4096,8192);assert a.shape==(1792,);assert a.dtype==np.uint8
    np.testing.assert_array_equal(a,b)
    assert a[:256].tobytes().hex()==molecule_record('CCO')[4]
    with pytest.raises(ValueError):fingerprint_targets('not a molecule')


def checkpoint(path,features=8200,heads=(2048,)):
    rng=np.random.default_rng(17);h=8
    arrays=dict(format=np.array(3),feature_dim=np.array(features),head_sizes=np.array(heads),
       w1=rng.normal(0,.01,(h,features)).astype('f4'),b1=np.zeros(h,'f4'),
       w2=rng.normal(0,.1,(sum(heads),h)).astype('f4'),b2=np.zeros(sum(heads),'f4'))
    np.savez_compressed(path,**arrays);return arrays


def test_model_numpy_batch_parity_and_validation(tmp_path):
    from casmi26.model_v3 import MultiFingerprintModel
    a=checkpoint(tmp_path/'model.npz');m=MultiFingerprintModel(tmp_path/'model.npz')
    x=np.ones((3,8200),'f4')
    expected=np.maximum(x@a['w1'].T+a['b1'],0)@a['w2'].T+a['b2']
    np.testing.assert_allclose(m.logits(x),expected,atol=2e-6)
    np.testing.assert_allclose(m.logits(x[0]),expected[0],atol=2e-6)
    with pytest.raises(ValueError):m.logits(np.zeros(4104))
    with pytest.raises(ValueError):m.logits(np.full(8200,np.nan))
    a['w1'][0,0]=np.nan;np.savez(tmp_path/'bad.npz',**a)
    with pytest.raises(ValueError):MultiFingerprintModel(tmp_path/'bad.npz')


def test_multihead_ranking_uses_nonlocal_information():
    from casmi26.model_v3 import candidate_scores
    sizes=(2,2,2);fp=np.array([[1,0,1,0,0,1],[1,0,0,1,1,0]],'u1')
    logits=np.array([2,-2,-3,3,3,-3])
    s=candidate_scores(logits,fp,[100,100],100,head_sizes=sizes,weights=(0,0,1))
    assert s[1]>s[0]
    with pytest.raises(ValueError):candidate_scores(logits,fp,[100,100],100,head_sizes=sizes,weights=(-1,0,1))
    with pytest.raises(ValueError):candidate_scores(logits,fp,[100,100],100,head_sizes=sizes,weights=(0,0,0))


def test_training_cpu_export_is_executable(tmp_path):
    pytest.importorskip('torch')
    from casmi26.model_v3 import train_model,MultiFingerprintModel
    rng=np.random.default_rng(4);x=rng.normal(size=(8,8200)).astype('f4')
    packed=np.packbits(rng.integers(0,2,(8,14336),'u1'),axis=1)
    report=train_model(x,packed,np.arange(8),tmp_path/'m.npz',epochs=2,hidden=8,batch_size=4,
                       device='cpu',head_sizes=(2048,),feature_dim=8200)
    assert report['training_rows']==8;assert report['epochs']==2
    assert np.isfinite(MultiFingerprintModel(tmp_path/'m.npz').logits(x)).all()
    assert report['export_max_absolute_error']<1e-4


def test_fresh_evaluation_split_is_disjoint_and_deterministic():
    from casmi26.pipeline_v3 import fresh_partition
    keys=[f'k{i}' for i in range(50)]
    a,b=fresh_partition(keys,{'k0','k1'},10,20)
    assert not set(a)&set(b);assert not (set(a)|set(b))&{'k0','k1'}
    assert (a,b)==fresh_partition(list(reversed(keys)),{'k1','k0'},10,20)
    with pytest.raises(ValueError):fresh_partition(keys,set(),30,30)


def test_bundle_rejects_tamper(tmp_path):
    from casmi26.pipeline_v3 import verify_bundle
    from casmi26.production import sha256
    (tmp_path/'catalog.json').write_text('[]');checkpoint(tmp_path/'model.npz')
    files={n:sha256(tmp_path/n) for n in ('catalog.json','model.npz')}
    (tmp_path/'v3-bundle.json').write_text(json.dumps({'format':3,'files':files,'head_sizes':[2048],
         'feature_dim':8200,'weights':[1],'mode':'neural','train_sha256':'x'}))
    with pytest.raises(ValueError):verify_bundle(tmp_path)


def fixture_v3(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from casmi26.production import molecule_record,sha256
    from casmi26.features_v3 import fingerprint_targets, FEATURE_VERSION
    data=tmp_path/'data';data.mkdir();bundle=tmp_path/'bundle';bundle.mkdir()
    records=[molecule_record('CCO'),molecule_record('CC(=O)C')]
    records.sort(key=lambda r:r[3]);catalog=[list(r[:4]) for r in records]
    train=[];test=[]
    for i,r in enumerate(records):
        p=r[3]+1.007276466621
        row={'normalized_smiles':r[0],'ms2_mzs':[20.,31.15],
             'ms2_normalized_intensities':[.2,.8],'precursor_mz':p,'adduct':'[M+H]+',
             'collision_energy_ev':[20.],'ingest_lib':'test-fixture','instrument_type':'Bruker'}
        train.append(row);test.append({**row,'molecule_id':str(i).zfill(3),'normalized_smiles':'DO_NOT_READ_TEST_LABEL'})
    pq.write_table(pa.Table.from_pylist(train),data/'train.parquet')
    pq.write_table(pa.Table.from_pylist(test+[test[0]]),data/'test.parquet')
    (data/'sample_submission.csv').write_text('molecule_id,smiles\n001,\n000,\n')
    (bundle/'catalog.json').write_text(json.dumps(catalog));checkpoint(bundle/'model.npz')
    np.save(bundle/'targets.npy',np.stack([fingerprint_targets(r[0]) for r in catalog]))
    manifest={'format':3,'feature_version':FEATURE_VERSION,'feature_dim':8200,'head_sizes':[2048],'weights':[1.],
              'mode':'neural','train_sha256':sha256(data/'train.parquet'),
              'files':{n:sha256(bundle/n) for n in ('model.npz','targets.npy','catalog.json')}}
    (bundle/'v3-bundle.json').write_text(json.dumps(manifest))
    return data,bundle,catalog


def test_v3_end_to_end_all_spectra_current_ids_labels_ignored(tmp_path):
    from casmi26.inference_v3 import infer
    data,bundle,catalog=fixture_v3(tmp_path)
    report=infer(data/'test.parquet',data/'train.parquet',bundle,tmp_path/'submission.csv',data/'sample_submission.csv')
    import csv
    rows=list(csv.DictReader((tmp_path/'submission.csv').open()))
    assert [r['molecule_id'] for r in rows]==['001','000']
    assert rows[1]['smiles']==catalog[0][1]
    assert report['test_spectra']==3;assert report['details']['000']['spectra_used']==2
    assert report['test_labels_used'] is False;assert report['official_score'] is None
    with pytest.raises(ValueError,match='overwrite'):
        infer(data/'test.parquet',data/'train.parquet',bundle,data/'test.parquet')
    (data/'sample_submission.csv').write_text('molecule_id,smiles\nstale-example-id,C\n')
    report=infer(data/'test.parquet',data/'train.parquet',bundle,tmp_path/'new.csv',data/'sample_submission.csv')
    assert report['prediction_count']==2;assert report['order_source']=='current_test_ids'


def test_bundle_actual_tamper_and_bad_weights(tmp_path):
    from casmi26.pipeline_v3 import verify_bundle
    data,bundle,_=fixture_v3(tmp_path);verify_bundle(bundle)
    a=np.load(bundle/'targets.npy');a[0,0]^=1;np.save(bundle/'targets.npy',a)
    with pytest.raises(ValueError,match='hash'):verify_bundle(bundle)


def test_real_cache_build_matches_training_transform(tmp_path):
    from casmi26.cache_v3 import prepare_cache
    from casmi26.features_v3 import arrow_highres
    from casmi26.production import sha256
    import pyarrow.parquet as pq
    data,bundle,catalog=fixture_v3(tmp_path);base=tmp_path/'old-cache';base.mkdir()
    table=pq.read_table(data/'train.parquet').combine_chunks();x,v,neutral=arrow_highres(table.to_batches()[0])
    (base/'catalog.json').write_text(json.dumps(catalog));np.save(base/'features.npy',x[:,:4104])
    np.save(base/'fingerprints.npy',np.load(bundle/'targets.npy')[:,:256]);np.save(base/'counts.npy',np.ones(2,'int64'))
    (base/'prepared.json').write_text(json.dumps({'signature':{'train_sha256':sha256(data/'train.parquet')}}))
    dest=tmp_path/'new-cache';report=prepare_cache(data/'train.parquet',base,dest,workers=1)
    assert report['accepted_spectra']==2;assert report['test_data_read'] is False
    np.testing.assert_allclose(np.load(dest/'features.npy'),x,atol=1e-7)
    assert prepare_cache(data/'train.parquet',base,dest,workers=1)==report
    a=np.load(dest/'features.npy');a[0,0]+=1;np.save(dest/'features.npy',a)
    with pytest.raises(ValueError,match='modified'):prepare_cache(data/'train.parquet',base,dest,workers=1)


def test_notebook_embeds_source_but_not_test_ids_and_compiles(tmp_path):
    from casmi26.notebook_v3 import build_notebook
    n=json.loads(build_notebook(tmp_path/'v3.ipynb').read_text())
    assert n['nbformat']==4
    for cell in n['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<cell>','exec')
    text=''.join(n['cells'][1]['source'])
    assert "'--no-index'" in text;assert 'inference_v3' in text
    assert 'KAGGLE_API_TOKEN' not in text

"""R06 checkpoint packaging and inference contracts, without provider access."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('torch')


def fixture(tmp_path, pooling='early'):
    from casmi26.architectures import make_model, save_model
    from casmi26.features_v3 import fingerprint_targets
    from casmi26.production import molecule_record, sha256, write_json
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = tmp_path/'experiment'; root.mkdir()
    trials=root/'trials';trials.mkdir()
    cache=tmp_path/'cache';cache.mkdir()
    catalog=[list(molecule_record(s)[:4]) for s in ['CCO','CC(=O)C']]
    catalog.sort(key=lambda r:r[3])
    write_json(cache/'catalog.json',catalog)
    np.save(cache/'targets.npy',np.stack([fingerprint_targets(r[0]) for r in catalog]))
    anchor=tmp_path/'anchor.npz'
    rng=np.random.default_rng(1)
    np.savez_compressed(anchor,format=np.array(3),feature_dim=np.array(8200),head_sizes=np.array([2048,4096,8192]),
        w1=rng.normal(0,.01,(8,8200)).astype('f4'),b1=np.zeros(8,'f4'),w2=rng.normal(0,.01,(14336,8)).astype('f4'),b2=np.zeros(14336,'f4'))
    model=make_model('deep_sets');save_model(trials/'best.npz',model)
    np.save(trials/'best.active.npy',np.ones(8200,'u1'))
    write_json(trials/'best.json',{'checkpoint_sha256':sha256(trials/'best.npz'),'mask_sha256':sha256(trials/'best.active.npy')})
    config={'component':'deep_sets','models':['best.npz'],'weights':[.5,.25,.25],'sigma_ppm':5.,'pooling':pooling}
    selected={'configuration':config,'selected_before_audit':True,'audit_keys_sha256':'a'*64,
              'model_hashes':{'best.npz':sha256(trials/'best.npz')}}
    write_json(root/'selection-before-audit.json',selected)
    write_json(root/'report.json',{'status':'completed','supported_improvement':True,'selection':selected,
               'audit':{'3':{'selected':{'mrr_at_25':.7}}},'test_data_read':False})
    train=tmp_path/'train.parquet';test=tmp_path/'test.parquet'
    rows=[]
    for i,c in enumerate(catalog):
        for j in range(5):
            rows.append({'molecule_id':str(i).zfill(3),'precursor_mz':c[3]+1.007276466621,
               'adduct':'[M+H]+','collision_energy_ev':[10.+j], 'ms2_mzs':[20.125+j*.01,31.25],
               'ms2_normalized_intensities':[.4,.6],'normalized_smiles':'IGNORED_TEST_ANSWER'})
    pq.write_table(pa.Table.from_pylist(rows),test)
    pq.write_table(pa.Table.from_pylist([dict(r, normalized_smiles=catalog[0][1]) for r in rows[:1]]),train)
    source={'train_sha256':sha256(train),'catalog_sha256':sha256(cache/'catalog.json'),
            'targets_sha256':sha256(cache/'targets.npy'),'anchor_sha256':sha256(anchor)}
    write_json(root/'protocol.json',{'source':source})
    return root,cache,anchor,train,test,rows


def test_build_validated_package_and_hash_guard(tmp_path):
    from casmi26.r06_candidate import build_bundle,verify_bundle
    root,cache,anchor,*_=fixture(tmp_path)
    b=tmp_path/'bundle';r=build_bundle(root,cache,anchor,b)
    assert r['format']==6 and r['candidate_not_champion']
    assert r['selection']['models']==['best.npz']
    assert build_bundle(root,cache,anchor,b)==r
    with (b/'best.active.npy').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='hash'):verify_bundle(b)


def test_unverified_audit_cannot_be_packaged_as_winner(tmp_path):
    from casmi26.r06_candidate import build_bundle
    root,cache,anchor,*_=fixture(tmp_path)
    report=json.loads((root/'report.json').read_text());report['supported_improvement']=False
    (root/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='improvement'):build_bundle(root,cache,anchor,tmp_path/'bundle')
    assert not (tmp_path/'bundle').exists()


def test_selected_checkpoint_cannot_be_silently_replaced(tmp_path):
    from casmi26.r06_candidate import build_bundle
    root,cache,anchor,*_=fixture(tmp_path)
    (root/'trials/best.npz').write_bytes(b'different')
    with pytest.raises(ValueError,match='hash'):build_bundle(root,cache,anchor,tmp_path/'bundle')


@pytest.mark.parametrize('pooling',['early','late_probability'])
def test_all_acquisitions_inference_and_permutation(tmp_path,pooling):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from casmi26.r06_candidate import build_bundle,infer
    root,cache,anchor,train,test,rows=fixture(tmp_path,pooling)
    b=tmp_path/'bundle';build_bundle(root,cache,anchor,b)
    output=tmp_path/'submission.csv'
    r=infer(test,b,output,device='cpu',budget='all')
    assert r['prediction_count']==2 and r['test_spectra']==10
    assert all(d['spectra_used']==5 for d in r['details'].values())
    assert r['test_labels_used'] is False and r['official_score'] is None
    original=output.read_bytes()
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))),test)
    r2=infer(test,b,output,device='cpu',budget='all')
    assert output.read_bytes()==original
    assert r2['selection']==r['selection']
    with pytest.raises(ValueError,match='overwrite'):infer(test,b,test,device='cpu')


def test_hidden_mass_without_strict_candidates_still_returns_valid_guesses(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from casmi26.r06_candidate import build_bundle,infer
    root,cache,anchor,train,test,rows=fixture(tmp_path)
    b=tmp_path/'bundle';build_bundle(root,cache,anchor,b)
    # Move every query far outside the tiny fixture catalog while keeping each
    # compound internally mass-consistent. A code-competition rerun must still
    # produce a nonempty, scorer-safe cell for every current test molecule.
    shifted=[dict(row,precursor_mz=float(row['precursor_mz'])+500.0) for row in rows]
    pq.write_table(pa.Table.from_pylist(shifted),test)
    out=tmp_path/'fallback.csv';report=infer(test,b,out,device='cpu',budget='all')
    with out.open(encoding='utf-8',newline='') as f:pred=list(csv.DictReader(f))
    assert len(pred)==2
    assert all(1<=len(r['smiles'].split(';'))<=25 for r in pred)
    assert report['empty_candidate_rows']==[]
    assert sorted(report['mass_incompatible_fallbacks'])==['000','001']
    assert all(d['candidate_mode']=='nearest_mass_no_compatible_structure' for d in report['details'].values())


def test_three_view_path_matches_training_transform(tmp_path):
    from casmi26.r06_candidate import prepare_group
    from casmi26.architecture_training import pool_acquisitions
    from casmi26.architectures import spectrum_digest,peak_tokens
    from casmi26.features_v3 import arrow_highres
    import pyarrow as pa
    root,cache,anchor,train,test,rows=fixture(tmp_path)
    group=rows[:5];x,t,m,used=prepare_group(group,budget=3)
    selected=sorted(group,key=spectrum_digest)[:3]
    a,valid,neutral=arrow_highres(pa.RecordBatch.from_pylist(selected))
    views=a.astype('f2').astype('f4')[None];tokens=np.stack([peak_tokens(r) for r in selected])[None]
    xx,tt,mm=pool_acquisitions(views,tokens,neutral[None],np.array([3]),budget=3)
    np.testing.assert_array_equal(x,xx[0]);np.testing.assert_array_equal(t,tt[0]);assert m==mm[0] and used==3


def test_refuse_conflicting_mass_and_missing_id(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from casmi26.r06_candidate import build_bundle,infer
    root,cache,anchor,train,test,rows=fixture(tmp_path)
    b=tmp_path/'bundle';build_bundle(root,cache,anchor,b)
    rows[0]['precursor_mz']+=3
    pq.write_table(pa.Table.from_pylist(rows),test)
    with pytest.raises(ValueError,match='mass'):infer(test,b,tmp_path/'s.csv',device='cpu')
    rows[0]['molecule_id']=''
    pq.write_table(pa.Table.from_pylist(rows),test)
    with pytest.raises(ValueError,match='ID'):infer(test,b,tmp_path/'s.csv',device='cpu')

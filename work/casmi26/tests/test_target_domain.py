"""Target-domain safeguards: split by graph, not source spelling or raw SMILES."""
import numpy as np
import pytest


def test_protocol_module_exists():
    from pathlib import Path
    assert (Path(__file__).parents[1]/'casmi26/target_domain.py').is_file()


def test_target_instrument_not_only_literal_bruker():
    from casmi26.target_domain import is_target_instrument
    assert is_target_instrument('timsTOF')
    assert is_target_instrument('Bruker timsTOF Pro')
    assert not is_target_instrument('Bruker Maxis')
    assert not is_target_instrument(None)


def test_split_forces_previous_examined_keys_into_calibration():
    from casmi26.target_domain import split_target
    keys=[f'key-{i}' for i in range(12)]
    a,b=split_target(keys,{'key-2','key-3','unrelated'},5)
    assert set(a)>= {'key-2','key-3'}
    assert not set(a)&set(b) and set(a)|set(b)==set(keys)
    assert len(a)==5 and len(b)==7
    assert (a,b)==split_target(list(reversed(keys)),{'key-3','key-2'},5)
    with pytest.raises(ValueError):split_target(keys,set(keys),5)


def test_training_removes_all_aliases_and_deduplicates_keys():
    from casmi26.target_domain import training_rows
    cat=[['a','a','A',10],['b','b','A',10],['c','c','B',20],['d','d','C',30]]
    ids=training_rows(cat,np.array([2,8,9,0]),{'B'},exclude_hash_holdout=False)
    assert ids.tolist()==[1]
    assert not training_rows(cat,np.array([2,8,9,0]),{'A','B'},exclude_hash_holdout=False).size


def test_mass_prior_does_not_depend_on_other_candidates():
    from casmi26.target_domain import mass_prior
    a=mass_prior(np.array([100.,100.001]),100.0002,offset_ppm=2,scale_ppm=5)
    b=mass_prior(np.array([100.,100.001,200.]),100.0002,offset_ppm=2,scale_ppm=5)
    np.testing.assert_array_equal(a,b[:2])
    assert a[0]>a[1]
    with pytest.raises(ValueError):mass_prior([100],100,scale_ppm=0)


def test_feature_scores_are_candidate_independent_and_use_bits():
    from casmi26.target_domain import fingerprint_evidence
    z=np.array([2.,-3.,1.,-2.]);b=np.array([[1,0,1,0],[0,1,0,1]],dtype='u1')
    s=fingerprint_evidence(z,b,head_sizes=(2,2))
    t=fingerprint_evidence(z,np.vstack([b,[1,1,1,1]]),head_sizes=(2,2))
    np.testing.assert_array_equal(s,t[:2]);assert (s[0]>s[1]).all()


def test_metric_counts_missing_truth_and_tautomer_alias_once():
    from casmi26.target_domain import rank_key, metrics
    assert rank_key([3,2,1],['A','A','B'],'B')==2
    assert rank_key([3,2,1],['A','A','B'],'C')==0
    assert metrics([1,2,0])['mrr_at_25']==pytest.approx(.5)


def test_hybrid_zero_spectral_evidence_does_not_demote_unreferenced_truth():
    from casmi26.target_domain import hybrid_scores
    n=np.array([.8,.1]);s=np.array([0.,0.]);m=np.zeros(2)
    assert np.argmax(hybrid_scores(n,s,m,spectral_weight=2))==0
    with pytest.raises(ValueError):hybrid_scores(n,s,m,spectral_weight=-1)


def test_full_domain_runner_with_isolated_synthetic_corpus(tmp_path):
    """Software integration only: the fabricated tiny corpus is not an accuracy test."""
    import importlib.util
    import json
    from pathlib import Path
    import zipfile
    import pyarrow as pa
    import pyarrow.parquet as pq
    from casmi26.production import molecule_record, sha256, write_json, is_validation
    from casmi26.features_v3 import arrow_highres, fingerprint_targets
    repo=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('domain_task',repo/'tasks/casmi_r07_domain.py')
    task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
    assert hasattr(task,'execute'), 'Domain experiment must expose a testable execution function'
    state=tmp_path/'state'; art=state/'artifacts/casmi26'; root=art/'research-r07'
    cache=state/'cache/casmi26/highres-v3';cache.mkdir(parents=True);root.mkdir(parents=True)
    smis=['C','CC','CCC','CCCC','CCCCC','CCCCCC','CO','CCO','CCCO','CCCCO','CN','CCN','CCCN','CCCCN','CC(=O)C','CC(=O)O']
    recs=[molecule_record(s) for s in smis]
    recs=sorted([r for r in recs if r[2] and not is_validation(r[2])],key=lambda r:(r[3],r[2]))
    queries=recs[-8:]; query_keys={r[2] for r in queries}
    rows=[]
    for i,r in enumerate(recs):
        common={'normalized_smiles':r[0],'ms2_mzs':[r[3]*.21,r[3]*.41],
                'ms2_normalized_intensities':[.2,1.], 'precursor_mz':r[3]+1.007276466621,
                'adduct':'[M+H]+','collision_energy_ev':[20.], 'instrument_type':'timsTOF'}
        rows.append({**common,'ingest_lib':'library'})
        if r[2] in query_keys:
            rows.append({**common,'ms2_mzs':[r[3]*.211,r[3]*.411], 'ingest_lib':'enveda-np-examples'})
    train=tmp_path/'train.parquet';pq.write_table(pa.Table.from_pylist(rows),train)
    npfile=root/'np-examples.parquet'
    pq.write_table(pa.Table.from_pylist([r for r in rows if r['ingest_lib']=='enveda-np-examples']),npfile)
    catalog=[list(r[:4]) for r in recs];write_json(cache/'catalog.json',catalog)
    features=[]
    for r in recs:
        b=pa.RecordBatch.from_pylist([next(x for x in rows if x['normalized_smiles']==r[0])])
        features.append(arrow_highres(b)[0][0])
    np.save(cache/'features.npy',np.stack(features));np.save(cache/'counts.npy',np.ones(len(recs),dtype='i8'))
    np.save(cache/'targets.npy',np.stack([fingerprint_targets(r[0]) for r in recs]))
    write_json(cache/'prepared-v3.json',{'signature':{'train_sha256':sha256(train)}})
    write_json(root/'inventory.json',{'train_path':str(train),'train_sha256':sha256(train),
                                     'np_examples':{'sha256':sha256(npfile)}})
    archive=state/'data/external/coconut-2026-08/coconut_csv_lite-08-2026.zip';archive.parent.mkdir(parents=True)
    import csv,io
    buffer=io.StringIO();w=csv.writer(buffer);w.writerow(['identifier','canonical_smiles','exact_molecular_weight'])
    for i,r in enumerate(recs):w.writerow([f'fixture-{i}',r[1],r[3]])
    with zipfile.ZipFile(archive,'w') as z:z.writestr('fixture.csv',buffer.getvalue())
    output=tmp_path/'output'
    result=task.execute(state,repo,output,device='cpu',epochs=1,calibration_size=4,
                        snapshot_sha256=sha256(archive),workers=1,run_tests=False)
    assert result==0
    report=json.loads((output/'report.json').read_text())
    assert report['training_target_overlap']==0 and report['warmstart'] is None
    assert report['calibration_molecules']==4 and report['audit_molecules']==4
    assert report['new_submissions']==0 and report['test_data_read'] is False
    assert set(report['audit'])=={'available','absent','external_recovery'}
    assert all(v['selected']['molecules']==4 for v in report['audit'].values())
    assert report['source']['train_sha256']==sha256(train)
    assert task.execute(state,repo,output,device='cpu',epochs=1,calibration_size=4,
                        snapshot_sha256=sha256(archive),workers=1,run_tests=False)==0

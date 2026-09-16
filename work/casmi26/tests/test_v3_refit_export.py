"""Positive-branch refit integration fixture; not validation/accuracy evidence."""
import json
import numpy as np
import pytest


def test_positive_gate_refits_and_exports_usable_bundle(tmp_path):
    pytest.importorskip('torch')
    import importlib.util
    from pathlib import Path
    from casmi26.features_v3 import arrow_highres
    from casmi26.production import sha256
    from casmi26.pipeline_v3 import refit_bundle,verify_bundle
    from casmi26.inference_v3 import infer
    import pyarrow.parquet as pq
    spec=importlib.util.spec_from_file_location('v3fixture',Path(__file__).with_name('test_v3.py'))
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    data,base,catalog=fixture.fixture_v3(tmp_path)
    x,_,_=arrow_highres(pq.read_table(data/'train.parquet').combine_chunks().to_batches()[0])
    np.save(base/'features.npy',x);np.save(base/'counts.npy',np.ones(len(catalog),'int64'))
    old=tmp_path/'old-models';old.mkdir()
    np.savez_compressed(old/'model.npz',version=np.array(1),
        w1=np.zeros((512,4104),'f4'),b1=np.zeros(512,'f4'),
        w2=np.zeros((2048,512),'f4'),b2=np.zeros(2048,'f4'))
    experiment=tmp_path/'synthetic-positive-gate';experiment.mkdir()
    np.savez_compressed(experiment/'selected.npz',format=np.array(3),feature_dim=np.array(8200),
        head_sizes=np.array([2048]),w1=np.zeros((512,8200),'f4'),b1=np.zeros(512,'f4'),
        w2=np.zeros((2048,512),'f4'),b2=np.zeros(2048,'f4'))
    selected={'model':'selected.npz','weights':[1.]}
    (experiment/'validation-v3.json').write_text(json.dumps({'eligible_for_candidate_bundle':True,'selection':selected,'fixture_only':True}))
    (experiment/'selection-before-audit.json').write_text(json.dumps(selected))
    (experiment/'protocol.json').write_text(json.dumps({'source':{'train_sha256':sha256(data/'train.parquet')},'epochs':1}))
    destination=tmp_path/'full-refit'
    result=refit_bundle(base,old,experiment,destination,device='cpu')
    assert result['status']=='bundle_created'
    manifest=verify_bundle(destination)
    assert manifest['training']['training_rows']==2
    assert manifest['contains_test_ids_or_predictions'] is False
    assert refit_bundle(base,old,experiment,destination,device='cpu')['status']=='bundle_verified'
    report=infer(data/'test.parquet',data/'train.parquet',destination,tmp_path/'out.csv')
    assert report['prediction_count']==2
    assert report['routing']=='confidence'

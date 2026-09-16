"""Small real-structure, synthetic-feature execution test; not accuracy evidence."""
import json
import numpy as np
import pytest


def test_fit_selection_resume_and_rejected_promotion(tmp_path):
    pytest.importorskip('torch')
    from casmi26.production import molecule_record,is_validation,sha256
    from casmi26.features_v3 import fingerprint_targets
    from casmi26.pipeline_v3 import fit_experiment,refit_bundle
    catalog=[]
    for i in range(1,101):
        r=molecule_record('C'*i)
        catalog.append(list(r[:4]))
    assert sum(is_validation(r[2]) for r in catalog)>=4
    cache=tmp_path/'cache';cache.mkdir()
    old=tmp_path/'artifacts/official-v1';old.mkdir(parents=True)
    experiment=tmp_path/'artifacts/research-v3'
    rng=np.random.default_rng(44)
    np.save(cache/'features.npy',rng.random((len(catalog),8200),dtype='f4'))
    np.save(cache/'targets.npy',np.stack([fingerprint_targets(r[0]) for r in catalog]))
    np.save(cache/'counts.npy',np.ones(len(catalog),'int64'))
    np.save(cache/'observed.npy',np.array([r[3] for r in catalog]))
    np.save(cache/'strata.npy',np.full(len(catalog),2,'uint8'))
    (cache/'catalog.json').write_text(json.dumps(catalog))
    files={p.name:sha256(p) for p in cache.iterdir()}
    (cache/'prepared-v3.json').write_text(json.dumps({'signature':{'train_sha256':'synthetic-only'},'files':files}))
    np.savez_compressed(old/'holdout-model.npz',version=np.array(1),
        w1=np.zeros((512,4104),'f4'),b1=np.zeros(512,'f4'),
        w2=np.zeros((2048,512),'f4'),b2=np.zeros(2048,'f4'))
    report=fit_experiment(cache,old,experiment,epochs=1,device='cpu',calibration_size=2,audit_size=2)
    assert report['training_query_overlap']==0
    # Every real alkane has a different exact mass, so all rankers tie. The
    # no-change baseline must win the predeclared deterministic tie-breaker.
    assert report['selection']['variant']=='v1_frozen'
    assert report['eligible_for_candidate_bundle'] is False
    assert refit_bundle(cache,old,experiment,tmp_path/'rejected')['status']=='not_promoted'
    assert not (tmp_path/'rejected').exists()
    assert fit_experiment(cache,old,experiment,epochs=1,device='cpu',calibration_size=2,audit_size=2)==report
    assert (experiment/'selection-before-audit.json').stat().st_mtime_ns <= (experiment/'validation-v3.json').stat().st_mtime_ns

import numpy as np
import pytest
from test_pipeline import row


def test_group_features_are_permutation_and_duplicate_invariant():
    from casmi26.learning import group_features
    from casmi26.spectra import from_row
    a=from_row(row('x',[(31,1),(29,.1)]));b=from_row(row('x',[(45,1),(29,.2)]))
    np.testing.assert_allclose(group_features([a,b]),group_features([b,a]))
    np.testing.assert_allclose(group_features([a,b,a,b]),group_features([a,b]))
    assert np.isfinite(group_features([a,b])).all()


def test_checkpoint_round_trip_and_real_optimization(tmp_path):
    torch=pytest.importorskip('torch')
    from casmi26.learning import train_arrays, FingerprintRanker, group_features, fingerprint
    from casmi26.spectra import from_row
    groups=[[from_row(row(str(i),[(31+i*4,1)]))] for i in range(4)]
    x=np.stack([group_features(g) for g in groups])
    smiles=['CCO','COC','CCN','CCC']
    y=np.stack([fingerprint(s) for s in smiles])
    path=tmp_path/'model.npz'
    report=train_arrays(x,y,path,epochs=15,batch_size=4,hidden=32,device='cpu',seed=13)
    assert report['final_loss'] < report['initial_loss']
    ranker=FingerprintRanker(path)
    p=ranker.posterior(groups[0])
    assert p.shape==(2048,)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    scores=ranker.scores(groups[0],smiles)
    assert set(scores)==set(smiles)


def test_checkpoint_rejects_shape_tampering(tmp_path):
    from casmi26.learning import FingerprintRanker
    p=tmp_path/'bad.npz';np.savez(p,version=np.array(1),w1=np.zeros((2,2)))
    with pytest.raises(ValueError): FingerprintRanker(p)

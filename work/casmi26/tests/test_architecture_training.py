import numpy as np
import pytest
from pathlib import Path

torch=pytest.importorskip('torch')


def test_acquisition_pooling_and_padding():
    from casmi26.architecture_training import pool_acquisitions
    v=np.zeros((2,3,8200),'f4');v[0,0]=1;v[0,1]=3;v[1,0]=9
    t=np.ones((2,3,32,16),'f4');m=np.array([[100,100.001,0],[200,0,0]])
    a,b,c=pool_acquisitions(v,t,m,np.array([2,1]),budget=3)
    assert a.dtype==np.float32
    assert np.allclose(a[:,0],[2,9]);assert c.tolist()==pytest.approx([100.0005,200])
    assert not b[1,32:].any()
    a,b,c=pool_acquisitions(v,t,m,np.array([2,1]),budget=1)
    assert np.allclose(a[:,0],[1,9]);assert not b[:,32:].any()


def test_score_factorization_matches_direct_bernoulli():
    from casmi26.architecture_training import fingerprint_basis,rank_from_basis
    z=np.array([2.,-1.,3.,-2.]);f=np.array([[1,0,1,0],[0,1,0,1]],'u1')
    b=fingerprint_basis(z,f,heads=(4,));assert b[0,0]>b[1,0]
    assert rank_from_basis(b,['A','B'],'A',[100,100],100,[1],5)==1
    assert rank_from_basis(b,['A','B'],'ABSENT',[100,100],100,[1],5)==0
    b2=fingerprint_basis(z*3,f,heads=(4,));np.testing.assert_allclose(b,b2)


def test_anchor_is_frozen_and_branch_updates(tmp_path):
    from casmi26.architecture_training import TorchAnchor,classification_loss
    from casmi26.architectures import make_model
    rng=np.random.default_rng(5);p=tmp_path/'anchor.npz'
    np.savez(p,format=3,feature_dim=8200,head_sizes=np.array([2048]),w1=rng.normal(0,.01,(4,8200)).astype('f4'),b1=np.zeros(4,'f4'),w2=np.zeros((2048,4),'f4'),b2=np.zeros(2048,'f4'))
    anchor=TorchAnchor(p);net=make_model('shallow_mlp',output_dim=2048)
    opt=torch.optim.AdamW(net.parameters(),lr=.001);x=torch.randn(2,8200);t=torch.randn(2,96,16)
    initial=[v.clone() for v in anchor.buffers()]
    y=torch.zeros(2,2048);y[:,0]=1
    loss=classification_loss(anchor(x)+net(x,t),y,torch.ones(2048),heads=(2048,))
    loss.backward();opt.step()
    assert net.head.weight.abs().sum()>0
    assert not list(anchor.parameters())
    for a,b in zip(initial,anchor.buffers()):torch.testing.assert_close(a,b)


def test_rank_missing_and_zero_sigma_validation():
    from casmi26.architecture_training import rank_from_basis
    assert rank_from_basis(np.empty((0,3)),[],'A',[],100,[.5,.25,.25],5)==0
    with pytest.raises(ValueError):rank_from_basis(np.ones((1,3)),['A'],'A',[100],100,[.5,.25,.25],0)


@pytest.mark.parametrize('loss_kind',['balanced','rank'])
def test_complete_training_trial_cpu_and_resume(tmp_path,loss_kind):
    from types import SimpleNamespace
    from casmi26.architecture_training import TorchAnchor,train_trial,pool_acquisitions
    from casmi26.architectures import load_model
    rng=np.random.default_rng(12);n=6
    views=rng.random((n,3,8200),dtype=np.float32).astype('f2')
    tokens=rng.random((n,3,32,16),dtype=np.float32)
    masses=np.full((n,3),100.)
    bits=np.packbits(rng.integers(0,2,(n,14336),dtype='u1'),axis=1)
    data=SimpleNamespace(views=views,targets=bits)
    def batch(ids,rng=None,budget=3):
        return pool_acquisitions(views[ids],tokens[ids],masses[ids],np.full(len(ids),3),budget=budget,rng=rng)
    data.batch=batch
    path=tmp_path/'anchor.npz'
    np.savez(path,format=3,feature_dim=8200,head_sizes=np.array([2048,4096,8192]),
             w1=rng.normal(0,.001,(4,8200)).astype('f4'),b1=np.zeros(4,'f4'),
             w2=np.zeros((14336,4),'f4'),b2=np.zeros(14336,'f4'))
    anchor=TorchAnchor(path);config={'architecture':'lowrank_linear','loss':loss_kind}
    neg=np.array([[(i+1)%n,(i+2)%n] for i in range(n)]) if loss_kind=='rank' else None
    p=tmp_path/'new.npz';r=train_trial(data,anchor,np.arange(n),config,p,epochs=1,batch_size=3,device='cpu',negatives=neg)
    assert r['steps']==2 and r['status']=='completed'
    assert r['export_max_abs_error']<1e-4
    assert np.isfinite(r['loss_history']).all()
    assert train_trial(data,anchor,np.arange(n),config,p,epochs=1,batch_size=3,device='cpu',negatives=neg)==r
    assert load_model(p).head.weight.abs().sum()>0

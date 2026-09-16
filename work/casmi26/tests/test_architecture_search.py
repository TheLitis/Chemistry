import json
import numpy as np
import pytest

torch = pytest.importorskip('torch')


def row(mz=(31.1, 45.2), intensity=(.7, .3)):
    return {'ms2_mzs':list(mz),'ms2_normalized_intensities':list(intensity),
            'precursor_mz':100.2,'adduct':'[M+H]+','collision_energy_ev':[20.]}


def test_tokens_ignore_order_scale_and_labels():
    from casmi26.architectures import peak_tokens, spectrum_digest
    a=row();b=row(tuple(reversed(a['ms2_mzs'])),tuple(v*100 for v in reversed(a['ms2_normalized_intensities'])))
    b['normalized_smiles']='SHOULD_NOT_BE_USED'
    np.testing.assert_allclose(peak_tokens(a),peak_tokens(b),rtol=1e-6,atol=1e-7)
    assert spectrum_digest(a)==spectrum_digest(b)
    assert peak_tokens(a).shape==(32,16)
    assert not peak_tokens(a)[2:].any()


def test_token_empty_after_precursor_filter_is_safe():
    from casmi26.architectures import peak_tokens
    assert not peak_tokens(row((100.2,), (1.,))).any()
    with pytest.raises(ValueError):peak_tokens(row((float('nan'),),(1.,)))


def test_fresh_split_no_prior_or_train_overlap():
    from casmi26.architectures import partition_keys
    keys=[f'k{i}' for i in range(40)]
    a,b,c=partition_keys(keys,{'k0'},5,5,10)
    assert not (set(a)|set(b)|set(c))&{'k0'}
    assert len(set(a+b+c))==20
    assert (a,b,c)==partition_keys(keys[::-1],{'k0'},5,5,10)


def test_model_registry_forward_finite_and_zero_delta():
    from casmi26.architectures import ARCHITECTURES, make_model
    x=torch.rand(2,8200);t=torch.rand(2,96,16);t[:,40:]=0
    for name in ARCHITECTURES:
        net=make_model(name,output_dim=24)
        y=net(x,t)
        assert y.shape==(2,24),name
        assert torch.isfinite(y).all(),name
        assert not y.any(),name
        (y.sum()).backward()
        assert net.head.weight.grad is not None,name


def test_set_models_permutation_invariance_after_nonzero_head():
    from casmi26.architectures import make_model
    torch.manual_seed(6);x=torch.rand(2,8200);t=torch.rand(2,96,16);t[:,30:]=0
    order=torch.randperm(96)
    for name in ('deep_sets','peak_transformer','latent_attention','peak_graph','massset_hybrid'):
        model=make_model(name,output_dim=8).eval()
        with torch.no_grad():model.head.weight.normal_(0,.1)
        a=model(x,t);b=model(x,t[:,order])
        torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5,msg=name)
        assert torch.isfinite(model(x,torch.zeros_like(t))).all()


def test_hard_negatives_exclude_same_key_and_nontraining():
    from casmi26.architectures import hard_negative_indices
    m=np.array([100.,100.,100.0001,150.,200.]);k=['A','A','B','C','D']
    neg=hard_negative_indices(m,k,[0,1,2,3],count=3)
    assert neg.shape==(4,3)
    for i,rr in zip([0,1,2,3],neg):
        assert all(j==-1 or (j in (0,1,2,3) and k[j]!=k[i]) for j in rr)
    assert (neg[3]==-1).all()


def test_rank_loss_discriminates_true_candidate_and_backpropagates():
    from casmi26.architectures import ranking_loss
    z=torch.tensor([[5.,-5.,3.,-3.]],requires_grad=True)
    pos=torch.tensor([[1.,0.,1.,0.]])
    neg=torch.tensor([[[0.,1.,0.,1.]]]);mask=torch.tensor([[True]])
    good=ranking_loss(z,pos,neg,mask,head_sizes=(4,))
    bad=ranking_loss(-z,pos,neg,mask,head_sizes=(4,))
    assert good<bad
    good.backward();assert torch.isfinite(z.grad).all()
    zero=ranking_loss(z,pos,neg,~mask,head_sizes=(4,))
    assert zero.item()==0


def test_checkpoint_roundtrip_numeric_only(tmp_path):
    from casmi26.architectures import make_model,save_model,load_model
    m=make_model('massset_hybrid',output_dim=12).eval()
    with torch.no_grad():m.head.weight.normal_(0,.01)
    p=tmp_path/'model.npz';save_model(p,m)
    n=load_model(p).eval();x=torch.rand(2,8200);t=torch.rand(2,96,16)
    torch.testing.assert_close(n(x,t),m(x,t))
    with np.load(p,allow_pickle=False) as z:assert all(v.dtype!=object for v in z.values())
    with pytest.raises(FileExistsError):save_model(p,m)


def test_metrics_include_missing_candidates():
    from casmi26.architectures import rank_metrics
    m=rank_metrics([1,2,0])
    assert m['mrr_at_25']==pytest.approx(.5)
    assert m['recall_at_25']==pytest.approx(2/3)

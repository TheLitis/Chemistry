import csv
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def fixture(tmp_path):
    spec=importlib.util.spec_from_file_location('r07_fixture',Path(__file__).with_name('test_r07_release.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.fixture(tmp_path)


def test_noop_extension_preserves_entire_r07_csv_and_raw_candidates(tmp_path):
    from casmi26.r07_release import infer
    bundle,train,test,_=fixture(tmp_path);seen=[]
    def ranker(cid,rows,queries,scores):
        assert all('normalized_smiles' not in q for q in queries)
        assert len(scores)==len(rows)
        seen.append((cid,len(queries)))
        return {'top_indices':np.argsort(-scores,kind='stable')[:25].tolist(), 'metadata':{'fixed':True}}
    base=infer(test,train,bundle,tmp_path/'base.csv',workers=1)
    ext=infer(test,train,bundle,tmp_path/'ext.csv',workers=1,ranking_adapter=ranker)
    assert (tmp_path/'base.csv').read_bytes()==(tmp_path/'ext.csv').read_bytes()
    assert base['submission_sha256']==ext['submission_sha256']
    assert sorted(seen)==[('000',2),('001',1)]
    assert ext['details']['000']['ranking_extension']=={'fixed':True}


@pytest.mark.parametrize('bad',[[0,0],[-1,0],[100,0],[0.0,1.0],[],[0]])
def test_invalid_rank_extension_cannot_emit_partial_submission(tmp_path,bad):
    from casmi26.r07_release import infer
    bundle,train,test,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='ranking'):
        infer(test,train,bundle,tmp_path/'bad.csv',workers=1,
              ranking_adapter=lambda *args:{'top_indices':bad,'metadata':{}})
    assert not (tmp_path/'bad.csv').exists()


def test_forward_extension_is_exact_and_uses_all_supported_raw_spectra():
    from casmi26.forward_release import ForwardRanker
    class Model:
        def __init__(self):self.calls=[]
        def predict(self,smiles,modes,energies):
            self.calls.append((smiles,modes,energies))
            mz=30.0 if smiles=='CCO' else 45.0
            return {(m,float(e)):np.array([[mz,1.]]) for m in modes for e in energies}
    q={'adduct':'[M+H]+','precursor':100.,'ce':[20.], 'peaks':[[30.,1.]]}
    raw={'x':[dict(q),dict(q)]};model=Model();adapter=ForwardRanker(raw,model=model)
    rows=[['a','CCO','KEY_A',46.],['b','COC','KEY_B',46.]]
    result=adapter('x',rows,[],np.array([0.,.1]))
    assert result['top_indices']==[0,1]
    assert result['metadata']['certified'] is True
    assert result['metadata']['raw_spectra']==2
    assert len(model.calls)==2
    adapter('x',rows,[],np.array([0.,.1]))
    assert len(model.calls)==2


def test_unsupported_adduct_keeps_r07_order_without_model_call():
    from casmi26.forward_release import ForwardRanker
    class Model:
        def predict(self,*args):raise AssertionError('Unsupported adduct called model')
    raw={'x':[{'adduct':'[M+Na]+','precursor':100.,'ce':[20.], 'peaks':[[30.,1.]]}]}
    adapter=ForwardRanker(raw,model=Model())
    result=adapter('x',[['a','CCO','K1',46.],['b','COC','K2',46.]],[],np.array([0.,.1]))
    assert result['top_indices']==[1,0]
    assert result['metadata']['evaluations']==0
    assert result['metadata']['supported_spectra']==0


def test_unknown_compound_and_extra_test_labels_are_not_silently_used(tmp_path):
    from casmi26.forward_release import read_raw_queries,ForwardRanker
    _,_,test,_=fixture(tmp_path)
    groups=read_raw_queries(test)
    assert len(groups['000'])==2 and len(groups)==2
    assert all(set(q)=={'adduct','precursor','ce','peaks'} for qs in groups.values() for q in qs)
    with pytest.raises(ValueError,match='query'):
        ForwardRanker(groups,model=object())('not-an-id',[],[],np.array([]))

"""Selected-feature integration must preserve the frozen scorer numerically."""
import numpy as np
import pytest
from casmi26.forward_release import ForwardRanker


class NumericModel:
    device = 'cpu'

    def predict(self, smiles, modes, energies):
        shift = float(smiles.split('-')[-1]) * .0001
        return {(mode, energy): np.array([[30. + shift, .6], [60. + shift, .4]])
                for mode in modes for energy in energies}


def example():
    queries = {'q': [dict(adduct='[M+H]+', precursor=180., ce=[10.,40.],
                           peaks=np.array([[30.,.7],[60.,.3]])),
                     dict(adduct='[M-H]-', precursor=178., ce=None,
                          peaks=np.array([[30.,.2],[60.,.8]]))]}
    rows = [['unused', 'candidate-'+str(i), 'key-'+str(i), 179.] for i in range(90)]
    scores = np.linspace(1., -.5, len(rows))
    return queries, rows, scores


def test_selected_feature_backend_preserves_full_top25_and_certificate():
    q, rows, scores = example()
    old = ForwardRanker(q, model=NumericModel())
    new = ForwardRanker(q, model=NumericModel(), nearest_only=True)
    assert old('q',rows,[],scores) == new('q',rows,[],scores)
    assert old.statistics == new.statistics


def test_fast_backend_reuses_same_prediction_cache(tmp_path):
    pytest.importorskip("torch")
    q,rows,scores = example();path=tmp_path/'forward.sqlite'
    old=ForwardRanker(q,model=NumericModel(),cache=path)
    expected=old('q',rows,[],scores);old.close()
    new=ForwardRanker(q,model=NumericModel(),cache=path,nearest_only=True)
    try:
        assert new('q',rows,[],scores)==expected
        assert new.statistics['graph_calls']==0
        assert new.statistics['disk_cache_hits']>0
    finally:
        new.close()


def test_unsupported_observations_preserve_original_ranking():
    q,rows,scores=example()
    for row in q['q']:row['adduct']='[M+Na]+'
    new=ForwardRanker(q,model=NumericModel(),nearest_only=True)
    result=new('q',rows,[],scores)
    assert result['top_indices']==list(range(25))
    assert result['metadata']['evaluations']==0


@pytest.mark.parametrize('value',[None,1,'yes',[]])
def test_backend_switch_rejects_ambiguous_values(value):
    with pytest.raises(ValueError,match='boolean'):
        ForwardRanker({},model=NumericModel(),nearest_only=value)

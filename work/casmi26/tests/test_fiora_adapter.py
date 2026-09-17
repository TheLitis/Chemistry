from pathlib import Path
import os
import numpy as np
import pytest


def source_model():
    path=os.environ.get('FIORA_TEST_MODEL')
    if not path:pytest.skip('Pinned FIORA state is not provisioned for this CI environment')
    return Path(path)


def test_cached_graph_predictions_match_author_full_calls():
    path=source_model()
    from casmi26.fiora_adapter import ForwardModel
    model=ForwardModel(path)
    for smi in ('CC(=O)OC1=CC=CC=C1C(=O)O','Cn1c(=O)c2c(ncn2C)n(C)c1=O'):
        cached=model.predict(smi,['[M+H]+','[M-H]-'],[10.,40.],cache_graph=True)
        full=model.predict(smi,['[M+H]+','[M-H]-'],[10.,40.],cache_graph=False)
        for key in cached:
            np.testing.assert_allclose(cached[key],full[key],rtol=2e-6,atol=2e-7)


def test_unsupported_modes_are_rejected_not_remapped():
    path=source_model()
    from casmi26.fiora_adapter import ForwardModel
    model=ForwardModel(path)
    with pytest.raises(ValueError,match='adduct'):model.predict('CCO',['[M+Na]+'],[20.])
    with pytest.raises(ValueError,match='energy'):model.predict('CCO',['[M+H]+'],[float('nan')])


def test_pinned_weights_required(tmp_path):
    from casmi26.fiora_adapter import verify_model_files
    p=tmp_path/'fiora_OS_v1.0.0.pt'
    p.with_name('fiora_OS_v1.0.0_state.pt').write_bytes(b'untrusted')
    p.with_name('fiora_OS_v1.0.0_params.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):verify_model_files(p)

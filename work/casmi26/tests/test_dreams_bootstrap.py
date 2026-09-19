from pathlib import Path
import importlib.util
import numpy as np
import pytest


def m():
    p=Path(__file__).resolve().parents[3]/'tasks/casmi_dreams_bootstrap.py'
    if not p.exists():pytest.fail('DreaMS bootstrap not implemented')
    spec=importlib.util.spec_from_file_location('db',p);x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x);return x


def test_exact_mass_preserved_and_intensities_normalized():
    a=m().pack_spectrum(np.array([[20.125,2.],[30.25,4.]]),100.5,3)
    np.testing.assert_allclose(a,[[100.5,1.1],[20.125,.5],[30.25,1.],[0.,0.]])
    assert a.dtype==np.float32


def test_select_strongest_preserving_input_order():
    x=m().pack_spectrum([[10,1],[20,4],[30,2],[40,3]],100,2)
    np.testing.assert_allclose(x,[[100,1.1],[20,1],[40,.75]])


@pytest.mark.parametrize('peaks,precursor',[([[20,-1]],100),([[20,0]],100),([[float('nan'),1]],100),([[20,1]],0)])
def test_invalid_input_rejected(peaks,precursor):
    with pytest.raises(ValueError):m().pack_spectrum(peaks,precursor,100)


def test_publisher_identity_required():
    d={'sha':'a'*40,'cardData':{'license':'mit'},'siblings':[{'rfilename':'DreaMS_embedding_model_torchscript.pt','size':4,'lfs':{'sha256':'b'*64,'size':4}}]}
    assert m().select_asset(d)['sha256']=='b'*64
    d['cardData']['license']='cc-by-nc-4.0'
    with pytest.raises(ValueError):m().select_asset(d)


def test_no_credential_redirect_hosts():
    for x in ['https://evil.com/model.pt','https://huggingface.co.evil.org/a','http://huggingface.co/a']:
        with pytest.raises(ValueError):m().validate_url(x)

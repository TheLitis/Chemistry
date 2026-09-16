import importlib.util
from pathlib import Path
import ssl
import pytest


def module():
    path=Path(__file__).resolve().parents[3]/'tasks/public_tls.py'
    assert path.exists(), 'Verified public TLS helper is absent'
    spec=importlib.util.spec_from_file_location('public_tls_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_ca_bundle_keeps_chain_and_hostname_checks():
    context=module().verified_context()
    assert context.verify_mode==ssl.CERT_REQUIRED
    assert context.check_hostname
    assert len(context.get_ca_certs())>0


def test_nonexistent_ca_file_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):module().verified_context(tmp_path/'absent.pem')

"""TLS contexts for public downloads in the isolated embedded Python runtime."""
from __future__ import annotations
from pathlib import Path
import ssl


def verified_context(cafile: str | Path | None = None) -> ssl.SSLContext:
    if cafile is None:
        import certifi
        cafile = certifi.where()
    context = ssl.create_default_context(cafile=str(cafile))
    if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
        raise RuntimeError('Public download requires certificate and hostname verification')
    return context

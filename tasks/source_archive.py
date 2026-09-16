"""Archive the explicitly trusted checkout without persisting Git trust changes."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile


def archive_source(repository: Path, destination: Path) -> dict:
    repository=Path(repository).resolve();destination=Path(destination).resolve()
    if not (repository/'.git').exists():raise ValueError('Expected an explicit Git checkout')
    if destination==repository or repository in destination.parents:
        raise ValueError('Source archive must be outside the checkout')
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='source-',suffix='.zip',dir=destination.parent);os.close(fd)
    command=['git','-c','safe.directory='+repository.as_posix(),'-C',str(repository),
             'archive','--format=zip','--output='+name,'HEAD']
    try:
        # One exact checkout, one command. No global configuration or ACL writes.
        subprocess.run(command,check=True,stdin=subprocess.DEVNULL,capture_output=True,timeout=60)
        with zipfile.ZipFile(name) as archive:
            if archive.testzip() is not None:raise ValueError('Corrupt source archive')
            members=len(archive.infolist())
        digest=hashlib.sha256(Path(name).read_bytes()).hexdigest()
        os.replace(name,destination)
    finally:
        if os.path.exists(name):os.unlink(name)
    return {'configuration_scope':'command','trusted_checkout':str(repository),'archive_members':members,
            'sha256':digest,'global_config_changed':False,'acl_changed':False}

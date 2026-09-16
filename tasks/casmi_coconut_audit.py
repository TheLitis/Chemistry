"""Acquire the fixed August-2026 public COCONUT snapshot, not test answers.

No credentials, account changes, Kaggle writes, or production model changes.
Only this named source and the project external-data directory are used.
"""
from __future__ import annotations
from collections import Counter
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

URL='https://coconut.s3.uni-jena.de/prod/downloads/2026-08/coconut_csv_lite-08-2026.zip'
SOURCE_PAGE='https://coconut.naturalproducts.net/download'


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1'})
    import pyarrow.parquet as pq
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    root=state/'data/external/coconut-2026-08';root.mkdir(parents=True,exist_ok=True)
    output=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    archive=root/'coconut_csv_lite-08-2026.zip';start=time.monotonic()
    report={'experiment':'R04-public-catalog-audit','source_url':URL,'source_page':SOURCE_PAGE,
            'release':'2026-08','license_on_download_page':'CC0; individual source provenance retained',
            'license_caveat':'Upstream source license question #767 has no maintainer clarification; do not relicense upstream annotations.',
            'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'commit':os.environ.get('GITHUB_SHA'),
            'test_data_read':False,'production_changed':False,'submission_made':False,'official_score':None}
    if not archive.exists():
        if shutil.disk_usage(root).free<3*1024**3:raise RuntimeError('Insufficient free space for external snapshot')
        tmp=archive.with_suffix('.part')
        try:
            with urllib.request.urlopen(urllib.request.Request(URL,headers={'User-Agent':'CASMI26-research/1.0'}),timeout=60) as inp,tmp.open('wb') as out:
                size=0
                for block in iter(lambda:inp.read(4*1024*1024),b''):
                    size+=len(block)
                    if size>1024**3:raise RuntimeError('Public snapshot exceeded 1 GiB safety bound')
                    out.write(block)
            with zipfile.ZipFile(tmp) as z:
                bad=z.testzip()
                if bad:raise RuntimeError('Snapshot CRC failure')
            os.replace(tmp,archive)
        finally:
            if tmp.exists():tmp.unlink()
    report.update(archive_bytes=archive.stat().st_size,archive_sha256=sha256(archive))
    csv.field_size_limit(64*1024*1024);tables=[]
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not member.filename.lower().endswith('.csv'):continue
            if member.file_size>5*1024**3:raise RuntimeError('CSV exceeds 5 GiB safety bound')
            with io.TextIOWrapper(z.open(member),encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream);examples=[];n=0;missing=Counter()
                for row in reader:
                    n+=1
                    if len(examples)<2:examples.append({k:str(v)[:300] for k,v in row.items() if v is not None})
                    for name in reader.fieldnames:
                        if not row.get(name):missing[name]+=1
                tables.append({'member':member.filename,'rows':n,'uncompressed_bytes':member.file_size,
                               'columns':reader.fieldnames,'missing':dict(missing),'examples':examples})
    report['tables']=tables
    catalog=json.loads((state/'cache/casmi26/official-v1/catalog.json').read_text())
    lookup={r[0]:r[2] for r in catalog};sources=Counter();source_keys={}
    data=state/'data/external/enveda-CASMI26-molecule-id-mass-spectra/train.parquet'
    if data.exists():
        with pq.ParquetFile(data) as table:
            for batch in table.iter_batches(batch_size=65536,columns=['ingest_lib','normalized_smiles','instrument_type']):
                libs=batch.column('ingest_lib').to_pylist();smiles=batch.column('normalized_smiles').to_pylist()
                sources.update(str(v) for v in libs)
                for lib,smi in zip(libs,smiles):
                    if 'np-example' in str(lib).lower():source_keys.setdefault(str(lib),set()).add(lookup.get(smi))
        report['official_train_sources']=dict(sources)
        report['np_example_structure_counts']={k:len(v-{None}) for k,v in source_keys.items()}
    report.update(status='snapshot_downloaded_and_schema_inspected',seconds=time.monotonic()-start,
                  next_step='Validate schema/mass fields before candidate expansion; no retrieval improvement is claimed by download alone.')
    dest=state/'artifacts/casmi26/research-r04';dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/'snapshot-audit.json',report);write_json(output/'snapshot-audit.json',report)
    print('COCONUT_AUDIT_BEGIN\n'+json.dumps(report,indent=2)+'\nCOCONUT_AUDIT_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

"""Import the user-provided competition archive; inspect schemas without guessing.

Only the named archive, CASMI state, and existing Kaggle provider configuration
are used. Never accepts rules, changes ACLs, reads browser cookies, or submits.
"""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
import zlib


def checksum(path):
    sha=hashlib.sha256();crc=0
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            sha.update(block);crc=zlib.crc32(block,crc)
    return sha.hexdigest(),crc


def import_archive(source: Path, destination: Path) -> list[dict]:
    destination=Path(destination).resolve(); source=Path(source)
    with zipfile.ZipFile(source) as archive:
        planned=[];seen=set();needed=0
        for item in archive.infolist():
            name=item.filename.replace('\\','/');parts=PurePosixPath(name)
            mode=item.external_attr>>16
            if (parts.is_absolute() or ':' in name or '..' in parts.parts or
                any(p.endswith((' ','.')) for p in parts.parts) or stat.S_ISLNK(mode)):
                raise ValueError('Unsafe archive path: '+name)
            if item.is_dir():continue
            if name.lower() in seen:raise ValueError('Duplicate archive member: '+name)
            seen.add(name.lower());target=destination.joinpath(*parts.parts)
            if destination not in target.resolve().parents:raise ValueError('Path escapes destination')
            if item.flag_bits & 1:raise ValueError('Encrypted archive member is unsupported')
            if target.exists():
                if target.is_symlink() or target.stat().st_size!=item.file_size or checksum(target)[1]!=item.CRC:
                    raise ValueError('Refusing to overwrite different existing data: '+str(target))
            else:needed+=item.file_size
            planned.append((item,target))
        ancestor=destination
        while not ancestor.exists():ancestor=ancestor.parent
        if needed>shutil.disk_usage(ancestor).free-1024**3:raise ValueError('Insufficient space for archive')
        result=[]
        for item,target in planned:
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                fd,name=tempfile.mkstemp(dir=target.parent,suffix='.part')
                try:
                    with os.fdopen(fd,'wb') as out,archive.open(item) as inp:
                        shutil.copyfileobj(inp,out,4*1024*1024)
                    if Path(name).stat().st_size!=item.file_size or checksum(name)[1]!=item.CRC:
                        raise ValueError('CRC/size mismatch: '+item.filename)
                    os.replace(name,target)
                finally:
                    if os.path.exists(name):os.unlink(name)
            result.append({'path':str(target),'bytes':item.file_size,'sha256':checksum(target)[0]})
    return result


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    repo=Path(__file__).resolve().parents[1]
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        if not python.exists():raise RuntimeError('Prepared CASMI Python is absent')
        return subprocess.call([str(python),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1'})
    import pyarrow.parquet as pq
    source=Path(r'C:\Users\loval\Downloads\enveda-CASMI26-molecule-id-mass-spectra.zip')
    target=state/'data/casmi26/official-v1'
    report={'request_id':os.environ.get('CHEMISTRY_REQUEST_ID'),'archive':str(source),'files':[],
            'official_score':None,'submission_made':False}
    print('IMPORT_START '+str(source),flush=True)
    report['files']=import_archive(source,target)
    report['tables']=[]
    for file in report['files']:
        path=Path(file['path'])
        if path.suffix.lower()=='.parquet':
            with pq.ParquetFile(path) as table:
                schema=table.schema_arrow
                item={'file':str(path),'rows':table.metadata.num_rows,
                      'schema':{field.name:str(field.type) for field in schema},'row_groups':table.num_row_groups}
                names=schema.names
                inspected=[k for k in ('molecule_id','compound_id','spectrum_id','adduct','ion_mode','instrument_type','source') if k in names]
                counters={k:Counter() for k in inspected}
                for batch in table.iter_batches(batch_size=65536,columns=inspected):
                    for key in inspected:counters[key].update(str(v) for v in batch.column(key).to_pylist())
                item['summary']={k:{'unique':len(v),'most_common':v.most_common(12)} for k,v in counters.items()}
                sample=next(table.iter_batches(batch_size=1)).to_pylist()[0] if item['rows'] else {}
                item['example']={k:({'array_length':len(v),'first_values':v[:3]} if isinstance(v,list) else v)
                                 for k,v in sample.items() if 'smiles' not in k.lower() and 'inchi' not in k.lower()}
                report['tables'].append(item)
        elif path.name=='sample_submission.csv':
            with path.open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            report['template']={'file':str(path),'rows':len(rows),'header':list(rows[0]) if rows else [],'examples':rows[:3]}
    prepare=load_module('casmi_prepare',repo/'tasks/casmi_prepare.py')
    access=load_module('casmi_access_check',repo/'tasks/casmi_access_check.py')
    candidates,warnings=access.windows_candidates(Path(r'C:\Users\loval'))
    env,kind=access.select_environment(dict(os.environ),candidates)
    env=prepare.kaggle_environment(state,env);env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
    result=prepare.run([str(python),'-c','from kaggle.cli import main;main()','competitions','files',prepare.SLUG,
                        '--page-size','200','-v'],env=env,timeout=60,log=out/'access.log',expose=False)
    report['kaggle']={'file_listing_exit_code':result['exit_code'],'configuration_kind':kind,
                      'cli_on_service_path':shutil.which('kaggle'),'credential_values_logged':False,
                      'warnings':sorted(set(warnings))}
    report['prior_experiments']=[]
    for path in (state/'artifacts/casmi26').glob('**/*.validation.json'):
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        report['prior_experiments'].append({'path':str(path),**{k:v for k,v in data.items()
                                  if k in ('status','counts','validation','mass_only_baseline','ambiguous_mass_validation',
                                           'ambiguous_mass_baseline','training_validation_key_overlap')}})
    artifact=state/'artifacts/casmi26';artifact.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    text=json.dumps(report,ensure_ascii=True,indent=2,default=str)
    (out/'official-import.json').write_text(text+'\n',encoding='utf-8')
    (artifact/'official-import.json').write_text(text+'\n',encoding='utf-8')
    print('CASMI_IMPORT_REPORT_BEGIN\n'+text+'\nCASMI_IMPORT_REPORT_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

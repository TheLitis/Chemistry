"""Execute the delivered R06 archive, not the repository imports. No Kaggle writes."""
from __future__ import annotations
import datetime as dt
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


def validate_members(archive,manifest):
    mandatory={'bundle/'+name for name in manifest['files']}|{'bundle/r06-bundle.json','predict_r06.py','README.txt','runtime.json'}
    names=set()
    for member in archive.infolist():
        name=member.filename;p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or stat.S_ISLNK(member.external_attr>>16):
            raise ValueError('Unsafe archive path')
        source=p.parent==PurePosixPath('work/casmi26/casmi26') and p.suffix=='.py'
        if name in names or (name not in mandatory and not source):raise ValueError('Unexpected archive member')
        names.add(name)
    if not mandatory<=names or 'work/casmi26/casmi26/__init__.py' not in names:
        raise ValueError('Incomplete archive')
    return names


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve())],env={**os.environ,
            'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    from casmi26.r06_candidate import verify_bundle
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    root=state/'artifacts/casmi26/research-r06/release'
    release=json.loads((root/'release.json').read_text());manifest=verify_bundle(root/'bundle')
    archive=root/'r06-research-candidate.zip'
    if sha256(archive)!=release['archive']['sha256']:raise ValueError('Release ZIP changed')
    tests=subprocess.run([str(python),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,
                        text=True,encoding='utf-8',errors='replace',timeout=300)
    (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8');print(tests.stdout,flush=True)
    if tests.returncode:raise RuntimeError('Tests failed')
    data=load('staged',repo/'tasks/casmi_staged.py').find_dataset(state/'data/external')
    with zipfile.ZipFile(archive) as z:
        names=validate_members(z,manifest)
        if sum(i.file_size for i in z.infolist())+512*1024**2>shutil.disk_usage(root).free:
            raise RuntimeError('Insufficient free space for standalone archive acceptance')
        with tempfile.TemporaryDirectory(prefix='archive-acceptance-',dir=root) as tmp:
            runtime=Path(tmp);z.extractall(runtime)
            extracted=verify_bundle(runtime/'bundle')
            if extracted!=manifest:raise ValueError('Extracted manifest changed')
            command=[str(python),str(runtime/'predict_r06.py'),'--test',str(data/'test.parquet'),
                '--bundle',str(runtime/'bundle'),'--output',str(out/'submission.csv'),
                '--sample-submission',str(data/'sample_submission.csv'),'--budget','all','--device','cpu']
            env=dict(os.environ);env.pop('PYTHONPATH',None)
            result=subprocess.run(command,cwd=runtime,env=env,stdin=subprocess.DEVNULL,capture_output=True,
                text=True,encoding='utf-8',errors='replace',timeout=900)
            (out/'archive-cli.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
            if result.returncode:raise RuntimeError('Standalone archive CLI failed')
            output=json.loads((out/'submission.csv.report.json').read_text())
            expected=release['outputs']['all']
            for key in ('test_sha256','prediction_count','test_spectra','submission_sha256','bundle_manifest_sha256','selection'):
                if output[key]!=expected[key]:raise ValueError('Standalone CLI differs: '+key)
            if sha256(out/'submission.csv')!=expected['submission_sha256']:raise ValueError('Standalone CSV hash mismatch')
    if sha256(state/'artifacts/casmi26/official-v1/model.npz')!='4d79a6d4f24060cd0f7c09bebefbced2c9e1672cb1cdf5f7831cd5a19c03b55d':
        raise ValueError('Original champion changed')
    source=load('source_archive',repo/'tasks/source_archive.py').archive_source(repo,out/'source.zip')
    report={'status':'completed','commit':os.environ.get('GITHUB_SHA'),'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'identity':subprocess.run(['whoami'],capture_output=True,text=True,check=True,timeout=15).stdout.strip(),
        'archive_sha256':release['archive']['sha256'],'archive_members':len(names),'source_archive':source,
        'standalone_extracted_cli_exit_code':result.returncode,'output_matches_repository_prediction':True,
        'prediction_count':output['prediction_count'],'spectra_used':sum(x['spectra_used'] for x in output['details'].values()),
        'submission_sha256':output['submission_sha256'],'original_champion_unchanged':True,
        'tests':tests.stdout.strip(),'new_training':False,'new_submissions':0,'new_kaggle_uploads':0,'official_score':None,
        'archive_contains_offline_dependency_wheels':False,'runtime':'existing isolated CASMI Python; source and weights from the extracted archive'}
    write_json(root/'archive-acceptance.json',report);write_json(out/'archive-acceptance.json',report)
    print('R06_ARCHIVE_ACCEPTANCE_BEGIN\n'+json.dumps(report,indent=2)+'\nR06_ARCHIVE_ACCEPTANCE_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

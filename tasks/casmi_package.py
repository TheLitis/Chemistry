"""Assemble an offline inference bundle without logging in to or submitting to Kaggle."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED


def download_arguments(folder: Path) -> list[str]:
    return ['-m','pip','download','--disable-pip-version-check','--dest',str(folder),
            '--platform','manylinux_2_28_x86_64','--platform','manylinux_2_17_x86_64',
            '--platform','manylinux2014_x86_64','--implementation','cp','--python-version','312',
            '--abi','cp312','--only-binary=:all:',
            'numpy==2.3.5','rdkit==2026.3.3','pyarrow==21.0.0']


def main() -> int:
    repo=Path(__file__).resolve().parents[1]
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('prepare',repo/'tasks/casmi_prepare.py')
    prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    python=state/'envs/casmi26/python.exe'
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8')
    def execute(args,label,timeout=900):
        result=prepare.run([str(python)]+args,env=env,timeout=timeout,log=out/(label+'.log'))
        if result['exit_code']:raise RuntimeError(label+' failed')
    assets=state/'artifacts/casmi26/public-v03'
    progress=json.loads((assets/'progress-report.json').read_text(encoding='utf-8'))
    if progress.get('request_id') != os.environ.get('CHEMISTRY_REQUEST_ID') or progress.get('status') != 'external_baseline_completed_no_official_score':
        raise RuntimeError('Refusing to package stale or unsuccessful experiment output')
    model=assets/'fingerprint-public.npz'
    validation=json.loads(model.with_suffix('.validation.json').read_text(encoding='utf-8'))
    if validation['checkpoint']['sha256'] != hashlib.sha256(model.read_bytes()).hexdigest():
        raise RuntimeError('Model differs from its validation evidence')
    wheels=assets/'offline-wheels-py312';wheels.mkdir(parents=True,exist_ok=True)
    execute(download_arguments(wheels),'download-linux-wheels')
    resolve=download_arguments(wheels)
    resolve += ['--no-index','--find-links',str(wheels)]
    execute(resolve,'offline-resolution')
    execute(['-m','pytest','-q',str(repo/'work/casmi26/tests')],'tests',240)
    notebook=assets/'casmi26-submission.ipynb'
    execute(['-c',f'import sys; sys.path.insert(0,{str(repo/"work/casmi26")!r}); from pathlib import Path; from casmi26.notebook import build_notebook; build_notebook(Path({str(notebook)!r}))'],'notebook',120)
    code=f'''import sys, tempfile, csv
from pathlib import Path
sys.path[:0]=[{str(repo/'work/casmi26')!r},{str(repo/'work/casmi26/tests')!r}]
from test_pipeline import fixture_data
from casmi26.engine import Config, predict
from casmi26.metric import require_official_rdkit
require_official_rdkit()
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);train,test,template=fixture_data(root)
    template.write_text('molecule_id,smiles\\n001,C\\n002,C\\n')
    report=predict(test,train,template,root/'submission.csv',config=Config(top_k=25),model=Path({str(model)!r}))
    assert report['prediction_count']==2 and report['fingerprint_model'] is not None
    assert report['official_score'] is None and report['metric_rdkit_matches']
    with (root/'submission.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert all(1<=len(r['smiles'].split(';'))<=25 for r in rows)
print('TRAINED_MODEL_PIPELINE_SMOKE_PASSED')
'''
    execute(['-c',code],'trained-inference',120)
    files=[model,model.with_suffix('.catalog.csv'),model.with_suffix('.validation.json'),notebook]
    files += sorted(wheels.glob('*.whl'))
    manifest={'official_score':None,'kaggle_execution_verified':False,
              'wheel_target':'CPython 3.12 Linux x86_64; Kaggle docker-python Dockerfile.tmpl package path',
              'offline_dependency_resolution':True,'trained_model_pipeline_smoke':True,
              'files':[{'name':str(p.relative_to(assets)).replace('\\','/'),'bytes':p.stat().st_size,
                        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    manifest_path=assets/'ARTIFACT_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');files.append(manifest_path)
    readme=assets/'OFFLINE_README.txt'
    readme.write_text(
        'Import casmi26-submission.ipynb into a Kaggle notebook. Attach the official competition data.\n'
        'Attach this bundle as a private dataset, keeping the model/catalog and wheel files.\n'
        'Disable Internet in notebook settings. The wheels target Linux CPython 3.12.\n'
        'The notebook uses the current test.parquet mount at run time, not fixed IDs.\n'
        'No official Kaggle execution or hidden score has been obtained.\n'
        'The model is an external-data fingerprint ranker, not a de novo molecular decoder.\n'
        'The validation catalog contains the true structures; this is not unseen-structure discovery.\n'
        'A missing mass-compatible structural candidate currently stops inference.\n',encoding='utf-8');files.append(readme)
    bundle=assets/'casmi26-offline-bundle.zip';temp=bundle.with_suffix('.tmp')
    with ZipFile(temp,'w',ZIP_DEFLATED,compresslevel=3) as archive:
        for path in files:archive.write(path,str(path.relative_to(assets)).replace('\\','/'))
    os.replace(temp,bundle)
    shutil.copy2(bundle,out/bundle.name)
    shutil.copy2(notebook,out/notebook.name)
    shutil.copy2(manifest_path,out/manifest_path.name)
    report={'status':'offline_bundle_created_not_kaggle_validated','bundle':str(bundle),
            'bytes':bundle.stat().st_size,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),
            'official_score':None,'files':len(files)}
    (out/'package-result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('CASMI_PACKAGE_RESULT '+json.dumps(report),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

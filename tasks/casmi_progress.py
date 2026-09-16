"""Verify CASMI26 code on ChemistryPC and train a public-data ranking baseline.

All downloads are public scientific data/packages. No account cookies are read,
no rules are accepted, and no Kaggle submission is made. Credentials, if already
provisioned, are used only for a sanitized access check. Generated evidence is
kept separate from any official score.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import urllib.request

SOURCE = 'https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/965c90b8d674c990e14535e09f0dc6e4e7ca9d85/data/MassSpecGym.tsv'
MAX_ROWS = 30000
MAX_BYTES = 100*1024*1024


def public_subset(folder: Path) -> tuple[Path,dict]:
    folder.mkdir(parents=True,exist_ok=True)
    raw=folder/'MassSpecGym-prefix.tsv'
    # Always record a bounded, complete-line prefix, not a supposed full dataset.
    if not raw.exists():
        temporary=raw.with_suffix('.part');count=0;size=0
        request=urllib.request.Request(SOURCE,headers={'User-Agent':'Chemistry-CASMI26-public-benchmark/0.2'})
        with urllib.request.urlopen(request,timeout=90) as response,temporary.open('wb') as f:
            for line in response:
                if size+len(line)>MAX_BYTES:break
                f.write(line);size+=len(line);count+=1
                if count>=MAX_ROWS+1:break
        if count<101:raise RuntimeError('Public corpus returned too few complete TSV rows')
        os.replace(temporary,raw)
    path=folder/'public-spectra.jsonl';rows=0
    with raw.open(encoding='utf-8-sig',newline='') as f,path.open('w',encoding='utf-8') as out:
        reader=csv.DictReader(f,delimiter='\t')
        required={'mzs','intensities','smiles','precursor_mz','adduct','inchikey'}
        if not required.issubset(reader.fieldnames or []):raise RuntimeError('Public TSV schema differs from its dataset card')
        for record in reader:
            item={'molecule_id':record['inchikey'],'spectrum_id':record.get('identifier'),
                  'ms2_mzs':[float(v) for v in record['mzs'].split(',')],
                  'ms2_normalized_intensities':[float(v) for v in record['intensities'].split(',')],
                  'normalized_smiles':record['smiles'],'precursor_mz':float(record['precursor_mz']),
                  'adduct':record['adduct'],'collision_energy_ev':[],
                  'collision_energy_orig':record.get('collision_energy',''),
                  'collision_energy_orig_units':'unknown',
                  'instrument_type':record.get('instrument_type')}
            out.write(json.dumps(item,allow_nan=False)+'\n');rows+=1
    evidence={'url':SOURCE,'license':'MIT (publisher dataset card)',
              'scope':'bounded first-row subset, not the full corpus or official benchmark split',
              'rows':rows,'raw_bytes':raw.stat().st_size,
              'prefix_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
    (folder/'source.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    return path,evidence


def main() -> int:
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('prepare',repo/'tasks/casmi_prepare.py')
    prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    out.mkdir(parents=True,exist_ok=True)
    env=prepare.kaggle_environment(state,os.environ);env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
    python=state/'envs/casmi26/python.exe'
    if not python.exists():raise RuntimeError('Run the existing CASMI environment preparation first')
    site=python.parent/'Lib/site-packages'
    report={'request_id':os.environ.get('CHEMISTRY_REQUEST_ID'),'official_score':None,
            'official_submission':False,'status':'running','python':str(python)}
    def execute(args,label,timeout=900):
        result=prepare.run([str(python)]+args,env=env,timeout=timeout,log=out/(label+'.log'))
        if result['exit_code']!=0:raise RuntimeError(label+' failed, exit='+str(result['exit_code']))
        return result
    # Update the isolated scientific runtime only, never the user's base Python/Torch.
    execute(['-m','pip','install','--disable-pip-version-check','--target',str(site),
             '--upgrade','--no-deps','rdkit==2026.3.3'],'rdkit-install')
    execute(['-c','import rdkit, numpy, pyarrow, torch; print(rdkit.__version__, numpy.__version__, pyarrow.__version__, torch.__version__); print("CUDA", torch.cuda.is_available())'],'versions',120)
    execute(['-m','pytest','-q',str(repo/'work/casmi26/tests')],'tests',240)
    execute(['-c',f'import sys; sys.path.insert(0,{str(repo/"work/casmi26")!r}); from casmi26.metric import require_official_rdkit; require_official_rdkit()'],'metric-version',60)
    access=prepare.run([str(python),'-c','from kaggle.cli import main; main()',
                        'competitions','files',prepare.SLUG,'--page-size','200','-v'],
                       env=env,timeout=60,log=out/'access.log',expose=False)
    report['kaggle_files_exit_code']=access['exit_code']
    report['tests_passed']=True
    artifact=state/'artifacts/casmi26/public-v02';artifact.mkdir(parents=True,exist_ok=True)
    try:
        data,provenance=public_subset(state/'data/external/massspecgym-prefix-v1')
        report['public_data']=provenance
        model=artifact/'fingerprint-public.npz'
        execute([str(repo/'train.py'),'--train',str(data),'--output',str(model),
                 '--epochs','30','--hidden','384','--max-molecules','10000','--device','cuda'],
                'public-training',1800)
        validation=json.loads(model.with_suffix('.validation.json').read_text(encoding='utf-8'))
        report['validation']={k:v for k,v in validation.items() if k not in ('molecules',)}
        for path in (model,model.with_suffix('.validation.json'),model.with_suffix('.catalog.csv')):
            shutil.copy2(path,out/path.name)
        report['model_path']=str(model)
    except Exception as exc:
        report['public_benchmark_error']=str(exc)[:1000]
        report['status']='tests_completed_public_benchmark_failed'
    notebook=artifact/'casmi26-submission.ipynb'
    execute(['-c',f'import sys; sys.path.insert(0,{str(repo/"work/casmi26")!r}); from pathlib import Path; from casmi26.notebook import build_notebook; build_notebook(Path({str(notebook)!r}))'],'build-notebook',120)
    shutil.copy2(notebook,out/notebook.name)
    visible=Path(r'C:\Users\loval\Chemistry\artifacts\casmi26\public-v02')
    visible.mkdir(parents=True,exist_ok=True)
    for path in artifact.iterdir():
        if path.is_file() and path.suffix in ('.npz','.csv','.json','.ipynb'):
            shutil.copy2(path,visible/path.name)
    report['user_artifact_directory']=str(visible)
    if 'public_benchmark_error' not in report:
        report['status']='external_baseline_completed_no_official_score'
    text=json.dumps(report,indent=2,ensure_ascii=True,allow_nan=False)
    (out/'progress-report.json').write_text(text+'\n',encoding='utf-8')
    (visible/'progress-report.json').write_text(text+'\n',encoding='utf-8')
    print('CASMI_PROGRESS_REPORT_BEGIN\n'+text+'\nCASMI_PROGRESS_REPORT_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

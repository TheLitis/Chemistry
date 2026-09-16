"""Package the locked R06 winner and verify CPU inference; no Kaggle writes."""
from __future__ import annotations
import csv
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve())],env={**os.environ,
            'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4'})
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.r06_candidate import build_bundle,infer
    from casmi26.production import sha256,write_json
    from casmi26.metric import require_official_rdkit,distinct_guesses
    require_official_rdkit()
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    experiment=state/'artifacts/casmi26/research-r06';dest=experiment/'release';dest.mkdir(exist_ok=True)
    research=json.loads((experiment/'report.json').read_text())
    tests=subprocess.run([str(python),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,
                         encoding='utf-8',errors='replace',timeout=300)
    print(tests.stdout,flush=True);(out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8')
    if tests.returncode:raise RuntimeError('Candidate tests failed')
    if sha256(state/'artifacts/casmi26/official-v1/model.npz')!='4d79a6d4f24060cd0f7c09bebefbced2c9e1672cb1cdf5f7831cd5a19c03b55d':
        raise RuntimeError('Original official champion changed')
    if not research.get('supported_improvement'):
        write_json(out/'release.json',{'status':'not_promoted','reason':'No supported audit improvement','new_submissions':0})
        return 0
    bundle=dest/'bundle'
    manifest=build_bundle(experiment,state/'cache/casmi26/highres-v3',state/'artifacts/casmi26/research-v3/multitarget.npz',bundle)
    spec=importlib.util.spec_from_file_location('staged',repo/'tasks/casmi_staged.py')
    staged=importlib.util.module_from_spec(spec);spec.loader.exec_module(staged)
    data=staged.find_dataset(state/'data/external')
    summaries={}
    for budget in (3,'all'):
        output=dest/('submission-'+str(budget)+'.csv')
        result=infer(data/'test.parquet',bundle,output,device='cpu',budget=budget,template=data/'sample_submission.csv')
        with output.open(encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f)
            if reader.fieldnames!=['molecule_id','smiles']:raise ValueError('Wrong output schema')
            rows=list(reader)
        if len(rows)!=result['prediction_count'] or len({r['molecule_id'] for r in rows})!=len(rows):raise ValueError('Wrong output IDs')
        for row in rows:
            values=row['smiles'].split(';') if row['smiles'] else []
            if len(values)>25 or len(distinct_guesses(values,25))!=len(values):raise ValueError('Invalid/duplicate guesses')
        if sha256(output)!=result['submission_sha256']:raise ValueError('Output checksum differs')
        summaries[str(budget)]={k:v for k,v in result.items() if k!='details'}
        summaries[str(budget)]['spectra_actually_used']=sum(d['spectra_used'] for d in result['details'].values())
        summaries[str(budget)]['all_guesses_valid']=True
        shutil.copy2(output,out/output.name);shutil.copy2(output.with_suffix('.csv.report.json'),out/(output.name+'.report.json'))
    for name in ('report.json','screen-results.json','refinement-results.json','selection-before-audit.json','audit-ranks.json'):
        shutil.copy2(experiment/name,out/name)
    entry="from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent/'work'/'casmi26'))\nfrom casmi26.r06_candidate import main\nif __name__=='__main__': raise SystemExit(main())\n"
    (dest/'predict_r06.py').write_text(entry,encoding='utf-8')
    readme=('R06 research candidate: unchanged audited holdout-trained checkpoints, not a full-data refit.\n'
        'Run with the existing CASMI Python: python predict_r06.py --test TEST.parquet --bundle bundle --output submission.csv\n'
        'Default --budget all uses all distinct acquisitions. --budget 3 matches the primary validation budget.\n'
        'The all-view late-fusion extension above three views is not an independent audit result.\n'
        'No Kaggle upload/submission is performed. Official score is unknown.\n')
    (dest/'README.txt').write_text(readme,encoding='utf-8')
    runtime={'python':sys.version,'torch':torch.__version__,'rdkit':__import__('rdkit').__version__,'numpy':__import__('numpy').__version__,
             'pyarrow':__import__('pyarrow').__version__,'tested_device':'cpu','offline_wheels_included':False}
    write_json(dest/'runtime.json',runtime)
    archive=dest/'r06-research-candidate.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name in sorted(set(manifest['files'])|{'r06-bundle.json'}):
            z.write(bundle/name,'bundle/'+name)
        for name in ('predict_r06.py','README.txt','runtime.json'):z.write(dest/name,name)
        for file in sorted((repo/'work/casmi26/casmi26').glob('*.py')):z.write(file,file.relative_to(repo).as_posix())
    result={'status':'completed','checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'commit':os.environ.get('GITHUB_SHA'),
        'identity':subprocess.run(['whoami'],capture_output=True,text=True,check=True,timeout=15).stdout.strip(),
        'selection':manifest['selection'],'bundle_manifest_sha256':sha256(bundle/'r06-bundle.json'),'files':manifest['files'],
        'outputs':summaries,'directory':str(dest),'archive':{'bytes':archive.stat().st_size,'sha256':sha256(archive),'path':str(archive)},
        'original_champion_unchanged':True,'new_submissions':0,'new_uploads':0,'new_training':False,
        'official_score':None,'candidate_not_champion':True,'weights_kind':manifest['weights_kind'],
        'tests':tests.stdout.strip(),'runtime':runtime}
    spec=importlib.util.spec_from_file_location('source_archive',repo/'tasks/source_archive.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    result['source_archive']=exporter.archive_source(repo,out/'source.zip')
    result['reconciles_archive_failure']={'research_run':35086289470,'release_run':35087391690,
        'reason':'Repository still owned by the former NETWORK SERVICE after user switched service to SYSTEM',
        'solution':'Exact checkout trusted for this archive command only; no global Git, ACL or service changes'}
    spec=importlib.util.spec_from_file_location('prepare',repo/'tasks/casmi_prepare.py')
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    env=prep.kaggle_environment(state,dict(os.environ));result['kaggle_reads']={}
    for label,args in [('history',['competitions','submissions',prep.SLUG,'--format','json']),
                       ('limits',['competitions','submission-limits',prep.SLUG,'--json'])]:
        read=subprocess.run([str(python),'-c','from kaggle.cli import main;main()']+args,env=env,
            stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        record={'exit_code':read.returncode}
        if read.returncode==0:record['data']=json.loads(read.stdout)
        else:record['error']=prep.redact(read.stderr,env)[:1000]
        result['kaggle_reads'][label]=record
    write_json(dest/'release.json',result);write_json(out/'release.json',result)
    print('R06_RELEASE_BEGIN\n'+json.dumps(result,indent=2)+'\nR06_RELEASE_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

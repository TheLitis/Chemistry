"""Verify R08B and run a separate forward-assisted full R07 release; no Kaggle writes."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

R07_BUNDLE='195e2a73b7d25ce570c178b2f1fb0603d2f8cf47a82e1050b4a8b18ab540baf6'
R07_MODEL='4a05a97b65276df6558e4c156f2129b5463b758f4ea730b0f1f93e99acf055a6'
SOURCE='e19ef82c9a6cb9dbac92bce23e914008f1aeb44e'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('verify','build'),required=True);a=p.parse_args()
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,
            'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
    repo=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(repo/'tasks'),str(repo/'work/casmi26')]
    from casmi26.production import sha256,write_json
    from casmi26.r07_release import verify_bundle,FILES
    import casmi_r08b_verify as independent
    import casmi_r08b_confirmation as original
    import casmi_r08_forward as forward
    import r08_python_command as imports
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    artifacts=state/'artifacts/casmi26';study=artifacts/'research-r08b/fixed-confirmation-v1'
    root=artifacts/'candidate-r08b-20260918';root.mkdir(parents=True,exist_ok=True)
    proof=independent.verify(study,previous_keys=original.used_keys(artifacts,study))
    write_json(out/'independent-verification.json',proof)
    if not proof['decision']['eligible']:raise ValueError('R08B candidate evidence gate failed')
    base=artifacts/'final-r07-v1/bundle'
    if sha256(base/'r07-bundle.json')!=R07_BUNDLE or sha256(base/'model.npz')!=R07_MODEL:
        raise ValueError('Frozen R07 changed')
    base_manifest=verify_bundle(base)
    fiora_root=artifacts/'research-r08/forward-v1';public=fiora_root/('fiora-'+SOURCE)
    deps=fiora_root/'deps'
    paths=[deps,public,repo/'work/casmi26',repo/'tasks'];env=forward.env_clean()
    env['FIORA_TEST_MODEL']=str(public/'fiora/resources/models/fiora_OS_v1.0.0.pt')
    test=subprocess.run(imports.python_command(py,paths,'import pytest,sys;raise SystemExit(pytest.main(sys.argv[1:]))',
        ['-q',repo/'work/casmi26/tests']),env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,
        encoding='utf-8',errors='replace',timeout=600)
    (out/'tests.log').write_text(test.stdout+'\n'+test.stderr,encoding='utf-8');print(test.stdout,flush=True)
    if test.returncode:raise RuntimeError('Tests failed before candidate execution')
    prep=load('prep',repo/'tasks/casmi_prepare.py');reader=load('reader',repo/'tasks/casmi_r07_submission.py')
    kenv=prep.kaggle_environment(state,dict(os.environ));kenv.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    kaggle={}
    for name,code,args in [('status',reader.API_READ,[]),('files','from kaggle.cli import main;main()',
        ['competitions','files',reader.SLUG,'--page-size','200','-v'])]:
        r=subprocess.run([str(py),'-c',code]+args,env=kenv,stdin=subprocess.DEVNULL,capture_output=True,
            text=True,encoding='utf-8',errors='replace',timeout=90)
        kaggle[name]={'exit_code':r.returncode}
        if name=='status' and r.returncode==0:kaggle[name]['data']=json.loads(r.stdout)
        else:kaggle[name]['text']=prep.redact(r.stdout+'\n'+r.stderr,kenv)[:18000]
    write_json(out/'kaggle-read-only.json',kaggle)
    inventory=json.loads((artifacts/'research-r07/inventory.json').read_text())
    train=Path(inventory['train_path']);testfile=train.parent/'test.parquet';template=train.parent/'sample_submission.csv'
    result={'stage':a.stage,'commit':os.environ.get('GITHUB_SHA'),'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'gate':proof['decision'],'tests':test.stdout.strip(),'kaggle':kaggle,
        'local_train':{'path':str(train),'bytes':train.stat().st_size,'sha256':sha256(train)},
        'official_training_update_notice':'2026-09-16: water-loss adduct correction; remote metadata is included for provenance review',
        'new_submissions':0,'new_uploads':0,'new_training':False,'r07_champion_changed':False,'official_score_for_candidate':None}
    if result['local_train']['sha256']!=base_manifest['train_sha256']:raise ValueError('Local train differs from validated R07 input')
    if a.stage=='build':
        started=time.monotonic();package=root/'package';package.mkdir(exist_ok=True)
        packaged_bundle=package/'bundle';packaged_bundle.mkdir(exist_ok=True)
        for name in FILES+('r07-bundle.json',):
            target=packaged_bundle/name
            if target.exists() and sha256(target)!=sha256(base/name):raise ValueError('Existing packaged model differs')
            if not target.exists():shutil.copy2(base/name,target)
        code=package/'work/casmi26/casmi26';code.mkdir(parents=True,exist_ok=True)
        for path in (repo/'work/casmi26/casmi26').glob('*.py'):shutil.copy2(path,code/path.name)
        third=package/'third_party';third.mkdir(exist_ok=True)
        for path in (public/'fiora').rglob('*'):
            if path.is_file() and (path.suffix=='.py' or path.name.endswith(('_state.pt','_params.json'))):
                target=third/path.relative_to(public);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        for path in public.glob('LICENSE*'):shutil.copy2(path,package/('FIORA-'+path.name))
        launcher=package/'predict_r08b.py'
        launcher.write_text("from pathlib import Path\nimport sys\nROOT=Path(__file__).resolve().parent\nsys.path[:0]=[str(ROOT/'work/casmi26'),str(ROOT/'third_party')]\nif __name__=='__main__':\n    from casmi26.forward_release import main\n    raise SystemExit(main())\n",encoding='utf-8')
        paths=[deps,third,package/'work/casmi26']
        output=root/'submission.csv'
        args=['--test',testfile,'--train',train,'--bundle',packaged_bundle,'--output',output,
            '--model-path',third/'fiora/resources/models/fiora_OS_v1.0.0.pt','--workers','8',
            '--cache',root/'forward-cache.sqlite','--device','cpu']
        if template.exists():args+=['--sample-submission',template]
        command=imports.python_command(py,paths,'import sys;from casmi26.forward_release import main;raise SystemExit(main(sys.argv[1:]))',args)
        print('R08B_FULL_CANDIDATE_START',flush=True)
        r=subprocess.run(command,env=env,cwd=package,stdin=subprocess.DEVNULL,capture_output=True,
            text=True,encoding='utf-8',errors='replace',timeout=10800)
        (out/'inference.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
        if r.returncode:raise RuntimeError('Full candidate failed; inspect inference.log')
        report=json.loads(output.with_suffix('.csv.report.json').read_text())
        from casmi26.metric import structure_key
        import pyarrow.parquet as pq
        ids=set(map(str,pq.read_table(testfile,columns=['molecule_id']).column('molecule_id').to_pylist()))
        with output.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        if len(rows)!=len(ids) or {r['molecule_id'] for r in rows}!=ids:raise ValueError('Candidate IDs differ')
        for row in rows:
            keys=[structure_key(s) for s in row['smiles'].split(';')]
            if not 1<=len(keys)<=25 or None in keys or len(set(keys))!=len(keys):raise ValueError('Invalid guesses')
        if report['format']!=8 or report['test_labels_used'] is not False or report['submission_sha256']!=sha256(output):
            raise ValueError('Candidate report contract failed')
        verify_bundle(packaged_bundle)
        manifest={'format':8,'algorithm':'R07-plus-fixed-R08B-forward','base_bundle_sha256':R07_BUNDLE,
            'base_model_sha256':R07_MODEL,'forward':report['forward'],'validation_report_sha256':sha256(study/'report.json'),
            'files':{p.relative_to(package).as_posix():sha256(p) for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts},
            'contains_test_ids_or_predictions':False,'dependencies_bundled':False,'official_score':None}
        write_json(package/'r08b-package.json',manifest)
        archive=root/'casmi-r08b-candidate.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for name in manifest['files']:z.write(package/name,name)
            z.write(package/'r08b-package.json','r08b-package.json')
        result.update(status='candidate_built_and_locally_executed',release_directory=str(root),
            package_sha256=sha256(package/'r08b-package.json'),archive_sha256=sha256(archive),archive_bytes=archive.stat().st_size,
            prediction={k:v for k,v in report.items() if k!='details'},rows=len(rows),
            guesses=sum(len(r['smiles'].split(';')) for r in rows),all_guesses_valid_and_distinct=True,
            seconds=time.monotonic()-started,kaggle_linux_preview_verified=False)
        shutil.copy2(output,out/'submission.csv');shutil.copy2(output.with_suffix('.csv.report.json'),out/'submission.csv.report.json')
    else:result['status']='independent_evidence_and_runtime_verified'
    write_json(out/'release-status.json',result);write_json(root/(a.stage+'-status.json'),result)
    subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip','--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
    print('R08B_RELEASE_BEGIN\n'+json.dumps(result,indent=2)+'\nR08B_RELEASE_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

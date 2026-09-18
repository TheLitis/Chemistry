"""At-most-once submission of the frozen, independently verified R08B preview.

Only one immutable private notebook is accepted. Check/status are read-only on
Kaggle. An attempted write, even a timeout without a ref, can never be retried.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

SLUG='enveda-CASMI26-molecule-id-mass-spectra'
KERNEL='thelindortis/casmi26-r08b-forward-hybrid'
DESCRIPTION='CASMI26 R08B - frozen R07 plus 0.25 FIORA; certified top25'
BUNDLE='195e2a73b7d25ce570c178b2f1fb0603d2f8cf47a82e1050b4a8b18ab540baf6'
MODEL='4a05a97b65276df6558e4c156f2129b5463b758f4ea730b0f1f93e99acf055a6'
FIORA='83221e187991f116a8aed1bf272dd656cf31721a177dcbb0239f414c8df31a0e'
PARAMS='c214195f945b1273b7b350cfe4bd964b9a410316433966e74c0a8bf9acdb86c5'
UTC=dt.timezone.utc


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4194304),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def utc(value):
    value=dt.datetime.fromisoformat(str(value).replace('Z','+00:00'))
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def candidate_identity(build,preview,assets,output):
    if (build.get('status')!='candidate_built_and_locally_executed' or
        build.get('gate',{}).get('eligible') is not True or
        build.get('all_guesses_valid_and_distinct') is not True or
        preview.get('preview_verified') is not True):
        raise ValueError('Candidate has not passed local and Kaggle preview acceptance')
    if preview.get('kernel')!=KERNEL or type(preview.get('kernel_version')) is not int or preview['kernel_version']!=1:
        raise ValueError('Wrong immutable notebook identity')
    if (assets.get('format')!=8 or assets.get('contains_test_ids_or_predictions') is not False or
        assets.get('base_bundle_manifest_sha256')!=BUNDLE):raise ValueError('Wrong asset contract')
    for key in ('package_sha256','asset_manifest_sha256','notebook_sha256','verified_submission_sha256'):
        if not isinstance(preview.get(key),str) or not re.fullmatch('[0-9a-f]{64}',preview[key]):
            raise ValueError('Missing verified identity: '+key)
    if build.get('package_sha256')!=preview['package_sha256'] or assets.get('package_sha256')!=preview['package_sha256']:
        raise ValueError('Mismatched package identity')
    expected=build['prediction']
    for key in ('format','model_sha256','bundle_manifest_sha256','prediction_count',
                'test_spectra','test_sha256','train_sha256','submission_sha256'):
        if key not in output or output[key]!=expected.get(key):raise ValueError('Preview differs: '+key)
    if (output.get('format')!=8 or output.get('model_sha256')!=MODEL or output.get('bundle_manifest_sha256')!=BUNDLE or
        output.get('test_labels_used') is not False or output.get('empty_candidate_rows')!=[]):
        raise ValueError('Invalid output contract')
    if type(output['prediction_count']) is not int or output['prediction_count']<1:
        raise ValueError('Invalid output count')
    if type(output['test_spectra']) is not int or output['test_spectra']<output['prediction_count']:
        raise ValueError('Invalid spectrum count')
    if output['submission_sha256']!=preview['verified_submission_sha256'] or output['submission_sha256']!=preview.get('visible_submission_sha256'):
        raise ValueError('Changed prediction hash')
    f=output.get('forward',{})
    if (f.get('weight')!=.25 or f.get('feature')!='cosine_nearest' or f.get('all_top25_numerically_certified') is not True or
        f.get('model_sha256')!=FIORA or f.get('params_sha256')!=PARAMS):raise ValueError('Forward score changed')
    seconds=output.get('seconds_total')
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<seconds<8*3600:
        raise ValueError('Preview runtime is unknown or leaves insufficient headroom')
    return {k:preview[k] for k in ('kernel','kernel_version','package_sha256','asset_manifest_sha256',
                                  'notebook_sha256','verified_submission_sha256')}


def submit_arguments(identity):
    if identity.get('kernel')!=KERNEL or type(identity.get('kernel_version')) is not int or identity['kernel_version']!=1:
        raise ValueError('Wrong immutable notebook target')
    return ['competitions','submit',SLUG,'-k',KERNEL,'-v','1','-f','submission.csv','-m',DESCRIPTION]


def reconcile(history,journal):
    if not journal.get('attempted'):return None
    if journal.get('description')!=DESCRIPTION:raise ValueError('Another candidate journal')
    matches=[];retained=journal.get('submission_ref');attempt=utc(journal['attempted_utc'])
    for row in history:
        if retained is not None:
            if str(row.get('ref'))!=str(retained):continue
            if row.get('description')!=DESCRIPTION:raise ValueError('Retained submission identity changed')
            matches.append(row)
        elif row.get('description')==DESCRIPTION and -60<=(utc(row['date'])-attempt).total_seconds()<=900:
            matches.append(row)
    if len(matches)>1:raise ValueError('Ambiguous submission reconciliation')
    return matches[0] if matches else None


def next_action(journal,budget,stage):
    if journal.get('attempted') or stage!='submit':return 'read_only'
    return 'one_attempt' if budget.get('may_submit') is True else 'wait'


def validate_history(response):
    rows=response.get('history');limits=response.get('limits',{})
    if not isinstance(rows,list) or type(limits.get('numTotal')) is not int or limits['numTotal']!=len(rows):
        raise ValueError('Incomplete account history; no new submission permitted')
    refs=[str(r['ref']) for r in rows]
    if len(refs)!=len(set(refs)):raise ValueError('Duplicate history reference')
    return True


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod);return mod


def verify_files(folder,files):
    folder=Path(folder).resolve()
    for name,h in files.items():
        path=folder/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or folder not in path.resolve().parents:
            raise ValueError('Unsafe manifest path')
        if sha256(path)!=h:raise ValueError('Modified artifact: '+name)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('check','submit','status'),required=True);a=p.parse_args()
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve())]+sys.argv[1:],
            env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import write_json
    shared=load('shared_r07',repo/'tasks/casmi_r07_submission.py')
    prep=load('prepare',repo/'tasks/casmi_prepare.py');env=prep.kaggle_environment(state,dict(os.environ))
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    art=state/'artifacts/casmi26';root=art/'kaggle-r08b-v1';root.mkdir(parents=True,exist_ok=True)
    release=art/'candidate-r08b-20260918';out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    lock=art/'competition-submit-exclusive.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    journal_path=root/'submission-journal.json';journal=read(journal_path) if journal_path.exists() else {}
    result={'stage':a.stage,'new_submissions':0,'new_uploads':0,'official_score':None,'commit':os.environ.get('GITHUB_SHA')}
    def save():write_json(journal_path,journal)
    def account():
        r=subprocess.run([str(python),'-c',shared.API_READ],env=env,stdin=subprocess.DEVNULL,
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        if r.returncode:
            (out/'api-error.log').write_text(prep.redact(r.stderr,env),encoding='utf-8')
            raise RuntimeError('Kaggle read failed; no automatic write retry')
        value=json.loads(r.stdout);validate_history(value);return value
    try:
        if not journal.get('attempted'):
            build=read(release/'build-status.json');preview=read(root/'preview-journal.json')
            assets=read(root/'assets/forward-assets.json');output=read(root/'verified-output/submission.csv.report.json')
            identity=candidate_identity(build,preview,assets,output)
            for path,digest in ((release/'package/r08b-package.json',identity['package_sha256']),
                (root/'assets/forward-assets.json',identity['asset_manifest_sha256']),
                (root/'notebook/casmi26-r08b.ipynb',identity['notebook_sha256']),
                (root/'verified-output/submission.csv',identity['verified_submission_sha256'])):
                if sha256(path)!=digest:raise ValueError('Verified file changed: '+path.name)
            verify_files(root/'assets',assets['files'])
            verify_files(release/'package',read(release/'package/r08b-package.json')['files'])
        else:
            identity=journal['identity'];submit_arguments(identity)
        current=account();now=dt.datetime.now(UTC)
        budget=shared.budget_decision(current['history'],current['limits'],now)
        result.update(checked_utc=now.isoformat(),identity=identity,budget=budget,limits=current['limits'])
        if a.stage=='check':
            r=subprocess.run([str(python),'-m','pytest','-q',str(repo/'work/casmi26/tests/test_r08b_submission.py'),
                str(repo/'work/casmi26/tests/test_r07_submission.py')],stdin=subprocess.DEVNULL,
                capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
            (out/'tests.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8');result['tests']=r.stdout.strip()
            if r.returncode:raise RuntimeError('Submission regression tests failed')
        action=next_action(journal,budget,a.stage)
        if not journal.get('attempted') and any(r.get('description')==DESCRIPTION for r in current['history']):
            raise RuntimeError('Matching manual submission already exists; reconcile without another write')
        if action=='one_attempt':
            journal.update(attempted=True,attempted_utc=now.isoformat(),description=DESCRIPTION,identity=identity)
            save();result['new_submissions']=1
            try:
                r=subprocess.run([str(python),'-c','from kaggle.cli import main;main()']+submit_arguments(identity),
                    env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
                text=prep.redact(r.stdout+'\n'+r.stderr,env)
                text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
                (out/'submit.log').write_text(text,encoding='utf-8');journal['command_exit_code']=r.returncode
            except subprocess.TimeoutExpired:journal['command_timeout']=True
            save();current=account()
        row=reconcile(current['history'],journal)
        if row:
            classification=shared.classify(row)
            journal.update(submission_ref=int(row['ref']),classification=classification,last_checked_utc=dt.datetime.now(UTC).isoformat());save()
            result.update(status=classification['state'],submission=row,classification=classification,
                          official_score=classification['public_score'])
        elif journal.get('attempted'):result['status']='attempt_not_yet_reconciled_no_resubmit'
        elif action=='wait':result['status']='waiting_for_shared_submission_budget'
        else:result['status']='ready_for_one_submission' if budget['may_submit'] else 'waiting_for_shared_submission_budget'
        result.update(limits=current['limits'],journal=journal)
        write_json(out/'submission-status.json',result);write_json(root/'submission-status.json',result)
        print('R08B_SUBMISSION_BEGIN\n'+json.dumps(result,indent=2)+'\nR08B_SUBMISSION_END',flush=True)
        return 0
    finally:lock.unlink()


if __name__=='__main__':raise SystemExit(main())

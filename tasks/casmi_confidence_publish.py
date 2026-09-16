"""Publish the locked R03 candidate, reuse private V1 assets, never spam submits.

Stages are explicit. No token changes, public sharing, rule acceptance or final
submission selection. Ambiguous network writes must not be automatically retried.
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

KERNEL='casmi26-confidence-v2'
DESCRIPTION='CASMI26 confidence v2 - R03 locked 0.95/0.05; no external data'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def adapt_notebook(book,threshold,margin):
    changes=0
    for cell in book['cells']:
        if cell['cell_type']!='code':continue
        text=''.join(cell['source'])
        if 'from casmi26.portable import main' in text:
            text=text.replace('from casmi26.portable import main','from casmi26.confident import main');changes+=1
        text=text.replace("CODE = WORK_ROOT/'_casmi26_code'","CODE = Path('/tmp/casmi26_v2_code') if sys.platform.startswith('linux') else WORK_ROOT/'_casmi26_code'")
        text=text.replace("DEPS = WORK_ROOT/'_casmi26_deps'","DEPS = Path('/tmp/casmi26_v2_deps') if sys.platform.startswith('linux') else WORK_ROOT/'_casmi26_deps'")
        text=text.replace("wheel_dir = ASSETS/'wheels'","wheel_dir = ASSETS/'wheels' if (ASSETS/'wheels').is_dir() else ASSETS")
        marker="if (DATA/'sample_submission.csv').is_file():"
        if marker in text:
            text=text.replace(marker,"command += ['--gate-threshold', "+repr(str(threshold))+", '--gate-margin', "+repr(str(margin))+"]\n"+marker);changes+=1
        compile(text,'v2-notebook-cell','exec')
        cell['source']=text.splitlines(True)
    if changes!=2:raise ValueError('Notebook source contract changed; do not silently produce old inference')
    book['cells'][0]['source']=['# CASMI26 confidence V2\n','R03 thresholds fixed from molecule-held-out calibration. Private assets, offline current-input inference. No external catalog.\n']
    return book


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('publish','submit','status'),required=True)
    args=parser.parse_args();state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    from casmi26.portable import verify_bundle
    from casmi26.submission_notebook import build_notebook
    prep=load('prep',repo/'tasks/casmi_prepare.py');pub=load('pub',repo/'tasks/casmi_kaggle_publish.py')
    score=load('score',repo/'tasks/casmi_score_status.py')
    env=prep.kaggle_environment(state,dict(os.environ));env['PYTHONIOENCODING']='utf-8'
    root=state/'artifacts/casmi26/kaggle-v2';root.mkdir(parents=True,exist_ok=True)
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    journal_path=root/'publish-journal.json';journal=json.loads(journal_path.read_text()) if journal_path.exists() else {}
    report={'stage':args.stage,'commit':os.environ.get('GITHUB_SHA'),'new_submissions':0,'official_score':None}
    def run(label,cli,timeout=180,check=True):
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+cli,env=env,stdin=subprocess.DEVNULL,
             capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        text=prep.redact(r.stdout+'\n'+r.stderr,env)
        text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',text)
        (out/(label+'.log')).write_text(text,encoding='utf-8')
        print(label.upper()+' '+str(r.returncode)+'\n'+text[-3500:],flush=True)
        if check and r.returncode:raise RuntimeError(label+' failed; no automatic write retry')
        return text,r.returncode
    def persist():write_json(journal_path,journal)
    try:
        pre=json.loads((state/'artifacts/casmi26/kaggle-v1/preflight.json').read_text())
        if 'maximum of five (5) Submissions per day' not in '\n'.join(pre.get('rules_excerpts',[])):
            raise ValueError('Submission limit not verified')
        owner=pre['kernel_init_metadata']['id'].split('/')[0];dataset=owner+'/casmi26-assets-v1';kernel=owner+'/'+KERNEL
        history,_=run('history',['competitions','submissions',pub.SLUG,'-v']);rows=pub.parse_history(history)
        limits,_=run('limits',['competitions','submission-limits',pub.SLUG,'--json']);limits=json.loads(limits)
        budget=pub.budget_status(rows,dt.datetime.now(dt.timezone.utc),5)
        report.update(kernel=kernel,dataset=dataset,budget=budget,live_limits=limits)
        if args.stage=='publish':
            if journal.get('push_attempted'):raise RuntimeError('V2 push already attempted; inspect status')
            if not budget['may_submit'] or limits.get('numAllowedNow',0)<1:raise RuntimeError('Quota guard blocked new candidate')
            experiment=json.loads((state/'artifacts/casmi26/research-r03/report.json').read_text())
            selection=experiment['selection']
            if not experiment['recommend_gate'] or selection['threshold']!=.95 or selection['margin']!=.05:
                raise RuntimeError('R03 did not authorize this frozen configuration')
            checks=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
            print(checks.stdout,flush=True);report['tests']=checks.stdout.strip()
            if checks.returncode:raise RuntimeError('Full suite failed before upload')
            bundle=state/'artifacts/casmi26/official-v1/bundle';manifest=verify_bundle(bundle)
            previous=json.loads((state/'artifacts/casmi26/kaggle-v1/publish-journal.json').read_text())
            if any(previous['asset_hashes'][n]!=digest for n,digest in manifest['files'].items()):
                raise ValueError('Private V1 assets differ from current verified model bundle')
            remote=root/'remote-metadata';remote.mkdir(exist_ok=True)
            run('private_metadata',['datasets','metadata',dataset,'-p',str(remote)])
            metadata=json.loads((remote/'dataset-metadata.json').read_text());info=metadata.get('info',metadata)
            if info.get('isPrivate') is not True:raise RuntimeError('Model assets are not verified private')
            folder=root/'notebook';folder.mkdir(exist_ok=True);notebook=folder/'casmi26-submission.ipynb'
            build_notebook(notebook);book=adapt_notebook(json.loads(notebook.read_text()),selection['threshold'],selection['margin'])
            write_json(notebook,book)
            meta=pub.kernel_metadata(owner,dataset);meta.update(id=kernel,title='CASMI26 Confidence V2')
            write_json(folder/'kernel-metadata.json',meta)
            journal.update(owner=owner,kernel=kernel,dataset=dataset,notebook_sha256=sha256(notebook),
                model_sha256=manifest['files']['model.npz'],threshold=.95,margin=.05,source_experiment='R03',
                push_attempted=True,description=DESCRIPTION);persist()
            text,_=run('push',['kernels','push','-p',str(folder)],timeout=180)
            match=re.search(r'Kernel version\s+(\d+)',text,re.I)
            if not match:raise RuntimeError('Push result has no version; inspect before retry')
            journal.update(kernel_version=int(match[1]),push_succeeded=True);persist()
            report['status']='private_v2_notebook_pushed'
        elif args.stage=='submit':
            if journal.get('submission_attempted'):raise RuntimeError('V2 submission already attempted; status only')
            if not budget['may_submit'] or limits.get('numAllowedNow',0)<1:raise RuntimeError('Quota guard blocked submit')
            if not journal.get('push_succeeded'):raise RuntimeError('No confirmed V2 notebook')
            text,_=run('kernel_status',['kernels','status',kernel])
            if 'COMPLETE' not in text or 'ERROR' in text:raise RuntimeError('Notebook is not successfully complete')
            directory=Path(tempfile.mkdtemp(prefix='verified-',dir=root))
            run('output',['kernels','output',kernel,'-p',str(directory),'--file-pattern',r'(^|/)submission\.csv(\.report\.json)?$'],timeout=300)
            report['output_validation']=pub.validate_downloaded_output(directory)
            prediction=json.loads((directory/'submission.csv.report.json').read_text())
            if prediction.get('ranking')!='r03-confidence' or prediction.get('gate_threshold')!=.95 or prediction.get('gate_margin')!=.05:
                raise RuntimeError('Notebook did not execute frozen R03 inference')
            report['visible_prediction']={k:v for k,v in prediction.items() if k!='details'}
            journal.update(submission_attempted=True,attempted_utc=dt.datetime.now(dt.timezone.utc).isoformat());persist()
            text,rc=run('submit',['competitions','submit',pub.SLUG,'-k',kernel,'-v',str(journal['kernel_version']),
                '-f','submission.csv','-m',DESCRIPTION],check=False)
            journal.update(submission_command_exit_code=rc,submission_accepted=rc==0);persist()
            if rc:raise RuntimeError('Submission outcome recorded; no retry')
            report.update(new_submissions=1,status='v2_submitted_for_scoring')
        else:
            if journal.get('push_attempted'):
                text,_=run('kernel_status',['kernels','status',kernel],check=False);report['kernel_status']=text.strip()
            matches=[r for r in rows if r.get('description')==DESCRIPTION]
            if len(matches)>1:raise RuntimeError('Ambiguous V2 history; no score inferred')
            if matches and journal.get('submission_accepted'):
                row=matches[0]
                if abs((score.utc(row['date'])-score.utc(journal['attempted_utc'])).total_seconds())>120:
                    raise RuntimeError('V2 submission timestamp differs from local journal')
                result=score.score_summary(row);report['score']=result
                report['official_score']=result['public_score'];journal['submission_ref']=result['submission_ref'];persist()
            report['status']='v2_status_checked_without_submission'
    except Exception as exc:
        report.update(status='stopped',error=prep.redact(str(exc),env)[:1600]);raise
    finally:
        report['journal']=journal
        write_json(root/(args.stage+'-report.json'),report);write_json(out/'v2-report.json',report)
        print('CONFIDENCE_V2_BEGIN\n'+json.dumps(report,indent=2)+'\nCONFIDENCE_V2_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

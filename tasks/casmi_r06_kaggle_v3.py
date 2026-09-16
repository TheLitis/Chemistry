"""Scorer-safe R06 v3 Kaggle workflow.

Reuses the existing private model and wheel datasets. The notebook embeds current
inference source so no model dataset re-upload is needed. Publish and competition
submit are separate stages; each write is journaled before execution.
"""
from __future__ import annotations
import argparse,base64,csv,datetime as dt,hashlib,importlib.util,io,json,os,re,shutil,subprocess,sys,time,zipfile
from pathlib import Path

SLUG='enveda-CASMI26-molecule-id-mass-spectra'
V1_DATASET='casmi26-assets-v1';R06_DATASET='casmi26-r06-assets-v1'
KERNEL='casmi26-r06-massset-rank-ensemble'
DESCRIPTION='CASMI26 R06.1 - MassSet rank ensemble + scorer-safe mass fallback'
OLD_REF=56278642
EXPECTED_BUNDLE_SHA256='7923285a3cb3d683f017255143bc18feb3caa78a4182a9680023e4c8c3d76699'
VISIBLE_HASH='750e3410dbb73e1cdf7dcb5b0b6a7e36c142ef05cd6eef8618b2fd016508cffe'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def kernel_metadata(owner):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',str(owner)):raise ValueError('Invalid owner')
    return {'id':owner+'/'+KERNEL,'title':'CASMI26 R06 MassSet Rank Ensemble','code_file':'casmi26-r06.ipynb',
      'language':'python','kernel_type':'notebook','is_private':'true','enable_gpu':'false','enable_internet':'false',
      'dataset_sources':[owner+'/'+V1_DATASET,owner+'/'+R06_DATASET],'competition_sources':[SLUG],
      'kernel_sources':[],'model_sources':[]}


def may_submit_correction(limits,pending,journal):
    try:today=int(limits.get('numToday',99));allowed=int(limits.get('numAllowedNow',0))
    except (TypeError,ValueError):return False
    return bool(journal.get('submission_attempted') and journal.get('submission_ref')==OLD_REF and
      journal.get('scoring_error_confirmed') and journal.get('kernel_v3_output_verified') and
      not journal.get('submission_v3_attempted') and pending==0 and today<5 and allowed>=1)


def _source_zip():
    package=Path(__file__).resolve().parents[1]/'work/casmi26/casmi26';stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(package.glob('*.py')):z.writestr('casmi26/'+path.name,path.read_bytes())
    return base64.b64encode(stream.getvalue()).decode('ascii')


def build_notebook(path):
    source64=_source_zip();cells=[]
    def md(t):cells.append({'cell_type':'markdown','id':f'cell-{len(cells)}','metadata':{},'source':t.splitlines(True)})
    def code(t):cells.append({'cell_type':'code','id':f'cell-{len(cells)}','metadata':{},'source':t.splitlines(True),'execution_count':None,'outputs':[]})
    md('# CASMI26 R06.1 — scorer-safe hidden rerun\n\nSame audited model weights and ranking. Current inference source is embedded; hidden molecules outside the strict catalog mass window receive an explicit nearest-mass fallback rather than an empty submission cell.\n')
    code('SOURCE_ZIP_BASE64 = '+repr(source64)+'\n')
    code("""from pathlib import Path
import base64,io,json,os,sys,subprocess,zipfile
INPUT=Path('/kaggle/input');WORK=Path('/kaggle/working');WORK.mkdir(parents=True,exist_ok=True)
data=[p.parent for p in INPUT.rglob('test.parquet') if (p.parent/'sample_submission.csv').is_file()]
if len(data)!=1:raise RuntimeError(f'Expected one competition mount, found {len(data)}')
DATA=data[0];manifests=list(INPUT.rglob('r06-bundle.json'))
if len(manifests)!=1:raise RuntimeError(f'Expected one R06 bundle, found {len(manifests)}')
BUNDLE=manifests[0].parent
CODE=WORK/'r06_v3_code';CODE.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ZIP_BASE64))) as z:
    for member in z.infolist():
        p=Path(member.filename)
        if p.is_absolute() or '..' in p.parts or p.parts[0]!='casmi26' or p.suffix!='.py':raise RuntimeError('Unsafe source member')
    z.extractall(CODE)
launcher=WORK/'run_r06_v3.py'
launcher.write_text("import sys\\nfrom pathlib import Path\\nsys.path.insert(0,"+repr(str(CODE))+ ")\\nfrom casmi26.r06_candidate import main\\nraise SystemExit(main())\\n",encoding='utf-8')
""")
    code("""DEPS=WORK/'r06_v3_deps';DEPS.mkdir(exist_ok=True)
if sys.platform.startswith('linux'):
    wheel_dirs=sorted({p.parent for p in INPUT.rglob('*.whl')})
    if not wheel_dirs:raise RuntimeError('Offline wheels missing')
    cmd=[sys.executable,'-m','pip','install','--no-index']
    for folder in wheel_dirs:cmd+=['--find-links',str(folder)]
    cmd+=['--target',str(DEPS),'--upgrade','numpy==2.3.5','rdkit==2026.3.3','pyarrow==21.0.0']
    subprocess.run(cmd,check=True,capture_output=True,text=True)
env=dict(os.environ,PYTHONPATH=str(DEPS)+os.pathsep+str(CODE),PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
subprocess.run([sys.executable,'-c',"import rdkit,numpy,pyarrow;assert rdkit.__version__=='2026.03.3';assert numpy.__version__=='2.3.5';assert pyarrow.__version__=='21.0.0'"],check=True,env=env)
""")
    code("""output=WORK/'submission.csv'
cmd=[sys.executable,str(launcher),'--test',str(DATA/'test.parquet'),'--bundle',str(BUNDLE),'--output',str(output),'--budget','all','--device','cpu']
if (DATA/'sample_submission.csv').is_file():cmd+=['--sample-submission',str(DATA/'sample_submission.csv')]
r=subprocess.run(cmd,check=True,env=env,capture_output=True,text=True);print(r.stdout[-8000:]);print(r.stderr[-3000:])
report=json.loads((WORK/'submission.csv.report.json').read_text())
assert report['format']==6 and report['budget']=='all' and report['test_labels_used'] is False
assert report['empty_candidate_rows']==[]
print(json.dumps({k:report.get(k) for k in ('prediction_count','test_spectra','submission_sha256','mass_incompatible_fallbacks')},indent=2))
""")
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':cells}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');return path


def _terminal(row):return str(row.get('status','')).split('.')[-1].upper() in ('COMPLETE','ERROR','FAILED','CANCELLED','CANCELED')

def _validate(folder,selection):
    folder=Path(folder);csvp=folder/'submission.csv';rp=folder/'submission.csv.report.json'
    if not csvp.is_file() or not rp.is_file():raise ValueError('Missing v3 output')
    report=json.loads(rp.read_text());digest=hashlib.sha256(csvp.read_bytes()).hexdigest()
    if report.get('format')!=6 or report.get('budget')!='all' or report.get('test_labels_used') is not False:raise ValueError('Wrong v3 contract')
    if report.get('empty_candidate_rows')!=[] or report.get('bundle_manifest_sha256')!=EXPECTED_BUNDLE_SHA256:raise ValueError('Unsafe v3 output')
    if report.get('selection')!=selection or report.get('submission_sha256')!=digest:raise ValueError('Changed model/output hash contract')
    with csvp.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);rows=list(reader)
        if reader.fieldnames!=['molecule_id','smiles']:raise ValueError('Wrong columns')
    if not rows or len({r['molecule_id'] for r in rows})!=len(rows):raise ValueError('Invalid IDs')
    for row in rows:
        g=[x.strip() for x in row['smiles'].split(';') if x.strip()]
        if not 1<=len(g)<=25:raise ValueError('Empty/invalid guesses')
    return {'rows':len(rows),'sha256':digest,'test_spectra':report.get('test_spectra'),'mass_incompatible_fallbacks':report.get('mass_incompatible_fallbacks',[]),'validated':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('publish','submit','status'),required=True);a=p.parse_args()
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    prep=load('prep',repo/'tasks/casmi_prepare.py');env=prep.kaggle_environment(state,dict(os.environ));env['PYTHONIOENCODING']='utf-8'
    root=state/'artifacts/casmi26/kaggle-r06-v1';out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    journal_path=root/'publish-journal.json';journal=json.loads(journal_path.read_text())
    release=state/'artifacts/casmi26/research-r06/release';bundle=release/'bundle/r06-bundle.json'
    if sha256(bundle)!=EXPECTED_BUNDLE_SHA256:raise RuntimeError('R06 bundle changed')
    selection=json.loads(bundle.read_text())['selection'];pre=json.loads((state/'artifacts/casmi26/kaggle-v1/preflight.json').read_text());owner=pre['kernel_init_metadata']['id'].split('/')[0];kernel=owner+'/'+KERNEL
    result={'stage':a.stage,'new_dataset_uploads':0,'new_submissions':0,'official_score':None,'kernel':kernel}
    def save():write_json(journal_path,journal)
    def cli(label,args,timeout=180,check=True):
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        text=prep.redact(r.stdout+'\n'+r.stderr,env);(out/(label+'.log')).write_text(text,encoding='utf-8')
        if check and r.returncode:raise RuntimeError(label+' failed')
        return text,r.returncode
    def history():
        t,_=cli('history',['competitions','submissions',SLUG,'--format','json']);v=json.loads(t.strip()) if t.strip() else []
        if not isinstance(v,list):raise ValueError('Bad history');return v
        return v
    if a.stage=='publish':
        tests=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=360)
        if tests.returncode:raise RuntimeError('Tests failed before v3 push')
        if journal.get('kernel_v3_push_attempted') and not journal.get('kernel_v3_push_succeeded'):raise RuntimeError('Ambiguous v3 push')
        if not journal.get('kernel_v3_push_succeeded'):
            folder=root/'notebook-v3';shutil.rmtree(folder,ignore_errors=True);folder.mkdir();nb=build_notebook(folder/'casmi26-r06.ipynb');(folder/'kernel-metadata.json').write_text(json.dumps(kernel_metadata(owner),indent=2)+'\n')
            journal.update(kernel_v3_push_attempted=True,kernel_v3_notebook_sha256=sha256(nb));save()
            text,_=cli('push_v3',['kernels','push','-p',str(folder),'-t','32400']);m=re.search(r'Kernel version\s+(\d+)',text,re.I)
            if not m:raise RuntimeError('No v3 version returned')
            version=int(m.group(1));journal.update(kernel_v3_version=version,kernel_v3_push_succeeded=True);save()
        version=int(journal['kernel_v3_version']);done=False
        for _ in range(120):
            text,rc=cli('kernel_v3_status',['kernels','status',kernel],check=False);low=text.lower()
            if rc==0 and 'complete' in low and not any(x in low for x in ('error','failed')):done=True;break
            if rc==0 and any(x in low for x in ('error','failed')):raise RuntimeError('v3 notebook failed')
            time.sleep(10)
        if not done:raise RuntimeError('v3 notebook timeout')
        verified=root/'verified-output-v3';shutil.rmtree(verified,ignore_errors=True);verified.mkdir()
        cli('kernel_v3_output',['kernels','output',kernel,'-p',str(verified),'--file-pattern',r'(^|/)submission\.csv(?:\.report\.json)?$','--page-size','200'],timeout=420)
        check=_validate(verified,selection)
        if check['sha256']!=VISIBLE_HASH:raise RuntimeError('v3 changed visible mass-compatible predictions')
        journal.update(kernel_v3_output_verified=True,kernel_v3_visible=check);save();result.update(status='v3_visible_verified',kernel_version=version,visible_output=check)
    elif a.stage=='submit':
        rows=history();limits=json.loads(cli('limits',['competitions','submission-limits',SLUG,'--json'])[0]);pending=sum(not _terminal(r) for r in rows)
        if not may_submit_correction(limits,pending,journal):raise RuntimeError('v3 correction gate blocked')
        version=int(journal['kernel_v3_version']);journal.update(submission_v3_attempted=True,submission_v3_attempted_utc=dt.datetime.now(dt.timezone.utc).isoformat());save()
        text,rc=cli('submit_v3',['competitions','submit',SLUG,'-k',kernel,'-v',str(version),'-f','submission.csv','-m',DESCRIPTION],check=False)
        journal['submission_v3_command_exit_code']=rc;match=re.search(r'Submission ref:\s*(\d+)',text,re.I)
        if match:journal['submission_v3_ref']=int(match.group(1));save()
        if rc:raise RuntimeError('v3 submit failed; no retry')
        result.update(status='v3_submitted',new_submissions=1,submission_ref=journal.get('submission_v3_ref'))
    else:
        rows=history();ref=journal.get('submission_v3_ref');match=[r for r in rows if ref and str(r.get('ref'))==str(ref)]
        row=match[0] if len(match)==1 else None;result.update(status='v3_status',row=row)
        if row and row.get('publicScore') not in (None,''):result['official_score']=float(row['publicScore'])
    result['journal']=journal;write_json(out/'r06-v3-report.json',result);print('R06_V3_BEGIN\n'+json.dumps(result,indent=2,default=str)+'\nR06_V3_END');return 0

if __name__=='__main__':raise SystemExit(main())

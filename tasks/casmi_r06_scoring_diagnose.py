"""Read-only diagnosis for R06 Kaggle Submission Scoring Error.

Never uploads, pushes a kernel, or creates a submission. Audits the exact locally
verified CSV and probes read-only Kaggle metadata for ref 56278642, including the
server-provided error_description hidden by Kaggle CLI 2.2.4.
"""
from __future__ import annotations
import csv
import datetime as dt
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

SLUG='enveda-CASMI26-molecule-id-mass-spectra'
REF=56278642
DESCRIPTION='CASMI26 R06 - two-seed MassSet hybrid rank ensemble; all spectra'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def classify_submission_row(row):
    state=str(row.get('status','')).split('.')[-1].upper()
    raw=row.get('publicScore')
    scored=raw not in (None,'')
    terminal=state in ('COMPLETE','ERROR','FAILED','CANCELLED','CANCELED')
    return {'state':state,'terminal':terminal,'scored':scored,
            'ambiguous_complete_without_score':state=='COMPLETE' and not scored,
            'public_score':float(raw) if scored else None}


def audit_submission(path,expected_ids=None):
    path=Path(path)
    repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.metric import structure_key
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fieldnames=reader.fieldnames;rows=list(reader)
    result={'path':str(path),'columns':fieldnames,'rows':len(rows),'guesses':0,'duplicate_ids':[],
            'empty_cells':[],'empty_guess_tokens':[],'invalid_guesses':[],'structure_key_failures':[],
            'too_many_guesses':[],'duplicate_structure_keys':[],'non_ascii_guesses':[]}
    ids=[str(r.get('molecule_id','')) for r in rows]
    counts={v:ids.count(v) for v in set(ids)};result['duplicate_ids']=sorted(v for v,n in counts.items() if n>1)
    expected=None if expected_ids is None else [str(v) for v in expected_ids]
    result['id_set_matches']=None if expected is None else set(ids)==set(expected) and len(ids)==len(expected)
    result['missing_ids']=[] if expected is None else sorted(set(expected)-set(ids))
    result['extra_ids']=[] if expected is None else sorted(set(ids)-set(expected))
    for row in rows:
        cid=str(row.get('molecule_id',''));cell=row.get('smiles')
        if cell is None or not str(cell).strip():
            result['empty_cells'].append(cid);continue
        tokens=str(cell).split(';')
        if any(not t.strip() for t in tokens):result['empty_guess_tokens'].append(cid)
        if len(tokens)>25:result['too_many_guesses'].append({'molecule_id':cid,'count':len(tokens)})
        keys=[]
        for rank,token in enumerate(tokens,1):
            token=token.strip();result['guesses']+=1
            if not token:continue
            if not token.isascii():result['non_ascii_guesses'].append({'molecule_id':cid,'rank':rank,'smiles':token})
            key=structure_key(token)
            if key is None:
                entry={'molecule_id':cid,'rank':rank,'smiles':token}
                result['invalid_guesses'].append(entry);result['structure_key_failures'].append(entry)
            else:keys.append(key)
        key_counts={k:keys.count(k) for k in set(keys)};dup=sorted(k for k,n in key_counts.items() if n>1)
        if dup:result['duplicate_structure_keys'].append({'molecule_id':cid,'keys':dup})
    result['format_ok']=fieldnames==['molecule_id','smiles'] and not any((result['duplicate_ids'],result['empty_cells'],
        result['empty_guess_tokens'],result['invalid_guesses'],result['too_many_guesses']))
    return result


def scalar_fields(obj):
    values={}
    for name in dir(obj):
        if name.startswith('_') or name.lower() in ('token','authorization'):continue
        try:value=getattr(obj,name)
        except Exception:continue
        if callable(value):continue
        if isinstance(value,(str,int,float,bool,type(None))):values[name]=value
        elif hasattr(value,'name'):values[name]=getattr(value,'name',str(value))
    return values


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    prep=load('prep',repo/'tasks/casmi_prepare.py');staged=load('staged',repo/'tasks/casmi_staged.py')
    env=prep.kaggle_environment(state,dict(os.environ));env['PYTHONIOENCODING']='utf-8'
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    root=state/'artifacts/casmi26/kaggle-r06-v1';verified=root/'verified-output-v2/submission.csv'
    if not verified.is_file():raise FileNotFoundError('Exact verified R06 CSV is missing')
    data=staged.find_dataset(state/'data/external')
    sample_ids=[];sample=data/'sample_submission.csv'
    if sample.is_file():
        with sample.open(encoding='utf-8-sig',newline='') as f:sample_ids=[r['molecule_id'] for r in csv.DictReader(f)]
    import pyarrow.parquet as pq
    test_ids=[]
    for batch in pq.ParquetFile(data/'test.parquet').iter_batches(columns=['molecule_id'],batch_size=8192):
        test_ids.extend(str(v) for v in batch.column(0).to_pylist())
    test_unique=list(dict.fromkeys(test_ids));audit=audit_submission(verified,expected_ids=sample_ids or test_unique)
    audit['test_id_set_matches']=set(test_unique)==set(sample_ids) if sample_ids else None
    audit['test_unique_ids']=len(test_unique);audit['sample_ids']=len(sample_ids)
    schema=pq.ParquetFile(data/'test.parquet').schema_arrow;local_metric=None
    if 'normalized_smiles' in schema.names:
        table=pq.read_table(data/'test.parquet',columns=['molecule_id','normalized_smiles']);targets={}
        for cid,smi in zip(table['molecule_id'].to_pylist(),table['normalized_smiles'].to_pylist()):
            if smi is not None:targets.setdefault(str(cid),str(smi))
        if targets and set(targets)==set(test_unique):
            with verified.open(encoding='utf-8-sig',newline='') as f:pred={r['molecule_id']:r['smiles'] for r in csv.DictReader(f)}
            from casmi26.metric import mrr_at_25
            local_metric={k:v for k,v in mrr_at_25(targets,pred).items() if k!='ranks'}
    def cli(label,args):
        r=subprocess.run([str(py),'-c','from kaggle.cli import main;main()']+args,env=env,stdin=subprocess.DEVNULL,
                         capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
        text=prep.redact(r.stdout+'\n'+r.stderr,env);(out/(label+'.log')).write_text(text,encoding='utf-8')
        return {'exit_code':r.returncode,'text':text[-12000:]}
    history=cli('history',['competitions','submissions',SLUG,'--format','json'])
    rows=json.loads(history['text'].strip()) if history['exit_code']==0 and history['text'].strip().startswith('[') else []
    match=[r for r in rows if int(r.get('ref',-1))==REF];row=match[0] if len(match)==1 else None
    verbose=cli('history_verbose',['competitions','submissions',SLUG,'-v'])
    # Current CLI 2.2.4 hides error_description, but the SDK response already has it.
    api_info={'methods':[],'listed_submission':None,'get_submission':None,'error_description':None,'error':None}
    try:
        old_token=os.environ.get('KAGGLE_API_TOKEN');os.environ['KAGGLE_API_TOKEN']=env['KAGGLE_API_TOKEN']
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api=KaggleApi();api.authenticate()
            api_info['methods']=[name for name,_ in inspect.getmembers(api,callable) if any(term in name.lower() for term in ('submission','score'))]
            if hasattr(api,'competition_submissions'):
                for obj in api.competition_submissions(SLUG):
                    values=scalar_fields(obj)
                    if str(values.get('ref',values.get('id','')))==str(REF):api_info['listed_submission']=values;break
            if hasattr(api,'get_submission'):
                obj=api.get_submission(REF);values=scalar_fields(obj);api_info['get_submission']=values
                api_info['error_description']=values.get('error_description') or values.get('errorDescription') or values.get('error')
        finally:
            if old_token is None:os.environ.pop('KAGGLE_API_TOKEN',None)
            else:os.environ['KAGGLE_API_TOKEN']=old_token
    except Exception as exc:api_info['error']=type(exc).__name__+': '+prep.redact(str(exc),env)[:1000]
    result={'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'submission_ref':REF,'description':DESCRIPTION,
            'new_submissions':0,'read_only':True,'history_row':row,'classification':classify_submission_row(row or {}),
            'csv_audit':audit,'local_example_metric':local_metric,'verbose_history':verbose,'api_info':api_info,
            'kaggle_scoring_error_interpretation':'Kaggle UI reports Submission Scoring Error; COMPLETE without public score is a failed/unscored result.'}
    (out/'r06-scoring-diagnosis.json').write_text(json.dumps(result,indent=2,default=str)+'\n',encoding='utf-8')
    print('R06_SCORING_DIAG_BEGIN\n'+json.dumps(result,indent=2,default=str)+'\nR06_SCORING_DIAG_END',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())

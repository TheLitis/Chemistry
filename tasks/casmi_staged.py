"""Inspect the explicitly staged official data; no rule acceptance or submissions."""
from collections import Counter
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

NAMES = ('train.parquet', 'test.parquet', 'sample_submission.csv')


def find_dataset(root: Path) -> Path:
    root = Path(root).resolve()
    folders = {root} | {p.parent for p in root.rglob('test.parquet') if not p.is_symlink()}
    complete = [p for p in folders if all((p/n).is_file() and not (p/n).is_symlink() and
                                         (p/n).stat().st_size > 0 for n in NAMES)]
    if len(complete) > 1:
        raise ValueError('Multiple complete datasets; provide a unique input folder')
    if not complete:
        raise FileNotFoundError('No complete train/test/template dataset under '+str(root))
    return complete[0]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    state = Path(os.environ['CHEMISTRY_STATE_ROOT'])
    py = state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve() != py.resolve():
        return subprocess.call([str(py), str(Path(__file__).resolve())],
                               env={**os.environ, 'PYTHONUTF8':'1', 'PYTHONIOENCODING':'utf-8'})
    import pyarrow.parquet as pq
    repo = Path(__file__).resolve().parents[1]
    out = Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']); out.mkdir(parents=True, exist_ok=True)
    data = find_dataset(state/'data/external')
    report = {'data_directory':str(data), 'request_id':os.environ.get('CHEMISTRY_REQUEST_ID'),
              'official_score':None, 'tables':[], 'submission_made':False}
    for name in NAMES[:2]:
        path = data/name
        with pq.ParquetFile(path) as table:
            item = {'name':name, 'path':str(path), 'bytes':path.stat().st_size,
                    'rows':table.metadata.num_rows, 'schema':str(table.schema_arrow)}
            columns = [k for k in ('molecule_id','compound_id','adduct','source','instrument_type')
                       if k in table.schema_arrow.names]
            counts = {k:Counter() for k in columns}
            for batch in table.iter_batches(columns=columns, batch_size=65536):
                for k in columns: counts[k].update(map(str,batch.column(k).to_pylist()))
            item['summary'] = {k:{'unique':len(v), 'top':v.most_common(10)} for k,v in counts.items()}
            example = next(table.iter_batches(batch_size=1)).to_pylist()[0]
            item['example'] = {k:({'length':len(v),'first':v[:3]} if isinstance(v,list) else v)
                               for k,v in example.items() if 'smiles' not in k and 'inchi' not in k}
            report['tables'].append(item)
    with (data/NAMES[2]).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    report['template'] = {'rows':len(rows), 'columns':list(rows[0]) if rows else [], 'examples':rows[:2]}
    prepare=load('prepare',repo/'tasks/casmi_prepare.py')
    access=load('access',repo/'tasks/casmi_access_check.py')
    candidates,warnings=access.windows_candidates(Path(r'C:\Users\loval'))
    env,kind=access.select_environment(dict(os.environ),candidates)
    env=prepare.kaggle_environment(state,env)
    result=prepare.run([str(py),'-c','from kaggle.cli import main;main()',
                        'competitions','files',prepare.SLUG,'--page-size','200','-v'],
                       env=env,timeout=60,log=out/'access.log',expose=False)
    report['kaggle']={'listing_succeeded':result['exit_code']==0,'source_kind':kind,
                      'warnings':warnings,'secret_values_logged':False}
    report['prior_results']=[]
    for p in (state/'artifacts/casmi26').glob('**/*.validation.json'):
        d=json.loads(p.read_text(encoding='utf-8-sig'))
        report['prior_results'].append({'path':str(p),**{k:v for k,v in d.items() if k in
            ('counts','validation','mass_only_baseline','ambiguous_mass_validation','ambiguous_mass_baseline')}})
    artifacts=state/'artifacts/casmi26';artifacts.mkdir(parents=True,exist_ok=True)
    text=json.dumps(report,indent=2,ensure_ascii=True,default=str)
    for p in (out/'staged-data.json', artifacts/'staged-data.json'):
        p.write_text(text+'\n',encoding='utf-8')
    print('STAGED_REPORT_BEGIN\n'+text+'\nSTAGED_REPORT_END',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())

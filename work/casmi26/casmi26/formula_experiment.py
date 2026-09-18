"""Fixed, exploratory subformula study on previously evaluated R08B cases.

No fitting, hidden-test reads, model changes, or leaderboard-driven selection.
The base scores and full candidate lists are held constant. FIORA is not rerun;
this compares a separate elemental-composition feature against full R07, NOT
an incomplete approximation of R08B plus a new feature.
"""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from . import formula_evidence as fe
from .ranking import paired_effect

SETTINGS={'weight':.25,'ppm':5.,'da':.001,'relative_floor':.01,'max_states':2_000_000}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,data):
    text=json.dumps(data,indent=2,allow_nan=False)+'\n'
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(text,encoding='utf-8');temporary.replace(path)


def rank(scores,keys,truth):
    for position,i in enumerate(np.argsort(-np.asarray(scores),kind='stable')[:25],1):
        if keys[int(i)]==truth:return position
    return 0


def evaluate_record(record):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')
    started=time.monotonic();smiles_to_formula={};formula_features={};counts=Counter()
    for case in record['cases'].values():
        keys=case['keys'];smiles=case['smiles'];scores=np.asarray(case['scores'],dtype='f8')
        if len(keys)!=len(smiles) or scores.shape!=(len(keys),) or not np.isfinite(scores).all() or len(keys)!=len(set(keys)):
            raise ValueError('Malformed fixed candidate list')
        for smi in smiles:
            if smi in smiles_to_formula:continue
            try:formula=fe.composition(smi)
            except ValueError:formula=None;counts['unsupported_structures']+=1
            smiles_to_formula[smi]=formula
            if formula is not None and formula not in formula_features:
                evidence=fe.score_formula(record['queries'],formula,**{k:v for k,v in SETTINGS.items() if k!='weight'})
                formula_features[formula]=evidence;counts[evidence['status']]+=1
    details={};coverage={}
    for name,case in record['cases'].items():
        base=np.asarray(case['scores'],dtype='f8');keys=case['keys'];features=[]
        for smi in case['smiles']:
            evidence=formula_features.get(smiles_to_formula[smi],{})
            features.append([evidence.get(k) or 0. for k in ('explained','excess')])
        evidence=np.asarray(features,dtype='f8').reshape(-1,2)
        details[name]={'baseline':rank(base,keys,record['key']),
                       'explained':rank(base+SETTINGS['weight']*evidence[:,0],keys,record['key']),
                       'excess':rank(base+SETTINGS['weight']*evidence[:,1],keys,record['key'])}
        coverage[name]=record['key'] in keys
    formulas=sorted(formula_features);index={f:i for i,f in enumerate(formulas)}
    return {'key':record['key'],'cohort':record['cohort'],'ranks':details,'correct_candidate_coverage':coverage,
        'candidate_counts':{k:len(v['keys']) for k,v in record['cases'].items()},
        'statistics':dict(counts),'seconds':time.monotonic()-started,
        'formulas':[{'composition':f,**formula_features[f]} for f in formulas],
        'smiles_to_formula_index':{s:index.get(f) for s,f in smiles_to_formula.items()}}


def metrics(ranks):
    r=np.asarray(ranks);rr=np.where(r>0,1/np.maximum(r,1),0.)
    return {'molecules':len(r),'mrr_at_25':float(rr.mean()),'top1':float((r==1).mean()),'recall_at_25':float((r>0).mean())}


def run_study(source,output,*,workers=4):
    import rdkit
    source=Path(source);output=Path(output)
    if type(workers) is not int or not 1<=workers<=8:raise ValueError('Invalid worker count')
    output.mkdir(parents=True,exist_ok=True)
    identity={'experiment':'R09-subformula-diagnostic-v1','source_sha256':digest(source),'settings':SETTINGS,
        'rdkit':rdkit.__version__,'code_sha256':digest(__file__),'feature_code_sha256':digest(fe.__file__),
        'quality_claim':'exploratory_reused_cohorts','parameter_selection_performed':False,
        'no_answer_formulas_passed_to_feature':True}
    protocol=output/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=identity:
        raise ValueError('Experiment identity changed; use a new output folder')
    report_path=output/'report.json'
    if report_path.exists():
        result=json.loads(report_path.read_text())
        if digest(output/'results.json.gz')!=result['results_sha256']:raise ValueError('Result artifact changed')
        return result
    records=json.loads(gzip.decompress(source.read_bytes()))
    if not records:raise ValueError('Empty research input')
    if len({r['key'] for r in records})!=len(records):raise ValueError('Duplicate query key')
    dump(protocol,identity)
    started=time.monotonic();results=[]
    if workers==1:
        iterator=map(evaluate_record,records)
        pool=None
    else:
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
        iterator=pool.map(evaluate_record,records,chunksize=1)
    try:
        with (output/'progress.jsonl').open('w',encoding='utf-8') as stream:
            for result in iterator:
                results.append(result);stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
                if len(results)%16==0:print('FORMULA_PROGRESS '+str(len(results))+'/'+str(len(records)),flush=True)
    finally:
        if pool is not None:pool.shutdown()
    payload=gzip.compress(json.dumps(results,allow_nan=False,separators=(',',':')).encode(),mtime=0)
    (output/'results.json.gz').write_bytes(payload)
    groups={}
    for cohort in sorted({r['cohort'] for r in results}):
        rows=[r for r in results if r['cohort']==cohort];group={}
        for case in rows[0]['ranks']:
            group[case]={method:metrics([r['ranks'][case][method] for r in rows]) for method in ('baseline','explained','excess')}
            rr=lambda method: np.array([1/r['ranks'][case][method] if r['ranks'][case][method]>0 else 0. for r in rows])
            for method in ('explained','excess'):
                group[case][method]['paired_vs_baseline']=paired_effect(rr('baseline'),rr(method))
            group[case]['true_candidate_coverage']=float(np.mean([r['correct_candidate_coverage'][case] for r in rows]))
        groups[cohort]=group
    counts=Counter()
    for r in results:counts.update(r['statistics'])
    report={**identity,'status':'completed','groups':groups,'query_count':len(results),'statistics':dict(counts),
        'formula_query_pairs':sum(len(r['formulas']) for r in results),'seconds':time.monotonic()-started,
        'results_sha256':digest(output/'results.json.gz'),'new_training':False,'new_submissions':0,
        'official_score':None,'production_scoring_changed':False,
        'limitations':['Previously spent R08B cases: this is exploratory, not an untouched validation set.',
            'Pure atom budgets cannot distinguish isomers sharing an elemental composition.',
            'Formula mass enumeration is permissive and does not establish fragment connectivity or chemical validity.',
            'Shifted mass controls are arithmetic controls, not chemical decoys or calibrated probabilities.',
            'Only neutral unlabeled CHNOPS/halogen candidates and [M+H]+/[M-H]- evidence are supported.',
            'No exact-MS1 isotope envelope, answer formula, full molecular generation, or SIRIUS code is used.',
            'No promotion or new Kaggle attempt follows from this diagnostic.']}
    dump(report_path,report);return report

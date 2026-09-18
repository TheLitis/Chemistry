"""R10: connected-fragment reranking of COMPLETE R08B top-25 lists.

This is an exploratory screen on spent cohorts, NOT a new validation. No
missing FIORA score is silently replaced by zero. Old certificates first prove
all retained scores exact; only their order changes, preserving Recall@25.
"""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time
import numpy as np
from . import connected_evidence as ce
from .formula_evidence import composition
from .ranking import paired_effect

SETTINGS={'weight':.25,'max_cuts':2,'max_scenarios':8192,'hydrogen_shift':1,'ppm':5.,'da':.001,'relative_floor':.01}
METHODS=('r08b','one_cut_global','two_cut_global','two_cut_isomer')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)


def fixed_top(case,certificate):
    keys,smiles,base=case['keys'],case['smiles'],case['scores'];c=certificate;n=len(keys)
    lo,mask,top,w=c['lower_scores'],c['evaluated_mask'],c['top_indices'],c['weight']
    if (c.get('certified') is not True or c.get('base_scores')!=base or len(base)!=n or len(smiles)!=n or
        len(set(keys))!=n or len(lo)!=n or len(mask)!=n or len(top)!=min(25,n) or len(set(top))!=len(top)):
        raise ValueError('Invalid complete-R08B certificate or input alignment')
    if any(type(i) is not int or not 0<=i<n for i in top) or any(type(v) is not bool for v in mask):
        raise ValueError('Invalid certificate indices/mask')
    if w not in (0.,.25) or not all(math.isfinite(v) for v in base+lo):raise ValueError('Non-finite or changed R08B scoring')
    if c.get('evaluations')!=sum(mask):raise ValueError('Certificate count mismatch')
    for i in range(n):
        if mask[i]:
            if not base[i]-1e-12<=lo[i]<=base[i]+w+1e-12:raise ValueError('Score outside certificate bound')
        elif lo[i]!=base[i]:raise ValueError('Changed uncomputed entry')
    if top!=sorted(range(n),key=lambda i:(-lo[i],i))[:25]:raise ValueError('Incorrect top25 certificate')
    if w and n:
        if not all(mask[i] for i in top):raise ValueError('Unknown top25 score')
        cutoff=lo[top[-1]]
        if any(not mask[i] and math.nextafter(base[i]+w,math.inf)>=cutoff for i in range(n)):
            raise ValueError('An uncomputed score could enter top25')
    return {'keys':[keys[i] for i in top],'smiles':[smiles[i] for i in top],
            'scores':[lo[i] for i in top],'original_candidate_count':n,'source_indices':top}


def reorder(scores,features,formulas,*,isomer_only=False):
    base=np.asarray(scores,dtype='f8');values=np.array([0. if v is None else v for v in features],dtype='f8')
    if (base.ndim!=1 or values.shape!=base.shape or len(formulas)!=len(base) or
        not np.isfinite(base).all() or not np.isfinite(values).all() or np.any((values<0)|(values>1))):
        raise ValueError('Invalid candidate evidence')
    final=base+SETTINGS['weight']*values
    if not isomer_only:return np.argsort(-final,kind='stable').tolist()
    groups={};order=list(range(len(base)))
    for i,f in enumerate(formulas):groups.setdefault(f if f is not None else ('unsupported',i),[]).append(i)
    for ids in groups.values():
        ranked=sorted(ids,key=lambda i:(-final[i],i))
        for position,i in zip(ids,ranked):order[position]=i
    return order


def _rank(keys,order,truth):
    return next((j for j,i in enumerate(order,1) if keys[i]==truth),0)


def evaluate_record(record):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error');started=time.monotonic();features={};formulas={};counts=Counter();cases={}
    for case in record['cases'].values():
        for smiles in case['smiles']:
            if smiles in features:continue
            features[smiles]=ce.score_structure(record['queries'],smiles,**{k:v for k,v in SETTINGS.items() if k!='weight'})
            counts[features[smiles]['status']]+=1
            try:formulas[smiles]=composition(smiles)
            except ValueError:formulas[smiles]=None
    for name,case in record['cases'].items():
        keys,smiles,scores=case['keys'],case['smiles'],case['scores'];n=len(keys)
        one=[features[s]['one_cut'] for s in smiles];two=[features[s]['two_cut'] for s in smiles];formula=[formulas[s] for s in smiles]
        orders={'r08b':list(range(n)),'one_cut_global':reorder(scores,one,formula),
                'two_cut_global':reorder(scores,two,formula),'two_cut_isomer':reorder(scores,two,formula,isomer_only=True)}
        ranks={m:_rank(keys,order,record['key']) for m,order in orders.items()}
        if any(set(order)!=set(range(n)) for order in orders.values()) or len(set(r>0 for r in ranks.values()))!=1:
            raise RuntimeError('Top25 set or query recall changed')
        if [formula[i] for i in orders['two_cut_isomer']]!=formula:raise RuntimeError('Formula slots changed')
        truth_index=keys.index(record['key']) if record['key'] in keys else None
        same_formula=(sum(f==formula[truth_index] for f in formula) if truth_index is not None and formula[truth_index] is not None else None)
        cases[name]={**case,'ranks':ranks,'orders':orders,'one_cut':one,'two_cut':two,'formulas':formula,
            'diagnostic_same_formula_competitors':same_formula}
    return {'key':record['key'],'cohort':record['cohort'],'cases':cases,'features':features,
            'statistics':dict(counts),'seconds':time.monotonic()-started}


def _metrics(r):
    a=np.asarray(r);return {'molecules':len(r),'mrr_at_25':float(np.where(a>0,1/np.maximum(a,1),0).mean()),
        'top1':float((a==1).mean()),'recall_at_25':float((a>0).mean())}


def run_study(source,certificates,output,*,workers=4):
    import rdkit
    if type(workers) is not int or not 1<=workers<=8:raise ValueError('Invalid workers')
    source,certificates,output=map(Path,(source,certificates,output));output.mkdir(parents=True,exist_ok=True)
    identity={'experiment':'R10-connected-fragment-screen-v1','source_sha256':digest(source),
        'certificates_sha256':digest(certificates),'rdkit':rdkit.__version__,'settings':SETTINGS,
        'baseline':'complete_R08B_certified_top25','quality_claim':'exploratory_reused_cohorts',
        'code_sha256':digest(__file__),'feature_code_sha256':digest(ce.__file__),
        'parameter_selection_performed':False,'top25_membership_changed':False}
    protocol=output/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=identity:raise ValueError('Experiment identity changed')
    if (output/'report.json').exists():
        report=json.loads((output/'report.json').read_text())
        if digest(output/'results.json.gz')!=report['results_sha256']:raise ValueError('Result changed')
        return report
    records=json.loads(gzip.decompress(source.read_bytes()));certs=json.loads(gzip.decompress(certificates.read_bytes()))
    if not records or len({r['key'] for r in records})!=len(records):raise ValueError('Missing or repeated queries')
    lookup={(c['query_id'],c['case']):c for c in certs};required={(r['key'],name) for r in records for name in r['cases']}
    if len(lookup)!=len(certs) or set(lookup)!=required:raise ValueError('Incomplete or duplicate certificates')
    prepared=[]
    for r in records:
        prepared.append({'key':r['key'],'cohort':r['cohort'],'queries':r['queries'],
            'cases':{name:fixed_top(case,lookup[(r['key'],name)]) for name,case in r['cases'].items()}})
    dump(protocol,identity);started=time.monotonic();rows=[]
    pool=None if workers==1 else ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
    iterator=map(evaluate_record,prepared) if pool is None else pool.map(evaluate_record,prepared,chunksize=1)
    try:
        with (output/'progress.jsonl').open('w',encoding='utf-8') as stream:
            for row in iterator:
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if len(rows)%16==0:print('CONNECTED_PROGRESS '+str(len(rows))+'/'+str(len(records)),flush=True)
    finally:
        if pool is not None:pool.shutdown()
    groups={};counts=Counter()
    for row in rows:counts.update(row['statistics'])
    for cohort in sorted({r['cohort'] for r in rows}):
        subset=[r for r in rows if r['cohort']==cohort];group={}
        for case in subset[0]['cases']:
            ranks={m:[r['cases'][case]['ranks'][m] for r in subset] for m in METHODS}
            result={m:_metrics(v) for m,v in ranks.items()}
            rr=lambda v:np.array([1/r if r else 0. for r in v])
            for m in METHODS[1:]:
                result[m]['paired_vs_r08b']=paired_effect(rr(ranks['r08b']),rr(ranks[m]))
                result[m]['top1_gained']=sum(a!=1 and b==1 for a,b in zip(ranks['r08b'],ranks[m]))
                result[m]['top1_lost']=sum(a==1 and b!=1 for a,b in zip(ranks['r08b'],ranks[m]))
            group[case]=result
        groups[cohort]=group
    payload=gzip.compress(json.dumps(rows,allow_nan=False,separators=(',',':')).encode(),mtime=0)
    (output/'results.json.gz').write_bytes(payload)
    report={**identity,'status':'completed','query_count':len(rows),'groups':groups,'statistics':dict(counts),
        'structure_query_pairs':sum(len(r['features']) for r in rows),
        'top25_cases':sum(len(r['cases']) for r in rows),'recall_preserved_for_every_query':True,
        'seconds':time.monotonic()-started,'results_sha256':digest(output/'results.json.gz'),
        'new_training':False,'new_submissions':0,'production_scoring_changed':False,'official_score':None,
        'limitations':['Spent R08B cohorts: descriptive screen, not an untouched quality estimate.',
            'Only reorder the exact existing R08B top25: cannot recover candidates ranked below 25 or absent from catalogs.',
            'Connected components after at most two cuts with bounded H shifts are hypotheses, not a mechanistic fragmentation simulation.',
            'Cut and H-shift penalties are fixed heuristics, not bond dissociation energies or learned probabilities.',
            'Only protonated/deprotonated evidence and neutral, unlabeled supported-element candidates are handled.',
            'FIORA pretraining membership remains unresolved for the baseline.',
            'Several fixed variants are reported; paired intervals are descriptive and not multiple-comparison adjusted.',
            'The candidate truth key is used only for evaluation and explicitly labelled formula-group diagnostics.']}
    dump(output/'report.json',report);return report

"""Independently verify fixed R08B cohorts, rank certificates, effects and gate.

Never imports the production ranker, predictor, or experiment's gate function.
Certificates prove numerical ranking under the fixed score, not correct chemistry.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from casmi_r08_exact_verify import check_certificate

COHORTS=('fresh_timsTOF','external_covered_mixed')
CASES=('available','absent','external_recovery')
GATES={'target_absent_delta_lower95_gt':0.,'external_recovery_delta_lower95_gt':0.,
       'external_available_delta_lower95_gt':-.03,'external_recovery_coverage_gte':.8}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def metric(ranks):
    if not ranks or any(type(r) is not int or not 0<=r<=25 for r in ranks):raise ValueError('Invalid ranks')
    n=len(ranks)
    return {'molecules':n,'mrr_at_25':sum(1/r for r in ranks if r)/n,
            'top1':sum(r==1 for r in ranks)/n,'recall_at_25':sum(r>0 for r in ranks)/n}


def effect(base,changed):
    if len(base)!=len(changed) or not base:raise ValueError('Unpaired comparison')
    delta=np.array([(1/b if b else 0)-(1/a if a else 0) for a,b in zip(base,changed)])
    rng=np.random.default_rng(26091603)
    means=[float(delta[rng.integers(len(delta),size=len(delta))].mean()) for _ in range(2000)]
    return {'delta_mrr':float(delta.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}


def near(a,b,label):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or abs(a-b)>1e-12:
        raise ValueError('Numerical mismatch: '+label)


def verify(root,*,previous_keys=(),expected_sizes=(128,96)):
    root=Path(root)
    def read(name):
        p=root/name
        if name.endswith('.gz'):
            with gzip.open(p,'rt',encoding='utf-8') as stream:return json.load(stream)
        return json.loads(p.read_text(encoding='utf-8-sig'))
    report,plan=read('report.json'),read('protocol.json')
    required={'protocol.json','evidence.json.gz','certificates.json.gz','features.json.gz','audit-ranks.json'}
    if not required.issubset(report.get('files',{})):raise ValueError('Incomplete integrity manifest')
    for name,h in report['files'].items():
        if Path(name).name!=name or digest(root/name)!=h:raise ValueError('Hash mismatch '+name)
    if plan['frozen_selection']!={'feature':'cosine_nearest','weight':.25} or plan['new_weight_search'] is not False:
        raise ValueError('Fixed score changed')
    if plan['gates']!=GATES:raise ValueError('Prospective gate changed')
    target,external=plan['target_keys'],plan['external_keys'];allkeys=target+external
    if len(target)!=expected_sizes[0] or len(external)!=expected_sizes[1]:raise ValueError('Prespecified cohort size changed')
    if len(set(allkeys))!=len(allkeys):raise ValueError('Cohort key overlap')
    if set(allkeys)&set(previous_keys):raise ValueError('Queries overlap previous evaluations')
    if any(int.from_bytes(hashlib.sha256(('official-split-v1:'+k).encode()).digest()[:8],'big')%10!=0 for k in allkeys):
        raise ValueError('Query not in model hash holdout')
    if plan['training_query_overlap']!=0 or report['fitting_overlap']!=0 or report['previous_query_keys_reused']!=0:
        raise ValueError('Reported fitting or previous overlap')
    records,raw,certs,features=[read(n) for n in ('evidence.json.gz','audit-ranks.json','certificates.json.gz','features.json.gz')]
    records_by={r['key']:r for r in records};raw_by={r['key']:r for r in raw}
    if len(records_by)!=len(records) or len(raw_by)!=len(raw):raise ValueError('Duplicate input rows')
    if set(records_by)!=set(allkeys) or set(raw_by)!=set(allkeys) or set(features)!=set(allkeys):raise ValueError('Query identity mismatch')
    targetset=set(target)
    for key,r in records_by.items():
        cohort=COHORTS[0] if key in targetset else COHORTS[1]
        if r['cohort']!=cohort or raw_by[key]['cohort']!=cohort:raise ValueError('Cohort reassignment')
        if set(r['cases'])!=set(CASES) or set(raw_by[key]['ranks'])!=set(CASES):raise ValueError('Missing case')
    seen=set();pairs=evaluated=0
    for c in certs:
        key,case=c['query_id'],c['case'];pair=(key,case)
        if pair in seen or key not in records_by or case not in CASES:raise ValueError('Repeated/unknown certificate')
        seen.add(pair);r=records_by[key];data=r['cases'][case];keys,base=data['keys'],data['scores']
        if len(keys)!=len(base) or len(keys)!=len(set(keys)):raise ValueError('Candidate alignment/duplicates')
        if c['base_scores']!=base:raise ValueError('Certificate not bound to original R07 scores')
        supported=any(q['adduct'] in ('[M+H]+','[M-H]-') for q in r['queries'])
        if c['weight']!=(.25 if supported else 0.):raise ValueError('Unsupported-adduct handling changed')
        check_certificate(c,k=25)
        for i,done in enumerate(c['evaluated_mask']):
            if done:
                if keys[i] not in features[key]:raise ValueError('Missing forward evidence')
                value=features[key][keys[i]]
                if value is not None and (type(value) not in (float,int) or not math.isfinite(value) or not 0<=value<=1):
                    raise ValueError('Invalid forward evidence')
                near(base[i]+c['weight']*(0. if value is None else value),c['lower_scores'][i],'computed evidence')
        def rank(order):
            names=[keys[i] for i in order[:25]]
            return names.index(key)+1 if key in names else 0
        original=sorted(range(len(base)),key=lambda i:(-base[i],i))
        row=raw_by[key]['ranks'][case]
        if row['r07']!=rank(original):raise ValueError('Wrong baseline rank')
        if row['r08']!=rank(c['top_indices']):raise ValueError('Wrong forward rank')
        if type(row['coverage']) is not bool or row['coverage']!=(key in keys):raise ValueError('Wrong coverage')
        if row['candidate_count']!=len(keys) or row['evaluated']!=c['evaluations']:raise ValueError('Wrong per-query counts')
        pairs+=len(keys);evaluated+=c['evaluations']
    if seen!={(key,case) for key in allkeys for case in CASES} or report['certificates']!=len(certs):
        raise ValueError('Missing certificates')
    computed={}
    for cohort in COHORTS:
        rows=[r for r in raw if r['cohort']==cohort];computed[cohort]={}
        for case in CASES:
            a=[r['ranks'][case]['r07'] for r in rows];b=[r['ranks'][case]['r08'] for r in rows]
            measured={'r07':metric(a),'r08':metric(b),'paired':effect(a,b),
                      'coverage':float(np.mean([r['ranks'][case]['coverage'] for r in rows]))}
            saved=report['cohorts'][cohort][case]
            for method in ('r07','r08'):
                for name,value in measured[method].items():near(value,saved[method][name],cohort+'.'+case+'.'+method+'.'+name)
            near(measured['coverage'],saved['coverage'],'coverage')
            near(measured['paired']['delta_mrr'],saved['paired']['delta_mrr'],'paired effect')
            for x,y in zip(measured['paired']['ci95'],saved['paired']['ci95']):near(x,y,'paired interval')
            computed[cohort][case]=measured
    checks={'fresh_target_absent_improves':computed[COHORTS[0]]['absent']['paired']['ci95'][0]>0,
        'fresh_external_recovery_improves':computed[COHORTS[1]]['external_recovery']['paired']['ci95'][0]>0,
        'reference_noninferiority_0_03':computed[COHORTS[1]]['available']['paired']['ci95'][0]>-.03,
        'external_mass_window_coverage_at_least_0_8':computed[COHORTS[1]]['external_recovery']['coverage']>=.8}
    decision={'eligible':all(checks.values()),'checks':checks}
    if report['decision']['eligible'] is not decision['eligible']:raise ValueError('Incorrect promotion decision')
    return {'status':'verified','queries':len(allkeys),'certificates':len(certs),
        'candidate_pairs':pairs,'forward_evaluations':evaluated,'certified_unnecessary_evaluations':pairs-evaluated,
        'all_rankings_recomputed':True,'paired_intervals_recomputed':True,'cohorts':computed,'decision':decision,
        'provided_previous_keys_checked':len(set(previous_keys)),'hash_holdout_verified':True,
        'production_ranker_imported':False,'certifies_chemical_correctness':False,
        'fiora_pretraining_disjointness_proven':False,'tolerance':1e-12}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--previous-keys',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    previous=json.loads(args.previous_keys.read_text()) if args.previous_keys else []
    report=verify(args.root,previous_keys=previous);text=json.dumps(report,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    print(text)

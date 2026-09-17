"""Independent numerical audit of sealed R07 + forward-evidence results.

Uses saved scores, features, protocol and ranks, never the production ranker.
No learning, candidate generation, hidden-test reads or Kaggle writes.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,math
from pathlib import Path
import numpy as np


def close(expected,actual,label,tol=1e-12):
    if expected is None or actual is None:
        if expected is not actual:raise ValueError('Mismatch: '+label)
    elif not math.isfinite(float(expected)) or not math.isfinite(float(actual)) or abs(float(expected)-float(actual))>tol:
        raise ValueError('Mismatch: '+label)


def metrics(ranks):
    if not ranks:return {'molecules':0,'mrr_at_25':None,'top1':None,'recall_at_25':None}
    if any(type(r) is not int or not 0<=r<=25 for r in ranks):raise ValueError('Invalid rank')
    n=len(ranks)
    return {'molecules':n,'mrr_at_25':sum(1./r for r in ranks if r)/n,
            'top1':sum(r==1 for r in ranks)/n,'recall_at_25':sum(r>0 for r in ranks)/n}


def paired(base,changed):
    if not base or len(base)!=len(changed):raise ValueError('Unpaired ranks')
    d=np.array([(1./b if b else 0.)-(1./a if a else 0.) for a,b in zip(base,changed)])
    rng=np.random.default_rng(26091603)
    means=[float(d[rng.integers(len(d),size=len(d))].mean()) for _ in range(2000)]
    return {'delta_mrr':float(d.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),
            'bootstrap_repeats':2000,'bootstrap_unit':'molecule'}


def rank_case(case,features,name,weight,limit,truth):
    keys=case['keys'];scores=[float(v) for v in case['scores']]
    if len(keys)!=len(scores) or len(set(keys))!=len(keys) or not all(math.isfinite(v) for v in scores):
        raise ValueError('Malformed candidate scores')
    if type(limit) is not int or limit<1 or not math.isfinite(weight) or weight<0:raise ValueError('Bad configuration')
    original=sorted(range(len(keys)),key=lambda i:(-scores[i],i))
    for i in original[:limit]:
        value=features.get(keys[i],{}).get(name)
        if value is not None:
            v=float(value)
            if not math.isfinite(v) or not 0<=v<=1:raise ValueError('Malformed forward evidence')
            scores[i]+=weight*v
    order=sorted(range(len(keys)),key=lambda i:(-scores[i],i))[:25]
    for r,i in enumerate(order,1):
        if keys[i]==truth:return r
    return 0


def recompute(plan,records,features):
    groups=[plan[k] for k in ('calibration_keys','reused_np_audit_keys','fresh_transfer_keys')]
    allkeys=sum(groups,[]);bykey={r['key']:r for r in records}
    if len(set(allkeys))!=len(allkeys) or len(bykey)!=len(records) or set(bykey)!=set(allkeys):
        raise ValueError('Molecular partitions overlap or input records differ')
    if plan.get('fitting_overlap')!=0:raise ValueError('Study reports fitting overlap')
    names=plan['features'];weights=plan['forward_weights'];limit=plan['shortlist_per_regime']
    if not names or 0. not in weights or not groups[0]:raise ValueError('No fixed no-change baseline/calibration')
    cases=list(records[0]['cases'])
    if not cases or any(set(r['cases'])!=set(cases) for r in records):raise ValueError('Inconsistent regimes')
    def ranks(querykeys,name,w):
        return {k:{case:rank_case(bykey[k]['cases'][case],features.get(k,{}),name,w,limit,k)
                   for case in cases} for k in querykeys}
    baseline=ranks(groups[0],names[0],0.);bm={c:metrics([r[c] for r in baseline.values()]) for c in cases}
    grid=[]
    for name in names:
        for w in weights:
            if w==0. and grid:continue
            rs=ranks(groups[0],name,w);mm={c:metrics([r[c] for r in rs.values()]) for c in cases}
            gains={c:mm[c]['mrr_at_25']-bm[c]['mrr_at_25'] for c in cases}
            grid.append({'feature':name,'weight':w,'metrics':mm,'gains':gains})
    selected=max(grid,key=lambda r:(min(r['gains'].values()),float(np.mean(list(r['gains'].values()))),-r['weight']))
    base=ranks(groups[1]+groups[2],selected['feature'],0.)
    altered=ranks(groups[1]+groups[2],selected['feature'],selected['weight'])
    return selected,{k:{'baseline':base[k],'selected':altered[k]} for k in base}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify(root):
    root=Path(root);load=lambda n:json.loads((root/n).read_text(encoding='utf-8-sig'))
    plan=load('protocol.json');prepared=load('prepared.json');report=load('report.json')
    for name,h in prepared['files'].items():
        if Path(name).name!=name or digest(root/name)!=h:raise ValueError('Prepared input mismatch: '+name)
    with gzip.open(root/'evidence.json.gz','rt',encoding='utf-8') as f:records=json.load(f)
    with gzip.open(root/'forward-features.json.gz','rt',encoding='utf-8') as f:features=json.load(f)
    locked=load('selection-before-transfer-audit.json');raw=load('audit-ranks.json')
    for name,h in locked['input_hashes'].items():
        if Path(name).name!=name or digest(root/name)!=h:raise ValueError('Locked input mismatch: '+name)
    winner,ranks=recompute(plan,records,features)
    for actual,label in ((locked['selected'],'locked selection'),(report['selected'],'reported selection')):
        if actual['feature']!=winner['feature']:raise ValueError('Mismatch: '+label)
        close(winner['weight'],actual['weight'],label)
    for case,mm in winner['metrics'].items():
        for name,v in mm.items():close(v,locked['selected']['metrics'][case][name],'calibration.'+case+'.'+name)
        close(winner['gains'][case],locked['selected']['gains'][case],'calibration gain.'+case)
    if len(raw)!=len(ranks) or {r['key'] for r in raw}!=set(ranks):raise ValueError('Raw rank cohort differs')
    decisions=0
    for r in raw:
        for variant in ('baseline','selected'):
            if ranks[r['key']][variant]!=r[variant]:raise ValueError('Raw rank mismatch: '+r['key'])
            decisions+=len(r[variant])
    cohorts={}
    for label,part in [('reused_np_diagnostic',plan['reused_np_audit_keys']),('fresh_timsTOF_transfer',plan['fresh_transfer_keys'])]:
        result={}
        for case in report['cohorts'][label]:
            b=[ranks[k]['baseline'][case] for k in part];s=[ranks[k]['selected'][case] for k in part]
            result[case]={'baseline':metrics(b),'selected':metrics(s),'paired_effect':paired(b,s)}
            expected=report['cohorts'][label][case]
            for v in ('baseline','selected'):
                for n,x in result[case][v].items():close(x,expected[v][n],label+'.'+case+'.'+v+'.'+n)
            effect=result[case]['paired_effect'];exp=expected['paired_effect']
            close(effect['delta_mrr'],exp['delta_mrr'],label+'.'+case+'.delta')
            for a,b in zip(effect['ci95'],exp['ci95']):close(a,b,label+'.'+case+'.ci95')
        cohorts[label]=result
    return {'status':'verified','selected':{'feature':winner['feature'],'weight':winner['weight']},
            'calibration_queries':len(plan['calibration_keys']),'raw_audit_records':len(raw),
            'independently_recomputed_audit_rank_decisions':decisions,'cohorts':cohorts,
            'input_hashes':{n:digest(root/n) for n in ('protocol.json','prepared.json','report.json','audit-ranks.json',
                'evidence.json.gz','forward-features.json.gz','selection-before-transfer-audit.json')},
            'numerical_tolerance':1e-12,'production_rank_implementation_imported':False,
            'new_training':False,'new_submissions':0,'validates_forward_model_pretraining_disjointness':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();result=verify(a.root);text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text,encoding='utf-8')
    print(text,end='');return 0


if __name__=='__main__':raise SystemExit(main())

"""Independent certificate verification. No production ranker imports."""
import argparse,gzip,hashlib,json,math
from pathlib import Path


def check_certificate(c,k=25):
    b=c['base_scores'];lo=c['lower_scores'];mask=c['evaluated_mask'];w=c['weight'];idx=c['top_indices'];n=len(b)
    if len(lo)!=n or len(mask)!=n or len(idx)!=min(k,n) or len(set(idx))!=len(idx):raise ValueError('Alignment mismatch')
    if not all(type(i) is int and 0<=i<n for i in idx):raise ValueError('Invalid index')
    if not all(type(m) is bool for m in mask):raise ValueError('Invalid evaluated mask')
    if not math.isfinite(w) or w<0 or not all(math.isfinite(x) for x in b+lo):raise ValueError('Invalid scores')
    if c['certified'] is not True:raise ValueError('Not certified')
    for i in range(n):
        if not mask[i] and lo[i]!=b[i]:raise ValueError('Uncomputed entry was changed')
        if mask[i] and not b[i]-1e-12<=lo[i]<=b[i]+w+1e-12:raise ValueError('Evidence outside [0,1]')
    expected=sorted(range(n),key=lambda i:(-lo[i],i))[:k]
    if idx!=expected:raise ValueError('Wrong stable ranking')
    if w and n:
        if not all(mask[i] for i in idx):raise ValueError('Unknown candidate appears in certified ranking')
        cutoff=lo[idx[-1]]
        for i in range(n):
            if not mask[i] and math.nextafter(b[i]+w,math.inf)>=cutoff:raise ValueError('Uncomputed candidate can enter top-k')
    if c['evaluations']!=sum(mask):raise ValueError('Wrong evaluation count')
    return True


def verify(root,source):
    load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    report=load(root/'report.json')
    plan=load(root/'protocol.json')
    if plan['selected'] != {'feature':'cosine_nearest','weight':.25}:raise ValueError('Scoring configuration changed')
    if hashlib.sha256((source/'evidence.json.gz').read_bytes()).hexdigest()!=plan['candidate_data_sha256']:raise ValueError('Original R07 evidence differs')
    for name,h in report['files'].items():
        if Path(name).name!=name or hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('Hash mismatch '+name)
    cert=json.load(gzip.open(root/'certificates.json.gz','rt'));features=json.load(gzip.open(root/'features.json.gz','rt'))
    records=json.load(gzip.open(source/'evidence.json.gz','rt'));bykey={r['key']:r for r in records}
    oldfeatures=json.load(gzip.open(source/'forward-features.json.gz','rt'))
    ranks=load(root/'audit-ranks.json');lookup={r['key']:r for r in ranks};count=pairs=known=0
    if len(lookup)!=len(ranks) or set(lookup)!=set(bykey):raise ValueError('Query records differ')
    seen=set()
    for c in cert:
        q,case=c['query_id'],c['case'];pair=(q,case)
        if pair in seen:raise ValueError('Repeated certificate')
        seen.add(pair);data=bykey[q]['cases'][case]
        if c['base_scores']!=data['scores']:raise ValueError('Not the original R07 scores')
        expected_weight=.25 if any(observation['adduct'] in ('[M+H]+','[M-H]-') for observation in bykey[q]['queries']) else 0.
        if c['weight']!=expected_weight:raise ValueError('Per-query scoring weight changed')
        check_certificate(c)
        for i,done in enumerate(c['evaluated_mask']):
            if done:
                key=data['keys'][i]
                if key not in features[q]:raise ValueError('Computed evidence absent from trace')
                bonus=features[q][key];expected=data['scores'][i]+c['weight']*(bonus or 0.)
                if abs(expected-c['lower_scores'][i])>1e-12:raise ValueError('Computed score disagrees with exported evidence')
        top=[data['keys'][i] for i in c['top_indices']]
        r=top.index(q)+1 if q in top else 0
        if lookup[q]['ranks'][case]['r08_exact']!=r:raise ValueError('Wrong reciprocal rank')
        base=list(data['scores']);original=sorted(range(len(base)),key=lambda i:(-base[i],i))
        def true_rank(order):
            names=[data['keys'][i] for i in order[:25]]
            return names.index(q)+1 if q in names else 0
        if lookup[q]['ranks'][case]['r07']!=true_rank(original):raise ValueError('Baseline rank differs')
        for i in original[:32]:
            value=oldfeatures.get(q,{}).get(data['keys'][i],{}).get('cosine_nearest')
            if value is not None:base[i]+=.25*value
        capped=sorted(range(len(base)),key=lambda i:(-base[i],i))
        if lookup[q]['ranks'][case]['r08_capped32']!=true_rank(capped):raise ValueError('Capped comparison rank differs')
        pairs+=len(data['keys']);known+=c['evaluations'];count+=1
    if seen!={(r['key'],c) for r in records for c in r['cases']}:raise ValueError('Incomplete certificates')
    if pairs!=report['candidate_pairs_across_regimes'] or known!=report['evaluated_pairs_across_regimes']:raise ValueError('Count mismatch')
    import numpy as np
    def paired(a,b):
        d=np.array([(1./y if y else 0.)-(1./x if x else 0.) for x,y in zip(a,b)])
        rng=np.random.default_rng(26091603)
        values=[float(d[rng.integers(len(d),size=len(d))].mean()) for _ in range(2000)]
        return {'delta_mrr':float(d.mean()),'ci95':np.quantile(values,[.025,.975]).tolist()}
    mm={}
    for cohort in report['comparison']:
        rows=[r for r in ranks if r['cohort']==cohort];mm[cohort]={}
        for case in report['comparison'][cohort]:
            mm[cohort][case]={}
            for version in ('r07','r08_capped32','r08_exact'):
                a=[r['ranks'][case][version] for r in rows];n=len(a)
                values={'molecules':n,'mrr_at_25':sum(1./r for r in a if r)/n,'top1':sum(r==1 for r in a)/n,'recall_at_25':sum(r>0 for r in a)/n}
                for key,v in values.items():
                    if abs(report['comparison'][cohort][case][version][key]-v)>1e-12:raise ValueError('Metric mismatch')
                mm[cohort][case][version]=values
            for base_name,label in (('r07','exact_minus_r07'),('r08_capped32','exact_minus_capped32')):
                effect=paired([r['ranks'][case][base_name] for r in rows],[r['ranks'][case]['r08_exact'] for r in rows])
                saved=report['comparison'][cohort][case][label]
                for x,y in zip([effect['delta_mrr']]+effect['ci95'],[saved['delta_mrr']]+saved['ci95']):
                    if abs(x-y)>1e-12:raise ValueError('Paired interval differs')
    return {'status':'verified','certificates':count,'candidate_pairs':pairs,'evaluated_pairs':known,
            'proven_unnecessary_evaluations':pairs-known,'metrics':mm,'production_ranker_imported':False,'all_three_rankings_recomputed':True,'all_paired_intervals_recomputed':True,
            'certifies_chemical_correctness':False,'numerical_tolerance':1e-12}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('source',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();r=verify(a.root,a.source);text=json.dumps(r,indent=2)+'\n'
    if a.output:a.output.write_text(text,encoding='utf-8')
    print(text)

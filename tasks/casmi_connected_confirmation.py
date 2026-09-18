"""Prospective R10-G confirmation, compared with full R08B on NEW keys.

The graph feature, its weight, candidate-set policy and pass/fail conditions
are fixed before new queries are scored. No Kaggle writes or model replacement.
"""
from __future__ import annotations
from collections import Counter
import datetime as dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT_RELATIVE='artifacts/casmi26/research-r10-connected/confirmation-v2'
SEED='R10G-prospective-20260918'
AMENDMENT={'original_stop':'insufficient_unused_external_keys','observed_external_pool_size':4,
           'scores_evaluated_before_amendment':0,'replace_source':'fixed_full_COCONUT_snapshot_membership',
           'aborted_run_id':35378155922,'aborted_artifact_sha256':'b9b61bc932387d64f7e9f142b4dfc0755b720eec711ec6b847f9c9112fbdf829'}
SELECTION={'feature':'two_cut','weight':.25,'scope':'complete_R08B_top25','max_cuts':2,
           'max_scenarios':8192,'hydrogen_shift':1,'ppm':5.,'da':.001,'relative_floor':.01,
           'base_forward_feature':'cosine_nearest','base_forward_weight':.25}
GATES={'target_absent_delta_lower95_gt':0.,'external_recovery_delta_lower95_gt':0.,
       'external_available_delta_lower95_gt':-.02,'external_recovery_coverage_gte':.8,
       'exact_recall_preservation':True}


def choose_keys(target,external,excluded,target_size=128,external_size=96):
    if min(target_size,external_size)<1:raise ValueError('Empty cohort')
    order=lambda k:hashlib.sha256((SEED+':'+k).encode()).digest()
    a=sorted(set(target)-set(excluded),key=order)[:target_size]
    b=sorted(set(external)-set(excluded)-set(a),key=order)[:external_size]
    if len(a)!=target_size or len(b)!=external_size:raise ValueError('Insufficient unused keys; do not resample')
    return a,b


def validate_partitions(plan,records,previous,*,expected_sizes=(128,96)):
    target,external=plan['target_keys'],plan['external_keys'];keys=target+external
    if (len(target),len(external))!=expected_sizes or len(set(keys))!=len(keys):
        raise ValueError('Changed or overlapping prespecified cohorts')
    if set(keys)&set(previous) or plan.get('training_query_overlap')!=0:
        raise ValueError('Reused query or model fitting overlap')
    if len(records)!=len(keys) or {r['key'] for r in records}!=set(keys):
        raise ValueError('Missing or repeated prepared query')
    for r in records:
        expected='fresh_timsTOF' if r['key'] in target else 'external_covered_mixed'
        if r['cohort']!=expected:raise ValueError('Query cohort changed')
    return True



def decide(cohorts,*,recall_preserved):
    checks={}
    try:
        a=cohorts['target']['absent']['paired']['ci95']
        e=cohorts['external']['external_recovery'];b=e['paired']['ci95']
        c=cohorts['external']['available']['paired']['ci95'];coverage=e['full_candidate_coverage']
        checks={'finite':all(math.isfinite(float(v)) for v in a+b+c+[coverage]),
            'target_absent_improvement':a[0]>0.,'external_recovery_improvement':b[0]>0.,
            'available_noninferiority':c[0]>-.02,'external_mass_coverage':coverage>=.8,
            'recall_preserved':recall_preserved is True}
    except (KeyError,TypeError,ValueError,IndexError):checks['complete']=False
    return {'eligible':bool(checks) and all(checks.values()),'checks':checks,
            'not_a_hidden_test_result':True,'official_champion_changed':False}


def refine_top(case,features):
    values=[]
    for s in case['smiles']:
        v=features[s]['two_cut'];v=0. if v is None else float(v)
        if not math.isfinite(v) or not 0<=v<=1:raise ValueError('Invalid graph evidence')
        values.append(v)
    return sorted(range(len(values)),key=lambda i:(-(case['scores'][i]+SELECTION['weight']*values[i]),i))


def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def evaluate(state,repo,out):
    import numpy as np
    import torch
    from casmi26.production import sha256,write_json
    from casmi26.bounded_forward import certified_rerank
    from casmi26.connected_experiment import fixed_top
    from casmi26.connected_evidence import score_structure
    from casmi26.forward_release import ForwardRanker
    from casmi26.forward_nearest import pool_cosine_nearest_only
    from casmi26.forward_ranking import SUPPORTED_ADDUCTS
    from casmi26.fiora_adapter import ForwardModel
    from casmi_r08b_verify import metric,effect
    torch.set_num_threads(4);root=state/ROOT_RELATIVE
    plan=load_json(root/'protocol.json');frozen=load_json(root/'frozen-graph-policy.json')
    if plan['frozen_selection']!=SELECTION or plan['gates']!=GATES:raise ValueError('Prospective policy changed')
    for name,h in frozen['code'].items():
        if sha256(repo/name)!=h:raise ValueError('Frozen research code changed')
    prepared=load_json(root/'prepared.json')
    for n,h in prepared['files'].items():
        if sha256(root/n)!=h:raise ValueError('Prepared inputs changed')
    if (root/'report.json').exists():
        done=load_json(root/'report.json')
        for name,h in done['files'].items():
            if sha256(root/name)!=h:raise ValueError('Completed confirmation changed')
        for name in list(done['files'])+['report.json']:shutil.copy2(root/name,out/name)
        return done
    records=json.load(gzip.open(root/'evidence.json.gz','rt',encoding='utf-8'))
    validate_partitions(plan,records,load_json(root/'previous-query-keys.json'))
    queries={r['key']:r['queries'] for r in records}
    art=state/'artifacts/casmi26';model_path=art/'research-r08/forward-v1/fiora-e19ef82c9a6cb9dbac92bce23e914008f1aeb44e/fiora/resources/models/fiora_OS_v1.0.0.pt'
    cache=root/'forward-cache.sqlite'
    if not cache.exists():
        original=art/'candidate-r08b-20260918/forward-cache.sqlite'
        with sqlite3.connect(original.resolve().as_uri()+'?mode=ro',uri=True) as a:
            with sqlite3.connect(cache) as b:a.backup(b)
    model=ForwardModel(model_path,device='cpu');ranker=ForwardRanker(queries,model=model,cache=cache,nearest_only=True)
    raw=[];certs=[];graph=[];started=time.monotonic();stats=Counter()
    try:
        for no,record in enumerate(records,1):
            key=record['key'];known={};graph_features={};case_results={};saved={}
            modes=sorted({q['adduct'] for q in record['queries']}&SUPPORTED_ADDUCTS)
            for name,case in record['cases'].items():
                def forward(i):
                    s=case['smiles'][i]
                    if s not in known:known[s]=pool_cosine_nearest_only(record['queries'],ranker.predictions(s,modes))
                    return known[s]
                c=certified_rerank(case['scores'],forward,weight=.25 if modes else 0.,k=25)
                c.update(query_id=key,case=name,base_scores=case['scores'],weight=.25 if modes else 0.)
                top=fixed_top(case,c);certs.append(c)
                for s in top['smiles']:
                    if s not in graph_features:
                        graph_features[s]=score_structure(record['queries'],s,**{k:SELECTION[k] for k in
                            ('max_cuts','max_scenarios','hydrogen_shift','ppm','da','relative_floor')})
                        stats[graph_features[s]['status']]+=1
                order=refine_top(top,graph_features)
                rank=lambda ids:next((pos for pos,i in enumerate(ids,1) if top['keys'][i]==key),0)
                a=rank(range(len(order)));b=rank(order)
                if (a>0)!=(b>0) or sorted(order)!=list(range(len(order))):raise RuntimeError('Top25 membership changed')
                case_results[name]={'r08b':a,'connected':b,'full_candidate_coverage':key in case['keys']}
                saved[name]={**top,'order':order,'r08b_rank':a,'connected_rank':b}
            cohort='target' if record['cohort']=='fresh_timsTOF' else 'external'
            raw.append({'key':key,'cohort':cohort,'ranks':case_results})
            graph.append({'key':key,'cohort':cohort,'cases':saved,'features':graph_features})
            if no%8==0:print('CONNECTED_CONFIRM '+json.dumps({'queries':no,'of':len(records),**ranker.statistics}),flush=True)
    finally:ranker.close()
    groups={}
    for cohort in ('target','external'):
        subset=[r for r in raw if r['cohort']==cohort];groups[cohort]={}
        for case in ('available','absent','external_recovery'):
            a=[r['ranks'][case]['r08b'] for r in subset];b=[r['ranks'][case]['connected'] for r in subset]
            groups[cohort][case]={'r08b':metric(a),'connected':metric(b),'paired':effect(a,b),
                'full_candidate_coverage':sum(r['ranks'][case]['full_candidate_coverage'] for r in subset)/len(subset),
                'top1_gained':sum(x!=1 and y==1 for x,y in zip(a,b)),'top1_lost':sum(x==1 and y!=1 for x,y in zip(a,b))}
    write_json(root/'audit-ranks.json',raw)
    for name,value in [('certificates.json.gz',certs),('graph-evidence.json.gz',graph)]:
        (root/name).write_bytes(gzip.compress(json.dumps(value,allow_nan=False,separators=(',',':')).encode(),mtime=0))
    result={'experiment':'R10G-prospective-confirmation-v2','status':'completed','selection':SELECTION,'amendment':AMENDMENT,'cohorts':groups,
        'decision':decide(groups,recall_preserved=True),'recall_preserved':True,'query_count':len(records),
        'previous_query_keys_reused':0,'training_query_overlap':0,'seconds':time.monotonic()-started,
        'graph_statistics':dict(stats),'forward_statistics':dict(ranker.statistics),
        'new_training':False,'new_submissions':0,'official_score':None,'champion_changed':False,
        'files':{n:sha256(root/n) for n in ('protocol.json','frozen-graph-policy.json','prepared.json','evidence.json.gz',
            'audit-ranks.json','certificates.json.gz','graph-evidence.json.gz','previous-query-keys.json','pool-membership.json')},
        'limitations':['FIORA pretraining membership unknown; not a fully model-disjoint or de novo result.',
            'External cohort is conditional on a fixed public catalog, not representative of arbitrary unknowns.',
            'Only reorders the R08B top25; absent answers remain unrecoverable.',
            'Cut-count and H-shift penalties are heuristic, not mechanistic chemical validation.',
            'No parameter changes are permitted after this confirmation.']}
    write_json(root/'report.json',result)
    for name in list(result['files'])+['report.json']:shutil.copy2(root/name,out/name)
    print('CONNECTED_CONFIRMATION_COMPLETE\n'+json.dumps(result,indent=2),flush=True);return result


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe';repo=Path(__file__).resolve().parents[1]
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    sys.path[:0]=[str(repo/'tasks'),str(repo/'work/casmi26')]
    from casmi26.production import sha256,write_json
    from casmi26.metric import require_official_rdkit
    require_official_rdkit();out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    if len(sys.argv)>1 and sys.argv[1]=='_prepare':
        from casmi_connected_cohort import prepare
        prepare(state,repo,out);return 0
    if len(sys.argv)>1 and sys.argv[1]=='_evaluate':evaluate(state,repo,out);return 0
    root=state/ROOT_RELATIVE;root.mkdir(parents=True,exist_ok=True)
    policy={'selection':SELECTION,'gates':GATES,'amendment':AMENDMENT,'seed':SEED,'target_size':128,'external_size':96,
        'code':{name:sha256(repo/name) for name in ('tasks/casmi_connected_confirmation.py','tasks/casmi_connected_cohort.py',
             'work/casmi26/casmi26/connected_evidence.py','work/casmi26/casmi26/connected_experiment.py')}}
    path=root/'frozen-graph-policy.json'
    if path.exists() and load_json(path)!=policy:raise ValueError('Cannot replace a prespecified confirmation')
    write_json(path,policy)
    test=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests/test_connected_confirmation.py'),
        str(repo/'work/casmi26/tests/test_connected_evidence.py'),str(repo/'work/casmi26/tests/test_connected_experiment.py')],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    (out/'tests.log').write_text(test.stdout+'\n'+test.stderr,encoding='utf-8')
    if test.returncode:raise RuntimeError('Confirmation tests failed')
    frozen=state/'artifacts/casmi26/candidate-r08b-20260918/package/r08b-package.json';before=sha256(frozen)
    subprocess.run([str(py),str(Path(__file__)),'_prepare'],check=True,stdin=subprocess.DEVNULL,timeout=5400)
    import r08_python_command as imports
    import casmi_r08_forward as runtime
    material=state/'artifacts/casmi26/research-r08/forward-v1'
    paths=[material/'deps',material/'fiora-e19ef82c9a6cb9dbac92bce23e914008f1aeb44e',repo/'work/casmi26',repo/'tasks']
    code='from pathlib import Path;import sys;import casmi_connected_confirmation as m;m.evaluate(*map(Path,sys.argv[1:]))'
    subprocess.run(imports.python_command(py,paths,code,[state,repo,out]),env=runtime.env_clean(),check=True,stdin=subprocess.DEVNULL,timeout=9000)
    if sha256(frozen)!=before:raise RuntimeError('Scored champion package changed')
    statusdir=out/'kaggle-read';statusdir.mkdir(exist_ok=True)
    status=subprocess.run([str(py),str(repo/'tasks/casmi_r08b_submission.py'),'--stage','status'],
        env={**os.environ,'CHEMISTRY_REQUEST_OUTPUT':str(statusdir)},stdin=subprocess.DEVNULL,
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    write_json(out/'execution.json',{'status':'completed','commit':os.environ.get('GITHUB_SHA'),
        'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'frozen_package_unchanged':True,
        'new_submissions':0,'new_training':False,'kaggle_status_exit_code':status.returncode,'tests':test.stdout.strip()})
    subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip',
        '--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60);return 0


if __name__=='__main__':raise SystemExit(main())

"""Fixed-score, label-blind certified screening after the R08 top-32 diagnostic.

Uses spent cohorts only for an exploratory comparison; does not alter the prior
promotion gate, retrain models, read hidden test, or make Kaggle writes.
"""
from __future__ import annotations
import argparse,datetime as dt,gzip,hashlib,importlib.util,json,os,sqlite3,subprocess,sys,time,zlib
from pathlib import Path


def make_requests(records,prior,*,weight=.25,k=25):
    from casmi26.bounded_forward import candidate_envelope
    requests={}
    for record in records:
        modes={q['adduct'] for q in record['queries'] if q['adduct'] in ('[M+H]+','[M-H]-')}
        if not modes:continue
        for case in record['cases'].values():
            for i in candidate_envelope(case['scores'],weight,k=k):
                key=case['keys'][int(i)]
                smiles=prior.get(key,{}).get('smiles',case['smiles'][int(i)])
                r=requests.setdefault(key,{'key':key,'smiles':smiles,'modes':set()})
                if key not in prior:r['smiles']=min(r['smiles'],smiles)
                r['modes'].update(modes)
    return [{**requests[key],'modes':sorted(requests[key]['modes'])} for key in sorted(requests)]


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def execute(state,repo,out):
    import numpy as np,torch
    from casmi26.production import sha256,write_json
    from casmi26.bounded_forward import certified_rerank
    from casmi26.forward_ranking import pool_forward_scores,ENERGIES
    from casmi26.fiora_adapter import ForwardModel,MODEL_HASH
    import casmi_r08_verify as independent
    torch.set_num_threads(4)
    parent=state/'artifacts/casmi26/research-r08/forward-v1'
    root=parent.parent/'exact-screen-v1';root.mkdir(parents=True,exist_ok=True)
    p=json.loads((parent/'protocol.json').read_text());prior_report=json.loads((parent/'report.json').read_text())
    locked=json.loads((parent/'selection-before-transfer-audit.json').read_text())
    for name,h in locked['input_hashes'].items():
        if Path(name).name!=name or sha256(parent/name)!=h:raise ValueError('Sealed input changed: '+name)
    chosen=prior_report['selected']
    if chosen!={'feature':'cosine_nearest','weight':.25}:raise ValueError('Fixed score has changed; this is not parameter search')
    with gzip.open(parent/'evidence.json.gz','rt',encoding='utf-8') as f:records=json.load(f)
    with gzip.open(parent/'forward-features.json.gz','rt',encoding='utf-8') as f:oldfeatures=json.load(f)
    prior_requests={r['key']:r for r in json.loads((parent/'requests.json').read_text())}
    requests=make_requests(records,prior_requests);req={r['key']:r for r in requests}
    protocol={'experiment':'R08-exact-screen-v1','scientific_status':'exploratory_reuse_of_spent_cohorts',
       'selected':chosen,'k':25,'no_top32_cap':True,'candidate_data_sha256':sha256(parent/'evidence.json.gz'),
       'previous_selection_sha256':sha256(parent/'selection-before-transfer-audit.json'),
       'source_train_sha256':p['source_train'],'model_sha256':MODEL_HASH,
       'requests_sha256':hashlib.sha256(json.dumps(requests,sort_keys=True).encode()).hexdigest(),
       'maximum_potential_graphs':len(requests),'forward_source':p['forward_source_commit'],
       'comparator':'full R07 and capped-top32 R08 on identical full candidate sets',
       'invariant':'top25 equal to evaluating bounded evidence for every candidate, under fixed score',
       'new_calibration':False,'new_training':False,'hidden_test_read':False,'new_submissions':0,
       'promotion_gate_relaxed':False,'original_gate_still_binding':True,
       'known_limitations':['No independent untouched audit in this diagnostic.',
           'Forward model pretraining membership remains unknown.',
           'External recovery still cannot find an absent structure.',
           'Numerical rank certificate does not certify molecular correctness.']}
    pp=root/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Do not mutate fixed diagnostic protocol')
    write_json(pp,protocol);write_json(root/'requests.json',requests)
    done=root/'report.json'
    if done.exists():
        result=json.loads(done.read_text())
        for name,h in result['files'].items():
            if sha256(root/name)!=h:raise ValueError('Completed diagnostic modified')
        for name in list(result['files'])+['report.json']: (out/name).write_bytes((root/name).read_bytes())
        print('R08_EXACT_REUSED '+json.dumps(result),flush=True);return
    source=parent/('fiora-'+p['forward_source_commit'])
    model=ForwardModel(source/'fiora/resources/models/fiora_OS_v1.0.0.pt',device='cpu')
    olddb=sqlite3.connect('file:'+str(parent/'simulations.sqlite').replace('\\','/')+'?mode=ro',uri=True)
    oldsig=json.loads(olddb.execute('SELECT v FROM meta WHERE k="signature"').fetchone()[0])
    if oldsig['model']!=MODEL_HASH or oldsig['source']!=p['forward_source_commit']:raise ValueError('Wrong source simulation cache')
    if oldsig['device']!='cpu' or oldsig['torch']!=torch.__version__:raise ValueError('Cannot silently mix forward execution environments')
    db=sqlite3.connect(root/'extension.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS predictions(k TEXT PRIMARY KEY,content BLOB NOT NULL)')
    sig=json.dumps(protocol,sort_keys=True);prev=db.execute('SELECT v FROM meta WHERE k="signature"').fetchone()
    if prev and prev[0]!=sig:raise ValueError('Extension cache belongs to another diagnostic')
    if not prev:db.execute('INSERT INTO meta VALUES(?,?)',('signature',sig));db.commit()
    cache={};stats={'new_graph_calls':0,'new_spectra':0,'prediction_failures':0,'old_cache_keys_used':0,'extension_cache_keys_used':0}
    def spectra(key):
        if key in cache:return cache[key]
        request=req[key]
        row=db.execute('SELECT content FROM predictions WHERE k=?',(key,)).fetchone()
        if row:
            data=json.loads(zlib.decompress(row[0]));stats['extension_cache_keys_used']+=1
        else:
            row=olddb.execute('SELECT content FROM predictions WHERE k=?',(key,)).fetchone()
            data=json.loads(zlib.decompress(row[0])) if row else {'key':key,'smiles':request['smiles'],'modes':[],'spectra':[],'status':'predicted'}
            if row:stats['old_cache_keys_used']+=1
            if data['smiles']!=request['smiles']:raise ValueError('Graph representation changed across caches')
            missing=sorted(set(request['modes'])-set(data['modes']))
            if missing and data['status']!='failed':
                try:
                    pred=model.predict(request['smiles'],missing,ENERGIES)
                    data['spectra'] += [{'adduct':m,'energy':e,'peaks':a.tolist()} for (m,e),a in pred.items()]
                    data['modes']=sorted(set(data['modes'])|set(missing));data['status']='predicted'
                    stats['new_spectra']+=len(pred)
                except (ValueError,RuntimeError,AssertionError,IndexError,KeyError) as exc:
                    data.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:240])
                    stats['prediction_failures']+=1
                stats['new_graph_calls']+=1
            db.execute('INSERT OR REPLACE INTO predictions VALUES(?,?)',(key,zlib.compress(json.dumps(data,allow_nan=False).encode(),1)));db.commit()
        value={(s['adduct'],float(s['energy'])):s['peaks'] for s in data['spectra']}
        cache[key]=value;return value
    started=time.monotonic();byquery={};raw=[];certificates=[];total_requested=total_evaluated=0
    calset=set(p['calibration_keys']);npset=set(p['reused_np_audit_keys'])
    try:
        for number,record in enumerate(records,1):
            qkey=record['key'];computed={}
            supported=any(q['adduct'] in ('[M+H]+','[M-H]-') for q in record['queries'])
            result={'key':qkey,'cohort':'calibration_reused' if qkey in calset else 'np_reused' if qkey in npset else 'transfer_reused','ranks':{}}
            for case,data in record['cases'].items():
                def evidence(i):
                    key=data['keys'][i]
                    if key not in computed:
                        pooled=pool_forward_scores(record['queries'],spectra(key))
                        computed[key]=pooled['features'].get(chosen['feature'])
                        if key in oldfeatures.get(qkey,{}) and chosen['feature'] in oldfeatures[qkey][key]:
                            expected=oldfeatures[qkey][key][chosen['feature']]
                            if computed[key] is None or abs(computed[key]-expected)>1e-10:raise ValueError('Existing forward evidence changed')
                    return computed[key]
                ranked=certified_rerank(data['scores'],evidence,weight=chosen['weight'] if supported else 0.,k=25)
                if not ranked['certified']:raise RuntimeError('Unlimited exact screening did not certify ranking')
                total_requested+=len(data['keys']);total_evaluated+=ranked['evaluations']
                guesses=[data['keys'][i] for i in ranked['top_indices']]
                rank=guesses.index(qkey)+1 if qkey in guesses else 0
                baseline=independent.rank_case(data,{},chosen['feature'],0.,32,qkey)
                capped=independent.rank_case(data,oldfeatures.get(qkey,{}),chosen['feature'],chosen['weight'],32,qkey)
                result['ranks'][case]={'r07':baseline,'r08_capped32':capped,'r08_exact':rank,
                    'evaluations':ranked['evaluations'],'candidates':len(data['keys']),
                    'truth_in_candidates':qkey in data['keys'],'certified':True}
                certificates.append({'query_id':qkey,'case':case,'weight':chosen['weight'] if supported else 0.,
                    'base_scores':data['scores'],**ranked})
            byquery[qkey]=computed;raw.append(result)
            if number%10==0:print('R08_EXACT_PROGRESS '+json.dumps({'queries':number,'total':len(records),**stats,'seconds':time.monotonic()-started}),flush=True)
    finally:db.close();olddb.close()
    comparison={}
    for name in ('calibration_reused','np_reused','transfer_reused'):
        rows=[r for r in raw if r['cohort']==name];comparison[name]={}
        for case in ('available','absent','external_recovery'):
            ranks={version:[r['ranks'][case][version] for r in rows] for version in ('r07','r08_capped32','r08_exact')}
            comparison[name][case]={**{v:independent.metrics(a) for v,a in ranks.items()},
                'exact_minus_r07':independent.paired(ranks['r07'],ranks['r08_exact']),
                'exact_minus_capped32':independent.paired(ranks['r08_capped32'],ranks['r08_exact'])}
    write_json(root/'audit-ranks.json',raw)
    for name,obj in [('certificates.json.gz',certificates),('features.json.gz',byquery)]:
        (root/name).write_bytes(gzip.compress(json.dumps(obj,allow_nan=False,separators=(',',':')).encode(),mtime=0))
    result={'status':'completed','experiment':protocol['experiment'],'commit':os.environ.get('GITHUB_SHA'),
        'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'comparison':comparison,'simulation':stats,
        'queries':len(records),'candidate_pairs_across_regimes':total_requested,'evaluated_pairs_across_regimes':total_evaluated,
        'all_top25_certified_under_fixed_score':True,'seconds':time.monotonic()-started,
        'new_training':False,'new_submissions':0,'new_uploads':0,'production_changed':False,
        'promotion_gate_relaxed':False,'official_score':None,'audit_is_independent':False,
        'files':{n:sha256(root/n) for n in ('protocol.json','requests.json','audit-ranks.json','certificates.json.gz','features.json.gz')},
        'limitations':protocol['known_limitations']}
    write_json(done,result)
    for name in list(result['files'])+['report.json']:(out/name).write_bytes((root/name).read_bytes())
    print('R08_EXACT_COMPLETE\n'+json.dumps(result,indent=2),flush=True)


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);repo=Path(__file__).resolve().parents[1]
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    root=state/'artifacts/casmi26/research-r08/forward-v1';py=state/'envs/casmi26/python.exe'
    helper=load('paths',repo/'tasks/r08_python_command.py');fw=load('environment',repo/'tasks/casmi_r08_forward.py')
    paths=[root/'deps',root/('fiora-'+fw.SOURCE),repo/'work/casmi26',repo/'tasks'];env=fw.env_clean()
    command=helper.python_command(py,paths,'import pytest,sys;raise SystemExit(pytest.main(sys.argv[1:]))',
         ['-q',str(repo/'work/casmi26/tests/test_bounded_forward.py'),str(repo/'work/casmi26/tests/test_r08_exact_task.py')])
    tests=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8');print(tests.stdout,flush=True)
    if tests.returncode:raise RuntimeError('Exact screening tests failed')
    code='from pathlib import Path;import sys;import casmi_r08_exact;casmi_r08_exact.execute(*map(Path,sys.argv[1:]))'
    result=subprocess.run(helper.python_command(py,paths,code,[state,repo,out]),env=env,stdin=subprocess.DEVNULL,timeout=7200)
    if result.returncode:raise RuntimeError('Exact comparison failed; caches retained')
    subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip','--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
    return 0


if __name__=='__main__':raise SystemExit(main())

"""Select forward evidence only on sealed calibration keys, then evaluate once."""
from __future__ import annotations
import argparse,gzip,hashlib,json,os,shutil,sqlite3,subprocess,sys,time,zlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    if a.root is None:
        state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
        if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4'})
        a.root=state/'artifacts/casmi26/research-r08/forward-v1';a.output=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.forward_ranking import pool_forward_scores,choose_configuration,evaluate_configuration,candidate_frontier
    from casmi26.production import sha256,write_json
    from casmi26.ranking import paired_effect
    import numpy as np
    root=a.root;out=a.output or root/'evaluation-artifacts';out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'protocol.json').read_text());prepared=json.loads((root/'prepared.json').read_text())
    for name,digest in prepared['files'].items():
        if sha256(root/name)!=digest:raise RuntimeError('Prepared evidence hash changed')
    simulation=json.loads((root/'simulation.json').read_text())
    if simulation['status']!='simulation_completed':raise RuntimeError('Forward simulation not completed')
    if simulation['identity']['requests']!=sha256(root/'requests.json'):raise RuntimeError('Predictions refer to another candidate pool')
    with gzip.open(root/'evidence.json.gz','rt',encoding='utf-8') as f:records=json.load(f)
    db=sqlite3.connect('file:'+str(root/'simulations.sqlite').replace('\\','/')+'?mode=ro',uri=True)
    sims={};failures=[]
    for key,blob in db.execute('SELECT k,content FROM predictions'):
        row=json.loads(zlib.decompress(blob))
        if row['key']!=key:raise ValueError('Corrupt simulation key')
        if row['status']=='failed':failures.append(row)
        sims[key]={(p['adduct'],float(p['energy'])):p['peaks'] for p in row['spectra']}
    db.close()
    requests=json.loads((root/'requests.json').read_text())
    if set(sims)!={r['key'] for r in requests}:raise RuntimeError('Incomplete candidate simulations')
    started=time.monotonic();features={};support={}
    for n,r in enumerate(records,1):
        features[r['key']]={};counts=[]
        for key in r['frontier']:
            result=pool_forward_scores(r['queries'],sims.get(key,{}))
            features[r['key']][key]=result['features'];counts.append(result['supported_spectra'])
        support[r['key']]={'queries':len(r['queries']),'supported_queries_max':max(counts,default=0),
            'successful_frontier_structures':sum(bool(v) for v in features[r['key']].values())}
        if n%25==0:print('R08_FORWARD_FEATURES '+str(n)+'/'+str(len(records)),flush=True)
    (root/'forward-features.json.gz').write_bytes(gzip.compress(json.dumps(features,sort_keys=True,allow_nan=False).encode(),mtime=0))
    bykey={r['key']:r for r in records}
    if len(bykey)!=len(records):raise ValueError('Duplicate query key')
    calkeys=plan['calibration_keys'];freshkeys=plan['fresh_transfer_keys'];oldkeys=plan['reused_np_audit_keys']
    if set(calkeys)&(set(freshkeys)|set(oldkeys)) or set(oldkeys)&set(freshkeys):raise ValueError('Evaluation key overlap')
    cal=[bykey[k] for k in calkeys]
    selection=choose_configuration(cal,features,plan['features'],plan['forward_weights'],plan['shortlist_per_regime'])
    selection['input_hashes']={n:sha256(root/n) for n in ('protocol.json','evidence.json.gz','requests.json','simulation.json','forward-features.json.gz')}
    selectedpath=root/'selection-before-transfer-audit.json'
    if selectedpath.exists() and json.loads(selectedpath.read_text())!=selection:raise RuntimeError('Do not overwrite a locked different selection')
    write_json(selectedpath,selection)
    chosen=selection['selected'];print('R08_SELECTION '+json.dumps({'feature':chosen['feature'],'weight':chosen['weight']}),flush=True)
    cohorts={};raw=[]
    def rr(r):return 1./r if r>0 else 0.
    for name,keys in (('reused_np_diagnostic',oldkeys),('fresh_timsTOF_transfer',freshkeys)):
        part=[bykey[k] for k in keys];base=evaluate_configuration(part,features,chosen['feature'],0.,plan['shortlist_per_regime'])
        changed=evaluate_configuration(part,features,chosen['feature'],chosen['weight'],plan['shortlist_per_regime'])
        comparisons={}
        for case in base['metrics']:
            effect=paired_effect([rr(r['ranks'][case]) for r in base['rows']],[rr(r['ranks'][case]) for r in changed['rows']])
            present=[r['key'] in r['cases'][case]['keys'] for r in part]
            shortlisted=[r['key'] in candidate_frontier({case:r['cases'][case]},plan['shortlist_per_regime']) for r in part]
            comparisons[case]={'baseline':base['metrics'][case],'selected':changed['metrics'][case],'paired_effect':effect,
                               'true_candidate_coverage':float(np.mean(present)),'true_shortlist_coverage':float(np.mean(shortlisted))}
        for b,s in zip(base['rows'],changed['rows']):raw.append({'key':b['key'],'cohort':name,'baseline':b['ranks'],'selected':s['ranks'],'support':support[b['key']]})
        cohorts[name]=comparisons
    report={'status':'completed','experiment':plan['experiment'],'selected':{'feature':chosen['feature'],'weight':chosen['weight']},
        'calibration_metrics':chosen['metrics'],'calibration_gain':chosen['gains'],'cohorts':cohorts,
        'fresh_transfer_molecules':len(freshkeys),'reused_np_diagnostic_molecules':len(oldkeys),'calibration_molecules':len(calkeys),
        'simulation':simulation,'forward_failures':len(failures),'query_support':support,
        'selected_before_fresh_audit':True,'fitting_overlap':plan['fitting_overlap'],
        'score_seconds':time.monotonic()-started,'new_submissions':0,'champion_changed':False,'official_score':None,
        'limitations':['Public forward-model pretraining overlap is unknown; not a de novo test.',
           'The 170 NP keys are reused diagnostics, not untouched validation.',
           'The 128 fresh keys are enveda-180 timsTOF structures, not necessarily novel natural products.',
           'Only the top 32 candidates per regime receive forward evidence; missing evidence preserves base score.',
           'HCD energy conditioning is evaluated on timsTOF as a transfer hypothesis.',
           'No hidden inputs, labels or Kaggle score were used for calibration.']}
    write_json(root/'audit-ranks.json',raw);write_json(root/'failures.json',failures);write_json(root/'report.json',report)
    for name in ('report.json','audit-ranks.json','selection-before-transfer-audit.json','failures.json','graph-parity.json','simulation.json','forward-features.json.gz','protocol.json'):
        shutil.copy2(root/name,out/name)
    print('R08_REPORT_BEGIN\n'+json.dumps({k:v for k,v in report.items() if k not in ('query_support','simulation')},indent=2)+'\nR08_REPORT_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

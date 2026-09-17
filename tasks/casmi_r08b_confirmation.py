"""Prospective fixed-score confirmation on NEW molecule keys; no Kaggle writes.

Original R08 remains a failed promotion gate. This separate experiment measures
reranking where external recovery is possible, rather than treating zero catalog
coverage as a failure of the scoring function. Its weights are NOT reselected.
"""
from __future__ import annotations
from collections import defaultdict, Counter
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import zlib

FORWARD_WEIGHT = .25
FEATURE = 'cosine_nearest'
TARGET_SIZE = 128
EXTERNAL_SIZE = 96
SOURCE = 'e19ef82c9a6cb9dbac92bce23e914008f1aeb44e'
SEED = 'R08B-fixed-confirmation-20260917'


def choose_keys(target, external, excluded, target_size=TARGET_SIZE, external_size=EXTERNAL_SIZE):
    if min(target_size, external_size) < 1:
        raise ValueError('Cohorts must be nonempty')
    order = lambda k: hashlib.sha256((SEED + ':' + k).encode()).digest()
    a = sorted(set(target) - set(excluded), key=order)[:target_size]
    b = sorted(set(external) - set(excluded) - set(a), key=order)[:external_size]
    if len(a) != target_size or len(b) != external_size:
        raise ValueError(f'Insufficient unused molecules: target={len(a)}, external={len(b)}')
    return a, b


def decide(cohorts):
    checks = {}
    try:
        target = cohorts['fresh_timsTOF']['absent']['paired']['ci95']
        covered = cohorts['external_covered_mixed']['external_recovery']
        recovery = covered['paired']['ci95']
        available = cohorts['external_covered_mixed']['available']['paired']['ci95']
        values = target + recovery + available + [covered['coverage']]
        checks['finite'] = all(math.isfinite(float(v)) for v in values)
        checks['fresh_target_absent_improves'] = float(target[0]) > 0.
        checks['fresh_external_recovery_improves'] = float(recovery[0]) > 0.
        checks['reference_noninferiority_0_03'] = float(available[0]) > -.03
        checks['external_mass_window_coverage_at_least_0_8'] = float(covered['coverage']) >= .8
    except (KeyError, IndexError, TypeError, ValueError):
        checks['complete'] = False
    return {'eligible': bool(checks) and all(checks.values()), 'checks': checks,
            'replaces_original_r08_gate': False, 'requires_new_independent_cohorts': True,
            'meaning': 'Prospectively specified candidate-evidence gate; not hidden-score or de novo proof'}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def used_keys(artifacts, root):
    from casmi26.pipeline_v3 import _excluded
    used = _excluded(artifacts, ignore=root)
    fields = ('calibration_keys', 'audit_keys', 'screen_keys', 'selection_keys',
              'reused_np_audit_keys', 'fresh_transfer_keys', 'target_keys', 'external_keys')
    for parent in Path(artifacts).glob('research-*'):
        for p in parent.rglob('protocol.json'):
            if root.resolve() in p.resolve().parents:
                continue
            data = json.loads(p.read_text(encoding='utf-8-sig'))
            for name in fields:
                used.update(k for k in data.get(name, []) if isinstance(k, str))
        for p in parent.rglob('audit-ranks.json'):
            if root.resolve() in p.resolve().parents:
                continue
            data = json.loads(p.read_text(encoding='utf-8-sig'))
            if isinstance(data, list):
                used.update(r['key'] for r in data if isinstance(r, dict) and isinstance(r.get('key'), str))
    return used


def prepare(state, repo, out):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from casmi26.production import sha256, write_json, mass_candidates, arrow_features, reference_score, is_validation
    from casmi26.r07_release import external_candidates, selected_scores
    from casmi26.model_v3 import MultiFingerprintModel
    from casmi26.target_domain import training_rows
    from casmi26.ranking import spectrum_signature
    art = state/'artifacts/casmi26'
    old = art/'research-r07/full-system-v1'
    cache = state/'cache/casmi26/highres-v3'
    root = art/'research-r08b/fixed-confirmation-v1'
    root.mkdir(parents=True, exist_ok=True)
    prior = json.loads((old/'report.json').read_text())
    for name, digest in prior['artifact_hashes'].items():
        if sha256(old/name) != digest:
            raise RuntimeError('Original R07 experiment changed')
    inv = json.loads((art/'research-r07/inventory.json').read_text())
    train = Path(inv['train_path'])
    if sha256(train) != prior['source']['train_sha256']:
        raise ValueError('Training corpus changed')
    done = root/'prepared.json'
    if done.exists():
        report = json.loads(done.read_text())
        if report['script_sha256'] != sha256(Path(__file__)):
            raise ValueError('Prepared confirmation code changed')
        for name, digest in report['files'].items():
            if sha256(root/name) != digest:
                raise ValueError('Prepared confirmation input changed')
        return root
    catalog = json.loads((cache/'catalog.json').read_text())
    counts = np.load(cache/'counts.npy', allow_pickle=False)
    packed = np.load(cache/'targets.npy', allow_pickle=False, mmap_mode='r')
    lookup = {r[0]: i for i, r in enumerate(catalog)}
    bykey = {r[2]: i for i, r in enumerate(catalog) if counts[i] > 0 and is_validation(r[2])}
    oldplan = json.loads((old/'protocol.json').read_text())
    npkeys = set(oldplan['calibration_keys'] + oldplan['audit_keys'])
    fit = training_rows(catalog, counts, npkeys)
    fitting_keys = {catalog[i][2] for i in fit}
    h = hashlib.sha256('\n'.join(sorted(catalog[i][2] for i in fit)).encode()).hexdigest()
    if h != prior['source']['training_key_sha256']:
        raise ValueError('Holdout model fitting lineage differs')
    excluded = used_keys(art, root) | npkeys
    external_known = {r[2] for r in json.loads((old/'external.json').read_text())}
    available_sources = defaultdict(set)
    with pq.ParquetFile(train) as table:
        for batch in table.iter_batches(batch_size=65536, columns=['normalized_smiles', 'ingest_lib']):
            for raw, source in zip(batch.column('normalized_smiles').to_pylist(), batch.column('ingest_lib').to_pylist()):
                i = lookup.get(raw)
                if i is not None:
                    key = catalog[i][2]
                    if key in bykey and key not in excluded and source != 'enveda-np-examples':
                        available_sources[key].add(str(source))
    target_pool = {k for k, sources in available_sources.items() if 'enveda-180' in sources}
    ext_pool = set(available_sources) & external_known
    write_json(out/'eligible-pools.json', {'target': len(target_pool), 'external': len(ext_pool),
        'previous_keys_excluded': len(excluded), 'no_candidate_scores_evaluated_yet': True})
    target, external = choose_keys(target_pool, ext_pool, excluded)
    selected = set(target + external)
    if selected & fitting_keys:
        raise ValueError('New queries overlap model fitting')
    query_source = {k: 'enveda-180' for k in target}
    for key in external:
        query_source[key] = min(available_sources[key], key=lambda s: hashlib.sha256((SEED+':source:'+key+':'+s).encode()).digest())
    protocol = {'experiment': 'R08B-fixed-confirmation-v1', 'target_keys': target, 'external_keys': external,
        'target_source': 'enveda-180', 'external_source_rule': 'one hashed source per previously unseen externally covered key',
        'external_pool': 'intersection of fixed prior COCONUT mass-selected public records and unused hash-holdout structures',
        'external_pool_not_random_all_COCONUT': True, 'query_source': query_source,
        'excluded_prior_key_count': len(excluded), 'training_query_overlap': 0,
        'frozen_selection': {'feature': FEATURE, 'weight': FORWARD_WEIGHT}, 'new_weight_search': False,
        'base_recipe': prior['selected'], 'base_mass_offset_ppm': prior['mass_offset_ppm'],
        'r07_report_sha256': sha256(old/'report.json'), 'train_sha256': sha256(train),
        'holdout_model_sha256': sha256(old/'v1-independent.npz'),
        'coconut_sha256': prior['source']['coconut_snapshot_sha256'], 'forward_source': SOURCE,
        'k': 25, 'shortlist_cap': None, 'all_inputs_before_ranks': True,
        'gates': {'target_absent_delta_lower95_gt': 0., 'external_recovery_delta_lower95_gt': 0.,
                  'external_available_delta_lower95_gt': -.03, 'external_recovery_coverage_gte': .8},
        'earlier_R08_gate': 'failed and unchanged; this is a new prospective fixed-score experiment',
        'fiora_pretraining_membership': 'unknown', 'hidden_test_read': False,
        'new_training': False, 'new_submissions': 0, 'script_sha256': sha256(Path(__file__))}
    plan = root/'protocol.json'
    if plan.exists() and json.loads(plan.read_text()) != protocol:
        raise ValueError('Do not mutate the prespecified confirmation')
    write_json(plan, protocol)
    cols = ['normalized_smiles', 'ingest_lib', 'ms2_mzs', 'ms2_normalized_intensities',
            'precursor_mz', 'adduct', 'collision_energy_ev', 'instrument_type']
    raw_queries = pa.array([r[0] for r in catalog if r[2] in selected], type=pa.string())
    groups = defaultdict(list); instruments = defaultdict(Counter)
    with pq.ParquetFile(train) as table:
        for batch in table.iter_batches(batch_size=8192, columns=cols):
            part = batch.filter(pc.is_in(batch.column('normalized_smiles'), value_set=raw_queries))
            if not len(part): continue
            x, valid, neutral = arrow_features(part)
            for j, row in enumerate(part.to_pylist()):
                key = catalog[lookup[row['normalized_smiles']]][2]
                if row['ingest_lib'] != query_source[key] or not valid[j]: continue
                peaks = np.column_stack([row['ms2_mzs'], row['ms2_normalized_intensities']])
                peaks = peaks[peaks[:, 1] > 0]; peaks[:, 1] /= peaks[:, 1].sum()
                groups[key].append({'x': x[j], 'mass': float(neutral[j]), 'peaks': peaks,
                    'precursor': row['precursor_mz'], 'adduct': row['adduct'], 'ce': row['collision_energy_ev']})
                instruments[key][str(row['instrument_type'])] += 1
    if set(groups) != selected:
        raise ValueError('Prespecified query has no valid observations; do not resample based on outcome')
    observed = {k: float(np.median([q['mass'] for q in qs])) for k, qs in groups.items()}
    coconut = state/'data/external/coconut-2026-08/coconut_csv_lite-08-2026.zip'
    if sha256(coconut) != protocol['coconut_sha256']: raise ValueError('COCONUT changed')
    extpath = root/'external.json'; extfp = root/'external.npy'
    if extpath.exists() and extfp.exists():
        erows=json.loads(extpath.read_text()); efp=np.load(extfp,allow_pickle=False)
    else:
        erows,efp,_=external_candidates(coconut,list(observed.values()),workers=8)
        write_json(extpath,erows);np.save(extfp,efp)
    masses=np.array([r[3] for r in catalog]);emasses=np.array([r[3] for r in erows])
    selections={};chosen=set()
    for key,mass in observed.items():
        oi=mass_candidates(masses,mass,50,.02);ei=mass_candidates(emasses,mass,50,.02)
        selections[key]=(oi,ei);chosen.update(map(int,oi))
    reflookup={catalog[i][0]:int(i) for i in chosen}
    refvalues=pa.array(list(reflookup),type=pa.string())
    query_signatures={spectrum_signature(q) for qs in groups.values() for q in qs}
    refs=defaultdict(list);seen=set();refcounts=Counter()
    with pq.ParquetFile(train) as table:
        for batch in table.iter_batches(batch_size=8192,columns=cols):
            refcounts['scanned']+=len(batch)
            part=batch.filter(pc.is_in(batch.column('normalized_smiles'),value_set=refvalues))
            if not len(part):continue
            _,valid,neutral=arrow_features(part)
            for j,row in enumerate(part.to_pylist()):
                i=reflookup[row['normalized_smiles']];key=catalog[i][2]
                if key in selected and row['ingest_lib']==query_source[key]:continue
                if not valid[j] or abs(neutral[j]-masses[i])>max(.003,masses[i]*50e-6):continue
                p=np.column_stack([row['ms2_mzs'],row['ms2_normalized_intensities']]);p=p[p[:,1]>0];p[:,1]/=p[:,1].sum()
                r={'peaks':p,'precursor':row['precursor_mz'],'adduct':row['adduct']}
                sig=spectrum_signature(r);pair=(key,sig)
                if sig in query_signatures:refcounts['identical_query_removed']+=1;continue
                if pair in seen:continue
                seen.add(pair);refs[key].append(r);refcounts['accepted']+=1
    model=MultiFingerprintModel(old/'v1-independent.npz');records=[]
    for number,key in enumerate(sorted(groups),1):
        qs=groups[key];oi,ei=selections[key];mass=observed[key]
        rows=[catalog[i] for i in oi]+[erows[i] for i in ei]
        bits=np.concatenate([packed[oi,:256],efp[ei]],axis=0)
        flags=np.concatenate([np.zeros(len(oi)),np.ones(len(ei))])
        z=model.logits(np.mean([q['x'] for q in qs],axis=0))
        sp={k:float(np.mean([max((reference_score(q,r) for r in refs[k]),default=0.) for q in qs])) for k in {r[2] for r in rows}}
        cases={}
        for case in ('available','absent','external_recovery'):
            indices=[];taken=set()
            for i,r in enumerate(rows):
                if case=='external_recovery' and not flags[i] and r[2] in selected:continue
                if r[2] not in taken:taken.add(r[2]);indices.append(i)
            ii=np.asarray(indices,dtype=np.int64);rs=[rows[i] for i in ii]
            spectral=np.array([sp[r[2]] if case=='available' or r[2] not in selected else 0. for r in rs])
            score=selected_scores(z,np.unpackbits(bits[ii],axis=1),spectral,np.array([r[3] for r in rs]),mass,
                                 flags[ii],prior['selected'],prior['mass_offset_ppm'])
            cases[case]={'keys':[r[2] for r in rs],'smiles':[r[1] for r in rs],'scores':score.tolist()}
        serial=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in q.items() if k!='x'} for q in qs]
        records.append({'key':key,'cohort':'fresh_timsTOF' if key in target else 'external_covered_mixed',
                        'queries':serial,'query_source':query_source[key],'instruments':dict(instruments[key]),'cases':cases})
        if number%16==0:print('R08B_PREPARE '+str(number),flush=True)
    with gzip.open(root/'evidence.json.gz','wt',encoding='utf-8') as f:json.dump(records,f,allow_nan=False,separators=(',',':'))
    prepared={'status':'prepared','script_sha256':sha256(Path(__file__)),'records':len(records),
        'query_spectra':sum(map(len,groups.values())),'reference_counts':dict(refcounts),
        'files':{n:sha256(root/n) for n in ('protocol.json','evidence.json.gz','external.json','external.npy')}}
    write_json(done,prepared);print('R08B_PREPARED '+json.dumps({k:v for k,v in prepared.items() if k!='files'}),flush=True)
    return root


def evaluate(state,repo,out,root):
    import numpy as np
    import torch
    from casmi26.production import sha256,write_json
    from casmi26.bounded_forward import certified_rerank
    from casmi26.forward_ranking import pool_forward_scores,ENERGIES,SUPPORTED_ADDUCTS
    from casmi26.fiora_adapter import ForwardModel,MODEL_HASH
    from casmi_r08_verify import paired,metrics
    torch.set_num_threads(4)
    plan=json.loads((root/'protocol.json').read_text())
    if plan['frozen_selection']!={'feature':FEATURE,'weight':FORWARD_WEIGHT}:
        raise ValueError('Fixed score changed')
    with gzip.open(root/'evidence.json.gz','rt',encoding='utf-8') as f:records=json.load(f)
    parent=state/'artifacts/casmi26/research-r08/forward-v1'
    model=ForwardModel(parent/('fiora-'+SOURCE)/'fiora/resources/models/fiora_OS_v1.0.0.pt',device='cpu')
    databases=[]
    for p in (parent/'simulations.sqlite',parent.parent/'exact-screen-v1/extension.sqlite'):
        databases.append(sqlite3.connect('file:'+str(p).replace('\\','/')+'?mode=ro',uri=True))
    own=sqlite3.connect(root/'simulations.sqlite')
    own.execute('CREATE TABLE IF NOT EXISTS predictions(k TEXT PRIMARY KEY,content BLOB NOT NULL)')
    identity={'model':MODEL_HASH,'source':SOURCE,'torch':torch.__version__,'energy':list(ENERGIES)}
    manifest=root/'simulation-identity.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Forward environment changed')
    write_json(manifest,identity)
    memory={};statistics=Counter()
    def prediction(key,smiles,modes):
        cachekey=key+':'+hashlib.sha256(smiles.encode()).hexdigest()
        if cachekey in memory:
            data=memory[cachekey]
        else:
            row=own.execute('SELECT content FROM predictions WHERE k=?',(cachekey,)).fetchone()
            data=json.loads(zlib.decompress(row[0])) if row else None
            if data is None:
                for db in reversed(databases):
                    row=db.execute('SELECT content FROM predictions WHERE k=?',(key,)).fetchone()
                    if row:
                        old=json.loads(zlib.decompress(row[0]))
                        if old.get('smiles')==smiles:
                            data=old;statistics['reused_prior_graphs']+=1;break
            if data is None:data={'smiles':smiles,'modes':[],'spectra':[],'status':'predicted'}
            memory[cachekey]=data
        missing=sorted(set(modes)-set(data['modes']))
        if missing and data['status']!='failed':
            try:
                predicted=model.predict(smiles,missing,ENERGIES)
                data['spectra'] += [{'adduct':m,'energy':e,'peaks':p.tolist()} for (m,e),p in predicted.items()]
                data['modes']=sorted(set(data['modes'])|set(missing))
                statistics['new_spectra']+=len(predicted)
            except (ValueError,RuntimeError,AssertionError,IndexError,KeyError) as e:
                data.update(status='failed',error_type=type(e).__name__,error=str(e)[:200]);statistics['failures']+=1
            statistics['new_graph_calls']+=1
            own.execute('INSERT OR REPLACE INTO predictions VALUES(?,?)',(cachekey,zlib.compress(json.dumps(data,allow_nan=False).encode(),1)));own.commit()
        return {(r['adduct'],float(r['energy'])):r['peaks'] for r in data['spectra']}
    done=root/'report.json'
    if done.exists():
        r=json.loads(done.read_text())
        for n,h in r['files'].items():
            if sha256(root/n)!=h:raise ValueError('Completed confirmation changed')
        for n in list(r['files'])+['report.json']:shutil.copy2(root/n,out/n)
        return r
    raw=[];certificates=[];features={};started=time.monotonic()
    try:
        for number,record in enumerate(records,1):
            key=record['key'];known={};ranked={};modes=sorted({q['adduct'] for q in record['queries']} & SUPPORTED_ADDUCTS)
            for case,data in record['cases'].items():
                def evidence(i):
                    k=data['keys'][i]
                    if k not in known:
                        spectra=prediction(k,data['smiles'][i],modes)
                        known[k]=pool_forward_scores(record['queries'],spectra)['features'].get(FEATURE)
                    return known[k]
                result=certified_rerank(data['scores'],evidence,weight=FORWARD_WEIGHT if modes else 0.,k=25)
                if not result['certified']:raise RuntimeError('Uncertified result')
                original=np.argsort(-np.array(data['scores']),kind='stable')[:25]
                def rank(order):
                    names=[data['keys'][int(i)] for i in order]
                    return names.index(key)+1 if key in names else 0
                ranked[case]={'r07':rank(original),'r08':rank(result['top_indices']),
                    'coverage':key in data['keys'],'candidate_count':len(data['keys']),'evaluated':result['evaluations']}
                certificates.append({'query_id':key,'case':case,'base_scores':data['scores'],
                    'weight':FORWARD_WEIGHT if modes else 0.,**result})
            raw.append({'key':key,'cohort':record['cohort'],'ranks':ranked,'supported_modes':modes})
            features[key]=known
            if number%8==0:print('R08B_FORWARD '+json.dumps({'queries':number,'total':len(records),**statistics}),flush=True)
    finally:
        own.close()
        for db in databases:db.close()
    cohorts={}
    for cohort in ('fresh_timsTOF','external_covered_mixed'):
        rows=[r for r in raw if r['cohort']==cohort];cohorts[cohort]={}
        for case in ('available','absent','external_recovery'):
            b=[r['ranks'][case]['r07'] for r in rows];s=[r['ranks'][case]['r08'] for r in rows]
            cohorts[cohort][case]={'r07':metrics(b),'r08':metrics(s),'paired':paired(b,s),
                'coverage':float(np.mean([r['ranks'][case]['coverage'] for r in rows]))}
    write_json(root/'audit-ranks.json',raw)
    for name,data in [('certificates.json.gz',certificates),('features.json.gz',features)]:
        with gzip.open(root/name,'wt',encoding='utf-8') as f:json.dump(data,f,allow_nan=False,separators=(',',':'))
    result={'status':'completed','experiment':'R08B-fixed-confirmation-v1','commit':os.environ.get('GITHUB_SHA'),
        'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'cohorts':cohorts,'decision':decide(cohorts),
        'frozen_selection':{'feature':FEATURE,'weight':FORWARD_WEIGHT},'new_calibration':False,
        'previous_query_keys_reused':0,'fitting_overlap':0,'statistics':dict(statistics),
        'certificates':len(certificates),'seconds_forward':time.monotonic()-started,
        'new_submissions':0,'new_training':False,'official_score':None,'champion_changed':False,
        'files':{n:sha256(root/n) for n in ('protocol.json','prepared.json','evidence.json.gz','simulation-identity.json',
            'audit-ranks.json','certificates.json.gz','features.json.gz')},
        'limitations':['Only the timsTOF cohort is a target-instrument test; externally covered cohort uses mixed instruments.',
            'External cohort was sampled conditional on presence in a prior fixed public catalog slice.',
            'FIORA pretraining membership is unknown; no claim of fully model-disjoint de novo identification.',
            'Reranking cannot recover a structure absent from the candidate set.',
            'This new prospective gate does not turn the original failed R08 gate into a pass.']}
    write_json(done,result)
    for n in list(result['files'])+['report.json']:shutil.copy2(root/n,out/n)
    print('R08B_COMPLETE\n'+json.dumps(result,indent=2),flush=True)
    return result


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);repo=Path(__file__).resolve().parents[1]
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    python=state/'envs/casmi26/python.exe'
    if len(sys.argv)>1 and sys.argv[1]=='_prepare':
        sys.path.insert(0,str(repo/'work/casmi26'));prepare(state,repo,out);return 0
    if len(sys.argv)>1 and sys.argv[1]=='_evaluate':
        evaluate(state,repo,out,state/'artifacts/casmi26/research-r08b/fixed-confirmation-v1');return 0
    env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'}
    tests=subprocess.run([str(python),'-m','pytest','-q',str(repo/'work/casmi26/tests/test_r08b_confirmation.py')],
        env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8')
    if tests.returncode:raise RuntimeError('Prospective protocol tests failed')
    subprocess.run([str(python),str(Path(__file__)),'_prepare'],env=env,check=True,stdin=subprocess.DEVNULL,timeout=5400)
    helper=load('command',repo/'tasks/r08_python_command.py');fw=load('forward',repo/'tasks/casmi_r08_forward.py')
    source=state/'artifacts/casmi26/research-r08/forward-v1'
    paths=[source/'deps',source/('fiora-'+SOURCE),repo/'work/casmi26',repo/'tasks']
    code='from pathlib import Path;import sys;import casmi_r08b_confirmation as m;m.evaluate(*map(Path,sys.argv[1:]))'
    root=state/'artifacts/casmi26/research-r08b/fixed-confirmation-v1'
    subprocess.run(helper.python_command(python,paths,code,[state,repo,out,root]),env=fw.env_clean(),
        check=True,stdin=subprocess.DEVNULL,timeout=7200)
    subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip',
        '--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
    return 0


if __name__=='__main__':raise SystemExit(main())

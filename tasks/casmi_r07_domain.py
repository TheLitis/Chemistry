"""R07/R08/R09: leakage-controlled timsTOF NP evaluation of complete systems.

Trains from scratch with ALL target NP keys excluded across every library.
External recovery uses actual public snapshot rows, never injected answers.
No provider credentials, Kaggle calls, test data or production weights touched.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def external_record(payload):
    from casmi26.production import molecule_record
    from casmi26.features_v3 import fingerprint_targets
    identifier,smi,hint=payload
    r=molecule_record(smi)
    if not r[2]:return None
    try:bits=fingerprint_targets(smi)
    except (ValueError,RuntimeError):return None
    return ['coconut:'+identifier,r[1],r[2],r[3]],bits,abs(r[3]-hint)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def execute(state,repo,out,*,device='cuda',epochs=30,calibration_size=80,
            snapshot_sha256='6e042af898b2fd3af3f52dc24536f15a26faac26dc0e27f0e5b072172a445b0d',
            workers=8,run_tests=True):
    state,repo,out=map(Path,(state,repo,out))
    if not 1<=epochs<=1000 or workers<1:raise ValueError('Invalid execution budget')
    py=Path(sys.executable)
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from rdkit import rdBase
    sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json,mass_candidates,reference_score,arrow_features
    from casmi26.features_v3 import arrow_highres
    from casmi26.portable import select_candidates
    from casmi26.target_domain import split_target,training_rows,rank_key,metrics,mass_prior,fingerprint_evidence,hybrid_scores,is_target_instrument
    from casmi26.model_v3 import train_model,MultiFingerprintModel
    from casmi26.pipeline_v3 import _excluded,export_old_model
    from casmi26.learning import train_arrays
    from casmi26.catalog_candidates import MassWindows,candidate_rows
    from casmi26.ranking import spectrum_signature,paired_effect
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    out.mkdir(parents=True,exist_ok=True)
    art=state/'artifacts/casmi26';root=art/'research-r07/full-system-v1';root.mkdir(parents=True,exist_ok=True)
    cache=state/'cache/casmi26/highres-v3'
    npfile=art/'research-r07/np-examples.parquet'
    inv=json.loads((art/'research-r07/inventory.json').read_text());train=Path(inv['train_path'])
    if sha256(train)!=inv['train_sha256'] or sha256(npfile)!=inv['np_examples']['sha256']:
        raise RuntimeError('Inventory inputs changed')
    started=time.monotonic();catalog=json.loads((cache/'catalog.json').read_text())
    lookup={r[0]:i for i,r in enumerate(catalog)};counts=np.load(cache/'counts.npy',allow_pickle=False)
    x=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    packed=np.load(cache/'targets.npy',mmap_mode='r',allow_pickle=False)
    masses=np.array([r[3] for r in catalog]);qtable=pq.read_table(npfile).combine_chunks()
    if not all(is_target_instrument(v) for v in qtable['instrument_type'].to_pylist()):
        raise RuntimeError('Non-timsTOF rows in target-domain corpus')
    prepared=json.loads((cache/'prepared-v3.json').read_text())
    if prepared['signature']['train_sha256']!=inv['train_sha256']:
        raise RuntimeError('Feature cache is based on a different training corpus')
    for name in ('features.npy','targets.npy','catalog.json','counts.npy'):
        if name in prepared.get('files',{}) and sha256(cache/name)!=prepared['files'][name]:
            raise RuntimeError('Modified training cache: '+name)
    groups=defaultdict(list);truthmass={};invalid=Counter();mass_diagnostics=[]
    for batch in qtable.to_batches(max_chunksize=2048):
        qx,valid,neutral=arrow_highres(batch)
        for j,row in enumerate(batch.to_pylist()):
            idx=lookup.get(row['normalized_smiles'])
            if idx is None or not valid[j]:
                invalid['unusable_rows']+=1;continue
            key=catalog[idx][2];p=np.column_stack([row['ms2_mzs'],row['ms2_normalized_intensities']])
            p=p[p[:,1]>0];p[:,1]/=p[:,1].sum()
            q={'x':qx[j],'mass':float(neutral[j]),'peaks':p,'precursor':row['precursor_mz'],'adduct':row['adduct']}
            groups[key].append(q);truthmass[key]=catalog[idx][3]
            mass_diagnostics.append({'key':key,'adduct':row['adduct'],'error_ppm':float((neutral[j]-truthmass[key])/truthmass[key]*1e6)})
    keys=sorted(groups);previous=_excluded(art,ignore=root)
    cal,audit=split_target(keys,previous,calibration_size)
    ids=training_rows(catalog,counts,set(keys))
    if {catalog[i][2] for i in ids}&set(keys):raise RuntimeError('Target leakage')
    implementation_files=[Path(__file__),*(repo/'work/casmi26/casmi26'/n for n in ('target_domain.py','production.py','learning.py','model_v3.py','features_v3.py','portable.py','catalog_candidates.py','ranking.py'))]
    implementation_hash=hashlib.sha256(''.join(sha256(p) for p in implementation_files).encode()).hexdigest()
    source={'implementation_sha256':implementation_hash,'coconut_snapshot_sha256':snapshot_sha256,'train_sha256':inv['train_sha256'],'target_np_sha256':sha256(npfile),
            'catalog_sha256':sha256(cache/'catalog.json'),'targets_sha256':sha256(cache/'targets.npy'),
            'training_key_sha256':hashlib.sha256('\n'.join(sorted(catalog[i][2] for i in ids)).encode()).hexdigest()}
    protocol={'experiment':'R07-domain-full-system','source':source,'calibration_keys':cal,'audit_keys':audit,
              'all_np_keys_excluded_from_training':True,'training_key_overlap':0,'training_rows':len(ids),
              'warmstart':None,'epochs':epochs,'device':device,
              'training_recipes':{'v1':{'learning_rate':.002,'seed':1729,'amp':False},'v3':{'learning_rate':.001,'seed':26091701,'amp':device.startswith('cuda')}},
              'queries':'only enveda-np-examples timsTOF; all valid acquisitions',
              'catalog_regimes':['available_other_source_references','no_target_references','target_keys_excised_then_public_recovery'],
              'reference_library':'all other sources; identical normalized query spectra removed',
              'external_snapshot':'2026-08','select':'calibration minimax gain against V1-style refit, then mean gain',
              'audit_frozen_before_model_selection':True,'test_data_read':False,'new_submissions':0,
              'note':'Target NP source is public training data, not a hidden test. Models trained from scratch, not exposed pretrained anchors.'}
    pp=root/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise RuntimeError('Sealed protocol changed')
    write_json(pp,protocol)
    done=root/'report.json'
    if done.exists():
        finished=json.loads(done.read_text())
        for name,h in finished['artifact_hashes'].items():
            if sha256(root/name)!=h:raise RuntimeError('Completed experiment artifact modified: '+name)
        for name in ('report.json','audit-ranks.json','protocol.json','selection-before-audit.json','mass-diagnostics.json'):
            shutil.copy2(root/name,out/name)
        print('R07_COMPLETE_REUSED',flush=True);return 0
    test_output='Tests not rerun inside this execution (caller verification required).'
    if run_tests:
        tests=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests/test_target_domain.py')],capture_output=True,text=True,timeout=180)
        test_output=tests.stdout+tests.stderr
        (out/'tests.log').write_text(test_output,encoding='utf-8')
        if tests.returncode:raise RuntimeError('Target-domain safeguards failed')
    training={};models={}
    for name,dim,heads in [('v1-independent',4104,(2048,)),('v3-independent',8200,(2048,4096,8192))]:
        path=root/(name+'.npz');meta=root/(name+'.training.json')
        recipe=protocol['training_recipes']['v1' if name=='v1-independent' else 'v3']
        identity={'source':source,'dimension':dim,'heads':list(heads),'epochs':epochs,'warmstart':None,**recipe}
        if path.exists():
            prior=json.loads(meta.read_text())
            if prior['identity']!=identity or prior['sha256']!=sha256(path):raise RuntimeError('Model identity changed')
            training[name]=prior['report']
        else:
            print('R07_TRAIN '+name,flush=True)
            if name=='v1-independent':
                # Reuse the original V1 optimizer/loss/seed, not the newer AMP recipe.
                legacy=root/'v1-independent-original.npz'
                r=train_arrays(np.asarray(x[ids,:4104]),np.unpackbits(packed[ids,:256],axis=1).astype(np.float32),
                    legacy,epochs=epochs,hidden=512,batch_size=256,device=device,seed=1729,learning_rate=.002)
                export_old_model(legacy,path)
            else:
                r=train_model(x,packed,ids,path,epochs=epochs,hidden=512,batch_size=256,device=device,
                              feature_dim=dim,head_sizes=heads,seed=26091701,learning_rate=.001)
            training[name]=r;write_json(meta,{'identity':identity,'report':r,'sha256':sha256(path)})
        models[name]=MultiFingerprintModel(path)
    print('R07_MODELS_READY',flush=True)
    # Independent structure source; only observed masses determine prefilter.
    observed={k:float(np.median([q['mass'] for q in groups[k]])) for k in keys}
    windows=MassWindows(observed.values(),ppm=50,da=.02,padding=.001)
    archive=state/'data/external/coconut-2026-08/coconut_csv_lite-08-2026.zip'
    expected=snapshot_sha256
    if sha256(archive)!=expected:raise RuntimeError('Public snapshot changed')
    emeta=root/'external.json';ebits=root/'external.npy'
    if emeta.exists() and ebits.exists():
        em=json.loads((root/'external-manifest.json').read_text())
        if em['snapshot_sha256']!=expected or em['catalog_sha256']!=sha256(emeta) or em['targets_sha256']!=sha256(ebits):
            raise RuntimeError('External candidate cache changed')
        erows=json.loads(emeta.read_text());epacked=np.load(ebits,allow_pickle=False)
    else:
        counter=Counter();pending=list(candidate_rows(archive,windows,counter));results=[]
        print('R07_PUBLIC_CANDIDATES '+str(len(pending)),flush=True)
        if workers==1:
            for r in map(external_record,pending):
                if r is not None and windows.contains(r[0][3]):results.append(r)
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                for r in pool.map(external_record,pending,chunksize=32):
                    if r is not None and windows.contains(r[0][3]):results.append(r)
        results.sort(key=lambda r:(r[0][3],r[0][2],r[0][0]));erows=[r[0] for r in results]
        epacked=np.stack([r[1] for r in results]);write_json(emeta,erows);np.save(ebits,epacked)
        write_json(root/'external-manifest.json',{'snapshot_sha256':expected,'processed':len(pending),'accepted':len(erows),
            'counters':dict(counter),'catalog_sha256':sha256(emeta),'targets_sha256':sha256(ebits)})
    emasses=np.array([r[3] for r in erows]);selections={};chosen=set()
    for key in keys:
        oi=mass_candidates(masses,observed[key],50,.02);ei=mass_candidates(emasses,observed[key],50,.02)
        selections[key]=(oi,ei);chosen.update(map(int,oi))
    lookupref={catalog[i][0]:int(i) for i in chosen}
    values=pa.array(list(lookupref),type=pa.string());library=defaultdict(list);lc=Counter();signatures=set()
    qsignatures={spectrum_signature(q) for rows in groups.values() for q in rows}
    cols=['normalized_smiles','ingest_lib','ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','collision_energy_ev']
    with pq.ParquetFile(train) as f:
        for b in f.iter_batches(batch_size=8192,columns=cols):
            lc['scanned']+=len(b)
            mask=pc.and_(pc.is_in(b.column('normalized_smiles'),value_set=values),pc.not_equal(b.column('ingest_lib'),'enveda-np-examples'))
            part=b.filter(mask)
            if not len(part):continue
            _,valid,neutral=arrow_features(part)
            for j,row in enumerate(part.to_pylist()):
                i=lookupref[row['normalized_smiles']];mass=masses[i]
                if not valid[j] or abs(neutral[j]-mass)>max(.003,mass*50e-6):lc['invalid']+=1;continue
                p=np.column_stack([row['ms2_mzs'],row['ms2_normalized_intensities']]);p=p[p[:,1]>0];p[:,1]/=p[:,1].sum()
                r={'peaks':p,'precursor':row['precursor_mz'],'adduct':row['adduct']}
                sig=spectrum_signature(r);pair=(catalog[i][2],sig)
                if sig in qsignatures:lc['query_identical_removed']+=1;continue
                if pair in signatures:lc['reference_duplicates_removed']+=1;continue
                signatures.add(pair);library[catalog[i][2]].append(r);lc['accepted']+=1
    print('R07_LIBRARY '+json.dumps(lc),flush=True)
    # Candidate-level evidence. Truth is consulted only by rank_key after scoring.
    saved=[];evidence={}
    for number,key in enumerate(keys,1):
        qs=groups[key];oi,ei=selections[key];obs=observed[key]
        rows=[catalog[i] for i in oi]+[erows[i] for i in ei]
        bits=np.concatenate([packed[oi],epacked[ei]])
        origin=np.concatenate([np.ones(len(oi),bool),np.zeros(len(ei),bool)])
        scache={k:float(np.mean([max((reference_score(q,r) for r in library[k]),default=0.) for q in qs])) for k in {r[2] for r in rows}}
        avg=np.mean([q['x'] for q in qs],axis=0);z1=models['v1-independent'].logits(avg[:4104]);z3=models['v3-independent'].logits(avg)
        p1=1/(1+np.exp(-np.clip(z1,-40,40)))
        unpack=np.unpackbits(bits,axis=1)
        f=unpack[:,:2048];dot=f@p1;tani=dot/np.maximum(p1.sum()+f.sum(1)-dot,1e-8)
        ll=fingerprint_evidence(z3,unpack)@np.array([.5,.25,.25])
        # R01-style scores retained as an explicit comparator, not as probabilities.
        llstd=(ll-ll.mean())/max(float(ll.std()),1e-8) if len(ll) else ll
        spectra=np.array([scache[r[2]] for r in rows]);spectra_abs=spectra.copy()
        spectra_abs[[i for i,r in enumerate(rows) if r[2] in groups]]=0
        allkeys=[r[2] for r in rows];cmass=np.array([r[3] for r in rows])
        cases={}
        for case in ('available','absent','external_recovery'):
            admitted=np.flatnonzero((~origin)|np.array([r[2] not in groups for r in rows])) if case=='external_recovery' else np.arange(len(rows))
            # Original representative preferred when present; no key appears twice.
            unique=[];seen=set()
            for i in admitted:
                if allkeys[i] not in seen:seen.add(allkeys[i]);unique.append(i)
            ii=np.array(unique,dtype=np.int64)
            cases[case]={'keys':[allkeys[i] for i in ii],'masses':cmass[ii],
                'v1_tanimoto':tani[ii],'v3_ll':ll[ii],'v3_standardized':llstd[ii],
                'spectral':(spectra if case=='available' else spectra_abs)[ii],
                'external':(~origin[ii]).astype(float)}
        # Baseline: ORIGINAL catalog only, original V1 .75/.25 recipe, same independent fit.
        strict,_=select_candidates(masses,obs)
        keep_original=np.isin(oi,strict)
        s1=spectra[:len(oi)];n1=tani[:len(oi)]
        baseline={};baseline_coverage={};baseline_counts={}
        for case in cases:
            ss=s1 if case=='available' else spectra_abs[:len(oi)]
            mask=np.array([catalog[i][2] not in groups for i in oi]) if case=='external_recovery' else np.ones(len(oi),bool)
            mask &= keep_original
            bk=[catalog[i][2] for i in oi[mask]]
            baseline[case]=rank_key((.75*ss+.25*n1)[mask],bk,key)
            baseline_coverage[case]=key in bk;baseline_counts[case]=len(set(bk))
        evidence[key]={'cases':cases,'baseline':baseline,'baseline_coverage':baseline_coverage,'baseline_counts':baseline_counts,'observed':obs,'has_reference':bool(library[key]),'spectra':len(qs)}
        if number%25==0:print('R07_EVIDENCE '+str(number)+'/'+str(len(keys)),flush=True)
    # Robust mass center fitted only on calibration labels; no audit mass correction fitting.
    offset=float(np.median([(observed[k]-truthmass[k])/truthmass[k]*1e6 for k in cal]))
    configurations=[]
    for kind in ('v1_tanimoto','v3_ll','v3_standardized'):
        for sw in (0.,.5,1.,2.,4.,8.):
            for mw in (0.,.1,.5):
                for penalty in (0.,.2):
                    configurations.append({'kind':kind,'spectral_weight':sw,'mass_weight':mw,'external_penalty':penalty})
    def trial(conf,which):
        result={c:[] for c in ('available','absent','external_recovery')}
        for key in which:
            for case,data in evidence[key]['cases'].items():
                mass=mass_prior(data['masses'],observed[key],offset_ppm=offset,scale_ppm=5)
                score=hybrid_scores(data[conf['kind']],data['spectral'],mass,spectral_weight=conf['spectral_weight'],mass_weight=conf['mass_weight'])
                score-=conf['external_penalty']*data['external']
                result[case].append(rank_key(score,data['keys'],key))
        return result
    bascal={c:metrics([evidence[k]['baseline'][c] for k in cal]) for c in ('available','absent','external_recovery')}
    candidates=[]
    for conf in configurations:
        ranks=trial(conf,cal);m={c:metrics(v) for c,v in ranks.items()}
        delta={c:m[c]['mrr_at_25']-bascal[c]['mrr_at_25'] for c in m}
        candidates.append({'configuration':conf,'metrics':m,'min_gain':min(delta.values()),'mean_gain':float(np.mean(list(delta.values())))})
    # Keep no-change option. The final holdout is untouched until this file is written.
    candidates.append({'configuration':{'kind':'baseline'},'metrics':bascal,'min_gain':0.,'mean_gain':0.})
    winner=max(candidates,key=lambda c:(c['min_gain'],c['mean_gain']))
    selection={'selected':winner,'mass_offset_ppm':offset,'mass_scale_ppm':5.,'configurations':len(configurations),
        'calibration':candidates,'audit_key_sha256':hashlib.sha256('\n'.join(audit).encode()).hexdigest(),
        'selected_before_audit':True,'criterion':'max minimum MRR gain across the three target-domain regimes, then mean gain'}
    write_json(root/'selection-before-audit.json',selection)
    chosenconf=winner['configuration']
    ra=trial(chosenconf,audit) if chosenconf['kind']!='baseline' else {c:[evidence[k]['baseline'][c] for k in audit] for c in bascal}
    rr=lambda v:np.where(np.array(v)>0,1/np.maximum(v,1),0.)
    auditresult={}
    for case in bascal:
        br=[evidence[k]['baseline'][case] for k in audit]
        auditresult[case]={'baseline':metrics(br),'selected':metrics(ra[case]),'paired':paired_effect(rr(br),rr(ra[case])),
            'selected_candidate_coverage':float(np.mean([evidence[k]['baseline_coverage'][case] if chosenconf['kind']=='baseline' else k in evidence[k]['cases'][case]['keys'] for k in audit])),
            'reference_available_fraction':float(np.mean([evidence[k]['has_reference'] for k in audit]))}
    details=[{'key':k,'spectra':evidence[k]['spectra'],'has_non_np_reference':evidence[k]['has_reference'],
        'baseline':evidence[k]['baseline'],'selected':{c:ra[c][i] for c in ra},
        'candidate_counts':{c:evidence[k]['baseline_counts'][c] if chosenconf['kind']=='baseline' else len(d['keys']) for c,d in evidence[k]['cases'].items()},
        'covered':{c:evidence[k]['baseline_coverage'][c] if chosenconf['kind']=='baseline' else k in d['keys'] for c,d in evidence[k]['cases'].items()}} for i,k in enumerate(audit)]
    write_json(root/'calibration-keys.json',cal)
    report={'experiment':'R07-domain-full-system','status':'completed','commit':os.environ.get('GITHUB_SHA'),
        'training':training,'source':source,'target_molecules':len(keys),'target_spectra':sum(len(v) for v in groups.values()),
        'calibration_molecules':len(cal),'audit_molecules':len(audit),'old_evaluated_np_keys_in_calibration':len(set(keys)&previous),
        'training_target_overlap':0,'warmstart':None,'selected':chosenconf,'calibration_min_gain':winner['min_gain'],
        'mass_offset_ppm':offset,'audit':auditresult,'external_rows':len(erows),'external_unique_keys':len({r[2] for r in erows}),
        'library':dict(lc),'invalid_query_rows':dict(invalid),'seconds':time.monotonic()-started,
        'tests':test_output.strip(),'official_score':None,'new_submissions':0,'test_data_read':False,
        'original_champion_replaced':False,
        'limitations':['Primary control is the complete V1 recipe retrained without NP targets, not its exposed production checkpoint.',
            '250 common NP molecules are not the hidden test or genuinely novel structures.',
            'External recovery is database retrieval after catalog excision, not de novo.',
            'All target keys are excluded from fitting across all libraries; their other-source spectra are available only in the library-present regime.',
            'Calibration grid results are not independent validation results.',
            'No exact hidden performance or candidate coverage can be inferred from this audit.']}
    write_json(root/'audit-ranks.json',details);write_json(root/'mass-diagnostics.json',mass_diagnostics)
    files=('audit-ranks.json','mass-diagnostics.json','selection-before-audit.json','v1-independent.npz','v3-independent.npz','external.json','external.npy')
    report['artifact_hashes']={name:sha256(root/name) for name in files}
    write_json(root/'report.json',report)
    for name in ('report.json','audit-ranks.json','protocol.json','selection-before-audit.json','mass-diagnostics.json'):
        shutil.copy2(root/name,out/name)
    print('R07_RESULT_BEGIN\n'+json.dumps(report,indent=2)+'\nR07_RESULT_END',flush=True)
    return 0


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,
            'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
    return execute(state,Path(__file__).resolve().parents[1],Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']))


if __name__=='__main__':raise SystemExit(main())

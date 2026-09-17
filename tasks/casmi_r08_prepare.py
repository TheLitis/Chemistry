"""Build frozen R07 candidate evidence for a forward-spectrum test, not a submission.

The 80/170 NP split is reused and explicitly diagnostic. A further 128 previously
unused hash-holdout timsTOF structures provide an untouched transfer audit.
"""
from __future__ import annotations
from collections import defaultdict,Counter
import gzip,hashlib,importlib.util,json,os,re,shutil,subprocess,sys,time
from pathlib import Path


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json,mass_candidates,reference_score,arrow_features,is_validation
    from casmi26.model_v3 import MultiFingerprintModel
    from casmi26.r07_release import external_candidates,selected_scores
    from casmi26.ranking import spectrum_signature
    from casmi26.target_domain import training_rows,rank_key
    from casmi26.pipeline_v3 import _excluded
    from casmi26.forward_ranking import candidate_frontier,ENERGIES
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    art=state/'artifacts/casmi26';study=art/'research-r07/full-system-v1';cache=state/'cache/casmi26/highres-v3'
    root=art/'research-r08/forward-v1';root.mkdir(parents=True,exist_ok=True)
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    prior=json.loads((study/'report.json').read_text());oldplan=json.loads((study/'protocol.json').read_text())
    for name,digest in prior['artifact_hashes'].items():
        if sha256(study/name)!=digest:raise RuntimeError('R07 evidence modified: '+name)
    inv=json.loads((art/'research-r07/inventory.json').read_text());train=Path(inv['train_path'])
    if sha256(train)!=prior['source']['train_sha256']:raise RuntimeError('Training data changed')
    catalog=json.loads((cache/'catalog.json').read_text());packed=np.load(cache/'targets.npy',mmap_mode='r',allow_pickle=False)
    counts=np.load(cache/'counts.npy',allow_pickle=False);masses=np.array([r[3] for r in catalog]);lookup={r[0]:i for i,r in enumerate(catalog)}
    npkeys=set(oldplan['calibration_keys'])|set(oldplan['audit_keys'])
    fitting=training_rows(catalog,counts,npkeys)
    keyhash=hashlib.sha256('\n'.join(sorted(catalog[i][2] for i in fitting)).encode()).hexdigest()
    if keyhash!=prior['source']['training_key_sha256']:raise RuntimeError('Training-key lineage mismatch')
    used=_excluded(art)|npkeys
    for path in art.glob('research-*/**/audit-ranks.json'):
        for r in json.loads(path.read_text()):
            if isinstance(r,dict) and isinstance(r.get('key'),str):used.add(r['key'])
    eligible={}
    with pq.ParquetFile(train) as pf:
        for batch in pf.iter_batches(batch_size=65536,columns=['normalized_smiles','ingest_lib']):
            part=batch.filter(pc.equal(batch.column('ingest_lib'),'enveda-180'))
            for raw in pc.unique(part.column('normalized_smiles')).to_pylist():
                i=lookup.get(raw)
                if i is not None and counts[i]>0 and is_validation(catalog[i][2]) and catalog[i][2] not in used:
                    eligible[catalog[i][2]]=i
    fresh=sorted(eligible,key=lambda k:hashlib.sha256(('R08-forward-transfer-20260917:'+k).encode()).digest())[:128]
    if len(fresh)!=128:raise RuntimeError('Insufficient fresh target-domain audit keys')
    if set(fresh)&{catalog[i][2] for i in fitting}:raise RuntimeError('Fresh audit leaked to fitted model')
    freshset=set(fresh);queries=defaultdict(list);groups=defaultdict(list)
    cols=['normalized_smiles','ingest_lib','ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','collision_energy_ev']
    tables=[pq.read_table(art/'research-r07/np-examples.parquet',columns=cols),
        pq.read_table(train,columns=cols,filters=[('ingest_lib','=','enveda-180'),('normalized_smiles','in',[r[0] for r in catalog if r[2] in freshset])])]
    for table in tables:
        for batch in table.to_batches(max_chunksize=2048):
            x,valid,neutral=arrow_features(batch)
            for j,row in enumerate(batch.to_pylist()):
                idx=lookup.get(row['normalized_smiles'])
                if idx is None or not valid[j]:continue
                key=catalog[idx][2];p=np.column_stack([row['ms2_mzs'],row['ms2_normalized_intensities']]);p=p[p[:,1]>0];p[:,1]/=p[:,1].sum()
                q={'x':x[j],'mass':float(neutral[j]),'peaks':p,'precursor':row['precursor_mz'],'adduct':row['adduct'],'ce':row['collision_energy_ev']}
                groups[key].append(q);queries[key].append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in q.items() if k!='x'})
    if set(groups)!=npkeys|freshset:raise RuntimeError('No usable spectrum for a prespecified query')
    observed={k:float(np.median([q['mass'] for q in v])) for k,v in groups.items()}
    plan={'experiment':'R08-forward-evidence-v1','calibration_keys':oldplan['calibration_keys'],
        'reused_np_audit_keys':oldplan['audit_keys'],'fresh_transfer_keys':fresh,'fitting_overlap':0,
        'fresh_keys_previously_used':0,'np_audit_is_independent':False,'source_train':sha256(train),
        'r07_report_sha256':sha256(study/'report.json'),'holdout_model_sha256':sha256(study/'v1-independent.npz'),
        'shortlist_per_regime':32,'collision_energy_grid':list(ENERGIES),'regimes':['available','absent','external_recovery'],
        'forward_source_commit':'e19ef82c9a6cb9dbac92bce23e914008f1aeb44e',
        'forward_pretraining_overlap':'unknown; frozen public model, not claimed de novo generalization',
        'forward_weights':[0.,.1,.25,.5,1.,2.],'features':['cosine_nearest','cosine_max','entropy_nearest','entropy_max','coverage_nearest','coverage_max'],
        'selection':'calibration only: max minimum gain across regimes then mean; no-change option always included',
        'test_data_read':False,'new_submissions':0,'primary_new_audit':'128 unseen enveda-180 structures, not representative of novel natural products'}
    pp=root/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=plan:raise RuntimeError('R08 protocol differs; refusing adaptive resampling')
    write_json(pp,plan)
    erows=json.loads((study/'external.json').read_text());efp=np.load(study/'external.npy',mmap_mode='r',allow_pickle=False)[:,:256]
    freshpath=root/'fresh-external.json'
    if freshpath.exists():
        fr=json.loads(freshpath.read_text());ff=np.load(root/'fresh-external.npy',allow_pickle=False)
    else:
        archive=state/'data/external/coconut-2026-08/coconut_csv_lite-08-2026.zip'
        if sha256(archive)!=prior['source']['coconut_snapshot_sha256']:raise RuntimeError('External snapshot mismatch')
        fr,ff,_=external_candidates(archive,[observed[k] for k in fresh],workers=8)
        write_json(freshpath,fr);np.save(root/'fresh-external.npy',ff)
    extmass=np.array([r[3] for r in erows]);fmass=np.array([r[3] for r in fr]);selections={};chosen=set()
    for key,obs in observed.items():
        oi=mass_candidates(masses,obs,50,.02);ei=mass_candidates(extmass if key in npkeys else fmass,obs,50,.02)
        selections[key]=(oi,ei);chosen.update(map(int,oi))
    lookupref={catalog[i][0]:int(i) for i in chosen};values=pa.array(list(lookupref),type=pa.string())
    library=defaultdict(list);seen=set();qc={spectrum_signature(q) for qs in groups.values() for q in qs};lc=Counter()
    with pq.ParquetFile(train) as pf:
        for batch in pf.iter_batches(batch_size=8192,columns=cols):
            lc['scanned']+=len(batch)
            mask=pc.and_(pc.is_in(batch.column('normalized_smiles'),value_set=values),pc.not_equal(batch.column('ingest_lib'),'enveda-np-examples'))
            part=batch.filter(mask)
            if not len(part):continue
            _,valid,neutral=arrow_features(part)
            for j,row in enumerate(part.to_pylist()):
                i=lookupref[row['normalized_smiles']]
                if not valid[j] or abs(neutral[j]-masses[i])>max(.003,masses[i]*50e-6):continue
                p=np.column_stack([row['ms2_mzs'],row['ms2_normalized_intensities']]);p=p[p[:,1]>0];p[:,1]/=p[:,1].sum()
                r={'peaks':p,'precursor':row['precursor_mz'],'adduct':row['adduct'],'source':row['ingest_lib']}
                sig=spectrum_signature(r);pair=(catalog[i][2],sig)
                if sig in qc:lc['identical_query_removed']+=1;continue
                if pair in seen:continue
                seen.add(pair);library[catalog[i][2]].append(r);lc['accepted']+=1
    model=MultiFingerprintModel(study/'v1-independent.npz');records=[];requests={};replayed={};start=time.monotonic()
    for n,key in enumerate(sorted(groups),1):
        qs=groups[key];obs=observed[key];oi,ei=selections[key];ext,fp=(erows,efp) if key in npkeys else (fr,ff)
        rows=[catalog[i] for i in oi]+[ext[i] for i in ei];bits=np.concatenate([packed[oi,:256],fp[ei]])
        flags=np.concatenate([np.zeros(len(oi)),np.ones(len(ei))]);target=npkeys if key in npkeys else freshset
        z=model.logits(np.mean([q['x'] for q in qs],axis=0))
        def spectrum_score(k):
            refs=library[k]
            if key in freshset and k in freshset:refs=[r for r in refs if r['source']!='enveda-180']
            return float(np.mean([max((reference_score(q,r) for r in refs),default=0.) for q in qs]))
        spect={k:spectrum_score(k) for k in {r[2] for r in rows}}
        cases={}
        for case in ('available','absent','external_recovery'):
            unique=[];seenkeys=set()
            for i,r in enumerate(rows):
                if case=='external_recovery' and not flags[i] and r[2] in target:continue
                if r[2] not in seenkeys:seenkeys.add(r[2]);unique.append(i)
            ii=np.asarray(unique,dtype=np.int64);rs=[rows[i] for i in ii]
            ss=np.array([spect[r[2]] if case=='available' or r[2] not in target else 0. for r in rs])
            scores=selected_scores(z,np.unpackbits(bits[ii],axis=1),ss,np.array([r[3] for r in rs]),obs,flags[ii],prior['selected'],prior['mass_offset_ppm'])
            cases[case]={'keys':[r[2] for r in rs],'smiles':[r[1] for r in rs],'masses':[r[3] for r in rs],
                         'scores':scores.tolist(),'spectral':ss.tolist(),'external':flags[ii].tolist()}
        if key in npkeys:replayed[key]={c:rank_key(d['scores'],d['keys'],key) for c,d in cases.items()}
        frontier=candidate_frontier(cases,plan['shortlist_per_regime'])
        structures={r[2]:r[1] for r in reversed(rows)}
        modes=sorted({q['adduct'] for q in qs if q['adduct'] in ('[M+H]+','[M-H]-')})
        for k in frontier:
            request=requests.setdefault(k,{'key':k,'smiles':structures[k],'modes':[]})
            request['modes']=sorted(set(request['modes'])|set(modes))
        records.append({'key':key,'cohort':'np' if key in npkeys else 'fresh','queries':queries[key],
                        'frontier':frontier,'cases':cases})
        if n%25==0:print('R08_EVIDENCE '+str(n)+'/'+str(len(groups)),flush=True)
    previous=json.loads((study/'audit-ranks.json').read_text());errors=[]
    for r in previous:
        if r['selected']!=replayed[r['key']]:errors.append({'key':r['key'],'old':r['selected'],'new':replayed[r['key']]})
    write_json(root/'replay.json',{'rows':len(previous),'matching':len(previous)-len(errors),'errors':errors})
    if errors:raise RuntimeError('R07 baseline replay did not match; new audit not evaluated')
    with gzip.open(root/'evidence.json.gz','wt',encoding='utf-8') as f:json.dump(records,f,allow_nan=False,separators=(',',':'))
    write_json(root/'requests.json',list(requests.values()))
    summary={'status':'prepared','commit':os.environ.get('GITHUB_SHA'),'molecules':len(groups),
        'raw_query_spectra':sum(map(len,groups.values())),'fresh_molecules':len(fresh),
        'candidate_structures_to_simulate':len(requests),'candidate_mode_pairs':sum(len(r['modes']) for r in requests.values()),
        'energy_conditions_per_mode':len(ENERGIES),'r07_replayed_audit_rows':len(previous),
        'r07_replay_mismatches':0,'library':dict(lc),'no_test_data':True,'new_submissions':0,
        'seconds_scoring':time.monotonic()-start,'files':{n:sha256(root/n) for n in ('protocol.json','evidence.json.gz','requests.json','replay.json')}}
    write_json(root/'prepared.json',summary)
    for name in ('prepared.json','protocol.json','evidence.json.gz','requests.json','replay.json'):shutil.copy2(root/name,out/name)
    print('R08_PREPARED_BEGIN\n'+json.dumps(summary,indent=2)+'\nR08_PREPARED_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

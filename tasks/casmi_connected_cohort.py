"""Prepare new independent query cohorts with the unchanged R07 base recipe.

Adapted from the verified R08B preparation; original experiment code and files
remain untouched. The new seed/root/selection distinguish this confirmation.
"""
from __future__ import annotations
from collections import defaultdict, Counter
import gzip,hashlib,json,os
from pathlib import Path
from casmi_r08b_confirmation import used_keys
from casmi_connected_confirmation import choose_keys,SEED,ROOT_RELATIVE,SELECTION,GATES
SOURCE='e19ef82c9a6cb9dbac92bce23e914008f1aeb44e'


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
    root = state/ROOT_RELATIVE
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
    write_json(root/'previous-query-keys.json',sorted(excluded))
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
    protocol = {'experiment': 'R10G-prospective-confirmation-v1', 'target_keys': target, 'external_keys': external,
        'target_source': 'enveda-180', 'external_source_rule': 'one hashed source per previously unseen externally covered key',
        'external_pool': 'intersection of fixed prior COCONUT mass-selected public records and unused hash-holdout structures',
        'external_pool_not_random_all_COCONUT': True, 'query_source': query_source,
        'excluded_prior_key_count': len(excluded), 'training_query_overlap': 0,
        'frozen_selection': SELECTION, 'new_weight_search': False,
        'base_recipe': prior['selected'], 'base_mass_offset_ppm': prior['mass_offset_ppm'],
        'r07_report_sha256': sha256(old/'report.json'), 'train_sha256': sha256(train),
        'holdout_model_sha256': sha256(old/'v1-independent.npz'),
        'coconut_sha256': prior['source']['coconut_snapshot_sha256'], 'forward_source': SOURCE,
        'k': 25, 'shortlist_cap': None, 'all_inputs_before_ranks': True,
        'gates': GATES,
        'screening_note': '224 prior R08B query keys used for R10 screening are excluded; fixed graph variant selected before new cases',
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
        if number%16==0:print('R10G_PREPARE '+str(number),flush=True)
    with gzip.open(root/'evidence.json.gz','wt',encoding='utf-8') as f:json.dump(records,f,allow_nan=False,separators=(',',':'))
    prepared={'status':'prepared','script_sha256':sha256(Path(__file__)),'records':len(records),
        'query_spectra':sum(map(len,groups.values())),'reference_counts':dict(refcounts),
        'files':{n:sha256(root/n) for n in ('protocol.json','evidence.json.gz','external.json','external.npy','previous-query-keys.json')}}
    write_json(done,prepared);print('R10G_PREPARED '+json.dumps({k:v for k,v in prepared.items() if k!='files'}),flush=True)
    return root

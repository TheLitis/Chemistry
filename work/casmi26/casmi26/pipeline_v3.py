"""Controlled training, frozen validation selection, and train-only v3 bundles."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .features_v3 import HEAD_SIZES, FEATURE_VERSION
from .model_v3 import MultiFingerprintModel, candidate_scores, train_model
from .production import is_validation, mass_candidates, sha256, write_json


def fresh_partition(keys, excluded, calibration_size=800, audit_size=1600):
    pool=sorted(set(keys)-set(excluded),key=lambda k:hashlib.sha256(('v3-implementation-r05:'+k).encode()).digest())
    if calibration_size<1 or audit_size<1 or len(pool)<calibration_size+audit_size:
        raise ValueError('Insufficient unused molecular keys')
    return pool[:calibration_size],pool[calibration_size:calibration_size+audit_size]


def _excluded(artifacts, ignore=None):
    used=set()
    base=Path(artifacts)/'official-v1/validation.json'
    if base.exists():used.update(v['key'] for v in json.loads(base.read_text())['details'])
    for root in Path(artifacts).glob('research-*'):
        if ignore is not None and root.resolve()==Path(ignore).resolve():continue
        for name in ('calibration-keys.json','audit-ranks.json'):
            path=root/name
            if path.exists():
                for row in json.loads(path.read_text()):used.add(row if isinstance(row,str) else row['key'])
        path=root/'selection-before-evaluation.json'
        if path.exists():used.update(json.loads(path.read_text()).get('keys',[]))
    return used


def _rank(scores, keys, truth):
    seen=set()
    for i in np.argsort(-scores,kind='stable'):
        key=keys[int(i)]
        if key in seen:continue
        seen.add(key)
        if len(seen)>25:break
        if key==truth:return len(seen)
    return 0


def _metrics(ranks):
    r=np.asarray(ranks)
    if not len(r):return {'molecules':0,'mrr_at_25':None,'top1':None,'recall_at_25':None}
    return {'molecules':len(r),'mrr_at_25':float(np.where(r>0,1/np.maximum(r,1),0).mean()),
            'top1':float((r==1).mean()),'recall_at_25':float((r>0).mean())}


def _keys_hash(keys):return hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def export_old_model(source, destination):
    from .learning import FingerprintRanker
    m=FingerprintRanker(source)
    np.savez_compressed(destination,format=np.array(3),feature_dim=np.array(4104),head_sizes=np.array([2048]),
                        w1=m.w1,b1=m.b1,w2=m.w2,b2=m.b2)
    return MultiFingerprintModel(destination)


def fit_experiment(cache, old_artifacts, destination, *, epochs=10, device='cuda',calibration_size=800,audit_size=1600):
    from .ranking import paired_effect
    cache,old_artifacts,destination=map(Path,(cache,old_artifacts,destination))
    destination.mkdir(parents=True,exist_ok=True)
    prepared=json.loads((cache/'prepared-v3.json').read_text())
    for name,h in prepared['files'].items():
        if sha256(cache/name)!=h:raise ValueError('Prepared feature/target cache was modified')
    source=prepared['signature'];catalog=json.loads((cache/'catalog.json').read_text())
    x=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    fp=np.load(cache/'targets.npy',mmap_mode='r',allow_pickle=False)
    counts=np.load(cache/'counts.npy',allow_pickle=False);observed=np.load(cache/'observed.npy',allow_pickle=False)
    masses=np.array([r[3] for r in catalog]);strata=np.load(cache/'strata.npy',allow_pickle=False)
    trainidx=np.array([i for i,r in enumerate(catalog) if counts[i]>0 and not is_validation(r[2])])
    representatives={}
    for i,r in enumerate(catalog):
        if counts[i]>0 and is_validation(r[2]):
            if r[2] not in representatives or counts[i]>counts[representatives[r[2]]]:representatives[r[2]]=i
    used=_excluded(old_artifacts.parent,ignore=destination)
    cal,audit=fresh_partition(representatives,used,calibration_size,audit_size)
    variants={'v1_frozen':{'model':'v1-frozen.npz','weights':[1.]},
              'v1_finetune':{'model':'v1-finetune.npz','weights':[1.]},
              'highres_morgan':{'model':'highres-morgan.npz','weights':[1.]},
              'multitarget_morgan':{'model':'multitarget.npz','weights':[1.,0.,0.]},
              'multitarget_balanced':{'model':'multitarget.npz','weights':[.5,.25,.25]},
              'multitarget_equal':{'model':'multitarget.npz','weights':[1.,1.,1.]}}
    protocol={'experiment':'v3-highres-multitarget','source':source,'epochs':epochs,'learning_rate':.0005,
              'warmstart_sha256':sha256(old_artifacts/'holdout-model.npz'),
              'calibration_keys':cal,'audit_keys':audit,'prior_used_keys_excluded':len(used),
              'training_keys_sha256':_keys_hash(sorted({catalog[i][2] for i in trainidx})),
              'variants':variants,'seed':26091605,'catalog':'inclusive','select_on':'calibration_mrr_only',
              'routing':'R03 fixed confidence 0.95/0.05 optional; neural audit does not prove routing gain',
              'test_data_read':False}
    plan=destination/'protocol.json'
    if plan.exists() and json.loads(plan.read_text())!=protocol:raise ValueError('Do not mutate a sealed experiment')
    write_json(plan,protocol)
    done=destination/'validation-v3.json'
    if done.exists():
        result=json.loads(done.read_text())
        for name,h in result['model_hashes'].items():
            if sha256(destination/name)!=h:raise ValueError('Evaluated model was modified')
        return result
    if {catalog[i][2] for i in trainidx}&(set(cal)|set(audit)):raise RuntimeError('Molecular split overlap')
    frozen=destination/'v1-frozen.npz'
    if not frozen.exists():export_old_model(old_artifacts/'holdout-model.npz',frozen)
    configurations=(('v1-finetune.npz',4104,(2048,)),('highres-morgan.npz',8200,(2048,)),('multitarget.npz',8200,HEAD_SIZES))
    training={}
    for name,dim,heads in configurations:
        path=destination/name;meta=destination/(name+'.training.json')
        identity={'source':source,'epochs':epochs,'feature_dim':dim,'head_sizes':list(heads),
                  'warmstart_sha256':protocol['warmstart_sha256'],'training_keys_sha256':protocol['training_keys_sha256']}
        if path.exists():
            prior=json.loads(meta.read_text())
            if prior['identity']!=identity or prior['model_sha256']!=sha256(path):raise ValueError('Stale checkpoint')
            training[name]=prior['training'];continue
        print('V3_TRAIN_START '+name,flush=True)
        report=train_model(x,fp,trainidx,path,epochs=epochs,device=device,feature_dim=dim,head_sizes=heads,
                           warmstart=old_artifacts/'holdout-model.npz')
        write_json(meta,{'identity':identity,'training':report,'model_sha256':sha256(path)});training[name]=report
    models={name:MultiFingerprintModel(destination/name) for name in sorted({v['model'] for v in variants.values()})}
    def predictions(keys):
        ids=np.array([representatives[k] for k in keys]);out={}
        for name,m in models.items():
            out[name]=np.concatenate([m.logits(x[ids[b:b+128],:m.feature_dim]) for b in range(0,len(ids),128)])
        return ids,out
    def measure(keys):
        queryidx,predictions_=predictions(keys);ranks={name:[] for name in variants};details=[]
        for q,key in enumerate(keys):
            i=queryidx[q];idx=mass_candidates(masses,observed[i]);keys_=[catalog[j][2] for j in idx]
            bits=np.unpackbits(fp[idx],axis=1);d={'key':key,'spectra':int(counts[i]),'candidates':len(set(keys_)),
                'strata':int(strata[i]),'ranks':{}}
            for name,v in variants.items():
                model=models[v['model']];n=sum(model.head_sizes)
                s=candidate_scores(predictions_[v['model']][q],bits[:,:n],masses[idx],observed[i],
                    head_sizes=model.head_sizes,weights=v['weights'])
                rank=_rank(s,keys_,key);ranks[name].append(rank);d['ranks'][name]=rank
            details.append(d)
        return ranks,details
    print('V3_CALIBRATION_START',flush=True)
    calr,_=measure(cal);calmetrics={name:_metrics(r) for name,r in calr.items()}
    # Include no-change baseline; deterministic insertion-order tie resolution.
    winner=max(calmetrics,key=lambda name:calmetrics[name]['mrr_at_25'])
    selection={'variant':winner,**variants[winner],'calibration':calmetrics,'audit_keys_sha256':_keys_hash(audit),
               'checkpoint_sha256':sha256(destination/variants[winner]['model']),'selected_before_audit':True}
    write_json(destination/'selection-before-audit.json',selection)
    print('V3_LOCKED_SELECTION '+winner,flush=True)
    auditr,details=measure(audit)
    metrics={name:_metrics(r) for name,r in auditr.items()}
    def rr(a):
        a=np.asarray(a);return np.where(a>0,1/np.maximum(a,1),0.)
    effect=paired_effect(rr(auditr['v1_frozen']),rr(auditr[winner]))
    # Subgroups are diagnostic only; they cannot choose another model or blend.
    groups={}
    for label,mask in [('bruker',1),('positive',2),('negative',4),('np_examples',8)]:
        ids=[i for i,d in enumerate(details) if d['strata']&mask]
        groups[label]={name:_metrics([r[i] for i in ids]) for name,r in auditr.items()}
    report={'experiment':'v3-highres-multitarget','status':'completed','selection':selection,'calibration':calmetrics,
            'audit':metrics,'audit_subgroups':groups,'selected_vs_frozen':effect,
            'model_hashes':{name:sha256(destination/name) for name in models},'training':training,
            'calibration_molecules':len(cal),'audit_molecules':len(audit),'training_query_overlap':0,
            'previous_keys_excluded':len(used),'eligible_for_candidate_bundle':winner!='v1_frozen' and effect['ci95'][0]>0,
            'official_score':None,'test_data_read':False,'production_champion_changed':False,
            'limitations':['Known-structure-inclusive catalog; not de novo.',
                'All-spectra aggregated train holdouts; not a guarantee of Bruker or hidden-test transfer.',
                'Subgroups overlap and are diagnostics, not model-selection targets.',
                'R03 routing is not revalidated by this neural-only comparison.',
                'No descriptor oracle is reported as a trained prediction score.']}
    write_json(destination/'calibration-keys.json',cal);write_json(destination/'audit-ranks.json',details)
    write_json(done,report)
    return report


def verify_bundle(folder):
    folder=Path(folder);m=json.loads((folder/'v3-bundle.json').read_text())
    required={'model.npz','targets.npy','catalog.json'}
    if m.get('format')!=3 or set(m.get('files',{}))!=required:
        raise ValueError('Incomplete/unsupported v3 bundle')
    if m.get('mode') not in ('neural','confidence','legacy'):
        raise ValueError('Unknown routing mode')
    if m.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Bundle feature version differs from the inference transform')
    if m['mode'] == 'confidence':
        for name in ('gate_threshold', 'gate_margin'):
            value = m.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid confidence configuration: ' + name)
    for name,h in m['files'].items():
        if sha256(folder/name)!=h:raise ValueError('Bundle hash mismatch: '+name)
    model=MultiFingerprintModel(folder/'model.npz')
    if tuple(m['head_sizes'])!=model.head_sizes or m['feature_dim']!=model.feature_dim:
        raise ValueError('Bundle/model feature contract mismatch')
    w=np.asarray(m['weights'])
    if w.shape!=(len(model.head_sizes),) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Invalid bundle head weights')
    catalog=json.loads((folder/'catalog.json').read_text());targets=np.load(folder/'targets.npy',mmap_mode='r',allow_pickle=False)
    if targets.shape!=(len(catalog),1792) or targets.dtype!=np.uint8:raise ValueError('Bundle target alignment mismatch')
    if not catalog or any(len(r)!=4 or not r[1] or not r[2] or not np.isfinite(r[3]) or r[3]<=0 for r in catalog):
        raise ValueError('Malformed catalog')
    if any(catalog[i][3]>catalog[i+1][3] for i in range(len(catalog)-1)):raise ValueError('Unsorted catalog')
    return m


def refit_bundle(cache, old_artifacts, experiment, destination, *, device='cuda'):
    cache,old_artifacts,experiment,destination=map(Path,(cache,old_artifacts,experiment,destination))
    report=json.loads((experiment/'validation-v3.json').read_text());plan=json.loads((experiment/'protocol.json').read_text())
    if not report['eligible_for_candidate_bundle']:
        return {'status':'not_promoted','reason':'No independently supported improvement; original champion preserved'}
    selected=report['selection'];model=MultiFingerprintModel(experiment/selected['model'])
    expected={'source':plan['source'],'selection_sha256':sha256(experiment/'selection-before-audit.json'),
              'warmstart_sha256':sha256(old_artifacts/'model.npz')}
    if (destination/'v3-bundle.json').exists():
        m=verify_bundle(destination)
        if m['training_identity']!=expected:raise ValueError('Cannot reuse a different refit')
        return {'status':'bundle_verified','bundle':str(destination),'manifest':m}
    destination.mkdir(parents=True,exist_ok=True)
    x=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False);fp=np.load(cache/'targets.npy',mmap_mode='r',allow_pickle=False)
    indices=np.flatnonzero(np.load(cache/'counts.npy',allow_pickle=False)>0)
    trained=train_model(x,fp,indices,destination/'model.npz',epochs=plan['epochs'],feature_dim=model.feature_dim,
             head_sizes=model.head_sizes,device=device,warmstart=old_artifacts/'model.npz')
    for n in ('targets.npy','catalog.json'):shutil.copy2(cache/n,destination/n)
    m={'format':3,'feature_version':FEATURE_VERSION,'feature_dim':model.feature_dim,'head_sizes':list(model.head_sizes),
       'weights':selected['weights'],'mode':'confidence','gate_threshold':.95,'gate_margin':.05,
       'train_sha256':plan['source']['train_sha256'],'training_identity':expected,'training':trained,
       'files':{n:sha256(destination/n) for n in ('targets.npy','catalog.json','model.npz')},
       'contains_test_ids_or_predictions':False,'official_score':None,'candidate_not_champion':True}
    write_json(destination/'v3-bundle.json',m);verify_bundle(destination)
    return {'status':'bundle_created','bundle':str(destination),'manifest':m}

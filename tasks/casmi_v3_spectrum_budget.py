"""Diagnostic only: fixed V3 audit models with 1/3/all spectra and Bruker rows.

No tuning, new model selection, refit, or Kaggle write. Reuses the spent audit
keys deliberately to characterize failure modes, not as a new selection set.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


class SpectrumBudget:
    def __init__(self):
        self.total=None;self.masses=[];self.count=0;self.best={};self.signatures=set()

    def add(self,row,features,mass):
        import numpy as np
        x=np.asarray(features,dtype=np.float64)
        if x.ndim!=1 or not np.isfinite(x).all() or not np.isfinite(mass) or mass<=0:
            raise ValueError('Invalid diagnostic spectrum')
        if self.total is None:self.total=np.zeros_like(x)
        if x.shape!=self.total.shape:raise ValueError('Feature shape changed')
        self.total+=x;self.masses.append(float(mass));self.count+=1
        fields=('adduct','precursor_mz','collision_energy_ev','ms2_mzs','ms2_normalized_intensities')
        key=hashlib.sha256(json.dumps({k:row[k] for k in fields},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if key in self.signatures:return
        self.signatures.add(key)
        if len(self.best)<3 or key<max(self.best):
            self.best[key]=(x.astype(np.float32),float(mass))
            if len(self.best)>3:del self.best[max(self.best)]

    def query(self,n=None):
        import numpy as np
        if not self.count:raise ValueError('No spectra')
        if n is None:return (self.total/self.count).astype(np.float32),float(np.median(self.masses)),self.count
        if n not in (1,3):raise ValueError('Only predeclared budgets 1, 3, all are supported')
        selected=[self.best[k] for k in sorted(self.best)[:n]]
        return np.mean([r[0] for r in selected],axis=0),float(np.median([r[1] for r in selected])),len(selected)


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.features_v3 import arrow_highres
    from casmi26.model_v3 import MultiFingerprintModel,candidate_scores
    from casmi26.pipeline_v3 import _rank,_metrics
    from casmi26.production import is_validation,mass_candidates,sha256,write_json
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    spec=importlib.util.spec_from_file_location('staged',repo/'tasks/casmi_staged.py')
    staged=importlib.util.module_from_spec(spec);spec.loader.exec_module(staged)
    train=staged.find_dataset(state/'data/external')/'train.parquet'
    cache=state/'cache/casmi26/highres-v3';experiment=state/'artifacts/casmi26/research-v3'
    dest=state/'artifacts/casmi26/highres-v3/spectrum-budget';out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    dest.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    checks=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
    print(checks.stdout,flush=True)
    (out/'tests.log').write_text(checks.stdout+'\n'+checks.stderr,encoding='utf-8')
    if checks.returncode:raise RuntimeError('Tests failed before diagnostic')
    done=dest/'report.json'
    if done.exists():raise RuntimeError('This diagnostic already exists; do not overwrite it for tuning')
    plan=json.loads((experiment/'protocol.json').read_text());validation=json.loads((experiment/'validation-v3.json').read_text())
    if sha256(train)!=plan['source']['train_sha256']:raise ValueError('Source train changed')
    keys=plan['audit_keys'];keyset=set(keys)
    if any(not is_validation(k) for k in keys):raise ValueError('Non-held-out key in diagnostic')
    catalog=json.loads((cache/'catalog.json').read_text());counts=np.load(cache/'counts.npy',allow_pickle=False)
    representatives={}
    for i,row in enumerate(catalog):
        if row[2] in keyset and counts[i]>0 and (row[2] not in representatives or counts[i]>counts[representatives[row[2]]]):
            representatives[row[2]]=i
    if set(representatives)!=keyset:raise ValueError('Missing predeclared audit keys')
    raw={catalog[i][0]:(k,catalog[i][3]) for k,i in representatives.items()}
    masses=np.array([r[3] for r in catalog]);packed=np.load(cache/'targets.npy',mmap_mode='r',allow_pickle=False)
    selected=validation['selection'];configurations={'frozen':('v1-frozen.npz',[1.]),'selected':(selected['model'],selected['weights'])}
    models={}
    for label,(name,weights) in configurations.items():
        if sha256(experiment/name)!=validation['model_hashes'][name]:raise ValueError('Evaluated checkpoint changed')
        models[label]=MultiFingerprintModel(experiment/name)
    write_json(dest/'protocol-before-diagnostics.json',{'audit_keys_sha256':hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
        'models':configurations,'budgets':[1,3,'all'],'cohorts':['all_sources','bruker_metadata'],
        'subset_selection':'lowest SHA256 of whitelisted spectrum fields; independent of labels and input order',
        'changes_model_selection':False,'reuses_spent_audit_keys':True,'test_data_read':False})
    groups={cohort:{} for cohort in ('all_sources','bruker_metadata')};values=pa.array(list(raw),type=pa.string())
    columns=['normalized_smiles','ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','collision_energy_ev','instrument_type']
    rejected=0;scanned=0;start=time.monotonic()
    with pq.ParquetFile(train) as table:
        for batch in table.iter_batches(batch_size=8192,columns=columns):
            scanned+=len(batch);part=batch.filter(pc.is_in(batch.column('normalized_smiles'),value_set=values))
            if not len(part):continue
            x,valid,neutral=arrow_highres(part)
            for i,row in enumerate(part.to_pylist()):
                key,mass=raw[row['normalized_smiles']]
                if not valid[i] or abs(neutral[i]-mass)>max(.003,mass*50e-6):rejected+=1;continue
                groups['all_sources'].setdefault(key,SpectrumBudget()).add(row,x[i],neutral[i])
                if 'bruker' in str(row['instrument_type']).lower():
                    groups['bruker_metadata'].setdefault(key,SpectrumBudget()).add(row,x[i],neutral[i])
    if set(groups['all_sources'])!=keyset:raise RuntimeError('Some audit molecules have no admitted spectra')
    results={};details=[]
    for cohort,budgets in groups.items():
        results[cohort]={'molecules':len(budgets),'spectra':sum(b.count for b in budgets.values()),'unique_spectra':sum(len(b.signatures) for b in budgets.values()),'by_budget':{}}
        for n in (1,3,None):
            ranks={label:[] for label in models};used=[]
            for key in keys:
                if key not in budgets:continue
                x,observed,count=budgets[key].query(n);idx=mass_candidates(masses,observed)
                bits=np.unpackbits(packed[idx],axis=1);candidate_keys=[catalog[j][2] for j in idx]
                d={'key':key,'cohort':cohort,'budget':n or 'all','spectra_used':count,'ranks':{}}
                for label,model in models.items():
                    logits=model.logits(x[:model.feature_dim]);score=candidate_scores(logits,bits[:,:sum(model.head_sizes)],masses[idx],observed,
                        head_sizes=model.head_sizes,weights=configurations[label][1])
                    rank=_rank(score,candidate_keys,key);ranks[label].append(rank);d['ranks'][label]=rank
                used.append(count);details.append(d)
            results[cohort]['by_budget'][str(n or 'all')]={'metrics':{label:_metrics(r) for label,r in ranks.items()},
                'average_spectra_used':float(np.mean(used)) if used else None}
            print('V3_SPECTRUM_BUDGET '+cohort+' '+str(n or 'all'),flush=True)
    report={'status':'completed','experiment':'v3-spectrum-budget-diagnostic','commit':os.environ.get('GITHUB_SHA'),
        'selection_fixed_to':selected['variant'],'cohorts':results,'rejected_spectra':rejected,'train_rows_scanned':scanned,
        'checkpoint_hashes':{label:sha256(experiment/configurations[label][0]) for label in models},
        'test_data_read':False,'new_training':False,'new_submissions':0,'model_selection_changed':False,
        'seconds':time.monotonic()-start,'official_score':None,'tests':checks.stdout.strip(),
        'limitations':['Diagnostics reuse spent audit keys; not a fresh model-selection benchmark.',
            'Structure catalog contains the true graph; this is not de novo or catalog-coverage evaluation.',
            'Bruker cohort is a literal instrument_type metadata match, not a guarantee of hidden instrument equivalence.',
            'Training-quality label/mass checks are used to exclude inconsistent labeled spectra.',
            'At most three distinct spectra are selected for reduced budgets; all includes repeated rows.']}
    for root in (dest,out):write_json(root/'report.json',report);write_json(root/'diagnostic-ranks.json',details)
    print('V3_BUDGET_DIAGNOSTIC_BEGIN\n'+json.dumps(report,indent=2)+'\nV3_BUDGET_DIAGNOSTIC_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

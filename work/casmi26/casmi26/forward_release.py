"""Fixed R08B forward evidence on the complete, unchanged R07 candidate universe.

Unsupported model chemistry contributes no bonus. Certified screening only skips
candidates whose maximum possible bonus cannot enter top-25. No scores are fit.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import zlib
import numpy as np

from .bounded_forward import certified_rerank
from .forward_ranking import pool_forward_scores, ENERGIES, SUPPORTED_ADDUCTS
from .fiora_adapter import ForwardModel, MODEL_HASH, PARAMS_HASH
from .production import sha256, write_json

SOURCE='e19ef82c9a6cb9dbac92bce23e914008f1aeb44e'
FORWARD_WEIGHT=.25
FORWARD_FEATURE='cosine_nearest'


def read_raw_queries(path):
    import pyarrow.parquet as pq
    groups=defaultdict(list)
    columns=['molecule_id','adduct','precursor_mz','collision_energy_ev','ms2_mzs','ms2_normalized_intensities']
    with pq.ParquetFile(path) as f:
        for batch in f.iter_batches(batch_size=2048,columns=columns):
            for row in batch.to_pylist():
                cid=row['molecule_id']
                if cid is None or not str(cid).strip():raise ValueError('Missing raw query ID')
                p=np.column_stack((row['ms2_mzs'],row['ms2_normalized_intensities'])).astype('f8')
                if not p.size or not np.isfinite(p).all() or np.any(p[:,0]<=0) or np.any(p[:,1]<0):
                    raise ValueError('Invalid raw query peaks')
                p=p[p[:,1]>0]
                if not len(p):raise ValueError('Raw query has no positive intensity')
                p[:,1]/=p[:,1].sum()
                groups[str(cid)].append({'adduct':row['adduct'],'precursor':row['precursor_mz'],
                                        'ce':row['collision_energy_ev'],'peaks':p})
    if not groups:raise ValueError('No raw queries')
    return dict(groups)


class ForwardRanker:
    def __init__(self,queries,*,model,cache=None):
        self.queries=queries;self.model=model;self.cache={};self.statistics=Counter();self.db=None
        if cache is not None:
            import torch,rdkit
            cache=Path(cache);cache.parent.mkdir(parents=True,exist_ok=True)
            self.db=sqlite3.connect(cache)
            self.db.execute('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS predictions(k TEXT PRIMARY KEY,v BLOB NOT NULL)')
            identity=json.dumps({'source':SOURCE,'model':MODEL_HASH,'params':PARAMS_HASH,'energies':list(ENERGIES),
                'adapter':sha256(Path(__file__).with_name('fiora_adapter.py')),'torch':torch.__version__,
                'rdkit':rdkit.__version__,'device':getattr(model,'device','unknown')},sort_keys=True)
            old=self.db.execute('SELECT v FROM meta WHERE k=?',('identity',)).fetchone()
            if old and old[0]!=identity:self.close();raise ValueError('Forward cache identity differs')
            if not old:self.db.execute('INSERT INTO meta VALUES(?,?)',('identity',identity));self.db.commit()

    def close(self):
        if self.db is not None:self.db.close();self.db=None

    def predictions(self,smiles,modes):
        key=hashlib.sha256((smiles+'\0'+';'.join(modes)).encode()).hexdigest()
        if key not in self.cache:
            old=self.db.execute('SELECT v FROM predictions WHERE k=?',(key,)).fetchone() if self.db else None
            if old:
                record=json.loads(zlib.decompress(old[0]));self.statistics['disk_cache_hits']+=1
                if record['smiles']!=smiles or record['modes']!=modes:raise ValueError('Forward cache key collision')
            else:
                record={'smiles':smiles,'modes':modes,'spectra':[],'status':'predicted'}
                try:
                    pred=self.model.predict(smiles,modes,ENERGIES)
                    record['spectra']=[{'mode':m,'energy':float(e),'peaks':np.asarray(p).tolist()} for (m,e),p in pred.items()]
                    self.statistics['simulated_spectra']+=len(pred)
                except (ValueError,RuntimeError,AssertionError,IndexError,KeyError) as exc:
                    record.update(status='unsupported_or_failed',error_type=type(exc).__name__)
                    self.statistics['candidate_failures']+=1
                self.statistics['graph_calls']+=1
                if self.db:
                    self.db.execute('INSERT INTO predictions VALUES(?,?)',(key,zlib.compress(json.dumps(record,allow_nan=False).encode(),1)))
                    self.db.commit()
            self.cache[key]={(r['mode'],r['energy']):r['peaks'] for r in record['spectra']}
        else:self.statistics['memory_cache_hits']+=1
        return self.cache[key]

    def __call__(self,cid,rows,queries,scores):
        if cid not in self.queries:raise ValueError('Missing raw query for ranking')
        raw=self.queries[cid];modes=sorted({q['adduct'] for q in raw}&SUPPORTED_ADDUCTS)
        n_supported=sum(q['adduct'] in SUPPORTED_ADDUCTS for q in raw)
        def evidence(i):
            pred=self.predictions(rows[i][1],modes)
            return pool_forward_scores(raw,pred)['features'].get(FORWARD_FEATURE)
        result=certified_rerank(scores,evidence,weight=FORWARD_WEIGHT if modes else 0.,k=25)
        if not result['certified']:raise RuntimeError('Uncertified forward ranking')
        self.statistics['query_groups']+=1;self.statistics['candidate_evaluations']+=result['evaluations']
        self.statistics['pruned_evaluations']+=result['pruned'];self.statistics['candidate_pairs']+=len(rows)
        return {'top_indices':result['top_indices'],'metadata':{'certified':True,'feature':FORWARD_FEATURE,
            'weight':FORWARD_WEIGHT if modes else 0.,'evaluations':result['evaluations'],'pruned':result['pruned'],
            'missing_evidence':result['missing_evidence'],'supported_spectra':n_supported,'raw_spectra':len(raw),
            'score_is_probability':False}}


def infer(test,train,bundle,output,*,model_path,template=None,workers=4,cache=None,device='cpu'):
    from .r07_release import infer as r07_infer
    started=time.monotonic()
    model_path=Path(model_path);test=Path(test);output=Path(output);bundle=Path(bundle)
    if cache is not None:
        resolved=Path(cache).resolve()
        protected={Path(p).resolve() for p in (test,train,output,output.with_suffix(output.suffix+'.report.json'),
            model_path.with_name(model_path.stem+'_state.pt'),model_path.with_name(model_path.stem+'_params.json'))}
        if template:protected.add(Path(template).resolve())
        if resolved in protected or bundle.resolve() in resolved.parents:
            raise ValueError('Cache would overwrite an input or model bundle')
    queries=read_raw_queries(test)
    model=ForwardModel(model_path,device=device)
    ranker=ForwardRanker(queries,model=model,cache=cache)
    try:report=r07_infer(test,train,bundle,output,template=template,workers=workers,ranking_adapter=ranker)
    finally:ranker.close()
    if report['prediction_count']!=len(queries) or report['test_spectra']!=sum(map(len,queries.values())):
        raise ValueError('Raw and base query group counts disagree')
    report.update(format=8,base_format=7,algorithm='R07-plus-fixed-R08B-forward',forward={
        'source':SOURCE,'model_sha256':MODEL_HASH,'params_sha256':PARAMS_HASH,
        'feature':FORWARD_FEATURE,'weight':FORWARD_WEIGHT,'energies':list(ENERGIES),'device':device,
        'statistics':dict(ranker.statistics),'all_top25_numerically_certified':True,
        'pretraining_membership':'unknown','score_is_probability':False},seconds_total=time.monotonic()-started)
    report['limitations']+=['FIORA supports only [M+H]+ and [M-H]- in this fixed adapter; other ion modes use R07 without a forward bonus.',
        'Numerical certificates establish a ranking under the fixed score, not chemical correctness.',
        'FIORA pretraining membership is unknown; no de novo generalization claim.']
    write_json(output.with_suffix(output.suffix+'.report.json'),report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('test','train','bundle','output','model-path'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--sample-submission',type=Path);p.add_argument('--cache',type=Path)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    a=p.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    r=infer(a.test,a.train,a.bundle,a.output,model_path=a.model_path,template=a.sample_submission,
        cache=a.cache,workers=a.workers,device=a.device)
    print(json.dumps({k:v for k,v in r.items() if k!='details'},indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

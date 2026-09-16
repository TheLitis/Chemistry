"""Training and molecule-held-out known-catalog retrieval validation.

Validation structures are present in the structure-only catalog by design.
Their spectra never enter optimization. This is NOT a de novo benchmark and
NOT the official MassSpecGym split/leaderboard or the hidden CASMI26 test.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .chemistry import info
from .engine import atomic_text, digest
from .learning import FP_BITS, FingerprintRanker, fingerprint, group_features, train_arrays
from .metric import structure_key, require_official_rdkit
from .spectra import records, from_row


def prepare_examples(path: Path, *, max_molecules: int = 30000) -> tuple[dict,dict]:
    if max_molecules < 2:raise ValueError('Need room for at least two molecules')
    groups={};counts=Counter();rejections=Counter()
    for row in records(path):
        counts['seen_spectra']+=1
        try:
            s=from_row(row,labeled=True);m=info(s.smiles);key=structure_key(s.smiles)
            if key is None:raise ValueError('No valid InChI')
            if abs(s.neutral-m.mass)>max(.003,m.mass*50e-6):
                rejections['precursor_mass_inconsistent']+=1;continue
            if key not in groups and len(groups)>=max_molecules:
                rejections['molecule_limit']+=1;continue
            x=group_features([s])
        except (ValueError,TypeError,KeyError):
            rejections['invalid_or_unsupported_row']+=1;continue
        if key not in groups:
            groups[key]={'smiles':s.smiles,'mass':m.mass,'spectra_count':0,
                         'feature_sum':np.zeros_like(x),'neutral_sum':0.}
        record=groups[key];record['feature_sum']+=x;record['spectra_count']+=1
        record['neutral_sum']+=s.neutral;counts['accepted_spectra']+=1
    for record in groups.values():
        record['features']=record.pop('feature_sum')/record['spectra_count']
        record['observed_mass']=record.pop('neutral_sum')/record['spectra_count']
    return groups,{**counts,'molecules':len(groups),'rejected':dict(rejections)}


def summarize_ranks(ranks: list[int | None]) -> dict:
    if not ranks:return {'molecules':0,'mrr_at_25':None,'top1_accuracy':None,'recall_at_25':None}
    return {'molecules':len(ranks),'mrr_at_25':sum(1/r if r is not None and r<=25 else 0 for r in ranks)/len(ranks),
            'top1_accuracy':sum(r==1 for r in ranks)/len(ranks),
            'recall_at_25':sum(r is not None and r<=25 for r in ranks)/len(ranks)}


def train_and_validate(path: Path, output: Path, *, epochs=30, hidden=384,
                       batch_size=128, device='cuda', max_molecules=30000, seed=1729) -> dict:
    examples,counts=prepare_examples(path,max_molecules=max_molecules)
    keys=sorted(examples)
    # This split is fixed before training and groups tautomer/stereo equivalents.
    val_keys=[k for k in keys if int.from_bytes(hashlib.sha256(k.encode()).digest()[:8],'big')%5==0]
    val_set=set(val_keys)
    train_keys=[k for k in keys if k not in val_set]
    if len(train_keys)<2 or len(val_keys)<1:raise ValueError('Insufficient molecules for a train/validation split')
    x=np.stack([examples[k]['features'] for k in train_keys])
    y=np.stack([fingerprint(examples[k]['smiles']) for k in train_keys])
    trained=train_arrays(x,y,output,epochs=epochs,hidden=hidden,batch_size=batch_size,device=device,seed=seed)
    model=FingerprintRanker(output)
    masses=np.array([examples[k]['mass'] for k in keys]);order=np.argsort(masses,kind='stable');masses=masses[order]
    candidate_fps=np.stack([fingerprint(examples[keys[i]]['smiles']) for i in order])
    ordered_keys=[keys[i] for i in order]
    ranks=[];baseline=[];ambiguous=[];ambiguous_baseline=[];details=[];no_candidate=0
    for key in val_keys:
        item=examples[key];mass=item['observed_mass'];tol=max(.003,mass*15e-6)
        left=np.searchsorted(masses,mass-tol);right=np.searchsorted(masses,mass+tol,side='right')
        ck=ordered_keys[left:right];fps=candidate_fps[left:right]
        p=model.posterior_features(item['features'])
        if len(ck):
            dots=fps@p;sim=dots/np.maximum(1e-8,p.sum()+fps.sum(axis=1)-dots)
            ranking=sorted(range(len(ck)),key=lambda i:(-float(sim[i]),ck[i]))
            predicted=[ck[i] for i in ranking]
            rank=predicted.index(key)+1 if key in predicted else None
            base=sorted(ck).index(key)+1 if key in ck else None
        else:
            rank=base=None;no_candidate+=1
        ranks.append(rank);baseline.append(base)
        if len(ck)>1:
            ambiguous.append(rank);ambiguous_baseline.append(base)
        details.append({'molecule_key':key,'spectra_used':item['spectra_count'],'candidate_count':len(ck),'rank':rank})
    report={'status':'completed_external_catalog_validation','official_score':None,
            'protocol':'canonical-tautomer-key hash holdout; structure-only catalog includes all held-out structures; no held-out spectra used for fitting',
            'de_novo_evaluation':False,'official_massspecgym_benchmark':False,'official_casmi_evaluation':False,
            'counts':counts,'optimization':trained,'validation':summarize_ranks(ranks),
            'mass_only_baseline':summarize_ranks(baseline),
            'ambiguous_mass_validation':summarize_ranks(ambiguous),
            'ambiguous_mass_baseline':summarize_ranks(ambiguous_baseline),
            'no_mass_candidate':no_candidate,'training_validation_key_overlap':len(set(train_keys)&set(val_keys)),
            'source':digest(path),'checkpoint':digest(output),'molecules':details,
            'limitations':['A small external subset is not representative of the CASMI26 hidden test.',
                          'The true structure is in the evaluation catalog by construction.',
                          'This ranker cannot generate a missing molecular graph.']}
    atomic_text(Path(output).with_suffix('.validation.json'),json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Catalog used for this external experiment only; never mislabel as an
    # independently acquired PubChem/COCONUT database.
    import csv
    with Path(output).with_suffix('.catalog.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['smiles']);writer.writerows([examples[k]['smiles']] for k in keys)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--hidden',type=int,default=384)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--max-molecules',type=int,default=30000)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--allow-rdkit-version-mismatch',action='store_true')
    a=p.parse_args(argv)
    if not a.allow_rdkit_version_mismatch:require_official_rdkit()
    if a.output.resolve()==a.train.resolve():raise ValueError('Refusing to overwrite input')
    report=train_and_validate(a.train,a.output,epochs=a.epochs,hidden=a.hidden,batch_size=a.batch_size,
                              device=a.device,max_molecules=a.max_molecules,seed=a.seed)
    print('TRAINING_SUMMARY '+json.dumps({k:v for k,v in report.items() if k not in ('molecules',)},allow_nan=False),flush=True)
    return 0

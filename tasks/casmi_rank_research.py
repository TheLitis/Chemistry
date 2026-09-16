"""R01: frozen-checkpoint ranking ablation, with separate calibration and audit.

Uses only official TRAIN-derived caches and the holdout checkpoint, never the
full-refit checkpoint or visible/hidden test. The structural catalog is inclusive:
these metrics measure known-catalog retrieval, not de novo structure recovery.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def rank_of(scores, keys, truth):
    import numpy as np
    if len(scores) != len(keys):
        raise ValueError('Score/key length mismatch')
    seen = set()
    for i in np.argsort(-np.asarray(scores), kind='stable'):
        key = keys[int(i)]
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > 25:
            break
        if key == truth:
            return len(seen)
    return 0


def unweight(p, weights):
    import numpy as np
    p = np.clip(p, 1e-7, 1-1e-7)
    return p / (weights * (1-p) + p)


def base_scores(p, fps, weights):
    import numpy as np
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1-1e-7)
    f = np.asarray(fps, dtype=np.float64)
    dot = f @ p
    q = unweight(p, weights)
    dq = f @ q
    return {
        'tanimoto': dot / np.maximum(p.sum()+f.sum(1)-dot, 1e-8),
        'cosine': dot / np.maximum(np.linalg.norm(p)*np.sqrt(f.sum(1)), 1e-8),
        'bernoulli': f @ (np.log(p)-np.log1p(-p)),
        'unweighted_bernoulli': f @ (np.log(q)-np.log1p(-q)),
        'unweighted_tanimoto': dq / np.maximum(q.sum()+f.sum(1)-dq, 1e-8),
    }


def mass_prior(masses, observed):
    import numpy as np
    sigma = max(.001, abs(float(observed))*5e-6)
    return -.5*((np.asarray(masses)-observed)/sigma)**2


def combine(scores, prior, weight):
    import numpy as np
    if not weight or not len(scores):
        return scores
    return (scores - np.mean(scores))/max(float(np.std(scores)), 1e-8) + weight*prior


def split_keys(keys, previously_used, n_calibration, n_audit):
    eligible = sorted(set(keys)-set(previously_used),
                      key=lambda k: hashlib.sha256(('rank-research-r01:'+k).encode()).digest())
    if len(eligible) < n_calibration+n_audit:
        raise ValueError('Insufficient previously unexamined held-out molecular keys')
    return eligible[:n_calibration], eligible[n_calibration:n_calibration+n_audit]


def metrics(ranks):
    import numpy as np
    r = np.asarray(ranks)
    if not len(r):
        raise ValueError('Empty evaluation')
    rr = np.where((r>0)&(r<=25), 1/np.maximum(r, 1), 0)
    return {'molecules': len(r), 'mrr_at_25': float(rr.mean()),
            'top1_accuracy': float((r==1).mean()),
            'recall_at_25': float(((r>0)&(r<=25)).mean())}


def paired_interval(baseline, selected, repeats=2000):
    import numpy as np
    a, b = np.asarray(baseline), np.asarray(selected)
    if a.shape != b.shape or not len(a):
        raise ValueError('Paired ranks must have equal nonzero length')
    delta = np.where((b>0)&(b<=25), 1/np.maximum(b,1), 0) - np.where((a>0)&(a<=25), 1/np.maximum(a,1), 0)
    rng = np.random.default_rng(260916)
    values = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(repeats)]
    return {'delta_mrr': float(delta.mean()), 'ci95': np.quantile(values, [.025,.975]).tolist(),
            'bootstrap_repeats': repeats}


def main():
    state = Path(os.environ['CHEMISTRY_STATE_ROOT'])
    py = state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve() != py.resolve():
        env = {**os.environ, 'PYTHONUTF8':'1', 'PYTHONIOENCODING':'utf-8',
               'OPENBLAS_NUM_THREADS':'4', 'OMP_NUM_THREADS':'4'}
        return subprocess.call([str(py), str(Path(__file__).resolve())], env=env)
    import numpy as np
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo/'work/casmi26'))
    from casmi26.production import is_validation, mass_candidates, sha256, write_json
    from casmi26.learning import FingerprintRanker
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    cache = state/'cache/casmi26/official-v1'
    art = state/'artifacts/casmi26/official-v1'
    dest = state/'artifacts/casmi26/research-r01'
    out = Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])/'research-r01'
    dest.mkdir(parents=True, exist_ok=True);out.mkdir(parents=True, exist_ok=True)
    if (dest/'report.json').exists():
        raise RuntimeError('R01 already evaluated: use existing evidence, do not overwrite or retune its audit')
    start = time.monotonic()
    checks = subprocess.run([str(py), '-m', 'pytest', '-q', str(repo/'work/casmi26/tests')],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=240)
    print(checks.stdout, flush=True)
    (out/'tests.log').write_text(checks.stdout+'\n'+checks.stderr, encoding='utf-8')
    if checks.returncode:
        raise RuntimeError('Tests failed before research')
    catalog = json.loads((cache/'catalog.json').read_text())
    x = np.load(cache/'features.npy', mmap_mode='r')
    counts = np.load(cache/'counts.npy')
    observed = np.load(cache/'observed.npy')
    packed = np.load(cache/'fingerprints.npy', mmap_mode='r')
    masses = np.array([r[3] for r in catalog])
    previous = json.loads((art/'validation.json').read_text())
    used = {d['key'] for d in previous['details']}
    representative = {}
    for i, r in enumerate(catalog):
        if counts[i] > 0 and is_validation(r[2]) and r[2] not in representative:
            representative[r[2]] = i
    calibration, audit = split_keys(representative, used, 2000, 4000)
    training = np.array([i for i, r in enumerate(catalog) if counts[i]>0 and not is_validation(r[2])])
    training_keys = {catalog[i][2] for i in training}
    if training_keys & (set(calibration)|set(audit)):
        raise RuntimeError('Training/evaluation structure leakage')
    fp_sum = np.zeros(2048, dtype=np.float64)
    for block in range(0, len(training), 4096):
        fp_sum += np.unpackbits(packed[training[block:block+4096]], axis=1).sum(0)
    prevalence = np.clip(fp_sum/len(training), .005, .995)
    weights = np.minimum(20., (1-prevalence)/prevalence)
    model_path = art/'holdout-model.npz'
    model = FingerprintRanker(model_path)
    queries = calibration + audit
    post = []
    for block in range(0, len(queries), 128):
        ids = [representative[k] for k in queries[block:block+128]]
        logits = np.maximum(np.asarray(x[ids])@model.w1.T + model.b1, 0)@model.w2.T + model.b2
        post.append(1/(1+np.exp(-np.clip(logits,-40,40))))
    post = np.concatenate(post)
    variants = [(method, w) for method in ('tanimoto','cosine','bernoulli','unweighted_bernoulli','unweighted_tanimoto')
                for w in (0., .25, 1.)]
    calibration_ranks = {f'{m}:{w:g}': [] for m,w in variants}
    def query_data(key):
        i = representative[key]
        idx = mass_candidates(masses, observed[i])
        return idx, np.unpackbits(packed[idx], axis=1), [catalog[j][2] for j in idx], mass_prior(masses[idx], observed[i])
    print('R01_CALIBRATION_START 2000', flush=True)
    for t, key in enumerate(calibration):
        idx, fps, keys, prior = query_data(key)
        scores = base_scores(post[t], fps, weights)
        for method, w in variants:
            calibration_ranks[f'{method}:{w:g}'].append(rank_of(combine(scores[method], prior, w), keys, key))
    cal_result = {name:metrics(rr) for name,rr in calibration_ranks.items()}
    winner = max(cal_result, key=lambda name: cal_result[name]['mrr_at_25'])
    method, weight = winner.split(':');weight = float(weight)
    selection = {'selected':winner, 'calibration':cal_result,
                 'audit_ids_sha256':hashlib.sha256('\n'.join(audit).encode()).hexdigest(),
                 'audit_size':len(audit), 'checkpoint_sha256':sha256(model_path),
                 'chosen_without_audit_labels':True}
    write_json(dest/'selection-before-audit.json', selection)
    print('R01_SELECTED '+winner+'; now evaluate locked audit', flush=True)
    audit_ranks=[];baseline=[];massonly=[];shuffled=[];details=[]
    massorder = np.argsort([observed[representative[k]] for k in audit], kind='stable')
    perm = np.empty(len(audit), dtype=np.int64)
    for b in range(0,len(audit),32):
        section = massorder[b:b+32];perm[section] = np.roll(section,1)
    for t, key in enumerate(audit):
        idx, fps, keys, prior = query_data(key)
        scores = base_scores(post[len(calibration)+t], fps, weights)
        br = rank_of(scores['tanimoto'], keys, key)
        sr = rank_of(combine(scores[method], prior, weight), keys, key)
        cr = base_scores(post[len(calibration)+perm[t]], fps, weights)[method]
        pr = rank_of(combine(cr, prior, weight), keys, key)
        mr = rank_of(-np.abs(masses[idx]-observed[representative[key]]), keys, key)
        baseline.append(br);audit_ranks.append(sr);shuffled.append(pr);massonly.append(mr)
        details.append({'key':key, 'candidate_keys':len(set(keys)), 'baseline_rank':br, 'selected_rank':sr,
                        'shuffled_rank':pr, 'mass_only_rank':mr, 'spectra':int(counts[representative[key]])})
    interval = paired_interval(baseline,audit_ranks)
    report = {'experiment':'R01-frozen-ranker-calibration', 'status':'completed',
              'commit':os.environ.get('GITHUB_SHA'), 'request_id':os.environ.get('CHEMISTRY_REQUEST_ID'),
              'protocol':'Inclusive structural catalog; fixed 90%-training checkpoint; untouched 2000-key calibration + 4000-key audit after excluding previous 4000 queried keys.',
              'selection':selection, 'training_validation_key_overlap':0,
              'previously_examined_keys_excluded':len(used),
              'audit':{'baseline':metrics(baseline), 'selected':metrics(audit_ranks),
                       'mass_only':metrics(massonly), 'mass_near_shuffled_posterior':metrics(shuffled),
                       'paired_bootstrap':interval},
              'recommend_promoting_rank_head': interval['ci95'][0]>0,
              'production_changed':False, 'new_training_performed':False,
              'official_score':None, 'submission_made':False, 'test_spectra_read':False,
              'limitations':['True structures are included in catalog; not a de novo score.',
                             'No end-to-end spectral/neural ensemble improvement is claimed.',
                             'Weighted-BCE correction is approximate because training also used a cosine loss.',
                             'This audit set is now spent: do not use it repeatedly to select later variants.'],
              'seconds':time.monotonic()-start}
    for root in (dest,out):
        write_json(root/'report.json',report)
        write_json(root/'audit-ranks.json',details)
        write_json(root/'calibration-keys.json',calibration)
    print('RESEARCH_R01_BEGIN\n'+json.dumps(report,indent=2,allow_nan=False)+'\nRESEARCH_R01_END',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

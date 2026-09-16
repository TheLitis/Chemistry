"""Small trainable spectrum-to-fingerprint ranker with offline NumPy inference.

This retrieves/reranks structures, it is not a de novo decoder. A checkpoint
contains numeric arrays only (no pickled executable Python objects).
"""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from .chemistry import canonical
from .spectra import Spectrum

BINS = 2048
FP_BITS = 2048
N_FEATURES = BINS*2 + 8
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS, includeChirality=False)


@lru_cache(maxsize=32768)
def fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(canonical(smiles))
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    out = _FP.GetFingerprintAsNumPy(mol).astype(np.float32)
    out.flags.writeable = False
    return out


def group_features(group: list[Spectrum]) -> np.ndarray:
    if not group:
        raise ValueError('At least one spectrum required')
    x = np.zeros(N_FEATURES, dtype=np.float32)
    for s in group:
        peaks = s.peaks[s.peaks[:,0] < s.precursor_mz-.01]
        for channel, mzs in enumerate((peaks[:,0], s.precursor_mz-peaks[:,0])):
            bins = np.floor(mzs).astype(np.int64)
            keep = (bins >= 0) & (bins < BINS)
            v = np.bincount(bins[keep], weights=np.sqrt(peaks[keep,1]), minlength=BINS).astype(np.float32)
            norm = np.linalg.norm(v)
            if norm:
                v /= norm
            x[channel*BINS:(channel+1)*BINS] += v
        energy = s.collision_energies or (() if s.collision_energy is None else (s.collision_energy,))
        x[-8:] += np.array([s.neutral/1200., s.precursor_mz/1200., s.mode == 1, s.mode == -1,
                           min(energy)/200. if energy else 0., max(energy)/200. if energy else 0.,
                           sum(energy)/len(energy)/200. if energy else 0., bool(energy)], dtype=np.float32)
    return x/len(group)


class FingerprintRanker:
    def __init__(self, path: Path):
        try:
            with np.load(path, allow_pickle=False) as data:
                if int(data['version']) != 1:
                    raise ValueError('Unsupported fingerprint model version')
                self.w1, self.b1, self.w2, self.b2 = [np.array(data[k],dtype=np.float32) for k in ('w1','b1','w2','b2')]
            h = len(self.b1)
            if not 1 <= h <= 4096 or self.w1.shape != (h,N_FEATURES) or self.w2.shape != (FP_BITS,h) or self.b2.shape != (FP_BITS,):
                raise ValueError('Invalid fingerprint model array shapes')
            if not all(np.isfinite(a).all() for a in (self.w1,self.b1,self.w2,self.b2)):
                raise ValueError('Nonfinite fingerprint model arrays')
        except (KeyError,TypeError) as exc:
            raise ValueError('Malformed fingerprint model checkpoint') from exc

    def posterior(self, group: list[Spectrum]) -> np.ndarray:
        return self.posterior_features(group_features(group))

    def posterior_features(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (N_FEATURES,) or not np.isfinite(x).all():
            raise ValueError("Invalid query feature vector")
        z = self.w2 @ np.maximum(self.w1 @ x + self.b1, 0) + self.b2
        return 1./(1.+np.exp(-np.clip(z,-40,40)))

    def scores(self, group: list[Spectrum], smiles: list[str]) -> dict[str,float]:
        p = self.posterior(group)
        result = {}
        for smi in smiles:
            fp = fingerprint(smi)
            dot = float(p@fp)
            result[smi] = dot/max(1e-8,float(p.sum()+fp.sum()-dot))
        return result


def train_arrays(x: np.ndarray, y: np.ndarray, output: Path, *, epochs: int=30,
                 batch_size: int=128, hidden: int=384, device: str='cuda', seed: int=1729,
                 learning_rate: float=.002) -> dict:
    import torch
    from torch import nn
    if x.ndim!=2 or x.shape[1]!=N_FEATURES or y.shape!=(len(x),FP_BITS) or not len(x):
        raise ValueError('Invalid training arrays')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not ((y>=0)&(y<=1)).all():
        raise ValueError('Invalid/nonfinite training values')
    if epochs<1 or batch_size<1 or hidden<1 or learning_rate<=0:
        raise ValueError('Invalid training settings')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    net=nn.Sequential(nn.Linear(N_FEATURES,hidden),nn.ReLU(),nn.Linear(hidden,FP_BITS)).to(device)
    prevalence=np.clip(y.mean(axis=0),.005,.995)
    with torch.no_grad():
        net[2].bias.copy_(torch.tensor(np.log(prevalence/(1-prevalence)),device=device))
    weights=torch.tensor(np.minimum(20.,(1-prevalence)/prevalence),device=device)
    bce=nn.BCEWithLogitsLoss(pos_weight=weights)
    def loss_fn(logits, targets):
        p=logits.sigmoid()
        similarity=(p*targets).sum(-1)/(p.norm(dim=-1)*targets.norm(dim=-1)).clamp_min(1e-8)
        return bce(logits,targets)+.5*(1-similarity).mean()
    opt=torch.optim.AdamW(net.parameters(),lr=learning_rate,weight_decay=1e-4)
    def measured_loss():
        net.eval();values=[]
        with torch.no_grad():
            for start in range(0,min(len(x),2048),batch_size):
                xb=torch.tensor(x[start:start+batch_size],device=device)
                yb=torch.tensor(y[start:start+batch_size],device=device)
                values.append(float(loss_fn(net(xb),yb).cpu()))
        return sum(values)/len(values)
    start_time=time.monotonic();initial=measured_loss();history=[]
    for epoch in range(epochs):
        net.train();permutation=rng.permutation(len(x));total=0.
        for start in range(0,len(x),batch_size):
            indices=permutation[start:start+batch_size]
            xb=torch.tensor(x[indices],device=device);yb=torch.tensor(y[indices],device=device)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(net(xb),yb)
            if not torch.isfinite(loss):raise RuntimeError('Training loss became nonfinite')
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            total+=float(loss.detach().cpu())*len(indices)
        value=total/len(x);history.append(value)
        print(json.dumps({'epoch':epoch+1,'epochs':epochs,'training_loss':value}),flush=True)
    # Never extrapolate through random weights for input columns absent from training.
    with torch.no_grad():
        net[0].weight[:, torch.tensor(np.all(x == 0, axis=0), device=device)] = 0
    final=measured_loss()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(suffix='.npz',dir=output.parent);os.close(fd)
    try:
        np.savez_compressed(name,version=np.array(1),w1=net[0].weight.detach().cpu().numpy(),
                            b1=net[0].bias.detach().cpu().numpy(),w2=net[2].weight.detach().cpu().numpy(),
                            b2=net[2].bias.detach().cpu().numpy())
        os.replace(name,output)
    finally:
        if os.path.exists(name):os.unlink(name)
    return {'initial_loss':initial,'final_loss':final,'loss_history':history,
            'device':device,'epochs':epochs,'seed':seed,'training_molecules':len(x),
            'parameters':sum(p.numel() for p in net.parameters()),
            'seconds':time.monotonic()-start_time,'official_score':None}

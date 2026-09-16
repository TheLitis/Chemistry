"""Trainable multi-target MLP; versioned numeric-only NumPy inference export."""
from __future__ import annotations
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from .features_v3 import FEATURE_DIM, HEAD_SIZES


class MultiFingerprintModel:
    def __init__(self, path):
        try:
            with np.load(path, allow_pickle=False) as z:
                if int(z['format']) != 3:
                    raise ValueError('Unsupported model format')
                self.feature_dim = int(z['feature_dim'])
                self.head_sizes = tuple(map(int, z['head_sizes']))
                self.w1, self.b1, self.w2, self.b2 = [np.asarray(z[k], dtype=np.float32) for k in ('w1','b1','w2','b2')]
            h = len(self.b1)
            if self.feature_dim not in (4104, FEATURE_DIM) or self.head_sizes not in ((2048,), HEAD_SIZES):
                raise ValueError('Unsupported feature/target contract')
            if not 1 <= h <= 4096 or self.w1.shape != (h,self.feature_dim) or self.w2.shape != (sum(self.head_sizes),h) or self.b2.shape != (sum(self.head_sizes),):
                raise ValueError('Malformed model shapes')
            if not all(np.isfinite(a).all() for a in (self.w1,self.b1,self.w2,self.b2)):
                raise ValueError('Nonfinite model weights')
        except (KeyError, TypeError, OverflowError) as e:
            raise ValueError('Malformed model checkpoint') from e

    def logits(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim not in (1,2) or x.shape[-1] != self.feature_dim or not np.isfinite(x).all():
            raise ValueError('Incorrect/nonfinite feature input')
        return np.maximum(x @ self.w1.T + self.b1, 0) @ self.w2.T + self.b2


def candidate_scores(logits, fingerprints, masses, observed, *, head_sizes=HEAD_SIZES, weights=(.5,.25,.25)):
    logits = np.asarray(logits,dtype=np.float64)
    fp = np.asarray(fingerprints,dtype=np.float64)
    masses = np.asarray(masses,dtype=np.float64)
    weights = np.asarray(weights,dtype=np.float64)
    if logits.shape != (sum(head_sizes),) or fp.shape != (len(masses),len(logits)) or weights.shape != (len(head_sizes),):
        raise ValueError('Head/candidate dimensions differ')
    if not all(np.isfinite(v).all() for v in (logits,fp,masses,weights)) or not np.isfinite(observed) or observed<=0:
        raise ValueError('Nonfinite/invalid scores')
    if (weights<0).any() or weights.sum()<=0 or ((fp!=0)&(fp!=1)).any():
        raise ValueError('Require nonnegative nonzero weights and binary targets')
    if not len(masses):
        return np.empty(0,dtype=np.float64)
    score = np.zeros(len(masses)); offset = 0
    for size,w in zip(head_sizes,weights/weights.sum()):
        # Matches R01's clipped probability log-odds; not a calibrated likelihood.
        values = fp[:,offset:offset+size] @ np.clip(logits[offset:offset+size],-16.11809555,16.11809555)
        if w:
            score += w * (values-values.mean()) / max(float(values.std()),1e-8)
        offset += size
    sigma = max(.001,abs(float(observed))*5e-6)
    return score - .5*((masses-observed)/sigma)**2


def _save(path, net, feature_dim, head_sizes):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,suffix='.npz');os.close(fd)
    try:
        np.savez_compressed(name,format=np.array(3),feature_dim=np.array(feature_dim),head_sizes=np.array(head_sizes),
            w1=net[0].weight.detach().cpu().numpy(),b1=net[0].bias.detach().cpu().numpy(),
            w2=net[2].weight.detach().cpu().numpy(),b2=net[2].bias.detach().cpu().numpy())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def train_model(x, packed, indices, output, *, epochs=10, hidden=512, batch_size=256,
                device='cuda', feature_dim=FEATURE_DIM, head_sizes=HEAD_SIZES,
                warmstart=None, seed=26091605, learning_rate=.0005):
    import torch
    from torch import nn
    indices=np.asarray(indices,dtype=np.int64);output=Path(output)
    if x.ndim!=2 or x.shape[1]<feature_dim or packed.shape!=(len(x),1792) or packed.dtype!=np.uint8:
        raise ValueError('Invalid training data contract')
    if not len(indices) or len(set(indices.tolist()))!=len(indices) or np.any(indices<0) or np.any(indices>=len(x)):
        raise ValueError('Invalid training row indices')
    if feature_dim not in (4104,8200) or tuple(head_sizes) not in ((2048,),HEAD_SIZES) or epochs<1 or batch_size<1 or not 0<learning_rate<1:
        raise ValueError('Invalid training configuration')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    if output.exists():
        raise FileExistsError('Refusing to overwrite a trained model: '+str(output))
    torch.set_num_threads(min(4,os.cpu_count() or 1));torch.manual_seed(seed)
    torch.set_float32_matmul_precision('highest')
    rng=np.random.default_rng(seed);nbits=sum(head_sizes);net=nn.Sequential(nn.Linear(feature_dim,hidden),nn.ReLU(),nn.Linear(hidden,nbits)).to(device)
    prevalence=np.zeros(nbits,dtype=np.float64);active=np.zeros(feature_dim,dtype=bool)
    for begin in range(0,len(indices),2048):
        ids=indices[begin:begin+2048];block=np.asarray(x[ids,:feature_dim])
        if not np.isfinite(block).all():raise ValueError('Nonfinite training features')
        active|=(block!=0).any(0)
        prevalence+=np.unpackbits(packed[ids],axis=1)[:,:nbits].sum(0)
    prevalence=np.clip(prevalence/len(indices),.005,.995)
    with torch.no_grad():
        net[2].bias.copy_(torch.as_tensor(np.log(prevalence/(1-prevalence)),dtype=torch.float32,device=device))
        if warmstart is not None:
            from .learning import FingerprintRanker
            old=FingerprintRanker(warmstart)
            if len(old.b1)!=hidden:raise ValueError('Warm-start hidden size differs')
            net[0].weight.zero_();net[0].weight[:,:4104].copy_(torch.as_tensor(old.w1,device=device))
            net[0].bias.copy_(torch.as_tensor(old.b1,device=device))
            net[2].weight[:2048].copy_(torch.as_tensor(old.w2,device=device))
            net[2].bias[:2048].copy_(torch.as_tensor(old.b2,device=device))
    pw=torch.as_tensor(np.minimum(20.,(1-prevalence)/prevalence),dtype=torch.float32,device=device)
    def loss_fn(logits,targets):
        logits=logits.float();values=[];pos=0
        for size in head_sizes:
            v=logits[:,pos:pos+size];y=targets[:,pos:pos+size]
            bce=nn.functional.binary_cross_entropy_with_logits(v,y,pos_weight=pw[pos:pos+size])
            p=v.sigmoid();cos=(p*y).sum(-1)/(p.norm(dim=-1)*y.norm(dim=-1)).clamp_min(1e-8)
            values.append(bce+.5*(1-cos).mean());pos+=size
        return torch.stack(values).mean()
    optimizer=torch.optim.AdamW(net.parameters(),lr=learning_rate,weight_decay=1e-4)
    cuda=device.startswith('cuda');scaler=torch.amp.GradScaler('cuda',enabled=cuda)
    def tensors(ids):
        a=np.ascontiguousarray(x[ids,:feature_dim],dtype=np.float32)
        b=np.unpackbits(packed[ids],axis=1)[:,:nbits].astype(np.float32)
        return torch.from_numpy(a).to(device),torch.from_numpy(b).to(device)
    start=time.monotonic();history=[]
    for epoch in range(epochs):
        net.train();order=rng.permutation(indices);total=0.
        for begin in range(0,len(order),batch_size):
            ids=order[begin:begin+batch_size];xb,yb=tensors(ids)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda' if cuda else 'cpu',dtype=torch.float16 if cuda else torch.bfloat16,enabled=cuda):
                loss=loss_fn(net(xb),yb)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            scaler.step(optimizer);scaler.update();total+=float(loss.detach().cpu())*len(ids)
        history.append(total/len(indices))
        print('V3_EPOCH '+str(epoch+1)+'/'+str(epochs)+' loss='+str(round(history[-1],6)),flush=True)
    with torch.no_grad():net[0].weight[:,torch.as_tensor(~active,device=device)]=0
    _save(output,net,feature_dim,head_sizes)
    model=MultiFingerprintModel(output);sample=np.asarray(x[indices[:8],:feature_dim],dtype=np.float32)
    net=net.cpu().eval()
    with torch.no_grad():expected=net(torch.from_numpy(sample)).numpy()
    parity=float(np.max(np.abs(model.logits(sample)-expected)))
    if parity>1e-3:raise RuntimeError('NumPy export parity failed: '+str(parity))
    return {'training_rows':len(indices),'epochs':epochs,'head_sizes':list(head_sizes),'feature_dim':feature_dim,
            'hidden':hidden,'batch_size':batch_size,'learning_rate':learning_rate,'seed':seed,
            'device':device,'amp':cuda,'loss_history':history,'parameters':sum(p.numel() for p in net.parameters()),
            'export_max_absolute_error':parity,'seconds':time.monotonic()-start,'official_score':None}

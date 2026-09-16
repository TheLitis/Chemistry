"""R06 controlled residual architecture search on a common frozen V3 anchor.

These are compact implementations, not reproductions of published pretrained
systems. Peak-set models consume the top 32 peaks of up to three acquisitions.
All models start with zero output correction and use the same anchor knowledge.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURES = ('lowrank_linear','shallow_mlp','residual_mlp','dual_branch',
                 'adduct_moe','mass_conv','deep_sets','peak_transformer',
                 'latent_attention','peak_graph','massset_hybrid')
TOKEN_DIM=16
PEAKS_PER_VIEW=32
HEAD_SIZES=(2048,4096,8192)


def _peaks(row):
    mz=np.asarray(row['ms2_mzs'],dtype=np.float64)
    it=np.asarray(row['ms2_normalized_intensities'],dtype=np.float64)
    if mz.ndim!=1 or mz.shape!=it.shape or not len(mz) or not np.isfinite(mz).all() or not np.isfinite(it).all() or (mz<=0).any() or (it<0).any() or it.sum()<=0:
        raise ValueError('Invalid peak arrays')
    good=it>0;mz,it=mz[good],it[good]
    order=np.argsort(mz,kind='stable');mz,it=mz[order],it[order]
    unique,inverse=np.unique(mz,return_inverse=True)
    it=np.bincount(inverse,weights=it);it/=it.sum()
    return unique,it


def spectrum_digest(row):
    mz,it=_peaks(row)
    payload={'mz':np.round(mz,8).tolist(),'intensity':np.round(it,12).tolist(),
             'precursor':round(float(row['precursor_mz']),8),'adduct':str(row['adduct']),
             'energy':row.get('collision_energy_ev')}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def peak_tokens(row):
    from .production import ion
    mz,it=_peaks(row);prec=float(row['precursor_mz'])
    n,z,shift=ion(row['adduct']);neutral=(prec*abs(z)-shift)/n
    if not math.isfinite(neutral) or neutral<=0:raise ValueError('Invalid precursor')
    good=mz<prec-.01;mz,it=mz[good],it[good]
    out=np.zeros((PEAKS_PER_VIEW,TOKEN_DIM),dtype=np.float32)
    if not len(mz):return out
    order=np.lexsort((mz,-it))[:PEAKS_PER_VIEW];mz,it=mz[order],it[order]
    order=np.argsort(mz);mz,it=mz[order],it[order];loss=prec-mz
    energy=row.get('collision_energy_ev')
    energy=[] if energy is None else ([energy] if isinstance(energy,(int,float)) else energy)
    ev=[float(v) for v in energy if v is not None and np.isfinite(float(v))]
    ef=float(np.mean(ev))/200 if ev else 0.;es=(max(ev)-min(ev))/200 if ev else 0.
    fm=mz-np.floor(mz);fl=loss-np.floor(loss)
    out[:len(mz)]=np.column_stack((mz/2048,loss/2048,np.sqrt(it),np.log1p(100*it)/np.log(101),
        np.full(len(mz),neutral/1200),np.full(len(mz),prec/1200),fm,fl,
        np.sin(2*np.pi*fm),np.cos(2*np.pi*fm),np.sin(2*np.pi*fl),np.cos(2*np.pi*fl),
        np.full(len(mz),float(z>0)),np.full(len(mz),float(z<0)),np.full(len(mz),ef),np.full(len(mz),es)))
    return out


def partition_keys(keys,excluded,n_screen=512,n_select=512,n_audit=2048):
    pool=sorted(set(keys)-set(excluded),key=lambda k:hashlib.sha256(('architecture-r06:'+k).encode()).digest())
    if min(n_screen,n_select,n_audit)<1 or len(pool)<n_screen+n_select+n_audit:
        raise ValueError('Insufficient fresh heldout molecular keys')
    return pool[:n_screen],pool[n_screen:n_screen+n_select],pool[n_screen+n_select:n_screen+n_select+n_audit]


def hard_negative_indices(masses,keys,training_indices,count=7):
    masses=np.asarray(masses,dtype=np.float64);ids=np.asarray(training_indices,dtype=np.int64)
    if count<1 or len(ids)!=len(set(ids.tolist())):raise ValueError('Invalid negative sampling inputs')
    order=ids[np.argsort(masses[ids],kind='stable')];mm=masses[order]
    out=np.full((len(ids),count),-1,dtype=np.int32)
    for row,i in enumerate(ids):
        tol=max(.005,masses[i]*20e-6);lo,hi=np.searchsorted(mm,[masses[i]-tol,masses[i]+tol],side='left')
        hi=np.searchsorted(mm,masses[i]+tol,side='right')
        candidates=[int(j) for j in order[lo:hi] if keys[int(j)]!=keys[int(i)]]
        candidates.sort(key=lambda j:(abs(masses[j]-masses[i]),keys[j],j))
        seen=set();chosen=[]
        for j in candidates:
            if keys[j] not in seen:chosen.append(j);seen.add(keys[j])
            if len(chosen)==count:break
        out[row,:len(chosen)]=chosen
    return out


class ResidualBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),nn.Dropout(.1),nn.Linear(width*2,width))
    def forward(self,x):return x+self.net(x)*.5


def pool(values,valid):
    mask=valid[...,None]
    mean=(values*mask).sum(1)/mask.sum(1).clamp_min(1)
    largest=values.masked_fill(~mask,-1e4).amax(1)
    largest=torch.where(valid.any(1,keepdim=True),largest,torch.zeros_like(largest))
    return torch.cat((mean,largest),-1)


class PeakEncoder(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.embed=nn.Sequential(nn.Linear(TOKEN_DIM,64),nn.GELU(),nn.Linear(64,64))
        if kind=='peak_transformer':
            layer=nn.TransformerEncoderLayer(64,4,128,dropout=.1,batch_first=True,norm_first=True)
            self.attention=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        if kind in ('latent_attention','massset_hybrid'):
            self.queries=nn.Parameter(torch.randn(16,64)*.02)
            self.cross=nn.MultiheadAttention(64,4,dropout=.1,batch_first=True)
            self.latent=ResidualBlock(64)
        if kind=='peak_graph':
            self.register_buffer('loss_centers',torch.tensor([17.026549,18.010565,27.994915,43.989829,97.976896,162.052824]))
            self.message=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,64))
    def forward(self,tokens):
        valid=tokens[:,:,2]>0;safe=valid.clone();safe[:,0]|=~safe.any(1)
        h=self.embed(tokens)
        if self.kind=='peak_transformer':
            h=self.attention(h,src_key_padding_mask=~safe)
        elif self.kind in ('latent_attention','massset_hybrid'):
            q=self.queries[None].expand(len(tokens),-1,-1)
            h,_=self.cross(q,h,h,key_padding_mask=~safe,need_weights=False)
            h=self.latent(h);active=valid.any(1)[:,None].expand(-1,h.shape[1])
            return pool(h,active)
        elif self.kind=='peak_graph':
            mz=tokens[:,:,0].float()*2048
            delta=(mz[:,:,None]-mz[:,None,:]).abs()
            weights=torch.exp(-.5*((delta[:,:,:,None]-self.loss_centers)/.05)**2).sum(-1)
            weights=weights*valid[:,:,None]*valid[:,None,:]
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1)
            message=weights.to(h.dtype)@h
            h=h+self.message(torch.cat((h,message),-1))
        return pool(h,valid)


class Architecture(nn.Module):
    def __init__(self,name,output_dim=14336):
        super().__init__()
        if name not in ARCHITECTURES or output_dim<1:raise ValueError('Unknown architecture/configuration')
        self.name=name;self.output_dim=output_dim
        self.input_norm=nn.LayerNorm(8200)
        if name=='lowrank_linear':
            self.encoder=nn.Linear(8200,128,bias=False);width=128
        elif name in ('shallow_mlp','residual_mlp'):
            blocks=[nn.Linear(8200,256),nn.GELU(),nn.Dropout(.1)]
            if name=='residual_mlp':blocks.extend(ResidualBlock(256) for _ in range(3))
            self.encoder=nn.Sequential(*blocks);width=256
        elif name=='dual_branch':
            self.fragments=nn.Sequential(nn.Linear(4096,128),nn.GELU())
            self.losses=nn.Sequential(nn.Linear(4096,128),nn.GELU());width=264
        elif name=='adduct_moe':
            self.experts=nn.ModuleList(nn.Sequential(nn.Linear(8200,128),nn.GELU()) for _ in range(3))
            self.gate=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,3));width=128
        elif name=='mass_conv':
            self.conv=nn.Sequential(nn.Conv1d(4,32,17,stride=8,padding=8),nn.GELU(),nn.Conv1d(32,64,9,stride=4,padding=4),nn.GELU(),nn.AdaptiveAvgPool1d(8))
            self.encoder=nn.Sequential(nn.Linear(520,256),nn.GELU());width=256
        else:
            self.peaks=PeakEncoder(name)
            if name=='massset_hybrid':
                self.dense=nn.Sequential(nn.Linear(8200,256),nn.GELU(),ResidualBlock(256),nn.Linear(256,128))
                self.gate=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,128),nn.Sigmoid())
                width=264
            else:width=136
        self.head=nn.Linear(width,output_dim)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x,tokens):
        if x.ndim!=2 or x.shape[1]!=8200 or tokens.ndim!=3 or tokens.shape[0]!=x.shape[0] or tokens.shape[2]!=16:
            raise ValueError('Incorrect architecture input shape')
        meta=x[:,4096:4104];a=self.input_norm(x)
        if self.name in ('lowrank_linear','shallow_mlp','residual_mlp'):h=self.encoder(a)
        elif self.name=='dual_branch':
            f=torch.cat((a[:,:2048],a[:,4104:6152]),-1);l=torch.cat((a[:,2048:4096],a[:,6152:8200]),-1)
            h=torch.cat((self.fragments(f),self.losses(l),meta),-1)
        elif self.name=='adduct_moe':
            experts=torch.stack([e(a) for e in self.experts],1);h=(experts*self.gate(meta).softmax(-1)[:,:,None]).sum(1)
        elif self.name=='mass_conv':
            channels=torch.stack((a[:,:2048],a[:,2048:4096],a[:,4104:6152],a[:,6152:8200]),1)
            h=self.encoder(torch.cat((self.conv(channels).flatten(1),meta),-1))
        else:
            peaks=self.peaks(tokens)
            if self.name=='massset_hybrid':h=torch.cat((self.dense(a),peaks*self.gate(meta),meta),-1)
            else:h=torch.cat((peaks,meta),-1)
        return self.head(h)


def make_model(name,output_dim=14336):return Architecture(name,output_dim)


def ranking_loss(logits,positives,negatives,valid,head_sizes=HEAD_SIZES):
    if negatives.shape[:2]!=valid.shape or negatives.shape[0]!=len(logits) or positives.shape!=logits.shape:
        raise ValueError('Invalid ranking-loss inputs')
    active=valid.any(1)
    if not active.any():return logits.sum()*0
    fp=torch.cat((positives[:,None],negatives),1).float();v=logits.float()
    offset=0;parts=[]
    for size in head_sizes:
        values=(fp[:,:,offset:offset+size]*v[:,None,offset:offset+size]).sum(-1)
        mask=torch.cat((torch.ones_like(valid[:,:1]),valid),1)
        mean=(values*mask).sum(1,keepdim=True)/mask.sum(1,keepdim=True)
        var=(((values-mean)*mask)**2).sum(1,keepdim=True)/mask.sum(1,keepdim=True)
        parts.append((values-mean)/(var+.01).sqrt());offset+=size
    score=torch.stack(parts).mean(0);score[:,1:]=score[:,1:].masked_fill(~valid,-1e4)
    return F.cross_entropy(score[active],torch.zeros(int(active.sum()),device=logits.device,dtype=torch.long))


def save_model(path,model):
    path=Path(path)
    if path.exists():raise FileExistsError(str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    arrays={'format':np.array(6),'architecture':np.array(model.name),'output_dim':np.array(model.output_dim)}
    for k,v in model.state_dict().items():
        a=v.detach().cpu().numpy()
        if not np.isfinite(a).all():raise ValueError('Nonfinite model tensor')
        arrays['state::'+k]=a
    fd,name=tempfile.mkstemp(dir=path.parent,suffix='.npz');os.close(fd)
    try:np.savez_compressed(name,**arrays);os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def load_model(path):
    with np.load(path,allow_pickle=False) as z:
        if int(z['format'])!=6:raise ValueError('Wrong architecture model format')
        model=make_model(str(z['architecture']),int(z['output_dim']))
        state={k[7:]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('state::')}
        if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite weights')
        model.load_state_dict(state,strict=True)
    return model


def rank_metrics(ranks):
    r=np.asarray(ranks,dtype=np.int64)
    if not len(r) or (r<0).any() or (r>25).any():raise ValueError('Invalid evaluation ranks')
    return {'molecules':len(r),'mrr_at_25':float(np.where(r>0,1/np.maximum(r,1),0).mean()),
            'top1':float((r==1).mean()),'recall_at_25':float((r>0).mean())}

"""Common training and evaluation primitives for R06. No provider access."""
from __future__ import annotations
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .architectures import HEAD_SIZES,make_model,load_model,save_model,ranking_loss
from .production import sha256,write_json,mass_candidates


class TorchAnchor(nn.Module):
    def __init__(self,path):
        super().__init__()
        from .model_v3 import MultiFingerprintModel
        model=MultiFingerprintModel(path);self.head_sizes=model.head_sizes
        for name in ('w1','b1','w2','b2'):self.register_buffer(name,torch.from_numpy(getattr(model,name).copy()))
    def forward(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            return F.linear(F.relu(F.linear(x.float(),self.w1,self.b1)),self.w2,self.b2)


def pool_acquisitions(views,tokens,masses,counts,*,budget=3,rng=None):
    views=np.asarray(views,dtype=np.float32);tokens=np.array(tokens,dtype=np.float32,copy=True)
    counts=np.asarray(counts,dtype=np.int64);masses=np.asarray(masses)
    if views.ndim!=3 or views.shape[1:]!=(3,8200) or tokens.shape!=(len(views),3,32,16) or np.any((counts<1)|(counts>3)):
        raise ValueError('Malformed acquisition batch')
    mask=np.arange(3)[None,:]<counts[:,None]
    if rng is not None:
        single=rng.random(len(views))<.5;slot=np.floor(rng.random(len(views))*counts).astype(int)
        mask[single]=np.arange(3)[None,:]==slot[single,None]
    elif budget==1:mask[:,1:]=False
    elif budget!=3:raise ValueError('Expected one or three acquisitions')
    x=(views*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
    tokens*=mask[:,:,None,None]
    mass=np.nanmedian(np.where(mask,masses,np.nan),axis=1)
    if not np.isfinite(x).all() or not np.isfinite(tokens).all() or not np.isfinite(mass).all():raise ValueError('Nonfinite acquisition batch')
    return x,tokens.reshape(len(views),96,16),mass


class Data:
    def __init__(self,folder):
        folder=Path(folder);self.folder=folder
        self.keys=json.loads((folder/'keys.json').read_text());self.index={k:i for i,k in enumerate(self.keys)}
        for name in ('views','tokens','masses','view_counts','targets','all_features','all_masses','strata','catalog_indices'):
            setattr(self,name,np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False))
    def batch(self,ids,*,budget=3,rng=None):
        ids=np.asarray(ids,dtype=np.int64)
        x,t,m=pool_acquisitions(self.views[ids],self.tokens[ids],self.masses[ids],self.view_counts[ids],budget=1 if budget==1 else 3,rng=rng)
        if budget=='all':x=np.array(self.all_features[ids],dtype=np.float32);m=np.array(self.all_masses[ids])
        return x,t,m


def classification_loss(logits,y,posweight,heads=HEAD_SIZES):
    values=[];offset=0
    for size in heads:
        z=logits[:,offset:offset+size].float();target=y[:,offset:offset+size]
        bce=F.binary_cross_entropy_with_logits(z,target,pos_weight=posweight[offset:offset+size])
        p=z.sigmoid();cos=(p*target).sum(-1)/(p.norm(dim=-1)*target.norm(dim=-1)).clamp_min(1e-8)
        values.append(bce+.5*(1-cos).mean());offset+=size
    return torch.stack(values).mean()


def train_trial(data,anchor,ids,configuration,path,*,epochs=6,seed=26091606,initial=None,negatives=None,device='cuda',batch_size=256,lr=.0003):
    path=Path(path);meta=path.with_suffix('.json')
    identity={'configuration':configuration,'epochs':epochs,'seed':seed,'n':len(ids),'initial_sha256':sha256(initial) if initial else None,
              'batch_size':batch_size,'learning_rate':lr,'indices_sha256':__import__('hashlib').sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()}
    if path.exists():
        old=json.loads(meta.read_text())
        if old['identity']!=identity or old['checkpoint_sha256']!=sha256(path):raise ValueError('A different trial already exists')
        return old
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA required for this stage')
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    net=load_model(initial) if initial else make_model(configuration['architecture'])
    net=net.to(device);anchor=anchor.to(device).eval()
    ids=np.asarray(ids,dtype=np.int64)
    if not len(ids) or len(set(ids.tolist()))!=len(ids):raise ValueError('Invalid training identities')
    frequency=np.zeros(sum(HEAD_SIZES),np.float64)
    active=np.zeros(8200,bool)
    for start in range(0,len(ids),1024):
        ii=ids[start:start+1024];frequency+=np.unpackbits(data.targets[ii],axis=1).sum(0)
        active|=np.any(np.asarray(data.views[ii])!=0,axis=(0,1))
    prevalence=np.clip(frequency/len(ids),.005,.995)
    weights=np.minimum(20,(1-prevalence)/prevalence).astype('f4')
    if configuration['loss']=='unweighted':weights[:]=1
    pw=torch.from_numpy(weights).to(device)
    active_tensor=torch.from_numpy(active.astype('f4')).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    cuda=device.startswith('cuda');scaler=torch.amp.GradScaler('cuda',enabled=cuda)
    if cuda:torch.cuda.reset_peak_memory_stats()
    started=time.monotonic();history=[];skipped=0;steps=0
    negative_map={int(i):row for i,row in zip(ids,negatives)} if negatives is not None else {}
    for epoch in range(epochs):
        net.train();order=rng.permutation(ids);loss_total=0.
        for start in range(0,len(order),batch_size):
            ii=order[start:start+batch_size];x,t,_=data.batch(ii,rng=rng)
            xb=torch.from_numpy(x).to(device);tb=torch.from_numpy(t).to(device)
            y=torch.from_numpy(np.unpackbits(data.targets[ii],axis=1).astype('f4')).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():base=anchor(xb)
            with torch.autocast('cuda' if cuda else 'cpu',dtype=torch.float16 if cuda else torch.bfloat16,enabled=cuda):delta=net(xb*active_tensor,tb)
            logits=base+delta.float();loss=classification_loss(logits,y,pw)+1e-4*delta.float().square().mean()
            if configuration['loss']=='rank':
                nj=np.stack([negative_map[int(i)] for i in ii]);ok=nj>=0
                neg=np.unpackbits(data.targets[np.maximum(nj,0)],axis=-1).astype('f4')
                loss=loss+.3*ranking_loss(logits,y,torch.from_numpy(neg).to(device),torch.from_numpy(ok).to(device))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();skipped+=int(scaler.get_scale()<old_scale)
            steps+=1;loss_total+=float(loss.detach().cpu())*len(ii)
            fraction=(epoch+(start+len(ii))/len(order))/epochs
            for group in optimizer.param_groups:group['lr']=lr*(.2+.8*.5*(1+math.cos(math.pi*fraction)))
        history.append(loss_total/len(ids))
        print('R06_EPOCH '+json.dumps({'trial':path.stem,'epoch':epoch+1,'loss':history[-1],'seconds':round(time.monotonic()-started,2)}),flush=True)
    net.register_buffer('training_active_mask',active_tensor.detach().cpu())
    del net._buffers['training_active_mask']
    save_model(path,net)
    np.save(path.with_suffix('.active.npy'),active.astype(np.uint8))
    net=net.eval();samplex,samplet,_=data.batch(ids[:4])
    samplex=samplex*active
    with torch.no_grad():expected=net(torch.from_numpy(samplex).to(device),torch.from_numpy(samplet).to(device)).cpu().numpy()
    restored=load_model(path).to(device).eval()
    with torch.no_grad():actual=restored(torch.from_numpy(samplex).to(device),torch.from_numpy(samplet).to(device)).cpu().numpy()
    parity=float(np.max(abs(expected-actual)))
    if parity>1e-4:raise RuntimeError('Architecture export parity failed')
    report={'status':'completed','identity':identity,'loss_history':history,'steps':steps,'amp_skipped_steps':skipped,
            'parameters':sum(p.numel() for p in net.parameters()),'seconds':time.monotonic()-started,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated() if cuda else None,'export_max_abs_error':parity,
            'checkpoint_sha256':sha256(path),'mask_sha256':sha256(path.with_suffix('.active.npy')),
            'negative_examples_fraction':float(np.any(negatives>=0,axis=1).mean()) if negatives is not None else None}
    write_json(meta,report)
    del net,restored,optimizer
    if cuda:torch.cuda.empty_cache()
    return report


def predict_trial(data,anchor,ids,path=None,*,budget=3,pooling='early',device='cuda',batch_size=128):
    if pooling not in ('early','late_probability'):raise ValueError('Unknown pooling')
    if budget=='all' and pooling!='early':raise ValueError('All-acquisition late fusion requires full raw acquisitions; not stored in this cache')
    net=load_model(path).to(device).eval() if path else None
    active=np.load(Path(path).with_suffix('.active.npy'),allow_pickle=False).astype('f4') if path else np.ones(8200,'f4')
    anchor=anchor.to(device).eval();output=[];observed=[]
    with torch.no_grad():
        for start in range(0,len(ids),batch_size):
            ii=np.asarray(ids[start:start+batch_size]);x,t,m=data.batch(ii,budget=budget);observed.extend(m)
            def predict(x_,t_):
                xb=torch.from_numpy(np.asarray(x_,dtype='f4')).to(device)
                base=anchor(xb)
                if net is not None:base=base+net(xb*torch.from_numpy(active).to(device),torch.from_numpy(np.asarray(t_,dtype='f4')).to(device))
                return base
            if pooling=='early' or budget==1:result=predict(x,t)
            else:
                n=data.view_counts[ii];prob=None
                for slot in range(3):
                    valid=(slot<n).astype('f4')
                    tt=np.zeros((len(ii),96,16),'f4');tt[:,:32]=data.tokens[ii,slot]
                    p=predict(np.asarray(data.views[ii,slot],dtype='f4'),tt).sigmoid()*torch.from_numpy(valid[:,None]).to(device)
                    prob=p if prob is None else prob+p
                prob=(prob/torch.from_numpy(np.asarray(n,dtype='f4')[:,None]).to(device)).clamp(1e-7,1-1e-7)
                result=prob.log()-torch.log1p(-prob)
            output.append(result.cpu().numpy())
    if net is not None:del net
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return np.concatenate(output),np.array(observed)


def fingerprint_basis(logits,bits,heads=HEAD_SIZES):
    values=[];offset=0
    for size in heads:
        s=np.asarray(bits[:,offset:offset+size],dtype='f4')@np.clip(np.asarray(logits[offset:offset+size],dtype='f4'),-16.11809555,16.11809555)
        s=s.astype('f8');values.append((s-s.mean())/max(float(s.std()),1e-8) if len(s) else s);offset+=size
    return np.stack(values,axis=1)


def rank_from_basis(basis,keys,truth,masses,observed,weights,sigma_ppm=5):
    if sigma_ppm is not None and sigma_ppm<=0:raise ValueError('Positive mass sigma required')
    if not len(keys):return 0
    w=np.asarray(weights,dtype='f8')
    if (w<0).any() or w.sum()<=0:raise ValueError('Invalid fingerprint weights')
    scores=basis@(w/w.sum())
    if sigma_ppm is not None:scores=scores-.5*((np.asarray(masses)-observed)/max(.001,abs(observed)*sigma_ppm*1e-6))**2
    seen=set()
    for i in np.argsort(-scores,kind='stable'):
        if keys[i] in seen:continue
        seen.add(keys[i])
        if len(seen)>25:break
        if keys[i]==truth:return len(seen)
    return 0


def evaluate_predictions(predictions,observed,truth,catalog,packed,*,weights=(.5,.25,.25),sigma_ppm=5,return_basis=False):
    masses=np.array([r[3] for r in catalog]);ranks=[];coverage=[];bases=[];counts=[]
    for z,m,k in zip(predictions,observed,truth):
        idx=mass_candidates(masses,m);keys=[catalog[int(i)][2] for i in idx]
        seen=set();keep=[]
        for i,key in zip(idx,keys):
            if key not in seen:seen.add(key);keep.append(int(i))
        idx=np.array(keep,dtype=np.int64);keys=[catalog[int(i)][2] for i in idx]
        bits=np.unpackbits(packed[idx],axis=1);b=fingerprint_basis(z,bits)
        rank=rank_from_basis(b,keys,k,masses[idx],m,weights,sigma_ppm)
        ranks.append(rank);coverage.append(k in keys);counts.append(len(keys))
        if return_basis:bases.append((b,keys,k,masses[idx],m))
    return {'ranks':ranks,'coverage':coverage,'candidate_counts':counts,'basis':bases}
